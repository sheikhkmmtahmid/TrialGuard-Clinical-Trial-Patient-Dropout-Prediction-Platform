"""
Feature engineering and synthetic data generation for TrialGuard.
Converts raw patient/visit records into ML-ready feature matrices.
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger('core')

SCALER_PATH = Path(__file__).resolve().parents[2] / 'ml_models' / 'scaler.pkl'
AACT_RATES_PATH = Path(__file__).resolve().parent / 'aact_dropout_rates.json'

# Maps the Trial model's phase codes (I, II, III, IV) to AACT's phase labels
# (PHASE1, PHASE2, PHASE3, PHASE4) so we can look up a real dropout rate to
# calibrate against. See core/utils/aact_benchmark.py and
# docs/aact_validation_report.md for where these numbers come from.
_PHASE_TO_AACT = {'I': 'PHASE1', 'II': 'PHASE2', 'III': 'PHASE3', 'IV': 'PHASE4'}

_FALLBACK_DROPOUT_RATE = 0.20  # AACT's overall mean, used if the calibration table is missing


def _load_aact_rates() -> dict:
    if AACT_RATES_PATH.exists():
        return json.loads(AACT_RATES_PATH.read_text())
    return {}


def target_dropout_rate(phase: str = None, therapeutic_area: str = None) -> float:
    """
    Look up a real-world dropout rate to calibrate synthetic generation
    against, given a trial phase (I, II, III, IV) and therapeutic area.
    Falls back from phase+area, to phase only, to the overall AACT mean,
    to a hardcoded default if the calibration table isn't present.
    """
    rates = _load_aact_rates()
    if not rates:
        return _FALLBACK_DROPOUT_RATE

    aact_phase = _PHASE_TO_AACT.get(phase)
    if aact_phase and therapeutic_area:
        key = f'{aact_phase}|{therapeutic_area}'
        if key in rates.get('by_phase_and_therapeutic_area', {}):
            return rates['by_phase_and_therapeutic_area'][key]
    if aact_phase and aact_phase in rates.get('by_phase', {}):
        return rates['by_phase'][aact_phase]
    return rates.get('overall_mean_dropout_rate', _FALLBACK_DROPOUT_RATE)

# age_group and distance_bucket used to be included here too, they were
# just age and distance_to_site_km rounded into buckets. Keeping both the
# raw number and a bucketed copy of the same fact means the model (and
# SHAP, when explaining it) has to arbitrarily split credit between two
# features that are really saying the same thing, which is a well known
# cause of unstable explanations when features are correlated with each
# other. Dropping the redundant bucketed copies removes that instability
# without losing any information, the raw values are still there.
FEATURE_COLUMNS = [
    'age', 'gender_encoded', 'ethnicity_encoded',
    'condition_severity_encoded', 'distance_to_site_km',
    'employment_encoded', 'prior_dropout_history',
    'visit_number', 'cumulative_missed_visits', 'visit_frequency_rate',
    'days_since_last_visit', 'days_between_visits_mean', 'days_between_visits_std',
    'adverse_events_count', 'adverse_event_rate', 'adverse_event_trend',
    'medication_adherence_score', 'medication_adherence_trend',
    'quality_of_life_score', 'qol_score_trend',
    'early_dropout_signal', 'high_adverse_event_flag', 'low_adherence_flag',
]

# Collected and engineered like every other feature above, but never fed to
# the scaler or any model. Every real-world source checked for this project
# (PDS, MUSIC, UCI Heart Failure, ImmPort, four independent data ecosystems)
# has never once had a real value for these three, so their apparent
# importance in earlier SHAP runs was only ever demonstrated on synthetic
# data that was built to make them predictive, never confirmed against a
# real patient. Rather than deleting them, they stay collected here so that
# if a real source with genuine values for one of these ever turns up, they
# can be moved back into MODEL_FEATURE_COLUMNS and retrained on, without
# having to rebuild the collection or encoding logic from scratch.
# See docs/pds_validation_report.md, "Recommendation: drop three features".
RESERVED_FEATURE_COLUMNS = ['distance_to_site_km', 'employment_encoded', 'prior_dropout_history']

# The list actually used to fit the scaler and every model. Keep this, not
# FEATURE_COLUMNS, as the single source of truth for what the model can see.
MODEL_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in RESERVED_FEATURE_COLUMNS]

_GENDER_MAP = {'M': 0, 'F': 1, 'O': 2, 'U': 3}
_ETHNICITY_MAP = {'white': 0, 'black': 1, 'hispanic': 2, 'asian': 3, 'other': 4, 'unknown': 5}
_SEVERITY_MAP = {'mild': 0, 'moderate': 1, 'severe': 2}
_EMPLOYMENT_MAP = {'employed': 0, 'unemployed': 1, 'retired': 2, 'student': 3, 'other': 4}


def _linear_trend(values: list) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    if np.std(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def engineer_features_for_patient(patient, visits_qs) -> pd.DataFrame:
    """
    Build one feature row per visit for a single patient.
    Returns a DataFrame with FEATURE_COLUMNS + 'dropout_status' + 'days_to_event'.
    """
    visits = list(visits_qs.order_by('visit_number'))
    if not visits:
        return pd.DataFrame()

    rows = []
    ae_history, adh_history, qol_history, days_history = [], [], [], []

    for v in visits:
        ae_history.append(v.adverse_events_count)
        adh_history.append(v.medication_adherence_score)
        qol_history.append(v.quality_of_life_score)
        days_history.append(v.days_since_last_visit)

        n = len(ae_history)
        total_ae = sum(ae_history)
        ae_rate = total_ae / n

        consec_missed = 0
        missed_so_far = list(range(n))
        for i in range(len(visits) - 1, -1, -1):
            if visits[i].missed_visits_to_date > 0:
                consec_missed += 1
            else:
                break

        # Days elapsed as of THIS visit, not the patient's final trial
        # duration. patient.days_to_event() looks at when the patient
        # eventually dropped out (or today, if still active), which is
        # only known in hindsight. Using it here would mean even a
        # patient's very first visit carries a feature that already knows
        # how their story ends, that is a leak, not a real signal, and it
        # would apply to real trial data too, not just synthetic data.
        days_elapsed_at_visit = max((v.visit_date - patient.enrollment_date).days, 1)

        row = {
            'age': patient.age,
            'gender_encoded': _GENDER_MAP.get(patient.gender, 3),
            'ethnicity_encoded': _ETHNICITY_MAP.get(patient.ethnicity, 5),
            'condition_severity_encoded': _SEVERITY_MAP.get(patient.condition_severity, 1),
            'distance_to_site_km': patient.distance_to_site_km,
            'employment_encoded': _EMPLOYMENT_MAP.get(patient.employment_status, 4),
            'prior_dropout_history': int(patient.prior_dropout_history),
            'visit_number': v.visit_number,
            'cumulative_missed_visits': v.missed_visits_to_date,
            'visit_frequency_rate': v.visit_number / days_elapsed_at_visit * 30,
            'days_since_last_visit': v.days_since_last_visit,
            'days_between_visits_mean': float(np.mean(days_history)) if days_history else 0.0,
            'days_between_visits_std': float(np.std(days_history)) if len(days_history) > 1 else 0.0,
            'adverse_events_count': v.adverse_events_count,
            'adverse_event_rate': ae_rate,
            'adverse_event_trend': _linear_trend(ae_history),
            'medication_adherence_score': v.medication_adherence_score,
            'medication_adherence_trend': _linear_trend(adh_history),
            'quality_of_life_score': v.quality_of_life_score,
            'qol_score_trend': _linear_trend(qol_history),
            'early_dropout_signal': int(consec_missed >= 2),
            'high_adverse_event_flag': int(ae_rate > 3.0),
            'low_adherence_flag': int(v.medication_adherence_score < 60.0),
            'dropout_status': int(patient.dropout_status),
            'days_to_event': patient.days_to_event(),
            'patient_id': patient.pk,
            'visit_id': v.pk,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_full_dataset() -> pd.DataFrame:
    """Pull all patients + visits from DB and build the ML feature matrix."""
    from core.models import Patient

    all_rows = []
    patients = Patient.objects.prefetch_related('visits').select_related('trial').all()
    for patient in patients:
        df = engineer_features_for_patient(patient, patient.visits.all())
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df = full_df.fillna(0)
    return full_df


def fit_and_save_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[MODEL_FEATURE_COLUMNS])
    dump(scaler, SCALER_PATH)
    logger.info("Scaler saved to %s", SCALER_PATH)
    return scaler


def load_scaler() -> StandardScaler:
    return load(SCALER_PATH)


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None) -> np.ndarray:
    if scaler is None:
        scaler = load_scaler()
    return scaler.transform(df[MODEL_FEATURE_COLUMNS])


def generate_synthetic_patients(
    n: int = 5000, phase: str = None, therapeutic_area: str = None, seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic clinical trial patient data using realistic distributions.
    Falls back to pure numpy synthesis if SDV is unavailable.

    If phase and/or therapeutic_area are given, the overall dropout rate for
    this batch is calibrated to match the real rate observed in AACT for that
    combination (see target_dropout_rate above), rather than using a single
    fixed rate for every trial. Individual patient risk (severity, distance,
    prior dropout history, age) still determines who is more likely to drop
    out relative to their peers in the same batch, calibration only sets the
    overall rate, not who specifically drops out.

    seed should differ between calls that are meant to represent different
    trials, otherwise two calls with the same n produce identical patients.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 80, size=n).astype(float)
    gender = rng.choice([0, 1, 2], size=n, p=[0.48, 0.48, 0.04])
    ethnicity = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.60, 0.13, 0.18, 0.06, 0.02, 0.01])
    severity = rng.choice([0, 1, 2], size=n, p=[0.30, 0.50, 0.20])
    distance = rng.exponential(scale=20, size=n).clip(0.5, 150)
    employment = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.55, 0.10, 0.20, 0.08, 0.07])
    prior_dropout = (rng.uniform(size=n) < 0.12).astype(int)
    days_enrolled = rng.integers(30, 730, size=n).astype(float)

    # latent_risk is a single hidden number per patient, from 0 (healthy,
    # easy trial) to 1 (severe, hard trial), built from their demographics
    # and history. This is the one thing that drives both how a patient's
    # visit trends look (in generate_synthetic_visits below) and how likely
    # they are to drop out. Neither of those two things is generated from
    # the other, they are both generated from this shared, noisy cause,
    # which is what makes a model's job realistic instead of trivial.
    #
    # The noise term below (currently std 0.25) stands in for everything
    # about a real patient that four demographic facts can never capture,
    # their home situation, their mental health that week, how much they
    # trust their coordinator, plain chance. Every regression-style model
    # is written as outcome = formula(known facts) + noise, and the size of
    # that noise term is what caps how well anyone, human or model, could
    # ever predict the outcome from those facts alone. A small noise term
    # means the four facts almost fully determine risk, which is not
    # realistic. This is deliberately large relative to the formula's own
    # spread, so demographics remain informative without being close to
    # deterministic.
    latent_risk = (
        0.3 * (severity / 2.0)
        + 0.2 * (distance / 150.0)
        + 0.2 * prior_dropout
        + 0.15 * (age / 80.0)
        + rng.normal(0, 0.25, size=n)
    ).clip(0, 1)

    # Deciding who actually drops out adds its own independent noise on top
    # of latent_risk, so two patients with the same risk dial do not always
    # get the same outcome. Real trials work this way, a high-risk patient
    # can still finish, a low-risk one can still leave. Without this noise,
    # dropout would just be a rounded copy of latent_risk, and since the
    # visit trends read that same dial, a model could reconstruct dropout
    # almost exactly just by combining them, the same problem as before,
    # one level removed.
    dropout_decision_score = (latent_risk + rng.normal(0, 0.22, size=n)).clip(0, 1)

    # The overall rate is still set by calibrating the threshold to a
    # real-world target instead of a fixed cutoff, so trials in different
    # phases and therapeutic areas end up with realistically different
    # dropout rates instead of all converging on the same number.
    rate = target_dropout_rate(phase, therapeutic_area)
    threshold = np.quantile(dropout_decision_score, 1 - rate)
    dropout_status = (dropout_decision_score > threshold).astype(int)

    dropout_days = np.where(
        dropout_status == 1,
        (rng.beta(2, 5, size=n) * days_enrolled).astype(int),
        days_enrolled.astype(int)
    )

    df = pd.DataFrame({
        'age': age, 'gender_encoded': gender, 'ethnicity_encoded': ethnicity,
        'condition_severity_encoded': severity, 'distance_to_site_km': distance,
        'employment_encoded': employment, 'prior_dropout_history': prior_dropout,
        'latent_risk': latent_risk,
        'days_to_event': dropout_days, 'dropout_status': dropout_status,
    })
    return df


def generate_synthetic_visits(patient_df: pd.DataFrame, visits_per_patient: int = 8) -> pd.DataFrame:
    """
    Synthesise visit-level records linked to patient rows.

    Visit trends are driven by each patient's latent_risk (the hidden risk
    dial from generate_synthetic_patients), not by their final dropout
    label. A patient's symptoms and adherence are not generated by looking
    up whether they eventually dropped out, they are generated from the
    same underlying risk that also, separately and noisily, influenced
    whether they dropped out. If latent_risk is missing (e.g. old cached
    data), fall back to dropout_status so nothing crashes, but that path
    reintroduces the leakage this function exists to avoid.
    """
    rng = np.random.default_rng(42)
    visit_rows = []
    has_latent_risk = 'latent_risk' in patient_df.columns

    for idx, pat in patient_df.iterrows():
        n_visits = rng.integers(2, visits_per_patient + 1)
        ae_baseline = 0.5 + pat['condition_severity_encoded'] * 0.8
        adh_baseline = 80 - pat['condition_severity_encoded'] * 10 - pat['distance_to_site_km'] * 0.2
        qol_baseline = 70 - pat['condition_severity_encoded'] * 8

        risk = pat['latent_risk'] if has_latent_risk else pat['dropout_status']
        ae_trend = rng.normal(0.05 * risk, 0.1)
        adh_trend = rng.normal(-1.5 * risk, 0.8)
        qol_trend = rng.normal(-1.0 * risk, 0.6)

        cumulative_missed = 0
        last_visit_days = 0

        for v in range(1, int(n_visits) + 1):
            days_gap = rng.integers(25, 40)
            missed = int(rng.uniform() < (0.05 + 0.15 * risk))
            cumulative_missed += missed

            #ae = max(0, int(rng.poisson(ae_baseline + ae_trend * v)))
            #fix
            lam = ae_baseline + ae_trend * v
            lam = np.nan_to_num(lam, nan=0.1)
            lam = max(lam, 0.1)

            ae = int(rng.poisson(lam))
            # fix end

            adh = float(np.clip(adh_baseline + adh_trend * v + rng.normal(0, 5), 0, 100))
            qol = float(np.clip(qol_baseline + qol_trend * v + rng.normal(0, 4), 0, 100))

            visit_rows.append({
                'patient_idx': idx,
                'visit_number': v,
                'adverse_events_count': ae,
                'missed_visits_to_date': cumulative_missed,
                'medication_adherence_score': adh,
                'quality_of_life_score': qol,
                'days_since_last_visit': days_gap if v > 1 else 0,
            })
            last_visit_days += days_gap

    return pd.DataFrame(visit_rows)
