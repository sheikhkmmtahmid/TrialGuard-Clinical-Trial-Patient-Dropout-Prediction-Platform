"""
Feature engineering and synthetic data generation for TrialGuard.
Converts raw patient/visit records into ML-ready feature matrices.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger('core')

SCALER_PATH = Path(__file__).resolve().parents[2] / 'ml_models' / 'scaler.pkl'

FEATURE_COLUMNS = [
    'age', 'age_group', 'gender_encoded', 'ethnicity_encoded',
    'condition_severity_encoded', 'distance_to_site_km', 'distance_bucket',
    'employment_encoded', 'prior_dropout_history',
    'visit_number', 'cumulative_missed_visits', 'visit_frequency_rate',
    'days_since_last_visit', 'days_between_visits_mean', 'days_between_visits_std',
    'adverse_events_count', 'adverse_event_rate', 'adverse_event_trend',
    'medication_adherence_score', 'medication_adherence_trend',
    'quality_of_life_score', 'qol_score_trend',
    'early_dropout_signal', 'high_adverse_event_flag', 'low_adherence_flag',
]

_GENDER_MAP = {'M': 0, 'F': 1, 'O': 2, 'U': 3}
_ETHNICITY_MAP = {'white': 0, 'black': 1, 'hispanic': 2, 'asian': 3, 'other': 4, 'unknown': 5}
_SEVERITY_MAP = {'mild': 0, 'moderate': 1, 'severe': 2}
_EMPLOYMENT_MAP = {'employed': 0, 'unemployed': 1, 'retired': 2, 'student': 3, 'other': 4}


def _age_group(age: int) -> int:
    if age < 30:
        return 0
    elif age < 45:
        return 1
    elif age < 60:
        return 2
    return 3


def _distance_bucket(km: float) -> int:
    if km < 10:
        return 0
    elif km < 25:
        return 1
    elif km < 50:
        return 2
    return 3


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

        row = {
            'age': patient.age,
            'age_group': _age_group(patient.age),
            'gender_encoded': _GENDER_MAP.get(patient.gender, 3),
            'ethnicity_encoded': _ETHNICITY_MAP.get(patient.ethnicity, 5),
            'condition_severity_encoded': _SEVERITY_MAP.get(patient.condition_severity, 1),
            'distance_to_site_km': patient.distance_to_site_km,
            'distance_bucket': _distance_bucket(patient.distance_to_site_km),
            'employment_encoded': _EMPLOYMENT_MAP.get(patient.employment_status, 4),
            'prior_dropout_history': int(patient.prior_dropout_history),
            'visit_number': v.visit_number,
            'cumulative_missed_visits': v.missed_visits_to_date,
            'visit_frequency_rate': v.visit_number / max(patient.days_to_event(), 1) * 30,
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
    scaler.fit(df[FEATURE_COLUMNS])
    dump(scaler, SCALER_PATH)
    logger.info("Scaler saved to %s", SCALER_PATH)
    return scaler


def load_scaler() -> StandardScaler:
    return load(SCALER_PATH)


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None) -> np.ndarray:
    if scaler is None:
        scaler = load_scaler()
    return scaler.transform(df[FEATURE_COLUMNS])


def generate_synthetic_patients(n: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic clinical trial patient data using realistic distributions.
    Falls back to pure numpy synthesis if SDV is unavailable.
    """
    rng = np.random.default_rng(42)

    age = rng.integers(18, 80, size=n).astype(float)
    gender = rng.choice([0, 1, 2], size=n, p=[0.48, 0.48, 0.04])
    ethnicity = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.60, 0.13, 0.18, 0.06, 0.02, 0.01])
    severity = rng.choice([0, 1, 2], size=n, p=[0.30, 0.50, 0.20])
    distance = rng.exponential(scale=20, size=n).clip(0.5, 150)
    employment = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.55, 0.10, 0.20, 0.08, 0.07])
    prior_dropout = (rng.uniform(size=n) < 0.12).astype(int)
    days_enrolled = rng.integers(30, 730, size=n).astype(float)

    # Dropout risk influenced by covariates
    risk_score = (
        0.3 * (severity / 2.0)
        + 0.2 * (distance / 150.0)
        + 0.2 * prior_dropout
        + 0.15 * (age / 80.0)
        + rng.normal(0, 0.1, size=n)
    ).clip(0, 1)
    dropout_status = (risk_score > 0.45).astype(int)
    dropout_days = np.where(
        dropout_status == 1,
        (rng.beta(2, 5, size=n) * days_enrolled).astype(int),
        days_enrolled.astype(int)
    )

    df = pd.DataFrame({
        'age': age, 'gender_encoded': gender, 'ethnicity_encoded': ethnicity,
        'condition_severity_encoded': severity, 'distance_to_site_km': distance,
        'employment_encoded': employment, 'prior_dropout_history': prior_dropout,
        'days_to_event': dropout_days, 'dropout_status': dropout_status,
    })
    return df


def generate_synthetic_visits(patient_df: pd.DataFrame, visits_per_patient: int = 8) -> pd.DataFrame:
    """Synthesise visit-level records linked to patient rows."""
    rng = np.random.default_rng(42)
    visit_rows = []

    for idx, pat in patient_df.iterrows():
        n_visits = rng.integers(2, visits_per_patient + 1)
        ae_baseline = 0.5 + pat['condition_severity_encoded'] * 0.8
        adh_baseline = 80 - pat['condition_severity_encoded'] * 10 - pat['distance_to_site_km'] * 0.2
        qol_baseline = 70 - pat['condition_severity_encoded'] * 8

        ae_trend = rng.normal(0.05 * pat['dropout_status'], 0.1)
        adh_trend = rng.normal(-1.5 * pat['dropout_status'], 0.8)
        qol_trend = rng.normal(-1.0 * pat['dropout_status'], 0.6)

        cumulative_missed = 0
        last_visit_days = 0

        for v in range(1, int(n_visits) + 1):
            days_gap = rng.integers(25, 40)
            missed = int(rng.uniform() < (0.05 + 0.15 * pat['dropout_status']))
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
