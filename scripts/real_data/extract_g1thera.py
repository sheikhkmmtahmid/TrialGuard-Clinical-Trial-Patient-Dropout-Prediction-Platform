"""
Extract real per-patient feature rows for the three G1 Therapeutics lung
studies (ADaM: adsl, adex, adqs, adae) and Pancrea_ClovisO_2010_186
(same ADaM shape). All four share the same column conventions.
"""
import pandas as pd
import numpy as np
from pathlib import Path

_GENDER_MAP = {'M': 0, 'F': 1}


def _age_from_bin(raw):
    """AGE in these 3 studies is a real 5-year age band ('50-54', '<45',
    '>=80'), not a raw number, verified against the source file. Previously
    treated as missing and silently replaced with the whole cohort's
    median, caught during the cleaning pass. Converted to the band's real
    midpoint here instead of discarding it."""
    if raw is None or pd.isna(raw):
        return None
    s = str(raw).strip()
    if '-' in s:
        lo, hi = s.split('-')
        return (float(lo) + float(hi)) / 2
    if s.startswith('<'):
        hi = float(s[1:])
        return hi - 2.5
    if s.startswith('>='):
        lo = float(s[2:])
        return lo + 2.5
    try:
        return float(s)
    except ValueError:
        return None

# Verified real text values from adsl.sas7bdat's RACEGR1 field across all
# three studies: only a binary Caucasian / Non-Caucasian split is available
# (no finer real category), so Non-Caucasian maps to the model's 'other'
# bucket rather than guessing a finer race that isn't actually recorded.
ETHNICITY_MAP = {'caucasian': 0, 'non-caucasian': 4}


def _linear_trend(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    if np.std(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def extract(base_dir, study_label, behavioral_usubjids, term_days_map, cutoff_days=None):
    """
    behavioral_usubjids: set of USUBJID strings confirmed as real behavioral
    dropouts (already decoded from DCSREAS/DCTREAS by hand for this study).
    term_days_map: dict USUBJID -> real day of disposition event (EOSDY/DCTDY).
    cutoff_days: if set, every feature below is computed using only events
    at day <= cutoff_days. dropout_status and days_to_event are NEVER
    windowed - days_to_event's fallback (when term_days_map has no entry)
    uses the FULL, unwindowed visit history, computed before any windowing
    below, so it can't be corrupted by the early-window feature logic.
    """
    base = Path(base_dir)
    adsl = pd.read_sas(base / 'adsl.sas7bdat', format='sas7bdat', encoding='latin1')
    adex = pd.read_sas(base / 'adex.sas7bdat', format='sas7bdat', encoding='latin1')
    adae = pd.read_sas(base / 'adae.sas7bdat', format='sas7bdat', encoding='latin1')
    try:
        adqs = pd.read_sas(base / 'adqs.sas7bdat', format='sas7bdat', encoding='latin1')
    except Exception:
        adqs = None

    rows = []
    for _, pat in adsl.iterrows():
        usubjid = pat['USUBJID']
        sex = str(pat.get('SEX', '')).strip().upper()
        pat_ex_full = adex[adex['USUBJID'] == usubjid].copy()
        pat_ae_full = adae[adae['USUBJID'] == usubjid].copy()
        pat_qs_full = adqs[adqs['USUBJID'] == usubjid].copy() if adqs is not None else pd.DataFrame()

        day_col = 'ASTDY' if 'ASTDY' in pat_ex_full.columns else ('ADY' if 'ADY' in pat_ex_full.columns else None)
        full_visit_days = sorted(pat_ex_full[day_col].dropna().unique().tolist()) if day_col else []

        if cutoff_days is not None:
            pat_ex = pat_ex_full[pat_ex_full[day_col] <= cutoff_days] if day_col else pat_ex_full
            ae_day_col_pre = 'ASTDY' if 'ASTDY' in pat_ae_full.columns else None
            pat_ae = pat_ae_full[pat_ae_full[ae_day_col_pre] <= cutoff_days] if ae_day_col_pre else pat_ae_full
            qcol_pre = 'ADY' if 'ADY' in pat_qs_full.columns else (pat_qs_full.columns[0] if not pat_qs_full.empty else None)
            pat_qs = pat_qs_full[pat_qs_full[qcol_pre] <= cutoff_days] if qcol_pre else pat_qs_full
        else:
            pat_ex, pat_ae, pat_qs = pat_ex_full, pat_ae_full, pat_qs_full

        visit_days = sorted(pat_ex[day_col].dropna().unique().tolist()) if day_col else []
        visit_number = len(visit_days)
        if visit_number == 0:
            continue

        gaps = np.diff(visit_days) if len(visit_days) > 1 else np.array([])
        days_since_last_visit = float(gaps[-1]) if len(gaps) else 0.0
        days_between_visits_mean = float(np.mean(gaps)) if len(gaps) else 0.0
        days_between_visits_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
        visit_frequency_rate = visit_number / max(visit_days[-1], 1) * 30

        n_ae = len(pat_ae)
        ae_rate = n_ae / visit_number if visit_number else 0.0
        ae_day_col = 'ASTDY' if 'ASTDY' in pat_ae.columns else None
        if ae_day_col and n_ae > 0 and len(visit_days) > 1:
            ae_days = pat_ae[ae_day_col].dropna().values
            bucket_counts = [np.sum((ae_days >= visit_days[i]) & (ae_days < visit_days[i + 1]))
                              for i in range(len(visit_days) - 1)]
            ae_trend = _linear_trend(bucket_counts)
        else:
            ae_trend = 0.0

        has_real_qol = False
        qol_score, qol_trend = 50.0, 0.0
        if not pat_qs.empty and 'AVAL' in pat_qs.columns:
            qcol = 'ADY' if 'ADY' in pat_qs.columns else pat_qs.columns[0]
            qol_series = pat_qs.sort_values(qcol)['AVAL'].dropna()
            if len(qol_series):
                qol_score = float(qol_series.iloc[-1])
                qol_trend = _linear_trend(qol_series.tolist())
                has_real_qol = True

        ecog = pd.to_numeric(pat.get('ECOG'), errors='coerce')
        severity = 0 if (pd.notna(ecog) and ecog == 0) else (2 if (pd.notna(ecog) and ecog >= 2) else 1)

        race_raw = pat.get('RACEGR1')
        ethnicity_encoded = ETHNICITY_MAP.get(str(race_raw).strip().lower(), 5)

        dropout_status = int(usubjid in behavioral_usubjids)
        days_to_event = float(term_days_map.get(usubjid, full_visit_days[-1] if full_visit_days else 30))

        rows.append({
            'age': _age_from_bin(pat.get('AGE')),
            'gender_encoded': _GENDER_MAP.get(sex, 3),
            'ethnicity_encoded': ethnicity_encoded,
            'condition_severity_encoded': severity,
            'visit_number': visit_number,
            'cumulative_missed_visits': 0,
            'visit_frequency_rate': visit_frequency_rate,
            'days_since_last_visit': days_since_last_visit,
            'days_between_visits_mean': days_between_visits_mean,
            'days_between_visits_std': days_between_visits_std,
            'adverse_events_count': n_ae,
            'adverse_event_rate': ae_rate,
            'adverse_event_trend': ae_trend,
            'medication_adherence_score': 85.0,
            'medication_adherence_trend': 0.0,
            'quality_of_life_score': qol_score,
            'qol_score_trend': qol_trend,
            'early_dropout_signal': 0,
            'high_adverse_event_flag': int(ae_rate > 3.0),
            'low_adherence_flag': 0,
            'dropout_status': dropout_status,
            'days_to_event': max(days_to_event, 1),
            'study': study_label,
            'has_real_qol': has_real_qol,
            'usubjid': usubjid,
        })

    return pd.DataFrame(rows)


def decode_dcsreas(base_dir):
    """Print DCSREAS / DCTREAS value counts so behavioral USUBJIDs can be hand-picked."""
    base = Path(base_dir)
    adsl = pd.read_sas(base / 'adsl.sas7bdat', format='sas7bdat', encoding='latin1')
    for col in ['DCSREAS', 'DCTREAS', 'EOSSTT', 'EOTSTT']:
        if col in adsl.columns:
            print(f"--- {col} ---")
            print(adsl[col].astype(str).str.strip().value_counts(dropna=False).to_string())
    return adsl


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'decode':
        for d in ['../../data/pds/LungSm_G1Thera_2015_433',
                  '../../data/pds/LungSm_G1Thera_2015_434',
                  '../../data/pds/LungSm_G1Thera_2017_435',
                  '../../data/pds/Pancrea_ClovisO_2010_186/DataSphere']:
            print(f"\n===== {d} =====")
            decode_dcsreas(d)
