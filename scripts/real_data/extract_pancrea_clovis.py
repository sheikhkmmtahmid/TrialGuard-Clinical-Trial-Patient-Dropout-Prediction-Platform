import pandas as pd
import numpy as np
from pathlib import Path

BEHAVIORAL = {'withdrawal by subject'}

# Verified real text values from adsl.sas7bdat's RACE field:
# {'WHITE': 327, nan: 19, 'OTHER': 13, 'ASIAN': 5, 'BLACK OR AFRICAN AMERICAN': 3}
ETHNICITY_MAP = {
    'white': 0, 'black or african american': 1, 'asian': 3, 'other': 4,
}


def _linear_trend(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    if np.std(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def extract(cutoff_days=None):
    """
    cutoff_days: if set, every feature below is computed using only events
    at day <= cutoff_days (a "truly prospective" early-window view).
    dropout_status and days_to_event are NEVER windowed - days_to_event
    here falls back to the patient's own last real visit day (no clean
    disposition-day field exists in this extract), which is computed from
    the FULL, unwindowed visit history even when cutoff_days is set, so
    windowing the predictive features never corrupts the real outcome time.
    """
    base = Path('../../data/pds/Pancrea_ClovisO_2010_186/DataSphere')
    adsl = pd.read_sas(base / 'adsl.sas7bdat', format='sas7bdat', encoding='latin1')
    adex = pd.read_sas(base / 'adex.sas7bdat', format='sas7bdat', encoding='latin1')
    adae = pd.read_sas(base / 'adae.sas7bdat', format='sas7bdat', encoding='latin1')
    adqs = pd.read_sas(base / 'adqs.sas7bdat', format='sas7bdat', encoding='latin1')

    adsl['_norm'] = adsl['DSREAS'].astype(str).str.strip().str.lower()
    beh_ids = set(adsl[adsl['_norm'].isin(BEHAVIORAL)]['USUBJID'])

    adqs_health = adqs[adqs['PARAM'].astype(str).str.strip() == 'Your own health state today']

    rows = []
    for _, pat in adsl.iterrows():
        usubjid = pat['USUBJID']
        sex = str(pat.get('SEX', '')).strip().upper()

        pat_ex_full = adex[adex['USUBJID'] == usubjid]
        pat_ae_full = adae[adae['USUBJID'] == usubjid]
        pat_qs_full = adqs_health[adqs_health['USUBJID'] == usubjid]

        # Real outcome timing: always from the FULL, unwindowed visit
        # history, never the early-window one below.
        full_visit_days = sorted(pat_ex_full['EXSTDY'].dropna().unique().tolist())
        days_to_event = float(full_visit_days[-1] if full_visit_days else 30)
        dropout_status = int(usubjid in beh_ids)

        if cutoff_days is not None:
            pat_ex = pat_ex_full[pat_ex_full['EXSTDY'] <= cutoff_days]
            pat_ae = pat_ae_full[pat_ae_full['AESTDY'] <= cutoff_days] if 'AESTDY' in pat_ae_full.columns else pat_ae_full
            pat_qs = pat_qs_full[pat_qs_full['ADY'] <= cutoff_days] if 'ADY' in pat_qs_full.columns else pat_qs_full
        else:
            pat_ex, pat_ae, pat_qs = pat_ex_full, pat_ae_full, pat_qs_full

        visit_days = sorted(pat_ex['EXSTDY'].dropna().unique().tolist())
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
        if 'AESTDY' in pat_ae.columns and n_ae > 0 and len(visit_days) > 1:
            ae_days = pat_ae['AESTDY'].dropna().values
            bucket_counts = [np.sum((ae_days >= visit_days[i]) & (ae_days < visit_days[i + 1]))
                              for i in range(len(visit_days) - 1)]
            ae_trend = _linear_trend(bucket_counts)
        else:
            ae_trend = 0.0

        has_real_qol = False
        qol_score, qol_trend = 50.0, 0.0
        if not pat_qs.empty:
            qol_series = pat_qs.sort_values('ADY')['AVAL'].dropna()
            if len(qol_series):
                qol_score = float(qol_series.iloc[-1])
                qol_trend = _linear_trend(qol_series.tolist())
                has_real_qol = True

        ecog = pd.to_numeric(pat.get('ECOG'), errors='coerce')
        severity = 0 if (pd.notna(ecog) and ecog == 0) else (2 if (pd.notna(ecog) and ecog >= 2) else 1)

        race_raw = pat.get('RACE')
        ethnicity_encoded = ETHNICITY_MAP.get(str(race_raw).strip().lower(), 5)

        # Real adherence proxy: verified EXDELAY (dose delay) and EXREDUC (dose
        # reduction) are clean Y/N fields, not mostly-blank. Percentage of this
        # patient's real dosing events with neither a delay nor a reduction.
        if len(pat_ex) > 0 and 'EXDELAY' in pat_ex.columns and 'EXREDUC' in pat_ex.columns:
            delayed = pat_ex['EXDELAY'].astype(str).str.strip().str.upper() == 'Y'
            reduced = pat_ex['EXREDUC'].astype(str).str.strip().str.upper() == 'Y'
            n_normal = (~(delayed | reduced)).sum()
            medication_adherence_score = 100.0 * n_normal / len(pat_ex)
        else:
            medication_adherence_score = 85.0

        rows.append({
            'age': pat.get('AGE'),
            'gender_encoded': {'M': 0, 'F': 1}.get(sex, 3),
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
            'medication_adherence_score': medication_adherence_score,
            'medication_adherence_trend': 0.0,
            'quality_of_life_score': qol_score,
            'qol_score_trend': qol_trend,
            'early_dropout_signal': 0,
            'high_adverse_event_flag': int(ae_rate > 3.0),
            'low_adherence_flag': int(medication_adherence_score < 60.0),
            'dropout_status': dropout_status,
            'days_to_event': max(days_to_event, 1),
            'study': 'Pancrea_ClovisO_2010_186',
            'has_real_qol': has_real_qol,
            'usubjid': usubjid,
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = extract()
    print(f"Pancrea_ClovisO: {len(df)} patients with real visit data, {df['dropout_status'].sum()} behavioral")
    df.to_csv('pancrea_clovis_real.csv', index=False)
