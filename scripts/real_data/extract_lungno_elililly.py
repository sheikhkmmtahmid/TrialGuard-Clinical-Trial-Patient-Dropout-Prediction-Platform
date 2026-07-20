import pandas as pd
import numpy as np
from pathlib import Path

BEHAVIORAL = {'withdrawal of consent without further f/u for survival',
              'withdrawal of consent with further f/u for survival',
              'lost to follow-up'}


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
    at day <= cutoff_days. dropout_status and days_to_event are NEVER
    windowed - days_to_event's fallback uses the FULL, unwindowed visit
    history.
    """
    base = Path('../../data/pds/LungNo_EliLill_2010_272/jfcc_deid_ctrl_trt')
    adsl = pd.read_sas(base / 'adsl.sas7bdat', format='sas7bdat', encoding='latin1')
    advs = pd.read_sas(base / 'advs.sas7bdat', format='sas7bdat', encoding='latin1')
    adae = pd.read_sas(base / 'adae.sas7bdat', format='sas7bdat', encoding='latin1')

    adsl['_norm'] = adsl['DSSTREAS'].astype(str).str.strip().str.lower()
    beh_ids = set(adsl[adsl['_norm'].isin(BEHAVIORAL)]['USUBJID'])

    rows = []
    for _, pat in adsl.iterrows():
        usubjid = pat['USUBJID']
        sex = str(pat.get('SEX', '')).strip().upper()

        pat_vs_full = advs[advs['USUBJID'] == usubjid]
        pat_ae_full = adae[adae['USUBJID'] == usubjid]

        full_visit_days = sorted(pat_vs_full['ADY'].dropna().unique().tolist())
        dropout_status = int(usubjid in beh_ids)
        days_to_event = float(full_visit_days[-1] if full_visit_days else 30)

        ae_day_col = 'ASTDY' if 'ASTDY' in pat_ae_full.columns else ('AESDY' if 'AESDY' in pat_ae_full.columns else None)
        if cutoff_days is not None:
            pat_vs = pat_vs_full[pat_vs_full['ADY'] <= cutoff_days]
            pat_ae = pat_ae_full[pat_ae_full[ae_day_col] <= cutoff_days] if ae_day_col else pat_ae_full
        else:
            pat_vs, pat_ae = pat_vs_full, pat_ae_full

        visit_days = sorted(pat_vs['ADY'].dropna().unique().tolist())
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
        ae_trend = 0.0
        if ae_day_col and n_ae > 0 and len(visit_days) > 1:
            ae_days = pat_ae[ae_day_col].dropna().values
            bucket_counts = [np.sum((ae_days >= visit_days[i]) & (ae_days < visit_days[i + 1]))
                              for i in range(len(visit_days) - 1)]
            ae_trend = _linear_trend(bucket_counts)

        ecog = pd.to_numeric(pat.get('ECOGBL'), errors='coerce')
        severity = 0 if (pd.notna(ecog) and ecog == 0) else (2 if (pd.notna(ecog) and ecog >= 2) else 1)

        rows.append({
            'age': pat.get('AGE'),
            'gender_encoded': {'M': 0, 'F': 1}.get(sex, 3),
            'ethnicity_encoded': 5,
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
            'quality_of_life_score': 50.0,
            'qol_score_trend': 0.0,
            'early_dropout_signal': 0,
            'high_adverse_event_flag': int(ae_rate > 3.0),
            'low_adherence_flag': 0,
            'dropout_status': dropout_status,
            'days_to_event': max(days_to_event, 1),
            'study': 'LungNo_EliLill_2010_272',
            'has_real_qol': False,
            'usubjid': usubjid,
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = extract()
    print(f"LungNo_EliLill_2010_272: {len(df)} patients with real visit data, {df['dropout_status'].sum()} behavioral")
    df.to_csv('lungno_elililly_real.csv', index=False)
