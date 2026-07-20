"""
Breast_EliLill_2008_168 (JVBC). advs has AVISITN (an ordinal visit number)
but no real day field, so visit_number comes from the count of distinct
visits recorded in advs (vital signs, taken at every real clinic visit),
and visit spacing uses a fixed 21-day cycle assumption since no other day
field is available for visit dates specifically (adae's AESDY is real and
used directly for AE timing).
"""
import pandas as pd
import numpy as np
from pathlib import Path

BEHAVIORAL = {'consent withdrawn from overall study participation', 'lost to follow-up'}
CYCLE_DAYS = 21

# Verified real text values from adsl.sas7bdat's RACEGR1 field:
# {'WHITE': 341, 'NON-WHITE': 44}. Only a binary split is available (no finer
# real category), so NON-WHITE maps to the model's 'other' bucket.
ETHNICITY_MAP = {'white': 0, 'non-white': 4}


def extract(cutoff_days=None):
    """
    cutoff_days: if set, every feature below is computed using only visit
    cycles / adverse events at day <= cutoff_days. dropout_status and
    days_to_event are NEVER windowed - days_to_event's fallback (this study
    has no clean disposition-day field) uses the FULL, unwindowed visit
    history, computed before any windowing below.
    """
    base = Path('../../data/pds/Breast_EliLill_2008_168/adam20161103')
    adsl = pd.read_sas(base / 'adsl.sas7bdat', format='sas7bdat', encoding='latin1')
    advs = pd.read_sas(base / 'advs.sas7bdat', format='sas7bdat', encoding='latin1')
    adae = pd.read_sas(base / 'adae.sas7bdat', format='sas7bdat', encoding='latin1')

    adsl['_norm'] = adsl['DSSDREAS'].astype(str).str.strip().str.lower()
    beh_ids = set(adsl[adsl['_norm'].isin(BEHAVIORAL)]['USUBJID'])

    rows = []
    for _, pat in adsl.iterrows():
        usubjid = pat['USUBJID']
        sex = str(pat.get('SEX', '')).strip().upper()

        pat_vs_full = advs[advs['USUBJID'] == usubjid]
        pat_ae_full = adae[adae['USUBJID'] == usubjid]

        race_raw = pat.get('RACEGR1')
        ethnicity_encoded = ETHNICITY_MAP.get(str(race_raw).strip().lower(), 5)

        full_visits = sorted(pat_vs_full['AVISITN'].dropna().unique().tolist())
        full_visit_days = [v * CYCLE_DAYS for v in full_visits]
        days_to_event = float(full_visit_days[-1] if full_visit_days else 30)
        dropout_status = int(usubjid in beh_ids)

        if cutoff_days is not None:
            max_cycle = cutoff_days / CYCLE_DAYS
            pat_vs = pat_vs_full[pat_vs_full['AVISITN'] <= max_cycle]
            pat_ae = pat_ae_full[pat_ae_full['AESDY'] <= cutoff_days] if 'AESDY' in pat_ae_full.columns else pat_ae_full
        else:
            pat_vs, pat_ae = pat_vs_full, pat_ae_full

        visits = sorted(pat_vs['AVISITN'].dropna().unique().tolist())
        visit_number = len(visits)
        if visit_number == 0:
            continue

        visit_days = [v * CYCLE_DAYS for v in visits]
        gaps = np.diff(visit_days) if len(visit_days) > 1 else np.array([])
        days_since_last_visit = float(gaps[-1]) if len(gaps) else 0.0
        days_between_visits_mean = float(np.mean(gaps)) if len(gaps) else float(CYCLE_DAYS)
        days_between_visits_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
        visit_frequency_rate = visit_number / max(visit_days[-1], 1) * 30

        n_ae = len(pat_ae)
        ae_rate = n_ae / visit_number if visit_number else 0.0
        ae_trend = 0.0
        if 'AESDY' in pat_ae.columns and n_ae > 0 and len(visit_days) > 1:
            ae_days = pat_ae['AESDY'].dropna().values
            bucket_counts = [np.sum((ae_days >= visit_days[i]) & (ae_days < visit_days[i + 1]))
                              for i in range(len(visit_days) - 1)]
            values = [v for v in bucket_counts if v is not None]
            if len(values) >= 2:
                x = np.arange(len(values), dtype=float)
                y = np.array(values, dtype=float)
                if np.std(x) > 0:
                    ae_trend = float(np.polyfit(x, y, 1)[0])

        rows.append({
            'age': pat.get('AGE'),
            'gender_encoded': {'M': 0, 'F': 1}.get(sex, 3),
            'ethnicity_encoded': ethnicity_encoded,
            'condition_severity_encoded': 1,
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
            'study': 'Breast_EliLill_2008_168',
            'has_real_qol': False,
            'usubjid': usubjid,
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = extract()
    print(f"Breast_EliLill_2008_168: {len(df)} patients with real visit data, {df['dropout_status'].sum()} behavioral")
    df.to_csv('breast_elililly_real.csv', index=False)
