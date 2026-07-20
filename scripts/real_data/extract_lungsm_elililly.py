import pandas as pd
import numpy as np
from pathlib import Path
from rebuild_missed_visits import planned_days_for_arm, count_missed

BEHAVIORAL = {'subject withdrew consent', 'withdrawal of consent', 'withdrew consent',
              'lost to follow-up', 'withdrawal by subject', 'protocol violation/lost to follow-up'}

# Real per-dose-event field, verified: 'None' = dose given as planned,
# 'Dose Held/Omitted' / 'Reduced Per Protocol' / 'Discontinued' = a real
# adherence problem for that dose. Most rows are blank (not assessed),
# those are excluded from the percentage rather than counted as "fine",
# only patients with at least one explicitly-recorded row get a real
# adherence score, everyone else keeps the imputed fallback.
DOSMOD_NORMAL = {'none'}


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
    cutoff_days: if set, every feature below (including the adherence
    proxy, since it's also drawn from EX dosing events) is computed using
    only events at day <= cutoff_days. dropout_status and days_to_event
    are NEVER windowed - days_to_event's fallback uses the FULL,
    unwindowed visit history. Same missed-visit-schedule note as
    extract_glioma.py: count_missed() already restricts planned visits to
    the patient's own last actual visit day, so only the actual visit_days
    need windowing.
    """
    base = Path('../../data/pds/LungSm_EliLill_2011_287/Lilly data LY2510924_CXAC')
    dm = pd.read_sas(base / 'dm.sas7bdat', format='sas7bdat', encoding='latin1')
    ds = pd.read_sas(base / 'ds_t.sas7bdat', format='sas7bdat', encoding='latin1')
    ex = pd.read_sas(base / 'ex.sas7bdat', format='sas7bdat', encoding='latin1')
    ae = pd.read_sas(base / 'ae.sas7bdat', format='sas7bdat', encoding='latin1')
    try:
        tv = pd.read_sas(base / 'tv.sas7bdat', format='sas7bdat', encoding='latin1')
    except Exception:
        tv = None
    ex['_dosmod_norm'] = ex['DOSMOD'].astype(str).str.strip().str.lower() if 'DOSMOD' in ex.columns else None

    ds['_norm'] = ds['DSDECOD'].astype(str).str.strip().str.lower()
    beh_ids = set(ds[ds['_norm'].isin(BEHAVIORAL)]['USUBJID'])
    term_days = ds[ds['_norm'].isin(BEHAVIORAL)].groupby('USUBJID')['DSSTDY'].max()

    rows = []
    for _, pat in dm.iterrows():
        usubjid = pat['USUBJID']
        sex = str(pat.get('SEX', '')).strip().upper()

        pat_ex_full = ex[ex['USUBJID'] == usubjid]
        pat_ae_full = ae[ae['USUBJID'] == usubjid]

        full_visit_days = sorted(pat_ex_full['EXSTDY'].dropna().unique().tolist()) if 'EXSTDY' in pat_ex_full.columns else []
        dropout_status = int(usubjid in beh_ids)
        days_to_event = float(term_days.get(usubjid, full_visit_days[-1] if full_visit_days else 30))

        if cutoff_days is not None:
            pat_ex = pat_ex_full[pat_ex_full['EXSTDY'] <= cutoff_days] if 'EXSTDY' in pat_ex_full.columns else pat_ex_full
            pat_ae = pat_ae_full[pat_ae_full['AESTDY'] <= cutoff_days] if 'AESTDY' in pat_ae_full.columns else pat_ae_full
        else:
            pat_ex, pat_ae = pat_ex_full, pat_ae_full

        visit_days = sorted(pat_ex['EXSTDY'].dropna().unique().tolist()) if 'EXSTDY' in pat_ex.columns else []
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
        if 'AESTDY' in pat_ae.columns and n_ae > 0 and len(visit_days) > 1:
            ae_days = pat_ae['AESTDY'].dropna().values
            bucket_counts = [np.sum((ae_days >= visit_days[i]) & (ae_days < visit_days[i + 1]))
                              for i in range(len(visit_days) - 1)]
            ae_trend = _linear_trend(bucket_counts)

        # Real missed-visit count vs this patient's own arm's real planned
        # schedule (tv.sas7bdat), same verified logic as the Glioma studies.
        cumulative_missed = 0
        if tv is not None and 'ARMCD' in pat.index and pd.notna(pat.get('ARMCD')):
            planned = planned_days_for_arm(tv, pat['ARMCD'])
            cumulative_missed = count_missed(visit_days, planned)

        # Real adherence proxy: percentage of this patient's explicitly-recorded
        # dosing events (DOSMOD not blank) that were given exactly as planned.
        # Patients with zero explicitly-recorded dosing events keep the
        # imputed fallback, there is genuinely no signal for them.
        explicit_dosmod = pat_ex[pat_ex['DOSMOD'].notna()] if 'DOSMOD' in pat_ex.columns else pd.DataFrame()
        if len(explicit_dosmod) > 0:
            n_normal = (explicit_dosmod['_dosmod_norm'] == 'none').sum()
            medication_adherence_score = 100.0 * n_normal / len(explicit_dosmod)
        else:
            medication_adherence_score = 85.0

        rows.append({
            'age': pat.get('AGE'),
            'gender_encoded': {'M': 0, 'F': 1}.get(sex, 3),
            'ethnicity_encoded': 5,
            'condition_severity_encoded': 1,
            'visit_number': visit_number,
            'cumulative_missed_visits': cumulative_missed,
            'visit_frequency_rate': visit_frequency_rate,
            'days_since_last_visit': days_since_last_visit,
            'days_between_visits_mean': days_between_visits_mean,
            'days_between_visits_std': days_between_visits_std,
            'adverse_events_count': n_ae,
            'adverse_event_rate': ae_rate,
            'adverse_event_trend': ae_trend,
            'medication_adherence_score': medication_adherence_score,
            'medication_adherence_trend': 0.0,
            'quality_of_life_score': 50.0,
            'qol_score_trend': 0.0,
            'early_dropout_signal': 0,
            'high_adverse_event_flag': int(ae_rate > 3.0),
            'low_adherence_flag': int(medication_adherence_score < 60.0),
            'dropout_status': dropout_status,
            'days_to_event': max(days_to_event, 1),
            'study': 'LungSm_EliLill_2011_287',
            'has_real_qol': False,
            'usubjid': usubjid,
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = extract()
    print(f"LungSm_EliLill_2011_287: {len(df)} patients with real visit data, {df['dropout_status'].sum()} behavioral")
    df.to_csv('lungsm_elililly_real.csv', index=False)
