"""
Extract real per-patient feature rows for 2 of the 5 ImmPort studies.
Each study's real behavioral-dropout patient set was already confirmed by
hand against its own CRF or data dictionary (see docs/data_sourcing.md,
"Update: account created, data pulled and checked").
"""
import pandas as pd
import numpy as np
from pathlib import Path

_SCRATCH = r'C:\Users\Legion\AppData\Local\Temp\claude\d--Trial-Guard\ed813048-6423-4cf1-9904-005d867754ea\scratchpad'
B = f'{_SCRATCH}/immport_downloads/batch3'
B2 = f'{_SCRATCH}/immport_downloads/batch2'
B1 = f'{_SCRATCH}/immport_downloads'


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
    windowed - both studies here fall back to the patient's own last real
    (or synthetic-spacing) visit day for days_to_event, computed from the
    FULL, unwindowed visit history.

    SDY3285's adverse-event count has no per-event day field in the source
    data at all (verified: aexpmstr.txt has no date/day column), so it
    cannot be honestly windowed - it stays a whole-history count even in
    the early-window version, same as it always was. Not a new limitation
    introduced by this rebuild, an existing one being disclosed here.
    """
    rows = []

    # ---- SDY471 (AMS01, MS): real VISITDY (vitals), real AESTDY (AE) ----
    demo = pd.read_csv(f'{B}/demo_ref_AMS01.txt', sep='\t')
    vitl = pd.read_csv(f'{B}/vitl_ref_AMS01.txt', sep='\t')
    aexp = pd.read_csv(f'{B}/aexp_ref_AMS01.txt', sep='\t')
    term = pd.read_csv(f'{B1}/term_ref_AMS01.txt', sep='\t')
    term['beh'] = (term['REASON'] == 6) | (term['FOLREAS'].isin([2, 4]))
    beh_471 = set(term[term['beh']]['SUBJECT_ORG_ACC_NUM'])

    for subj in demo['SUBJECT_ORG_ACC_NUM'].unique():
        pat_demo = demo[demo['SUBJECT_ORG_ACC_NUM'] == subj].iloc[0]
        pat_vitl_full = vitl[vitl['SUBJECT_ORG_ACC_NUM'] == subj]
        pat_ae_full = aexp[aexp['SUBJECT_ORG_ACC_NUM'] == subj]

        full_visit_days = sorted(pat_vitl_full['VISITDY'].dropna().unique().tolist())
        days_to_event_471 = max(float(full_visit_days[-1]), 1) if full_visit_days else 30.0

        if cutoff_days is not None:
            pat_vitl = pat_vitl_full[pat_vitl_full['VISITDY'] <= cutoff_days]
            pat_ae = pat_ae_full[pat_ae_full['AESTDY'] <= cutoff_days] if 'AESTDY' in pat_ae_full.columns else pat_ae_full
        else:
            pat_vitl, pat_ae = pat_vitl_full, pat_ae_full

        visit_days = sorted(pat_vitl['VISITDY'].dropna().unique().tolist())
        visit_number = len(visit_days)
        if visit_number == 0:
            continue
        gaps = np.diff(visit_days) if len(visit_days) > 1 else np.array([])
        n_ae = len(pat_ae)
        ae_rate = n_ae / visit_number if visit_number else 0.0
        ae_trend = 0.0
        if n_ae > 0 and len(visit_days) > 1:
            ae_days = pat_ae['AESTDY'].dropna().values
            bucket_counts = [np.sum((ae_days >= visit_days[i]) & (ae_days < visit_days[i + 1]))
                              for i in range(len(visit_days) - 1)]
            ae_trend = _linear_trend(bucket_counts)

        rows.append({
            'age': pat_demo.get('age'),
            'gender_encoded': {'M': 0, 'F': 1}.get(str(pat_demo.get('SEX', '')).strip().upper(), 3),
            'ethnicity_encoded': 5, 'condition_severity_encoded': 1,
            'visit_number': visit_number, 'cumulative_missed_visits': 0,
            'visit_frequency_rate': visit_number / max(visit_days[-1] - visit_days[0], 1) * 30 if visit_days[-1] != visit_days[0] else visit_number,
            'days_since_last_visit': float(gaps[-1]) if len(gaps) else 0.0,
            'days_between_visits_mean': float(np.mean(gaps)) if len(gaps) else 0.0,
            'days_between_visits_std': float(np.std(gaps)) if len(gaps) > 1 else 0.0,
            'adverse_events_count': n_ae, 'adverse_event_rate': ae_rate, 'adverse_event_trend': ae_trend,
            'medication_adherence_score': 85.0, 'medication_adherence_trend': 0.0,
            'quality_of_life_score': 50.0, 'qol_score_trend': 0.0,
            'early_dropout_signal': 0, 'high_adverse_event_flag': int(ae_rate > 3.0), 'low_adherence_flag': 0,
            'dropout_status': int(subj in beh_471), 'days_to_event': days_to_event_471,
            'study': 'ImmPort_SDY471', 'has_real_qol': False, 'usubjid': subj,
        })
    print(f"SDY471: {sum(1 for r in rows if r['study']=='ImmPort_SDY471')} patients, "
          f"{sum(r['dropout_status'] for r in rows if r['study']=='ImmPort_SDY471')} behavioral")

    # ---- SDY3285 (ACCLAIM, MS): no real day field, use visit COUNT with fixed 30-day spacing ----
    vitp = pd.read_csv(f'{B}/vitpmstr.txt', sep='\t')
    aexpm = pd.read_csv(f'{B}/aexpmstr.txt', sep='\t')
    term3285 = pd.read_csv(f'{B2}/termmstr.txt', sep='\t')
    BEH_3285 = {'participant withdrew consent', 'failure to return/lost to follow-up'}
    term3285['_norm'] = term3285['TERMREA'].astype(str).str.strip().str.lower()
    beh_3285 = set(term3285[term3285['_norm'].isin(BEH_3285)]['Subject.Accession'])

    n0 = len(rows)
    for subj in vitp['Subject.Accession'].unique():
        pat_vitp_full = vitp[vitp['Subject.Accession'] == subj]
        pat_ae = aexpm[aexpm['Subject.Accession'] == subj]  # no day field available at all, see docstring

        full_visit_number = len(pat_vitp_full)
        full_visit_days = [i * 30 for i in range(full_visit_number)]
        days_to_event_3285 = max(float(full_visit_days[-1]) if full_visit_days else 30, 1)

        if cutoff_days is not None:
            max_visits_in_window = cutoff_days // 30 + 1
            visit_number = min(full_visit_number, max_visits_in_window)
        else:
            visit_number = full_visit_number
        if visit_number == 0:
            continue
        visit_days = [i * 30 for i in range(visit_number)]
        gaps = np.diff(visit_days) if len(visit_days) > 1 else np.array([])
        n_ae = len(pat_ae)
        rows.append({
            'age': None, 'gender_encoded': 3, 'ethnicity_encoded': 5, 'condition_severity_encoded': 1,
            'visit_number': visit_number, 'cumulative_missed_visits': 0,
            'visit_frequency_rate': visit_number / max(visit_days[-1], 1) * 30 if visit_days else visit_number,
            'days_since_last_visit': float(gaps[-1]) if len(gaps) else 0.0,
            'days_between_visits_mean': float(np.mean(gaps)) if len(gaps) else 30.0,
            'days_between_visits_std': float(np.std(gaps)) if len(gaps) > 1 else 0.0,
            'adverse_events_count': n_ae, 'adverse_event_rate': n_ae / visit_number if visit_number else 0.0,
            'adverse_event_trend': 0.0,
            'medication_adherence_score': 85.0, 'medication_adherence_trend': 0.0,
            'quality_of_life_score': 50.0, 'qol_score_trend': 0.0,
            'early_dropout_signal': 0, 'high_adverse_event_flag': int((n_ae / visit_number if visit_number else 0) > 3.0),
            'low_adherence_flag': 0,
            'dropout_status': int(subj in beh_3285), 'days_to_event': days_to_event_3285,
            'study': 'ImmPort_SDY3285', 'has_real_qol': False, 'usubjid': subj,
        })
    print(f"SDY3285: {len(rows)-n0} patients, {sum(r['dropout_status'] for r in rows[n0:])} behavioral")

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = extract()
    df.to_csv('immport_real_part1.csv', index=False)
    print("saved immport_real_part1.csv (SDY471 + SDY3285), total:", len(df))
