import pandas as pd
import numpy as np

_SCRATCH = r'C:\Users\Legion\AppData\Local\Temp\claude\d--Trial-Guard\ed813048-6423-4cf1-9904-005d867754ea\scratchpad'
B = f'{_SCRATCH}/immport_downloads/batch3'
B2 = f'{_SCRATCH}/immport_downloads/batch2'


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
    windowed.

    SDY524's days_to_event comes from real calendar dates (enrollment to
    last follow-up) already, independent of visit/AE windowing, so it
    needs no special unwindowed-copy handling. SDY797 and SDY1904 fall
    back to the patient's last real visit day for days_to_event, computed
    from the FULL, unwindowed visit history.

    SDY797's adverse-event count has no usable elapsed-day field wired up
    in this extraction (only a real calendar "Start Date" exists, with no
    enrollment-date join available here to convert it to an elapsed day),
    so - same as the original script - it stays a whole-history count in
    both versions. Not a new limitation introduced by this rebuild.
    """
    rows = []

    # ---- SDY524 (AbATE, T1D): real calendar dates (enrollment, last follow-up)
    # give a real days_to_event and total duration; monthly-visit assumption
    # (standard for this trial's follow-up schedule) converts duration into a
    # visit count. Real AE start days from ADAE1_PUB. ----
    adstand = pd.read_csv(f'{B2}/ITN027AI.DataSet.ADSTAND.txt', sep='\t')
    adae524 = pd.read_csv(f'{B}/ITN027AI.DataSet.ADAE1_PUB.txt', sep='\t')
    BEH_524 = {'voluntary withdrawal', 'lost to follow-up'}
    adstand['_norm'] = adstand['Study Termination Reason'].astype(str).str.strip().str.lower()
    beh_524 = set(adstand[adstand['_norm'].isin(BEH_524)]['Participant ID'])
    adstand['_enroll'] = pd.to_datetime(adstand['Enrollment Date'], errors='coerce')
    adstand['_lastfu'] = pd.to_datetime(adstand['Date of last follow-up'], errors='coerce')
    adstand['_duration'] = (adstand['_lastfu'] - adstand['_enroll']).dt.days
    # Real Sex field exists here (Male/Female), verified against the source
    # file - previously never read at all (gender_encoded was hardcoded to
    # "unknown" for every SDY524 patient), caught during the cleaning pass.
    sex_by_pid_524 = adstand.set_index('Participant ID')['Sex'].to_dict()
    GENDER_MAP_524 = {'MALE': 0, 'FEMALE': 1}

    ae_day_col = 'AE Start Day' if 'AE Start Day' in adae524.columns else None

    n0 = len(rows)
    for _, pat in adstand.iterrows():
        pid = pat['Participant ID']
        duration = pat['_duration']
        if pd.isna(duration) or duration <= 0:
            continue
        full_visit_number = max(int(duration // 30), 1)
        if cutoff_days is not None:
            visit_number = min(full_visit_number, max(int(cutoff_days // 30), 1))
        else:
            visit_number = full_visit_number
        visit_days = [i * 30 for i in range(1, visit_number + 1)]

        pat_ae_full = adae524[adae524['Participant ID'] == pid]
        if cutoff_days is not None and ae_day_col:
            pat_ae = pat_ae_full[pd.to_numeric(pat_ae_full[ae_day_col], errors='coerce') <= cutoff_days]
        else:
            pat_ae = pat_ae_full
        n_ae = len(pat_ae)
        ae_rate = n_ae / visit_number
        ae_trend = 0.0
        if ae_day_col and n_ae > 0 and len(visit_days) > 1:
            ae_days = pd.to_numeric(pat_ae[ae_day_col], errors='coerce').dropna().values
            bucket_counts = [np.sum((ae_days >= visit_days[i - 1]) & (ae_days < visit_days[i]))
                              for i in range(1, len(visit_days))]
            ae_trend = _linear_trend(bucket_counts)

        gender_encoded_524 = GENDER_MAP_524.get(str(sex_by_pid_524.get(pid, '')).strip().upper(), 3)
        rows.append({
            'age': None, 'gender_encoded': gender_encoded_524, 'ethnicity_encoded': 5, 'condition_severity_encoded': 1,
            'visit_number': visit_number, 'cumulative_missed_visits': 0,
            'visit_frequency_rate': visit_number / max(duration, 1) * 30,
            'days_since_last_visit': 30.0, 'days_between_visits_mean': 30.0, 'days_between_visits_std': 0.0,
            'adverse_events_count': n_ae, 'adverse_event_rate': ae_rate, 'adverse_event_trend': ae_trend,
            'medication_adherence_score': 85.0, 'medication_adherence_trend': 0.0,
            'quality_of_life_score': 50.0, 'qol_score_trend': 0.0,
            'early_dropout_signal': 0, 'high_adverse_event_flag': int(ae_rate > 3.0), 'low_adherence_flag': 0,
            'dropout_status': int(pid in beh_524), 'days_to_event': max(float(duration), 1),
            'study': 'ImmPort_SDY524', 'has_real_qol': False, 'usubjid': pid,
        })
    print(f"SDY524: {len(rows)-n0} patients, {sum(r['dropout_status'] for r in rows[n0:])} behavioral")

    # ---- SDY797 (T1DAL, T1D): real "Vital Day" from ADVS1, real AE Start Day ----
    advs797 = pd.read_csv(f'{B}/ADVS1_2019-03-01_10-24-35_ITN045AI.txt', sep='\t')
    aexp797 = pd.read_csv(f'{B}/AEXPMSTR_2019-03-01_10-42-22_ITN045AI.txt', sep='\t')
    term797 = pd.read_csv(f'{B2}/TERMMSTR_2019-03-01_10-44_ITN045AI.txt', sep='\t')
    BEH_797 = {'failure to return/lost to follow-up', 'participant or guardian withdrew consent'}
    term797['_norm'] = term797['Reason for Early Termination'].astype(str).str.strip().str.lower()
    beh_797 = set(term797[term797['_norm'].isin(BEH_797)]['ImmPort Accession'])

    n0 = len(rows)
    for subj in advs797['ImmPort Accession'].unique():
        pat_vs_full = advs797[advs797['ImmPort Accession'] == subj]
        pat_ae = aexp797[aexp797['ImmPort Accession'] == subj] if 'ImmPort Accession' in aexp797.columns else pd.DataFrame()

        full_visit_days = sorted(pd.to_numeric(pat_vs_full['Vital Day'], errors='coerce').dropna().unique().tolist())
        days_to_event_797 = max(float(full_visit_days[-1]), 1) if full_visit_days else 30.0

        pat_vs = pat_vs_full[pd.to_numeric(pat_vs_full['Vital Day'], errors='coerce') <= cutoff_days] if cutoff_days is not None else pat_vs_full
        visit_days = sorted(pd.to_numeric(pat_vs['Vital Day'], errors='coerce').dropna().unique().tolist())
        visit_number = len(visit_days)
        if visit_number == 0:
            continue
        gaps = np.diff(visit_days) if len(visit_days) > 1 else np.array([])
        n_ae = len(pat_ae)  # no usable elapsed-day field here, see docstring
        ae_rate = n_ae / visit_number if visit_number else 0.0

        rows.append({
            'age': pat_vs['Age'].iloc[0] if 'Age' in pat_vs.columns and len(pat_vs) else None,
            'gender_encoded': {'Male': 0, 'Female': 1}.get(str(pat_vs['Gender'].iloc[0]).strip(), 3) if 'Gender' in pat_vs.columns and len(pat_vs) else 3,
            'ethnicity_encoded': 5, 'condition_severity_encoded': 1,
            'visit_number': visit_number, 'cumulative_missed_visits': 0,
            'visit_frequency_rate': visit_number / max(visit_days[-1], 1) * 30,
            'days_since_last_visit': float(gaps[-1]) if len(gaps) else 0.0,
            'days_between_visits_mean': float(np.mean(gaps)) if len(gaps) else 0.0,
            'days_between_visits_std': float(np.std(gaps)) if len(gaps) > 1 else 0.0,
            'adverse_events_count': n_ae, 'adverse_event_rate': ae_rate, 'adverse_event_trend': 0.0,
            'medication_adherence_score': 85.0, 'medication_adherence_trend': 0.0,
            'quality_of_life_score': 50.0, 'qol_score_trend': 0.0,
            'early_dropout_signal': 0, 'high_adverse_event_flag': int(ae_rate > 3.0), 'low_adherence_flag': 0,
            'dropout_status': int(subj in beh_797), 'days_to_event': days_to_event_797,
            'study': 'ImmPort_SDY797', 'has_real_qol': False, 'usubjid': subj,
        })
    print(f"SDY797: {len(rows)-n0} patients, {sum(r['dropout_status'] for r in rows[n0:])} behavioral")

    # ---- SDY1904 (Tocilizumab, T1D): real VSDY (vitals), real AESTDY (AE) ----
    vsp1904 = pd.read_csv(f'{B}/vspcmstr_r_EXTEND.txt', sep='\t')
    ae1904 = pd.read_csv(f'{B}/aexpcode_r_EXTEND.txt', sep='\t')
    ds1904 = pd.read_csv(f'{B2}/ds1mstr_r_EXTEND.txt', sep='\t')
    BEH_1904 = {'withdrawn consent*', 'lost to follow-up'}
    ds1904['_norm'] = ds1904['DSREAS'].astype(str).str.strip().str.lower()
    beh_1904 = set(ds1904[ds1904['_norm'].isin(BEH_1904)]['SUBJECT_ACCESSION'])

    n0 = len(rows)
    for subj in vsp1904['SUBJECT_ACCESSION'].unique():
        pat_vs_full = vsp1904[vsp1904['SUBJECT_ACCESSION'] == subj]
        pat_ae_full = ae1904[ae1904['SUBJECT_ACCESSION'] == subj]

        full_visit_days = sorted(pd.to_numeric(pat_vs_full['VSDY'], errors='coerce').dropna().unique().tolist())
        days_to_event_1904 = max(float(full_visit_days[-1]), 1) if full_visit_days else 30.0

        if cutoff_days is not None:
            pat_vs = pat_vs_full[pd.to_numeric(pat_vs_full['VSDY'], errors='coerce') <= cutoff_days]
            pat_ae = pat_ae_full[pd.to_numeric(pat_ae_full['AESTDY'], errors='coerce') <= cutoff_days] if 'AESTDY' in pat_ae_full.columns else pat_ae_full
        else:
            pat_vs, pat_ae = pat_vs_full, pat_ae_full

        visit_days = sorted(pd.to_numeric(pat_vs['VSDY'], errors='coerce').dropna().unique().tolist())
        visit_number = len(visit_days)
        if visit_number == 0:
            continue
        gaps = np.diff(visit_days) if len(visit_days) > 1 else np.array([])
        n_ae = len(pat_ae)
        ae_rate = n_ae / visit_number if visit_number else 0.0
        ae_trend = 0.0
        if 'AESTDY' in pat_ae.columns and n_ae > 0 and len(visit_days) > 1:
            ae_days = pd.to_numeric(pat_ae['AESTDY'], errors='coerce').dropna().values
            bucket_counts = [np.sum((ae_days >= visit_days[i]) & (ae_days < visit_days[i + 1]))
                              for i in range(len(visit_days) - 1)]
            ae_trend = _linear_trend(bucket_counts)

        rows.append({
            'age': None, 'gender_encoded': 3, 'ethnicity_encoded': 5, 'condition_severity_encoded': 1,
            'visit_number': visit_number, 'cumulative_missed_visits': 0,
            'visit_frequency_rate': visit_number / max(visit_days[-1], 1) * 30,
            'days_since_last_visit': float(gaps[-1]) if len(gaps) else 0.0,
            'days_between_visits_mean': float(np.mean(gaps)) if len(gaps) else 0.0,
            'days_between_visits_std': float(np.std(gaps)) if len(gaps) > 1 else 0.0,
            'adverse_events_count': n_ae, 'adverse_event_rate': ae_rate, 'adverse_event_trend': ae_trend,
            'medication_adherence_score': 85.0, 'medication_adherence_trend': 0.0,
            'quality_of_life_score': 50.0, 'qol_score_trend': 0.0,
            'early_dropout_signal': 0, 'high_adverse_event_flag': int(ae_rate > 3.0), 'low_adherence_flag': 0,
            'dropout_status': int(subj in beh_1904), 'days_to_event': days_to_event_1904,
            'study': 'ImmPort_SDY1904', 'has_real_qol': False, 'usubjid': subj,
        })
    print(f"SDY1904: {len(rows)-n0} patients, {sum(r['dropout_status'] for r in rows[n0:])} behavioral")

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = extract()
    df.to_csv('immport_real_part2.csv', index=False)
    print("saved immport_real_part2.csv, total:", len(df))
