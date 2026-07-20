"""
Extract real per-patient feature rows from PPMI (Parkinson's Progression
Markers Initiative, downloaded via IDA/LONI). Verified field-by-field
against the actual PPMI data dictionary before use - see
docs/ppmi_data_build_log.md for the full read-through of all 63
downloaded files and why each was or wasn't used.

Population: patients with a real ENROLL_STATUS indicating they actually
started (Enrolled/Withdrew/Complete/Baseline/Baseline Withdraw/Withdraw
Deceased) AND at least one real visit record in Age_at_visit. Screen
failures, declines, and pending/scheduled patients never began and are
excluded, same treatment as every other real source in this project.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r'D:\Trial Guard\data\ida')

TRULY_ENROLLED = {'Enrolled', 'Withdrew', 'Complete', 'Baseline',
                   'Baseline Withdraw', 'Withdraw Deceased'}
BEHAVIORAL_WD_COLS = ['WDDISINT', 'WDFAMILY', 'WDLTFU', 'WDNONCOMP',
                       'WDTRANSPORT', 'WDBURDEN', 'WDOTHER']

# Verified against Code_List_-__Annotated__.csv: SCREEN.SEX 0=Female, 1=Male
# (opposite of this project's gender_encoded convention), flipped below.
GENDER_FLIP = {0: 1, 1: 0}

COHORT_STUDY_LABEL = {
    "Parkinson's Disease": 'PPMI_PD',
    'Prodromal': 'PPMI_Prodromal',
    'Healthy Control': 'PPMI_HealthyControl',
    'SWEDD': 'PPMI_SWEDD',
}


def parse_my(date_str):
    """PPMI dates are MM/YYYY (day-of-month stripped for de-identification).
    Parsed to the 15th of the month as a reasonable mid-month anchor."""
    if pd.isna(date_str):
        return pd.NaT
    try:
        m, y = str(date_str).split('/')
        return pd.Timestamp(year=int(y), month=int(m), day=15)
    except (ValueError, TypeError):
        return pd.NaT


def ethnicity_for(patno, demo_by_patno, myppmi_by_patno):
    row = demo_by_patno.get(patno)
    if row is not None:
        if row.get('HISPLAT') == 1:
            return 2
        if row.get('RAWHITE') == 1:
            return 0
        if row.get('RABLACK') == 1:
            return 1
        if row.get('RAASIAN') == 1:
            return 3
        if row.get('RAHAWOPI') == 1 or row.get('RAINDALS') == 1 or row.get('RANOS') == 1:
            return 4
    row2 = myppmi_by_patno.get(patno)
    if row2 is not None:
        if row2.get('HISPANIC') == 1:
            return 2
        if row2.get('WHITE') == 1:
            return 0
        if row2.get('AFRICAN_AMERICAN') == 1:
            return 1
        if row2.get('ASIAN') == 1:
            return 3
        if row2.get('INDIGENOUS') == 1 or row2.get('MIDDLE_EASTERN') == 1 or row2.get('NATIVE_HAWAIIAN') == 1:
            return 4
    return 5


def _linear_trend(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0]) if np.std(x) > 0 else 0.0


def extract(cutoff_days=None):
    """
    cutoff_days: if set, every feature is computed using only real
    assessments at day <= cutoff_days, where "day" is elapsed days since
    this patient's own real ENROLL_DATE (both are real calendar dates,
    month/year precision - see parse_my). dropout_status and days_to_event
    are NEVER windowed - days_to_event's fallback uses the FULL,
    unwindowed visit history.

    Severity (NP3TOT) and QoL (MSEADLG) use their own real INFODT
    (assessment date) fields for windowing, converted to elapsed days via
    each patient's own enroll date - not the visit-age-based proxy used
    for visit_number/AE, since these come from separate real assessment
    forms with their own real dates. The snapshot value taken is the LAST
    one recorded at or before the cutoff (not the global last, and not
    forced to be the very first) - the same "window first, then take the
    most recent point in that window" rule applied everywhere else in
    this rebuild.
    """
    ps = pd.read_csv(BASE / 'Participant_Status_13Jul2026.csv')
    concl = pd.read_csv(BASE / 'Study_Enrollment' / 'Conclusion_of_Study_Participation_13Jul2026.csv')
    demo = pd.read_csv(BASE / 'Subject_Demographics' / 'Demographics_13Jul2026.csv')
    myppmi = pd.read_csv(BASE / 'Subject_Demographics' / 'Race_and_Ethnicity_Question_in_myPPMI_13Jul2026.csv')
    av = pd.read_csv(BASE / 'Subject_Demographics' / 'Age_at_visit_13Jul2026.csv')

    updrs3 = pd.read_csv(BASE / 'Motor___MDS-UPDRS' / 'MDS-UPDRS_Part_III_13Jul2026.csv', low_memory=False)
    updrs3 = updrs3.dropna(subset=['NP3TOT']).copy()
    updrs3['_infodt'] = updrs3['INFODT'].apply(parse_my)
    updrs3_by_patno = {p: g for p, g in updrs3.groupby('PATNO')}

    mseadl = pd.read_csv(BASE / 'Motor___MDS-UPDRS' / 'Modified_Schwab___England_Activities_of_Daily_Living_13Jul2026.csv')
    mseadl = mseadl.dropna(subset=['MSEADLG']).copy()
    mseadl['_infodt'] = mseadl['INFODT'].apply(parse_my)
    mseadl_by_patno = {p: g for p, g in mseadl.groupby('PATNO')}

    # Real adverse-event signal: the full "AE" module (with severity, MedDRA
    # term, seriousness) was NOT part of this download - what's available
    # instead is the in-clinic and telephone "was an adverse event observed
    # at this visit" flag (AERPRT / TELAERPT, verified 0/1 Yes-No against
    # the code list). Coarser than the detailed AE logs used for the PDS
    # trials, but real: counts how many of a patient's real visits had an
    # AE flagged, not a guess or a default.
    ae_clinic = pd.read_csv(BASE / 'Medical' / 'Adverse_Event_In-Clinic_Assessment_13Jul2026.csv')
    ae_tel = pd.read_csv(BASE / 'Medical' / 'Adverse_Event_Telephone_Assessment_13Jul2026.csv')
    ae_clinic['_infodt'] = ae_clinic['INFODT'].apply(parse_my)
    ae_tel['_infodt'] = ae_tel['INFODT'].apply(parse_my)
    ae_clinic_flags = ae_clinic[['PATNO', '_infodt', 'AERPRT']].rename(columns={'AERPRT': 'flag'})
    ae_tel_flags = ae_tel[['PATNO', '_infodt', 'TELAERPT']].rename(columns={'TELAERPT': 'flag'})
    ae_all = pd.concat([ae_clinic_flags, ae_tel_flags], ignore_index=True).dropna(subset=['flag'])
    ae_by_patno = {p: g for p, g in ae_all.groupby('PATNO')}

    # One row per patient for lookup tables (verified: Demographics/myPPMI
    # occasionally have >1 row per PATNO across visit re-confirmations;
    # keep the first non-null real record for each).
    demo_by_patno = {p: r for p, r in demo.sort_values('INFODT').groupby('PATNO').first().to_dict('index').items()}
    myppmi_by_patno = {p: r for p, r in myppmi.sort_values('CREATED_AT').groupby('PATNO').first().to_dict('index').items()}
    # CONCL: keep the most-complete record where a patient has >1 (verified
    # only 1 real duplicate patient, 165031, keep the later non-null row).
    concl = concl.sort_values('ORIG_ENTRY').groupby('PATNO').last().reset_index()
    concl_by_patno = concl.set_index('PATNO').to_dict('index')

    pop = ps[ps['ENROLL_STATUS'].isin(TRULY_ENROLLED)].copy()
    pop = pop[pop['PATNO'].isin(av['PATNO'])]
    pop = pop[pop['COHORT_DEFINITION'].isin(COHORT_STUDY_LABEL)]

    rows = []
    for _, pat in pop.iterrows():
        patno = pat['PATNO']
        pat_visits = av[av['PATNO'] == patno].copy()
        if pat_visits.empty:
            continue

        # Real per-visit age -> visit count and spacing. No per-visit
        # calendar date was in this download, so age (years, ~0.1yr
        # precision) x 365.25 stands in for day-level spacing - coarser
        # than PDS/ImmPort's real dates, disclosed in the build log.
        full_ages = sorted(pat_visits['AGE_AT_VISIT'].dropna().unique().tolist())
        if not full_ages:
            continue
        full_visit_days = [(a - full_ages[0]) * 365.25 for a in full_ages]

        c = concl_by_patno.get(patno)
        if c is not None:
            dropout_status = int(any(c.get(col) == 1 for col in BEHAVIORAL_WD_COLS))
        else:
            dropout_status = int(pat['ENROLL_STATUS'] in ('Withdrew', 'Baseline Withdraw'))

        enroll_dt = parse_my(pat['ENROLL_DATE'])
        wd_dt = parse_my(c['WDDT']) if c is not None else pd.NaT
        status_dt = parse_my(pat['STATUS_DATE'])
        end_dt = wd_dt if pd.notna(wd_dt) else (status_dt if pd.notna(status_dt) else pd.NaT)
        if pd.notna(enroll_dt) and pd.notna(end_dt):
            days_to_event = max((end_dt - enroll_dt).days, 1)
        else:
            days_to_event = max(float(full_visit_days[-1]) if full_visit_days else 30, 1)

        if cutoff_days is not None:
            ages = [a for a, d in zip(full_ages, full_visit_days) if d <= cutoff_days]
        else:
            ages = full_ages
        visit_number = len(ages)
        if visit_number == 0:
            continue
        visit_days = [(a - full_ages[0]) * 365.25 for a in ages]
        gaps = np.diff(visit_days) if len(visit_days) > 1 else np.array([])
        days_since_last_visit = float(gaps[-1]) if len(gaps) else 0.0
        days_between_visits_mean = float(np.mean(gaps)) if len(gaps) else 0.0
        days_between_visits_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
        visit_frequency_rate = visit_number / max(visit_days[-1], 1) * 30

        sex = demo_by_patno.get(patno, {}).get('SEX')
        gender_encoded = GENDER_FLIP.get(sex, 3)
        ethnicity_encoded = ethnicity_for(patno, demo_by_patno, myppmi_by_patno)

        # Real day-based windowing needs this patient's own enroll date to
        # convert each assessment's real INFODT into an elapsed day. If
        # cutoff_days is set and enroll_dt is missing, this patient's real
        # severity/QoL/AE values genuinely can't be windowed - honestly
        # treated as unavailable for the windowed view specifically (NOT
        # for the unwindowed v1 view, which never needed enroll_dt at all
        # and must not lose real values just because a date is missing).
        can_window = cutoff_days is None or pd.notna(enroll_dt)

        # Severity: window by real elapsed days (INFODT - enroll date),
        # take the most recent value within that window.
        pat_updrs = updrs3_by_patno.get(patno)
        np3_val = None
        if pat_updrs is not None and can_window:
            pat_updrs = pat_updrs.copy()
            if cutoff_days is not None:
                pat_updrs['_elapsed'] = (pat_updrs['_infodt'] - enroll_dt).dt.days
                pat_updrs = pat_updrs[pat_updrs['_elapsed'] <= cutoff_days]
            pat_updrs = pat_updrs.sort_values('_infodt')
            if len(pat_updrs):
                np3_val = float(pat_updrs['NP3TOT'].iloc[-1])
        has_real_severity = np3_val is not None
        severity = severity_bucket_global(np3_val) if has_real_severity else 1

        # QoL: same real-elapsed-day windowing, snapshot + trend from
        # whatever real assessments fall within the window.
        pat_mse = mseadl_by_patno.get(patno)
        qol_score, qol_trend, has_real_qol = 50.0, 0.0, False
        if pat_mse is not None and can_window:
            pat_mse = pat_mse.copy()
            if cutoff_days is not None:
                pat_mse['_elapsed'] = (pat_mse['_infodt'] - enroll_dt).dt.days
                pat_mse = pat_mse[pat_mse['_elapsed'] <= cutoff_days]
            pat_mse = pat_mse.sort_values('_infodt')
            if len(pat_mse):
                qol_score = float(pat_mse['MSEADLG'].iloc[-1])
                qol_trend = _linear_trend(pat_mse['MSEADLG'].tolist())
                has_real_qol = True

        # Adverse events: same real-elapsed-day windowing.
        pat_ae = ae_by_patno.get(patno)
        adverse_events_count, adverse_event_rate, adverse_event_trend = 0, 0.0, 0.0
        if pat_ae is not None and can_window:
            pat_ae = pat_ae.copy()
            if cutoff_days is not None:
                pat_ae['_elapsed'] = (pat_ae['_infodt'] - enroll_dt).dt.days
                pat_ae = pat_ae[pat_ae['_elapsed'] <= cutoff_days]
            pat_ae = pat_ae.sort_values('_infodt')
            n_ae_assessed_visits = len(pat_ae)
            if n_ae_assessed_visits > 0:
                adverse_events_count = int(pat_ae['flag'].sum())
                adverse_event_rate = adverse_events_count / n_ae_assessed_visits
                adverse_event_trend = _linear_trend(pat_ae['flag'].tolist())

        rows.append({
            'age': pat.get('ENROLL_AGE'),
            'gender_encoded': gender_encoded,
            'ethnicity_encoded': ethnicity_encoded,
            'condition_severity_encoded': severity,
            'visit_number': visit_number,
            'cumulative_missed_visits': 0,  # no planned-visit schedule in this download
            'visit_frequency_rate': visit_frequency_rate,
            'days_since_last_visit': days_since_last_visit,
            'days_between_visits_mean': days_between_visits_mean,
            'days_between_visits_std': days_between_visits_std,
            'adverse_events_count': adverse_events_count,
            'adverse_event_rate': adverse_event_rate,
            'adverse_event_trend': adverse_event_trend,
            'medication_adherence_score': 85.0,  # no adherence-style field in what's available (observational study, not a drug-dosing trial)
            'medication_adherence_trend': 0.0,
            'quality_of_life_score': qol_score,
            'qol_score_trend': qol_trend,
            'early_dropout_signal': 0,
            'high_adverse_event_flag': int(adverse_event_rate > 3.0),
            'low_adherence_flag': 0,
            'dropout_status': dropout_status,
            'days_to_event': days_to_event,
            'study': COHORT_STUDY_LABEL[pat['COHORT_DEFINITION']],
            'has_real_qol': has_real_qol,
            'usubjid': f'PPMI_{patno}',
        })

    return pd.DataFrame(rows)


# Tertile cutoffs for severity, computed once globally from the FULL (v1,
# unwindowed) real NP3TOT distribution, verified/logged in
# docs/ppmi_data_build_log.md. Kept fixed across v1 and v2 so the meaning
# of "mild/moderate/severe" doesn't silently shift between the two
# datasets - only which real value gets bucketed changes with windowing,
# not the bucket boundaries themselves.
_NP3_Q1, _NP3_Q2 = None, None


def _init_severity_cutoffs():
    global _NP3_Q1, _NP3_Q2
    updrs3 = pd.read_csv(BASE / 'Motor___MDS-UPDRS' / 'MDS-UPDRS_Part_III_13Jul2026.csv', low_memory=False)
    updrs3 = updrs3.dropna(subset=['NP3TOT']).copy()
    # Sort by the real PARSED date, not the raw MM/YYYY string - string
    # sort would misorder e.g. "3/2015" before "12/2014" (caught by
    # comparing against the main extraction loop's own sort and finding a
    # mismatch in the resulting tertile counts).
    updrs3['_infodt'] = updrs3['INFODT'].apply(parse_my)
    np3_last = updrs3.sort_values('_infodt').groupby('PATNO')['NP3TOT'].last()
    _NP3_Q1, _NP3_Q2 = np3_last.quantile([1 / 3, 2 / 3])


def severity_bucket_global(val):
    if _NP3_Q1 is None:
        _init_severity_cutoffs()
    if val is None or pd.isna(val):
        return None
    return 0 if val <= _NP3_Q1 else (2 if val > _NP3_Q2 else 1)


if __name__ == '__main__':
    df = extract()
    for study, sub in df.groupby('study'):
        print(f"{study}: {len(sub)} patients, {sub['dropout_status'].sum()} behavioral "
              f"({sub['dropout_status'].mean()*100:.1f}%)")
    print(f"\nTOTAL: {len(df)} patients, {df['dropout_status'].sum()} behavioral")
    df.to_csv('ppmi_real.csv', index=False)
    print("saved ppmi_real.csv")
