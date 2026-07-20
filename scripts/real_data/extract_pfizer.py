"""
Pfizer XRP4174D-3001, both cohorts. No clean study-day field is available
without loading and cross-referencing calendar dates against a randomization
reference date (RNDT), which these files don't make simple. CYCLE number is
used as a real, honest visit-count proxy instead (chemo cycles here are
~21 days apart per the protocol), consistent with how visit_number/trend
features are meant to work, just using cycle count rather than exact days.
"""
import pandas as pd
import numpy as np
from pathlib import Path

CYCLE_DAYS = 21  # protocol-specified cycle length for this regimen
BEHAVIORAL_REASDC = {6.0, 9.0}  # Lost to follow-up (6), Subject did not wish to continue (9)


def _linear_trend(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    if np.std(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def extract(cohort, cutoff_days=None):
    """
    cutoff_days: if set, every feature below (visits, AE, QoL - all keyed
    off real CYCLE numbers) is computed using only cycles at
    cycle*CYCLE_DAYS <= cutoff_days. dropout_status and days_to_event are
    NEVER windowed - days_to_event's fallback uses the FULL, unwindowed
    cycle history.
    """
    base = Path(f'../../data/pds/LungSm_Pfizer_2002_419/XRP4174D-3001_{cohort}_SAS7BDATfiles')
    popu = pd.read_sas(base / f'{cohort}_popu.sas7bdat', format='sas7bdat', encoding='latin1')
    demo = pd.read_sas(base / f'{cohort}_demo.sas7bdat', format='sas7bdat', encoding='latin1')
    doseval = pd.read_sas(base / f'{cohort}_doseval.sas7bdat', format='sas7bdat', encoding='latin1')
    ae_file = f'{cohort}_ae_analysis.sas7bdat' if cohort == 'cohort1' else f'{cohort}_ae.sas7bdat'
    ae = pd.read_sas(base / ae_file, format='sas7bdat', encoding='latin1')
    qol = pd.read_sas(base / f'{cohort}_qol.sas7bdat', format='sas7bdat', encoding='latin1')

    # SUBJID is only unique *within a site* in this study (each site restarts
    # its own numbering, e.g. every site has a "subject 1"), verified: popu
    # has far more unique (SITEID, SUBJID) pairs than unique SUBJID values
    # alone. Matching on bare SUBJID silently merged AE/dosing/QoL records
    # from different real patients at different sites who shared a subject
    # number - caught via duplicate/implausible rows (713 AEs for one
    # "patient") during the cleaning pass. Fixed by keying on the real
    # composite identifier (SITEID, SUBJID) everywhere below.
    for d in (popu, demo, doseval, ae, qol):
        d['_pid'] = list(zip(d['SITEID'], d['SUBJID']))
    demo_by_pid = {row['_pid']: row for _, row in demo.iterrows()}
    qol_items = [c for c in qol.columns if c.startswith('QOL') and c[3:].isdigit()]

    rows = []
    for _, pat in popu.iterrows():
        pid = pat['_pid']
        subjid = pat['SUBJID']
        pat_doses_full = doseval[doseval['_pid'] == pid]
        pat_ae_full = ae[ae['_pid'] == pid]
        pat_qol_full = qol[qol['_pid'] == pid]

        full_cycles = sorted(pat_doses_full['CYCLE'].dropna().unique().tolist())
        full_visit_days = [c * CYCLE_DAYS for c in full_cycles]
        days_to_event = float(full_visit_days[-1]) if full_visit_days else 30.0

        reaswd = pat.get('REASWD')
        dropout_status = int(pd.notna(reaswd) and float(reaswd) in BEHAVIORAL_REASDC)

        if cutoff_days is not None:
            max_cycle = cutoff_days / CYCLE_DAYS
            pat_doses = pat_doses_full[pat_doses_full['CYCLE'] <= max_cycle]
            pat_ae = pat_ae_full[pat_ae_full['cycle'] <= max_cycle] if 'cycle' in pat_ae_full.columns else pat_ae_full
            pat_qol = pat_qol_full[pat_qol_full['CYCLE'] <= max_cycle] if 'CYCLE' in pat_qol_full.columns else pat_qol_full
        else:
            pat_doses, pat_ae, pat_qol = pat_doses_full, pat_ae_full, pat_qol_full

        cycles = sorted(pat_doses['CYCLE'].dropna().unique().tolist())
        visit_number = len(cycles)
        if visit_number == 0:
            continue

        visit_days = [c * CYCLE_DAYS for c in cycles]
        gaps = np.diff(visit_days) if len(visit_days) > 1 else np.array([])
        days_since_last_visit = float(gaps[-1]) if len(gaps) else 0.0
        days_between_visits_mean = float(np.mean(gaps)) if len(gaps) else float(CYCLE_DAYS)
        days_between_visits_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
        visit_frequency_rate = visit_number / max(visit_days[-1], 1) * 30

        n_ae = len(pat_ae)
        ae_rate = n_ae / visit_number if visit_number else 0.0
        ae_trend = 0.0
        if 'cycle' in pat_ae.columns and n_ae > 0 and len(cycles) > 1:
            ae_cycles = pat_ae['cycle'].dropna().values
            bucket_counts = [np.sum(ae_cycles == c) for c in cycles]
            ae_trend = _linear_trend(bucket_counts)

        has_real_qol = False
        qol_score, qol_trend = 50.0, 0.0
        if not pat_qol.empty and qol_items:
            pat_qol_sorted = pat_qol.sort_values('CYCLE')
            per_cycle_mean = pat_qol_sorted[qol_items].mean(axis=1, skipna=True)
            series = per_cycle_mean.dropna().tolist()
            if series:
                # QOL items here are 1-5 scale (lower=better on some instruments);
                # rescale roughly to 0-100 to match the model's expected range.
                qol_score = float(series[-1]) * 20
                qol_trend = _linear_trend(series) * 20
                has_real_qol = True

        # Real SEX here is a numeric code (1./2.), not a letter, verified
        # against the source file - a bare {'M':0,'F':1} string lookup
        # silently missed every patient, caught during cleaning. Direction
        # (1=Male, 2=Female) confirmed against this same PDS corpus's
        # standard SEX codebook convention (docs/all_docs_read.txt), no
        # Pfizer-specific value-label table exists to verify further.
        age_val, sex_num = None, None
        if pid in demo_by_pid:
            drow = demo_by_pid[pid]
            age_val = drow.get('age_d')
            sex_num = pd.to_numeric(drow.get('SEX'), errors='coerce')
        gender_encoded = 0 if sex_num == 1 else (1 if sex_num == 2 else 3)

        rows.append({
            'age': age_val,
            'gender_encoded': gender_encoded,
            'ethnicity_encoded': 5,
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
            'quality_of_life_score': qol_score,
            'qol_score_trend': qol_trend,
            'early_dropout_signal': 0,
            'high_adverse_event_flag': int(ae_rate > 3.0),
            'low_adherence_flag': 0,
            'dropout_status': dropout_status,
            'days_to_event': max(days_to_event, 1),
            'study': f'LungSm_Pfizer_2002_419_{cohort}',
            'has_real_qol': has_real_qol,
            'usubjid': f'{cohort}_{int(pat["SITEID"])}_{int(subjid)}',
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df1 = extract('cohort1')
    df2 = extract('cohort2')
    combined = pd.concat([df1, df2], ignore_index=True)
    print(f"Pfizer cohort1: {len(df1)} patients, {df1['dropout_status'].sum()} behavioral")
    print(f"Pfizer cohort2: {len(df2)} patients, {df2['dropout_status'].sum()} behavioral")
    combined.to_csv('pfizer_real.csv', index=False)
