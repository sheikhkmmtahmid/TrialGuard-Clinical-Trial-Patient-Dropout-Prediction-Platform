"""
Shared extractor for the 4 Amgen studies (3 colorectal, 1 head-and-neck),
which all use the same file structure: exposure.sas7bdat (real dose visit
days via DOSREFDY), ae.sas7bdat (real AE onset day via AESTDY),
disposit.sas7bdat (reason field name differs: DSEOS or DSREAS, and each
study's real behavioral-reason label set, both already confirmed by hand
against each study's data in this project's PDS validation work).
"""
import pandas as pd
import numpy as np
from pathlib import Path

BEHAVIORAL = {'consent withdrawn', 'lost to follow-up'}

# Verified real text values from RACCAT (Colorec_Amgen studies) / RACE
# (HeadNe_Amgen_2007_265), mapped to the model's existing ethnicity codes.
ETHNICITY_MAP = {
    'white or caucasian': 0, 'black or african american': 1,
    'hispanic or latino': 2, 'asian': 3, 'other': 4,
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


def extract(base_dir, study_label, disp_file, reason_col, day_col, cutoff_days=None):
    """
    cutoff_days: if set, every feature below is computed using only events
    that happened at day <= cutoff_days (a "truly prospective" early-window
    view). dropout_status and days_to_event are NEVER windowed - they must
    reflect the real, eventual outcome, or this stops testing early-warning
    validity and starts testing something else.
    """
    base = Path(base_dir)
    exposure = pd.read_sas(base / 'exposure.sas7bdat', format='sas7bdat', encoding='latin1')
    ae = pd.read_sas(base / 'ae.sas7bdat', format='sas7bdat', encoding='latin1')
    disp = pd.read_sas(base / disp_file, format='sas7bdat', encoding='latin1')

    disp['_norm'] = disp[reason_col].astype(str).str.strip().str.lower()
    beh_ids = set(disp[disp['_norm'].isin(BEHAVIORAL)]['SUBJID'])
    days_map = disp.set_index('SUBJID')[day_col].to_dict() if day_col in disp.columns else {}

    rows = []
    for subjid in disp['SUBJID'].unique():
        pat_disp = disp[disp['SUBJID'] == subjid].iloc[0]
        pat_exp = exposure[exposure['SUBJID'] == subjid]
        pat_ae = ae[ae['SUBJID'] == subjid]
        if cutoff_days is not None:
            if 'DOSREFDY' in pat_exp.columns:
                pat_exp = pat_exp[pat_exp['DOSREFDY'] <= cutoff_days]
            if 'AESTDY' in pat_ae.columns:
                pat_ae = pat_ae[pat_ae['AESTDY'] <= cutoff_days]

        visit_days = sorted(pat_exp['DOSREFDY'].dropna().unique().tolist()) if 'DOSREFDY' in pat_exp.columns else []
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

        # Real SEX values in these 4 studies are spelled out ("Male"/"Female"),
        # not single letters, verified directly against the source files -
        # a bare {'M':0,'F':1} lookup silently missed every one of them and
        # fell back to "unknown" for 100% of these patients, caught during
        # the post-harmonization cleaning pass. Handling both forms now.
        sex = str(pat_disp.get('SEX', '')).strip().upper()
        ecog = pd.to_numeric(pat_disp.get('B_ECOG'), errors='coerce')
        severity = 0 if (pd.notna(ecog) and ecog == 0) else (2 if (pd.notna(ecog) and ecog >= 2) else 1)

        race_raw = pat_disp.get('RACCAT')
        if race_raw is None or pd.isna(race_raw):
            race_raw = pat_disp.get('RACE')
        ethnicity_encoded = ETHNICITY_MAP.get(str(race_raw).strip().lower(), 5)

        # Real adherence proxy where DOSCHGYN exists (only Colorec_Amgen_2005_262,
        # verified: N = dose given without change, Y = dose changed), percentage
        # of this patient's real dosing events given without a change.
        if 'DOSCHGYN' in pat_exp.columns and len(pat_exp) > 0:
            n_normal = (pat_exp['DOSCHGYN'].astype(str).str.strip().str.upper() == 'N').sum()
            medication_adherence_score = 100.0 * n_normal / len(pat_exp)
        else:
            medication_adherence_score = 85.0

        dropout_status = int(subjid in beh_ids)
        term_day = days_map.get(subjid)
        days_to_event = float(term_day) if pd.notna(term_day) else float(visit_days[-1] if visit_days else 30)

        rows.append({
            'age': pat_disp.get('AGE'),
            'gender_encoded': {'M': 0, 'MALE': 0, 'F': 1, 'FEMALE': 1}.get(sex, 3),
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
            'quality_of_life_score': 50.0,
            'qol_score_trend': 0.0,
            'early_dropout_signal': 0,
            'high_adverse_event_flag': int(ae_rate > 3.0),
            'low_adherence_flag': int(medication_adherence_score < 60.0),
            'dropout_status': dropout_status,
            'days_to_event': max(days_to_event, 1),
            'study': study_label,
            'has_real_qol': False,
            'usubjid': f'{study_label}_{subjid}',
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    configs = [
        ('../../data/pds/Colorec_Amgen_2005_262/SAS dataset - 20040249', 'Colorec_Amgen_2005_262',
         'disposit.sas7bdat', 'DSEOS', 'LASTOSDY'),
        ('../../data/pds/Colorec_Amgen_2006_263', 'Colorec_Amgen_2006_263',
         'disposit.sas7bdat', 'DSREAS', 'DSDY'),
        ('../../data/pds/Colorec_Amgen_2006_264/SAS dataset - 20050203', 'Colorec_Amgen_2006_264',
         'disposit.sas7bdat', 'DSREAS', 'DSDY'),
        ('../../data/pds/HeadNe_Amgen_2007_265', 'HeadNe_Amgen_2007_265',
         'disposit.sas7bdat', 'DSEOS', 'DSDY'),
    ]
    results = []
    for base_dir, label, disp_file, reason_col, day_col in configs:
        df = extract(base_dir, label, disp_file, reason_col, day_col)
        print(f"{label}: {len(df)} patients with real visit data, {df['dropout_status'].sum()} behavioral")
        results.append(df)

    combined = pd.concat(results, ignore_index=True)
    combined.to_csv('amgen_group_real.csv', index=False)
    print("saved amgen_group_real.csv, total rows:", len(combined))
