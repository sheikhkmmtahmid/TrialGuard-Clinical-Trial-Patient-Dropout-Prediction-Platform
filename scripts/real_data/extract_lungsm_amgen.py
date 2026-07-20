import pandas as pd
import numpy as np
from pathlib import Path

BEHAVIORAL = {'consent withdrawn', 'lost to follow-up'}

# Named-timepoint QoL columns (wide format, not a day-stamped series) mapped
# to their real approximate study day, verified against the field names
# themselves (VASB=baseline, VASW7=week 7, etc.). VASEOS (end-of-study) is
# excluded from the early-window view since its real collection day varies
# per patient and isn't known - including it would leak how close to the
# end of the study the assessment actually happened.
QOL_COL_DAYS = {'VASB': 0, 'VASW7': 49, 'VASW13': 91, 'VASW24': 168, 'VASW50': 350}


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
    windowed - days_to_event's fallback (when the real disposition day is
    missing) uses the FULL, unwindowed visit history.
    """
    base = Path('../../data/pds/LungSm_Amgen_2002_266/SAS dataset - 20010145')
    disp = pd.read_sas(base / 'c_disp.sas7bdat', format='sas7bdat', encoding='latin1')
    doses = pd.read_sas(base / 'c_doses.sas7bdat', format='sas7bdat', encoding='latin1')
    ae = pd.read_sas(base / 'c_ae.sas7bdat', format='sas7bdat', encoding='latin1')
    qol = pd.read_sas(base / 'a_qol.sas7bdat', format='sas7bdat', encoding='latin1')

    disp['_norm'] = disp['EOS'].astype(str).str.strip().str.lower()
    beh_ids = set(disp[disp['_norm'].isin(BEHAVIORAL)]['SUBJID'])
    days_map = disp.set_index('SUBJID')['EOSDAY'].to_dict()

    qol_indexed = qol.set_index('SUBJID')
    if cutoff_days is None:
        # Full-history v1 behavior: all 6 named timepoints, including
        # VASEOS, exactly as the original script always did.
        qol_cols_for_run = ['VASB', 'VASW7', 'VASW13', 'VASW24', 'VASW50', 'VASEOS']
    else:
        # Early-window: only timepoints with a known real day <= cutoff.
        # VASEOS excluded - its real collection day varies per patient and
        # isn't known, so it can't be honestly windowed.
        qol_cols_for_run = [c for c in QOL_COL_DAYS if QOL_COL_DAYS[c] <= cutoff_days]

    rows = []
    for subjid in disp['SUBJID'].unique():
        pat_disp = disp[disp['SUBJID'] == subjid].iloc[0]
        # Real SEX values here are spelled out ("Male"/"Female"), verified
        # against the source file, not single letters - a bare {'M':0,'F':1}
        # lookup silently missed every patient, caught during cleaning.
        sex = str(pat_disp.get('SEX', '')).strip().upper()

        pat_doses_full = doses[doses['SUBJID'] == subjid]
        pat_ae_full = ae[ae['SUBJID'] == subjid]

        full_visit_days = sorted(pat_doses_full['STUDYDAY'].dropna().unique().tolist())
        term_day = days_map.get(subjid)
        days_to_event = float(term_day) if pd.notna(term_day) else float(full_visit_days[-1] if full_visit_days else 30)
        dropout_status = int(subjid in beh_ids)

        if cutoff_days is not None:
            pat_doses = pat_doses_full[pat_doses_full['STUDYDAY'] <= cutoff_days]
            pat_ae = pat_ae_full[pat_ae_full['STUDYDAY'] <= cutoff_days] if 'STUDYDAY' in pat_ae_full.columns else pat_ae_full
        else:
            pat_doses, pat_ae = pat_doses_full, pat_ae_full

        visit_days = sorted(pat_doses['STUDYDAY'].dropna().unique().tolist())
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
        if 'STUDYDAY' in pat_ae.columns and n_ae > 0 and len(visit_days) > 1:
            ae_days = pat_ae['STUDYDAY'].dropna().values
            bucket_counts = [np.sum((ae_days >= visit_days[i]) & (ae_days < visit_days[i + 1]))
                              for i in range(len(visit_days) - 1)]
            ae_trend = _linear_trend(bucket_counts)
        else:
            ae_trend = 0.0

        has_real_qol = False
        qol_score, qol_trend = 50.0, 0.0
        if subjid in qol_indexed.index:
            qrow = qol_indexed.loc[subjid]
            series = [qrow[c] for c in qol_cols_for_run if c in qrow.index and pd.notna(qrow[c])]
            if series:
                qol_score = float(series[-1])
                qol_trend = _linear_trend(series)
                has_real_qol = True

        ecog = pd.to_numeric(pat_disp.get('B_ECOG2'), errors='coerce')
        severity = 0 if (pd.notna(ecog) and ecog == 0) else (2 if (pd.notna(ecog) and ecog >= 2) else 1)

        rows.append({
            'age': pat_disp.get('AGE'),
            'gender_encoded': {'M': 0, 'MALE': 0, 'F': 1, 'FEMALE': 1}.get(sex, 3),
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
            'quality_of_life_score': qol_score,
            'qol_score_trend': qol_trend,
            'early_dropout_signal': 0,
            'high_adverse_event_flag': int(ae_rate > 3.0),
            'low_adherence_flag': 0,
            'dropout_status': dropout_status,
            'days_to_event': max(days_to_event, 1),
            'study': 'LungSm_Amgen_2002_266',
            'has_real_qol': has_real_qol,
            'usubjid': subjid,
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = extract()
    print(f"LungSm_Amgen_2002_266: {len(df)} patients with real visit data, {df['dropout_status'].sum()} behavioral")
    df.to_csv('lungsm_amgen_real.csv', index=False)
