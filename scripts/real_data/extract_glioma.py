"""
Extract real per-patient feature rows for Glioma_EMDSero_2008_441 and
Glioma_EMDSero_2009_440 (SDTM: ds, ex, qs, ae, sv, dm domains).

Output columns match core.utils.data_pipeline.FEATURE_COLUMNS exactly, plus
dropout_status and days_to_event, one row per patient (last known state,
matching how the live app and train_models.py build a feature row: the
final visit's cumulative values).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from rebuild_missed_visits import planned_days_for_arm, count_missed

BEHAVIORAL = {'subject withdrew consent', 'withdrawal of consent', 'withdrew consent',
              'lost to follow-up'}

_SEVERITY_DEFAULT = 1  # moderate, used only if no severity signal exists in the SDTM extract
_GENDER_MAP = {'M': 0, 'F': 1, 'F ': 1, 'MALE': 0, 'FEMALE': 1}


def _linear_trend(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    if np.std(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def extract(base_dir, study_label, cutoff_days=None):
    """
    cutoff_days: if set, every feature below is computed using only events
    at day <= cutoff_days. dropout_status and days_to_event are NEVER
    windowed - days_to_event's fallback (when term_days has no entry) uses
    the FULL, unwindowed visit history.

    count_missed() already only counts planned visits up to the patient's
    own LAST ACTUAL visit day as "relevant" (see rebuild_missed_visits.py),
    so passing it the windowed visit_days automatically gives the right
    early-window answer - the full planned schedule doesn't need separate
    windowing, the function's own logic handles that truncation.
    """
    base = Path(base_dir)
    dm = pd.read_sas(base / 'dm.sas7bdat', format='sas7bdat', encoding='latin1')
    ds = pd.read_sas(base / 'ds.sas7bdat', format='sas7bdat', encoding='latin1')
    ex = pd.read_sas(base / 'ex.sas7bdat', format='sas7bdat', encoding='latin1')
    ae = pd.read_sas(base / 'ae.sas7bdat', format='sas7bdat', encoding='latin1')
    try:
        qs = pd.read_sas(base / 'qs.sas7bdat', format='sas7bdat', encoding='latin1')
    except Exception:
        qs = None
    try:
        sv = pd.read_sas(base / 'sv.sas7bdat', format='sas7bdat', encoding='latin1')
    except Exception:
        sv = None
    try:
        tv = pd.read_sas(base / 'tv.sas7bdat', format='sas7bdat', encoding='latin1')
    except Exception:
        tv = None

    ds['_norm'] = ds['DSDECOD'].astype(str).str.strip().str.lower()
    beh_ids = set(ds[ds['_norm'].isin(BEHAVIORAL)]['USUBJID'])

    # Termination day: DSSTDY (study day of the disposition event) for the
    # behavioral row itself, gives a real days_to_event for these patients.
    term_days = (
        ds[ds['_norm'].isin(BEHAVIORAL)]
        .groupby('USUBJID')['DSSTDY'].max()
    )

    rows = []
    for _, pat in dm.iterrows():
        usubjid = pat['USUBJID']
        age = pat.get('AGE')
        sex = str(pat.get('SEX', '')).strip().upper()
        gender_encoded = _GENDER_MAP.get(sex, 3)

        pat_ex_full = ex[ex['USUBJID'] == usubjid].copy()
        pat_ae_full = ae[ae['USUBJID'] == usubjid].copy()
        pat_qs_full = qs[qs['USUBJID'] == usubjid].copy() if qs is not None else pd.DataFrame()

        full_visit_days = sorted(pat_ex_full['EXSTDY'].dropna().unique().tolist()) if 'EXSTDY' in pat_ex_full.columns else []
        dropout_status = int(usubjid in beh_ids)
        days_to_event = float(term_days.get(usubjid, full_visit_days[-1] if full_visit_days else 30))

        if cutoff_days is not None:
            pat_ex = pat_ex_full[pat_ex_full['EXSTDY'] <= cutoff_days] if 'EXSTDY' in pat_ex_full.columns else pat_ex_full
            pat_ae = pat_ae_full[pat_ae_full['AESTDY'] <= cutoff_days] if 'AESTDY' in pat_ae_full.columns else pat_ae_full
            if not pat_qs_full.empty:
                qcol_pre = 'QSDY' if 'QSDY' in pat_qs_full.columns else pat_qs_full.columns[0]
                pat_qs = pat_qs_full[pat_qs_full[qcol_pre] <= cutoff_days]
            else:
                pat_qs = pat_qs_full
        else:
            pat_ex, pat_ae, pat_qs = pat_ex_full, pat_ae_full, pat_qs_full

        # Visit cadence: real dosing visit days from EX (EXSTDY = study day of dose)
        visit_days = sorted(pat_ex['EXSTDY'].dropna().unique().tolist()) if 'EXSTDY' in pat_ex.columns else []
        visit_number = len(visit_days)
        if visit_number == 0:
            continue  # no real visit history, cannot build a real feature row

        gaps = np.diff(visit_days) if len(visit_days) > 1 else np.array([])
        days_since_last_visit = float(gaps[-1]) if len(gaps) else 0.0
        days_between_visits_mean = float(np.mean(gaps)) if len(gaps) else 0.0
        days_between_visits_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
        visit_frequency_rate = visit_number / max(visit_days[-1], 1) * 30

        n_ae = len(pat_ae)
        ae_rate = n_ae / visit_number if visit_number else 0.0
        # AE trend: bin AE onset study-day into per-visit buckets, count per bucket
        if 'AESTDY' in pat_ae.columns and n_ae > 0 and len(visit_days) > 1:
            ae_days = pat_ae['AESTDY'].dropna().values
            bucket_counts = [np.sum((ae_days >= visit_days[i]) & (ae_days < visit_days[i + 1]))
                              for i in range(len(visit_days) - 1)]
            ae_trend = _linear_trend(bucket_counts)
        else:
            ae_trend = 0.0

        # QoL: real score if the qs domain has a total/composite score row
        if not pat_qs.empty and 'QSSTRESN' in pat_qs.columns:
            qol_series = pat_qs.sort_values('QSDY' if 'QSDY' in pat_qs.columns else pat_qs.columns[0])['QSSTRESN'].dropna()
            qol_score = float(qol_series.iloc[-1]) if len(qol_series) else 50.0
            qol_trend = _linear_trend(qol_series.tolist())
            has_real_qol = len(qol_series) > 0
        else:
            qol_score, qol_trend, has_real_qol = 50.0, 0.0, False

        # Real missed-visit count: compare this patient's actual visit days
        # (already windowed above, if cutoff_days is set) against their own
        # arm's real planned schedule (tv.sas7bdat), verified against a
        # hand-checked example first, see scripts/real_data/rebuild_missed_visits.py.
        cumulative_missed = 0
        if tv is not None and 'ARMCD' in pat.index and pd.notna(pat.get('ARMCD')):
            planned = planned_days_for_arm(tv, pat['ARMCD'])
            cumulative_missed = count_missed(visit_days, planned)

        rows.append({
            'age': age,
            'gender_encoded': gender_encoded,
            'ethnicity_encoded': 5,  # not collected in this SDTM extract; imputed as "unknown"
            'condition_severity_encoded': _SEVERITY_DEFAULT,
            'visit_number': visit_number,
            'cumulative_missed_visits': cumulative_missed,
            'visit_frequency_rate': visit_frequency_rate,
            'days_since_last_visit': days_since_last_visit,
            'days_between_visits_mean': days_between_visits_mean,
            'days_between_visits_std': days_between_visits_std,
            'adverse_events_count': n_ae,
            'adverse_event_rate': ae_rate,
            'adverse_event_trend': ae_trend,
            'medication_adherence_score': 85.0,  # not collected in this extract; imputed
            'medication_adherence_trend': 0.0,
            'quality_of_life_score': qol_score,
            'qol_score_trend': qol_trend,
            'early_dropout_signal': 0,
            'high_adverse_event_flag': int(ae_rate > 3.0),
            'low_adherence_flag': 0,
            'dropout_status': dropout_status,
            'days_to_event': max(days_to_event, 1),
            'study': study_label,
            'has_real_qol': has_real_qol,
            'usubjid': usubjid,
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df1 = extract('../../data/pds/Glioma_EMDSero_2008_441/DATA_SAS', 'Glioma_2008_441')
    df2 = extract('../../data/pds/Glioma_EMDSero_2009_440/DATA', 'Glioma_2009_440')
    combined = pd.concat([df1, df2], ignore_index=True)
    print(f"Glioma 2008: {len(df1)} patients, {df1['dropout_status'].sum()} behavioral dropouts")
    print(f"Glioma 2009: {len(df2)} patients, {df2['dropout_status'].sum()} behavioral dropouts")
    combined.to_csv('glioma_real.csv', index=False)
    print("saved glioma_real.csv")
