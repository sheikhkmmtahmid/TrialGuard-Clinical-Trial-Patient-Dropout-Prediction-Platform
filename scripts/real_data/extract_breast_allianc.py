"""
Breast_Allianc_2008_158 (CALGB 40502). No day-level fields exist in any of
the three CSVs (cycles, efficacy, ae), so cycleno is used as the visit
count, with 21 days/cycle as the protocol-typical spacing (documented
here rather than silently assumed). No per-visit AE date exists either,
so adverse_event_trend cannot be honestly computed and is left at 0
(no signal claimed) rather than guessed. AGECAT is a 3-bucket category,
not raw age, converted to each bucket's midpoint.
"""
import pandas as pd
import numpy as np
from pathlib import Path

CYCLE_DAYS = 21
BEHAVIORAL_TXENDREAS = {22}  # w/d after starting rx (documented code only, see docs/pds_validation_report.md)
AGECAT_MIDPOINT = {1: 35, 2: 60, 3: 75}


def extract(cutoff_days=None):
    """
    cutoff_days: if set, visit/cycle-based features and the adherence
    proxy (both drawn from real cycle numbers) are computed using only
    cycles at day <= cutoff_days. dropout_status and days_to_event are
    NEVER windowed - days_to_event's fallback uses the FULL, unwindowed
    cycle history. adverse_events_count has no per-event day field at
    all in this source (verified, same as the original script's own
    note), so it stays a whole-history count in both versions - a real,
    pre-existing limitation of this study, not a new one from windowing.
    """
    base = Path('../../data/pds/Breast_Allianc_2008_158')
    cyc = pd.read_csv(base / 'c40502_cycles.csv')
    eff = pd.read_csv(base / 'c40502_efficacy.csv')
    ae = pd.read_csv(base / 'c40502_ae.csv')

    beh_ids = set(eff[eff['txendreas'].isin(BEHAVIORAL_TXENDREAS)]['MASK_ID'])

    rows = []
    for mask_id in eff['MASK_ID'].unique():
        pat_eff = eff[eff['MASK_ID'] == mask_id].iloc[0]
        pat_cyc_full = cyc[cyc['MASK_ID'] == mask_id]
        pat_ae = ae[ae['MASK_ID'] == mask_id]  # no usable day field, see docstring

        full_cycles = sorted(pat_cyc_full['cycleno'].dropna().unique().tolist())
        full_visit_days = [c * CYCLE_DAYS for c in full_cycles]
        days_to_event = float(full_visit_days[-1] if full_visit_days else 30)
        dropout_status = int(mask_id in beh_ids)

        if cutoff_days is not None:
            max_cycle = cutoff_days / CYCLE_DAYS
            pat_cyc = pat_cyc_full[pat_cyc_full['cycleno'] <= max_cycle]
        else:
            pat_cyc = pat_cyc_full

        cycles = sorted(pat_cyc['cycleno'].dropna().unique().tolist())
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

        agecat = pat_eff.get('AGECAT')
        age = AGECAT_MIDPOINT.get(int(agecat) if pd.notna(agecat) else None, 55)
        sex_id = pat_eff.get('sex_id')
        gender_encoded = 1 if sex_id == 2 else (0 if sex_id == 1 else 3)

        # Real adherence proxy: verified against c40502_datadictionary.pdf,
        # "Pac - dose modified pac_dosemod 1=yes" (blank means not modified,
        # standard CRF checkbox convention, confirmed by the dictionary
        # wording itself, not assumed). Percentage of this patient's real
        # cycles with no recorded dose modification.
        if len(pat_cyc) > 0 and 'pac_dosemod' in pat_cyc.columns:
            n_normal = pat_cyc['pac_dosemod'].isna().sum()
            medication_adherence_score = 100.0 * n_normal / len(pat_cyc)
        else:
            medication_adherence_score = 85.0

        rows.append({
            'age': age,
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
            'adverse_event_trend': 0.0,
            'medication_adherence_score': medication_adherence_score,
            'medication_adherence_trend': 0.0,
            'quality_of_life_score': 50.0,
            'qol_score_trend': 0.0,
            'early_dropout_signal': 0,
            'high_adverse_event_flag': int(ae_rate > 3.0),
            'low_adherence_flag': int(medication_adherence_score < 60.0),
            'dropout_status': dropout_status,
            'days_to_event': max(days_to_event, 1),
            'study': 'Breast_Allianc_2008_158',
            'has_real_qol': False,
            'usubjid': mask_id,
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = extract()
    print(f"Breast_Allianc_2008_158: {len(df)} patients with real visit data, {df['dropout_status'].sum()} behavioral")
    df.to_csv('breast_allianc_real.csv', index=False)
