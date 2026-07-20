"""
Extract real per-patient feature rows from MUSIC (MUerte Subita en
Insuficiencia Cardiaca / Sudden Cardiac Death in Heart Failure), a real,
already-verified 992-patient European heart-failure cohort sitting in
data/music/ since earlier in this project but never extracted into a
usable study.

Real fields used, all verified against subject-info_codes.csv /
subject-info_definitions.csv before use:
- Age, Gender (male=1) -> real demographics
- NYHA class -> the standard heart-failure severity scale (real severity,
  same role ECOG played for oncology and MDS-UPDRS Part III for PPMI)
- Follow-up period from enrollment (days) -> real, day-precision
  days_to_event (better precision than PPMI's month-only dates)
- Exit of the study -> outcome. The shipped codebook only documents code
  0 ("survivor"); codes 1/2/3 are undefined in the file itself. Resolved
  by cross-referencing against Cause of death instead of guessing: every
  patient coded 3 has a real non-zero cause-of-death code, and every
  patient coded 1 or 2 has none - so 3 = death (not behavioral), 1 and 2
  = confirmed non-death exits (behavioral, matches this project's
  standing definition), NaN = no exit recorded (still followed/censored).

No per-visit longitudinal data exists in this file (one baseline
assessment per patient, not a visit series), so visit_number is honestly
set to 1 and the visit-cadence features are left at their default -
there is no real spacing to compute from a single row.
"""
import pandas as pd
from pathlib import Path

BASE = Path(r'D:\Trial Guard\data\music')

# Verified via cross-tab against Cause of death (see docstring): 3 = death,
# 1 and 2 = confirmed non-death exits.
BEHAVIORAL_EXIT_CODES = {1, 2}
DEATH_EXIT_CODE = 3


def extract(cutoff_days=None):
    """
    cutoff_days: accepted for API consistency with every other extract()
    in this project, but has no effect here - this source has exactly one
    baseline assessment per patient, no visit series exists to window.
    Already effectively a "day-0 only" view by construction.
    """
    df = pd.read_csv(BASE / 'subject-info.csv', sep=';')

    df['age_clean'] = df['Age'].replace('>89', '90').astype(float)
    df['gender_encoded'] = df['Gender (male=1)'].map({1: 0, 0: 1})  # flip to this project's M=0/F=1
    df['severity'] = df['NYHA class'].map({2: 1, 3: 2})  # only classes II/III present in this cohort

    def dropout_for(exit_code):
        if pd.isna(exit_code):
            return 0  # no exit recorded - still followed, treated as retained like every other real source
        return int(int(exit_code) in BEHAVIORAL_EXIT_CODES)

    df['dropout_status'] = df['Exit of the study'].apply(dropout_for)

    rows = []
    for _, pat in df.iterrows():
        rows.append({
            'age': pat['age_clean'],
            'gender_encoded': pat['gender_encoded'],
            'ethnicity_encoded': 5,  # not collected in this extract; honestly marked unknown
            'condition_severity_encoded': pat['severity'] if pd.notna(pat['severity']) else 1,
            'visit_number': 1,  # single baseline assessment, no real visit series in this file
            'cumulative_missed_visits': 0,
            'visit_frequency_rate': 0.0,
            'days_since_last_visit': 0.0,
            'days_between_visits_mean': 0.0,
            'days_between_visits_std': 0.0,
            'adverse_events_count': 0,
            'adverse_event_rate': 0.0,
            'adverse_event_trend': 0.0,
            'medication_adherence_score': 85.0,  # medication list exists, but that's what drugs, not adherence
            'medication_adherence_trend': 0.0,
            'quality_of_life_score': 50.0,  # no QoL instrument in this extract
            'qol_score_trend': 0.0,
            'early_dropout_signal': 0,
            'high_adverse_event_flag': 0,
            'low_adherence_flag': 0,
            'dropout_status': pat['dropout_status'],
            'days_to_event': max(float(pat['Follow-up period from enrollment (days)']), 1),
            'study': 'MUSIC',
            'has_real_qol': False,
            'usubjid': f"MUSIC_{pat['Patient ID']}",
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = extract()
    print(f"MUSIC: {len(df)} patients, {df['dropout_status'].sum()} behavioral "
          f"({df['dropout_status'].mean()*100:.1f}%)")
    df.to_csv('music_real.csv', index=False)
    print("saved music_real.csv")
