import pandas as pd
import numpy as np

MODEL_FEATURE_COLUMNS = [
    'age', 'gender_encoded', 'ethnicity_encoded', 'condition_severity_encoded',
    'visit_number', 'cumulative_missed_visits', 'visit_frequency_rate',
    'days_since_last_visit', 'days_between_visits_mean', 'days_between_visits_std',
    'adverse_events_count', 'adverse_event_rate', 'adverse_event_trend',
    'medication_adherence_score', 'medication_adherence_trend',
    'quality_of_life_score', 'qol_score_trend',
    'early_dropout_signal', 'high_adverse_event_flag', 'low_adherence_flag',
]

df = pd.read_csv('real_combined_v2.csv')
df['age'] = pd.to_numeric(df['age'], errors='coerce')

n_missing_age = df['age'].isna().sum()
real_median_age = df['age'].median()
df['age'] = df['age'].fillna(real_median_age)
print(f"Imputed {n_missing_age} missing ages with the real cohort's own median ({real_median_age:.0f}), "
      f"not a synthetic value.")

# Recompute the two flags that depend on adverse_event_rate / medication_adherence_score
# (already set at extraction time, but recompute here for consistency across all sources).
# early_dropout_signal must be built only from things known WITHOUT already
# knowing the outcome (visit behavior), never from dropout_status itself,
# that would be feeding the model the answer as one of its own inputs.
# An earlier version of this line accidentally OR'd in a dropout_status
# condition, caught and fixed here before any training ran on it.
df['high_adverse_event_flag'] = (df['adverse_event_rate'] > 3.0).astype(int)
df['low_adherence_flag'] = (df['medication_adherence_score'] < 60.0).astype(int)
df['early_dropout_signal'] = (df['cumulative_missed_visits'] >= 2).astype(int)

for c in MODEL_FEATURE_COLUMNS:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df[MODEL_FEATURE_COLUMNS] = df[MODEL_FEATURE_COLUMNS].fillna(0)

df['days_to_event'] = pd.to_numeric(df['days_to_event'], errors='coerce')
n_missing_days = df['days_to_event'].isna().sum()
df['days_to_event'] = df['days_to_event'].fillna(df['days_to_event'].median()).clip(lower=1)
print(f"Filled {n_missing_days} missing days_to_event values with the real cohort's own median.")

print(f"\nFinal real dataset: {len(df)} patients, {df['dropout_status'].sum()} behavioral dropouts")
print(f"Studies included: {df['study'].nunique()}")
print(f"Patients with real quality-of-life data: {df['has_real_qol'].sum()}")

df.to_csv('real_dataset_v2_final.csv', index=False)
print("saved real_dataset_v2_final.csv")
