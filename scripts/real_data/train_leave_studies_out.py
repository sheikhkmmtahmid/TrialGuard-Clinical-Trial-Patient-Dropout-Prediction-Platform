"""
Harder, more honest test than a random patient-level split: hold out
ENTIRE STUDIES the model never sees during training, not just random
patients from the same 22 studies. This checks whether the model is
learning genuine patient-risk patterns, or partly just learning
"which study does this patient's data look like it came from", since
dropout rate varies 3.0% to 19.2% across the pooled studies.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss
from lifelines import CoxPHFitter
import xgboost as xgb
import json

MODEL_FEATURE_COLUMNS = [
    'age', 'gender_encoded', 'ethnicity_encoded', 'condition_severity_encoded',
    'visit_number', 'cumulative_missed_visits', 'visit_frequency_rate',
    'days_since_last_visit', 'days_between_visits_mean', 'days_between_visits_std',
    'adverse_events_count', 'adverse_event_rate', 'adverse_event_trend',
    'medication_adherence_score', 'medication_adherence_trend',
    'quality_of_life_score', 'qol_score_trend',
    'early_dropout_signal', 'high_adverse_event_flag', 'low_adherence_flag',
]
COX_COVARIATES = ['age', 'condition_severity_encoded', 'adverse_event_rate']

# Held-out studies: picked to cover a real spread of dropout rates
# (19.2% down to 3.3%) and different data sources (PDS + ImmPort), not
# cherry-picked to make the number look good either way.
HOLDOUT_STUDIES = [
    'Colorec_Amgen_2005_262',   # 19.2% rate, largest single study
    'Breast_EliLill_2008_168',  # 10.4% rate
    'Glioma_2008_441',          # 5.0% rate
    'LungSm_Amgen_2002_266',    # 3.3% rate, lowest
]

df = pd.read_csv('real_dataset_final.csv')
test_df = df[df['study'].isin(HOLDOUT_STUDIES)].copy()
train_df = df[~df['study'].isin(HOLDOUT_STUDIES)].copy()

print(f"Train: {len(train_df)} patients from {train_df['study'].nunique()} studies, "
      f"{train_df['dropout_status'].sum()} events")
print(f"Test (entirely unseen studies): {len(test_df)} patients from {test_df['study'].nunique()} studies, "
      f"{test_df['dropout_status'].sum()} events")
print(f"Held-out studies: {HOLDOUT_STUDIES}")

X_train = train_df[MODEL_FEATURE_COLUMNS].values
y_train = train_df['dropout_status'].values
X_test = test_df[MODEL_FEATURE_COLUMNS].values
y_test = test_df['dropout_status'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cox_df = train_df[COX_COVARIATES + ['days_to_event', 'dropout_status']].copy()
cox_df['days_to_event'] = cox_df['days_to_event'].clip(lower=1)
cph = CoxPHFitter(penalizer=0.5)
cph.fit(cox_df, duration_col='days_to_event', event_col='dropout_status')
c_index_train = cph.concordance_index_


def hazard_ratios_for(frame):
    sub = frame[COX_COVARIATES].copy()
    log_hr = cph.predict_log_partial_hazard(sub)
    return np.exp(log_hr.values if hasattr(log_hr, 'values') else log_hr)


hr_train = hazard_ratios_for(train_df)
hr_test = hazard_ratios_for(test_df)
X_train_full = np.hstack([X_train_scaled, hr_train.reshape(-1, 1)])
X_test_full = np.hstack([X_test_scaled, hr_test.reshape(-1, 1)])

scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
    reg_alpha=0.3, reg_lambda=1.2, scale_pos_weight=scale_pos_weight,
    random_state=42, eval_metric='logloss',
)
model.fit(X_train_full, y_train)

y_prob = model.predict_proba(X_test_full)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
brier = brier_score_loss(y_test, y_prob)

print(f"\n=== Leave-studies-out test (n={len(y_test)}, events={int(y_test.sum())}, "
      f"studies never seen in training) ===")
print(f"ROC-AUC:   {auc:.4f}")
print(f"F1:        {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Brier:     {brier:.4f}")

rng = np.random.default_rng(42)
boot_aucs = []
idx = np.arange(len(y_test))
for _ in range(2000):
    sample = rng.choice(idx, size=len(idx), replace=True)
    if len(np.unique(y_test[sample])) < 2:
        continue
    boot_aucs.append(roc_auc_score(y_test[sample], y_prob[sample]))
ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
print(f"\nBootstrap 95% CI for AUC ({len(boot_aucs)} resamples): {ci_low:.4f} - {ci_high:.4f}")

results = {
    'n_train': int(len(X_train)), 'n_test': int(len(X_test)),
    'n_events_test': int(y_test.sum()), 'holdout_studies': HOLDOUT_STUDIES,
    'xgb_roc_auc': round(float(auc), 4),
    'xgb_roc_auc_ci_95': [round(float(ci_low), 4), round(float(ci_high), 4)],
    'xgb_f1': round(float(f1), 4), 'xgb_precision': round(float(precision), 4),
    'xgb_recall': round(float(recall), 4), 'calibration_brier_score': round(float(brier), 4),
}
with open('leave_studies_out_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nsaved leave_studies_out_results.json")
