"""
Train Cox PH + XGBoost on the real combined dataset (real_dataset_final.csv),
using the exact same MODEL_FEATURE_COLUMNS the live app now trains on.
Evaluated with a held-out test split plus a 2000-resample bootstrap CI,
same methodology used earlier in this project's real-data validation work.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
# cumulative_missed_visits and medication_adherence_score are dropped from
# the real-data Cox fit specifically: neither could be honestly
# reconstructed for most of these real studies (no scheduled-visit
# calendar to detect missed visits against; no clean adherence score in
# almost any source), so they ended up constant (0 and 85 respectively)
# across nearly the whole real dataset, and a constant column has no
# variance for Cox to fit against. This is a real data limitation, not a
# methodology choice, flagged here rather than silently worked around.
COX_COVARIATES = ['age', 'condition_severity_encoded', 'adverse_event_rate']

df = pd.read_csv('real_dataset_final.csv')
print(f"Training on {len(df)} real patients, {df['dropout_status'].sum()} real behavioral dropouts "
      f"({df['dropout_status'].mean()*100:.1f}% rate)")

X = df[MODEL_FEATURE_COLUMNS].values
y = df['dropout_status'].values

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.25, stratify=y, random_state=42
)
print(f"Train: {len(X_train)} ({y_train.sum()} events) | Test: {len(X_test)} ({y_test.sum()} events)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Cox PH on the training split only ---
cox_df = df_train[COX_COVARIATES + ['days_to_event', 'dropout_status']].copy()
cox_df['days_to_event'] = cox_df['days_to_event'].clip(lower=1)
cph = CoxPHFitter(penalizer=0.5)
cph.fit(cox_df, duration_col='days_to_event', event_col='dropout_status')
c_index = cph.concordance_index_
print(f"Cox concordance index (train): {c_index:.4f}")


def hazard_ratios_for(frame):
    sub = frame[COX_COVARIATES].copy()
    log_hr = cph.predict_log_partial_hazard(sub)
    return np.exp(log_hr.values if hasattr(log_hr, 'values') else log_hr)


hr_train = hazard_ratios_for(df_train)
hr_test = hazard_ratios_for(df_test)

X_train_full = np.hstack([X_train_scaled, hr_train.reshape(-1, 1)])
X_test_full = np.hstack([X_test_scaled, hr_test.reshape(-1, 1)])

# --- XGBoost, reasonable defaults (no Optuna search here, time-boxed;
# flagged explicitly in the report as a methodology difference from the
# synthetic model, which was tuned with 50 Optuna trials) ---
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

print(f"\n=== Held-out test results (real data, n={len(y_test)}, events={y_test.sum()}) ===")
print(f"ROC-AUC:   {auc:.4f}")
print(f"F1:        {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Brier:     {brier:.4f}")

# --- threshold sweep ---
sweep = []
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    yp = (y_prob >= t).astype(int)
    sweep.append({
        'threshold': t,
        'recall': round(float(recall_score(y_test, yp, zero_division=0)), 4),
        'precision': round(float(precision_score(y_test, yp, zero_division=0)), 4),
        'fraction_flagged': round(float(yp.mean()), 4),
    })
print("\nThreshold sweep:")
for s in sweep:
    print(s)

# --- bootstrap CI for AUC ---
rng = np.random.default_rng(42)
n_boot = 2000
boot_aucs = []
idx = np.arange(len(y_test))
for _ in range(n_boot):
    sample = rng.choice(idx, size=len(idx), replace=True)
    if len(np.unique(y_test[sample])) < 2:
        continue
    boot_aucs.append(roc_auc_score(y_test[sample], y_prob[sample]))
ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
print(f"\nBootstrap 95% CI for AUC ({len(boot_aucs)} resamples): {ci_low:.4f} - {ci_high:.4f}")


# --- SHAP: does the model give consistent, trustworthy explanations? ---
import shap as shap_lib
sys_path_note = None
import sys
sys.path.insert(0, r'D:\Trial Guard')
from core.utils.shap_explainer import compute_shap_stability_score, compute_shap_robustness_score

bg_rng = np.random.default_rng(42)
bg_size = min(100, len(X_train_full))
background_X = X_train_full[bg_rng.choice(len(X_train_full), size=bg_size, replace=False)]
explainer = shap_lib.TreeExplainer(model, data=background_X, feature_perturbation='interventional')

shap_sample = X_test_full[:min(500, len(X_test_full))]
shap_vals = explainer.shap_values(shap_sample)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

stability = compute_shap_stability_score(shap_vals)
robustness = compute_shap_robustness_score(explainer, X_test_full)
print(f"\n=== SHAP (real data) ===")
print(f"cross-patient similarity: {stability:.4f}")
print(f"same-patient robustness:  {robustness:.4f}")

results = {
    'n_total': int(len(df)), 'n_events_total': int(df['dropout_status'].sum()),
    'n_train': int(len(X_train)), 'n_test': int(len(X_test)),
    'n_events_test': int(y_test.sum()),
    'cox_concordance_index': round(float(c_index), 4),
    'xgb_roc_auc': round(float(auc), 4),
    'xgb_roc_auc_ci_95': [round(float(ci_low), 4), round(float(ci_high), 4)],
    'xgb_f1': round(float(f1), 4),
    'xgb_precision': round(float(precision), 4),
    'xgb_recall': round(float(recall), 4),
    'calibration_brier_score': round(float(brier), 4),
    'threshold_sweep': sweep,
    'shap_stability_score': round(float(stability), 4),
    'shap_robustness_score': round(float(robustness), 4),
}
with open('real_data_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nsaved real_data_results.json")
