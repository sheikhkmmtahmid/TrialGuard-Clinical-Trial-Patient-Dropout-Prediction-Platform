"""
Same size, same 20-feature schema, same train/test/bootstrap methodology
as train_on_real.py, but on synthetic data, for a direct, fair comparison.
"""
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trialguard.settings')
import sys
sys.path.insert(0, r'D:\Trial Guard')
django.setup()

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss
from lifelines import CoxPHFitter
import xgboost as xgb
import json

from core.utils.data_pipeline import (
    generate_synthetic_patients, generate_synthetic_visits, MODEL_FEATURE_COLUMNS,
)

COX_COVARIATES = ['age', 'condition_severity_encoded', 'cumulative_missed_visits',
                   'adverse_event_rate', 'medication_adherence_score']

# Same n as the real combined dataset (6,478), same overall structure:
# generate patients, generate their visits, take each patient's LAST visit
# as their feature row, exactly how the real dataset and the live app both
# build a one-row-per-patient feature table.
patients = generate_synthetic_patients(n=6478, seed=99)
visits = generate_synthetic_visits(patients, visits_per_patient=8)

last_visits = visits.sort_values('visit_number').groupby('patient_idx').tail(1).set_index('patient_idx')
patients = patients.join(last_visits, how='inner')
patients['visit_number'] = patients['visit_number'].fillna(0)

# adverse_event_rate / trend, medication_adherence_trend, qol_trend need full
# visit history per patient, not just the last row, mirroring engineer_features_for_patient.
rows = []
for idx, pat in patients.iterrows():
    pat_visits = visits[visits['patient_idx'] == idx].sort_values('visit_number')
    if pat_visits.empty:
        continue
    ae_hist = pat_visits['adverse_events_count'].tolist()
    adh_hist = pat_visits['medication_adherence_score'].tolist()
    qol_hist = pat_visits['quality_of_life_score'].tolist()

    def trend(vals):
        vals = [v for v in vals if v is not None and not pd.isna(v)]
        if len(vals) < 2:
            return 0.0
        x = np.arange(len(vals), dtype=float)
        y = np.array(vals, dtype=float)
        return float(np.polyfit(x, y, 1)[0]) if np.std(x) > 0 else 0.0

    last = pat_visits.iloc[-1]
    ae_rate = sum(ae_hist) / len(ae_hist)
    rows.append({
        'age': pat['age'], 'gender_encoded': pat['gender_encoded'],
        'ethnicity_encoded': pat['ethnicity_encoded'],
        'condition_severity_encoded': pat['condition_severity_encoded'],
        'visit_number': last['visit_number'],
        'cumulative_missed_visits': last['missed_visits_to_date'],
        'visit_frequency_rate': last['visit_number'] / max(pat_visits['days_since_last_visit'].sum(), 1) * 30,
        'days_since_last_visit': last['days_since_last_visit'],
        'days_between_visits_mean': float(np.mean(pat_visits['days_since_last_visit'][1:])) if len(pat_visits) > 1 else 0.0,
        'days_between_visits_std': float(np.std(pat_visits['days_since_last_visit'][1:])) if len(pat_visits) > 2 else 0.0,
        'adverse_events_count': last['adverse_events_count'],
        'adverse_event_rate': ae_rate,
        'adverse_event_trend': trend(ae_hist),
        'medication_adherence_score': last['medication_adherence_score'],
        'medication_adherence_trend': trend(adh_hist),
        'quality_of_life_score': last['quality_of_life_score'],
        'qol_score_trend': trend(qol_hist),
        'early_dropout_signal': int(last['missed_visits_to_date'] >= 2),
        'high_adverse_event_flag': int(ae_rate > 3.0),
        'low_adherence_flag': int(last['medication_adherence_score'] < 60.0),
        'dropout_status': int(pat['dropout_status']),
        'days_to_event': pat['days_to_event'],
    })

df = pd.DataFrame(rows)
print(f"Synthetic dataset: {len(df)} patients, {df['dropout_status'].sum()} dropouts "
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

cox_df = df_train[COX_COVARIATES + ['days_to_event', 'dropout_status']].copy()
cox_df['days_to_event'] = cox_df['days_to_event'].clip(lower=1)
cph = CoxPHFitter(penalizer=0.1)
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

print(f"\n=== Held-out test results (synthetic data, n={len(y_test)}, events={y_test.sum()}) ===")
print(f"ROC-AUC:   {auc:.4f}")
print(f"F1:        {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Brier:     {brier:.4f}")

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


# --- SHAP: does the model give consistent, trustworthy explanations? ---
import shap as shap_lib
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
print(f"\n=== SHAP (synthetic data) ===")
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
with open(r'D:\Trial Guard\scripts\real_data\synthetic_data_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nsaved synthetic_data_results.json")
