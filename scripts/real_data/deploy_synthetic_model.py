"""
Regenerates the 4 files ml_models/ actually serves (xgb_model.pkl,
cox_model.pkl, scaler.pkl, shap_explainer.pkl) plus evaluation_results.json,
from today's properly Optuna-tuned synthetic model (tuned_model_synthetic.pkl),
using the current 20-feature MODEL_FEATURE_COLUMNS schema.

This fixes the drift found earlier: the live app's serving code
(core/views.py) already builds inputs as [scaled 20 features + Cox hazard
ratio] and expects cox_model.pkl's covariates to be the 5-covariate set
in core/utils/survival_model.py's COX_COVARIATES - which is exactly what
today's synthetic training used, so no code changes are needed, only the
model files.

The stale files were backed up to ml_models_stale_23feature_backup/
before this ran.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load, dump

sys.path.insert(0, r'D:\Trial Guard')

REAL_DATA_DIR = Path(r'D:\Trial Guard\scripts\real_data')
ML_MODELS_DIR = Path(r'D:\Trial Guard\ml_models')
ROOT = Path(r'D:\Trial Guard')

bundle = load(REAL_DATA_DIR / 'tuned_model_synthetic.pkl')
xgb_model = bundle['model']
scaler = bundle['scaler']
cph = bundle['cox']
feature_columns = bundle['feature_columns']
cox_covariates = bundle['cox_covariates']

# Confirm this matches what the live serving code expects before writing
# anything - if it doesn't match, better to fail loudly here than silently
# serve something broken.
from core.utils.data_pipeline import MODEL_FEATURE_COLUMNS
from core.utils.survival_model import COX_COVARIATES
assert feature_columns == MODEL_FEATURE_COLUMNS, \
    f"feature mismatch: {feature_columns} vs {MODEL_FEATURE_COLUMNS}"
assert cox_covariates == COX_COVARIATES, \
    f"cox covariate mismatch: {cox_covariates} vs {COX_COVARIATES}"
print("Schema check passed: feature columns and Cox covariates match the live serving code exactly.")

# ---- cox_model.pkl ----
dump({'model': cph, 'concordance_index': cph.concordance_index_, 'covariates': cox_covariates},
     ML_MODELS_DIR / 'cox_model.pkl')
print(f"Saved cox_model.pkl (concordance_index={cph.concordance_index_:.4f})")

# ---- scaler.pkl ----
dump(scaler, ML_MODELS_DIR / 'scaler.pkl')
print("Saved scaler.pkl")

# ---- rebuild a background sample + full eval set (scaled + hazard ratio),
#      same representation used at serve time, for the SHAP explainer and
#      for regenerating evaluation_results.json ----
df = pd.read_csv(REAL_DATA_DIR / 'synthetic_dataset_matched.csv')
X_raw = df[feature_columns].values
y = df['dropout_status'].values
X_scaled = scaler.transform(X_raw)

sub = df[cox_covariates].copy()
log_hr = cph.predict_log_partial_hazard(sub)
hazard_ratio = np.exp(log_hr.values if hasattr(log_hr, 'values') else log_hr)
X_full = np.hstack([X_scaled, hazard_ratio.reshape(-1, 1)])

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.25, stratify=y, random_state=42)

y_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# ---- shap_explainer.pkl ----
import shap
rng = np.random.default_rng(42)
bg_size = min(100, len(X_train))
background_X = X_train[rng.choice(len(X_train), size=bg_size, replace=False)]
explainer = shap.TreeExplainer(xgb_model, data=background_X, feature_perturbation='interventional')
dump(explainer, ML_MODELS_DIR / 'shap_explainer.pkl')
print("Saved shap_explainer.pkl")

from core.utils.shap_explainer import compute_shap_stability_score, compute_shap_robustness_score
shap_sample = X_test[:min(500, len(X_test))]
shap_vals = explainer.shap_values(shap_sample, check_additivity=False)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
stability = compute_shap_stability_score(shap_vals)
robustness = compute_shap_robustness_score(explainer, X_test)

# ---- xgb_model.pkl + evaluation_results.json ----
from core.utils.xgboost_model import evaluate_at_thresholds

metrics = {
    'xgb_roc_auc': round(float(roc_auc_score(y_test, y_prob)), 4),
    'xgb_f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    'xgb_precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
    'xgb_recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
    'calibration_brier_score': round(float(brier_score_loss(y_test, y_prob)), 4),
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'best_params': {k: v for k, v in bundle['xgb_params'].items()
                     if k not in ('use_label_encoder', 'eval_metric', 'tree_method')},
    'threshold_sweep': evaluate_at_thresholds(y_test, y_prob),
    'shap_stability_score': stability,
    'shap_robustness_score': robustness,
    'cox_concordance_index': round(float(cph.concordance_index_), 4),
    'trained_on': 'synthetic (freshly generated, n=6476, Optuna-tuned 50 trials, '
                   'grouped/stratified CV) - see docs/model_training_log.md',
}

dump({'model': xgb_model, 'feature_columns': feature_columns, 'metrics': metrics},
     ML_MODELS_DIR / 'xgb_model.pkl')
print(f"Saved xgb_model.pkl (AUC={metrics['xgb_roc_auc']})")

(ROOT / 'evaluation_results.json').write_text(json.dumps(metrics, indent=2))
print("Saved evaluation_results.json")

print("\nDONE. Live app model files now match the current 20-feature schema.")
