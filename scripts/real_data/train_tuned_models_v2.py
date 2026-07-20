"""
Trains Cox+XGBoost+SHAP on real_dataset_v2_final.csv (the early-window,
genuinely prospective rebuild - see docs/real_dataset_v2_early_window_log.md)
using the exact same methodology as train_tuned_models.py's real-data
pipeline (Optuna 50 trials, grouped CV, same 4 held-out studies), so the
V1 vs V2 comparison isolates the effect of windowing the features, not a
methodology difference. Does not retrain synthetic - V1's synthetic
results (already saved) are the reference point, there is no "early
window" version of the synthetic generator to compare against.
"""
import pandas as pd
from pathlib import Path
from train_tuned_models import run_pipeline, COX_COVARIATES_REAL, HOLDOUT_STUDIES, OUT_DIR

print("=" * 70)
print("REAL DATA V2 (60-day early window, prospective)")
print("=" * 70)
df_real_v2 = pd.read_csv(OUT_DIR / 'real_dataset_v2_final.csv')
result, model, scaler, cox, train_pool, test_pool, holdout = run_pipeline(
    'real_v2', df_real_v2, COX_COVARIATES_REAL, group_col='study', holdout_studies=HOLDOUT_STUDIES
)
print("DONE")
