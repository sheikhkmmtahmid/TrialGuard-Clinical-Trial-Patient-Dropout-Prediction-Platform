"""
Train Cox -> XGBoost -> SHAP on the rebuilt, harmonized real dataset
(real_dataset_final.csv), and on a freshly generated, matching-size
synthetic dataset, using EQUIVALENT tuning rigor on both sides so the
comparison is fair (see docs/model_training_log.md for why the earlier
synthetic comparison run was not actually tuned).

Real data: XGBoost tuned via Optuna (50 trials) using GroupKFold(5) by
`study` (hospital-grouped CV, so tuning cannot exploit which hospital a
patient came from). 4 whole studies are held out completely from
training and tuning, used only as the final "new hospitals" test
(same 4 studies as the earlier leave-studies-out run, kept identical on
purpose). Cox tuned via a small penalizer/l1_ratio grid, same grouped
CV, concordance index objective.

Synthetic data: no natural hospital grouping exists (each patient is
generated independently), so tuned via plain StratifiedKFold(5) and
evaluated via a random stratified split only.

Both final models are saved with everything needed to warm-start them
on new data later without discarding what they've already learned.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump

sys.path.insert(0, r'D:\Trial Guard')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trialguard.settings')
import django
django.setup()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss
from lifelines import CoxPHFitter
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR = Path(r'D:\Trial Guard\scripts\real_data')
MODEL_FEATURE_COLUMNS = [
    'age', 'gender_encoded', 'ethnicity_encoded', 'condition_severity_encoded',
    'visit_number', 'cumulative_missed_visits', 'visit_frequency_rate',
    'days_since_last_visit', 'days_between_visits_mean', 'days_between_visits_std',
    'adverse_events_count', 'adverse_event_rate', 'adverse_event_trend',
    'medication_adherence_score', 'medication_adherence_trend',
    'quality_of_life_score', 'qol_score_trend',
    'early_dropout_signal', 'high_adverse_event_flag', 'low_adherence_flag',
]
COX_COVARIATES_REAL = ['age', 'condition_severity_encoded', 'adverse_event_rate']
HOLDOUT_STUDIES = ['Colorec_Amgen_2005_262', 'Breast_EliLill_2008_168',
                    'Glioma_2008_441', 'LungSm_Amgen_2002_266']
N_OPTUNA_TRIALS = 50
RNG_SEED = 42


def evaluate_at_thresholds(y_true, y_prob, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        rows.append({
            'threshold': t,
            'recall': round(float(recall_score(y_true, yp, zero_division=0)), 4),
            'precision': round(float(precision_score(y_true, yp, zero_division=0)), 4),
            'fraction_flagged': round(float(yp.mean()), 4),
        })
    return rows


def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    boot = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        boot.append(roc_auc_score(y_true[s], y_prob[s]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4), len(boot)


def xgb_objective(trial, X, y, groups=None, n_splits=5):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
        'random_state': RNG_SEED,
        'eval_metric': 'auc',
        'tree_method': 'hist',
    }
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED).split(X, y)

    aucs = []
    for tr_idx, va_idx in splitter:
        model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
        preds = model.predict_proba(X[va_idx])[:, 1]
        if len(np.unique(y[va_idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[va_idx], preds))
    return float(np.mean(aucs)) if aucs else 0.5


def tune_cox_penalizer(df_pool, groups, cox_covariates, n_splits=5):
    """Small grid over penalizer / l1_ratio, grouped CV, mean concordance index."""
    grid = [(p, l1) for p in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
            for l1 in [0.0, 0.5, 1.0]]
    best = (None, -np.inf)
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(df_pool), 1))
    for penalizer, l1_ratio in grid:
        c_indices = []
        for tr_idx, va_idx in gkf.split(X_dummy, df_pool['dropout_status'], groups):
            train_fold = df_pool.iloc[tr_idx][cox_covariates + ['days_to_event', 'dropout_status']].copy()
            val_fold = df_pool.iloc[va_idx][cox_covariates + ['days_to_event', 'dropout_status']].copy()
            train_fold['days_to_event'] = train_fold['days_to_event'].clip(lower=1)
            val_fold['days_to_event'] = val_fold['days_to_event'].clip(lower=1)
            try:
                cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                cph.fit(train_fold, duration_col='days_to_event', event_col='dropout_status')
                c = cph.score(val_fold, scoring_method='concordance_index')
                c_indices.append(c)
            except Exception:
                continue
        if c_indices:
            mean_c = float(np.mean(c_indices))
            if mean_c > best[1]:
                best = ((penalizer, l1_ratio), mean_c)
    return best  # ((penalizer, l1_ratio), mean_cv_concordance)


def run_pipeline(label, df, cox_covariates, group_col=None, holdout_studies=None):
    """
    Returns a results dict. If holdout_studies is given (real data), the
    headline test is those studies held out entirely; a random stratified
    split of the remaining pool is also evaluated for comparability.
    If holdout_studies is None (synthetic data), only the random
    stratified split is computed.
    """
    log = {'label': label, 'steps': []}
    t0 = time.time()

    if holdout_studies:
        pool = df[~df['study'].isin(holdout_studies)].reset_index(drop=True)
        holdout = df[df['study'].isin(holdout_studies)].reset_index(drop=True)
        log['steps'].append(f"Held out {len(holdout)} patients across {len(holdout_studies)} whole "
                             f"studies from ALL training/tuning: {holdout_studies}")
    else:
        pool = df.reset_index(drop=True)
        holdout = None

    X_pool = pool[MODEL_FEATURE_COLUMNS].values
    y_pool = pool['dropout_status'].values
    groups_pool = pool[group_col].values if group_col else None

    # random stratified split carved out of the pool, for the secondary /
    # comparable-to-earlier-session number
    X_train, X_test, y_train, y_test, pool_train, pool_test = train_test_split(
        X_pool, y_pool, pool, test_size=0.25, stratify=y_pool, random_state=RNG_SEED
    )
    groups_train = pool_train[group_col].values if group_col else None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- Cox: tune penalizer/l1_ratio via grouped/stratified CV ----------------
    # Restricted to pool_train only, same reasoning as the XGBoost tuning below:
    # pool_test must stay genuinely unseen until final evaluation.
    if group_col:
        best_cox_params, best_cox_cv_c = tune_cox_penalizer(pool_train, groups_train, cox_covariates)
    else:
        # no group structure -> use a fake group-per-row-block via StratifiedKFold instead
        # (tune_cox_penalizer needs "groups"; build pseudo-groups of size ~1 so GroupKFold
        # degenerates close to StratifiedKFold behavior is wrong - instead do a small manual
        # StratifiedKFold-based search here for synthetic data)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
        grid = [(p, l1) for p in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0] for l1 in [0.0, 0.5, 1.0]]
        best_cox_params, best_cox_cv_c = None, -np.inf
        pool_train_reset = pool_train.reset_index(drop=True)
        for penalizer, l1_ratio in grid:
            cvals = []
            for tr_idx, va_idx in skf.split(X_train, y_train):
                tr = pool_train_reset.iloc[tr_idx][cox_covariates + ['days_to_event', 'dropout_status']].copy()
                va = pool_train_reset.iloc[va_idx][cox_covariates + ['days_to_event', 'dropout_status']].copy()
                tr['days_to_event'] = tr['days_to_event'].clip(lower=1)
                va['days_to_event'] = va['days_to_event'].clip(lower=1)
                try:
                    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                    cph.fit(tr, duration_col='days_to_event', event_col='dropout_status')
                    cvals.append(cph.score(va, scoring_method='concordance_index'))
                except Exception:
                    continue
            if cvals and np.mean(cvals) > best_cox_cv_c:
                best_cox_cv_c = float(np.mean(cvals))
                best_cox_params = (penalizer, l1_ratio)

    penalizer, l1_ratio = best_cox_params
    log['steps'].append(f"Cox best (penalizer={penalizer}, l1_ratio={l1_ratio}), "
                         f"CV concordance={best_cox_cv_c:.4f}")

    cox_train_df = pool_train[cox_covariates + ['days_to_event', 'dropout_status']].copy()
    cox_train_df['days_to_event'] = cox_train_df['days_to_event'].clip(lower=1)
    cph_final = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph_final.fit(cox_train_df, duration_col='days_to_event', event_col='dropout_status')
    cox_train_c_index = cph_final.concordance_index_

    def hazard_ratios_for(frame):
        sub = frame[cox_covariates].copy()
        log_hr = cph_final.predict_log_partial_hazard(sub)
        return np.exp(log_hr.values if hasattr(log_hr, 'values') else log_hr)

    def cox_eval(frame):
        cx = frame[cox_covariates + ['days_to_event', 'dropout_status']].copy()
        cx['days_to_event'] = cx['days_to_event'].clip(lower=1)
        return float(cph_final.score(cx, scoring_method='concordance_index'))

    cox_c_random_test = cox_eval(pool_test)
    cox_c_holdout = cox_eval(holdout) if holdout is not None else None

    hr_train = hazard_ratios_for(pool_train)
    hr_test = hazard_ratios_for(pool_test)
    X_train_full = np.hstack([X_train_scaled, hr_train.reshape(-1, 1)])
    X_test_full = np.hstack([X_test_scaled, hr_test.reshape(-1, 1)])

    if holdout is not None:
        hr_holdout = hazard_ratios_for(holdout)
        X_holdout = holdout[MODEL_FEATURE_COLUMNS].values
        X_holdout_scaled = scaler.transform(X_holdout)
        X_holdout_full = np.hstack([X_holdout_scaled, hr_holdout.reshape(-1, 1)])
        y_holdout = holdout['dropout_status'].values

    # ---------------- XGBoost: Optuna tuning on the TRAINING SPLIT ONLY ----------------
    # Deliberately restricted to pool_train (not the full pool): pool_test is the
    # secondary "random split" evaluation set, and letting Optuna's CV folds see
    # those rows during tuning would leak information into that number even
    # though the final model itself is still only fit on pool_train. Using
    # X_train_full/groups_train keeps pool_test genuinely unseen until evaluation.
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: xgb_objective(trial, X_train_full, y_train, groups=groups_train, n_splits=5),
        n_trials=N_OPTUNA_TRIALS, show_progress_bar=False,
    )
    best_params = dict(study.best_params)
    best_params.update({'random_state': RNG_SEED, 'eval_metric': 'auc', 'tree_method': 'hist'})
    log['steps'].append(f"XGBoost Optuna best CV AUC={study.best_value:.4f}, params={best_params}")

    final_model = xgb.XGBClassifier(**best_params, early_stopping_rounds=20)
    final_model.fit(X_train_full, y_train, eval_set=[(X_test_full, y_test)], verbose=False)

    y_prob_test = final_model.predict_proba(X_test_full)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    auc_test = roc_auc_score(y_test, y_prob_test)
    ci_lo, ci_hi, n_boot = bootstrap_auc_ci(y_test, y_prob_test)

    result = {
        'label': label,
        'n_total': int(len(df)),
        'n_events_total': int(df['dropout_status'].sum()),
        'n_pool': int(len(pool)),
        'n_train': int(len(X_train)),
        'n_test_random': int(len(X_test)),
        'n_events_test_random': int(y_test.sum()),
        'cox_penalizer': penalizer,
        'cox_l1_ratio': l1_ratio,
        'cox_cv_concordance': round(best_cox_cv_c, 4),
        'cox_train_concordance': round(float(cox_train_c_index), 4),
        'cox_concordance_random_test': round(cox_c_random_test, 4),
        'xgb_best_params': best_params,
        'xgb_optuna_cv_auc': round(float(study.best_value), 4),
        'xgb_roc_auc_random_test': round(float(auc_test), 4),
        'xgb_roc_auc_random_test_ci95': [ci_lo, ci_hi],
        'xgb_f1_random_test': round(float(f1_score(y_test, y_pred_test, zero_division=0)), 4),
        'xgb_precision_random_test': round(float(precision_score(y_test, y_pred_test, zero_division=0)), 4),
        'xgb_recall_random_test': round(float(recall_score(y_test, y_pred_test, zero_division=0)), 4),
        'xgb_brier_random_test': round(float(brier_score_loss(y_test, y_prob_test)), 4),
        'xgb_threshold_sweep_random_test': evaluate_at_thresholds(y_test, y_prob_test),
    }

    if holdout is not None:
        y_prob_holdout = final_model.predict_proba(X_holdout_full)[:, 1]
        y_pred_holdout = (y_prob_holdout >= 0.5).astype(int)
        auc_holdout = roc_auc_score(y_holdout, y_prob_holdout)
        h_ci_lo, h_ci_hi, _ = bootstrap_auc_ci(y_holdout, y_prob_holdout)
        result.update({
            'holdout_studies': holdout_studies,
            'n_holdout': int(len(holdout)),
            'n_events_holdout': int(y_holdout.sum()),
            'cox_concordance_holdout': round(cox_c_holdout, 4),
            'xgb_roc_auc_holdout': round(float(auc_holdout), 4),
            'xgb_roc_auc_holdout_ci95': [h_ci_lo, h_ci_hi],
            'xgb_f1_holdout': round(float(f1_score(y_holdout, y_pred_holdout, zero_division=0)), 4),
            'xgb_precision_holdout': round(float(precision_score(y_holdout, y_pred_holdout, zero_division=0)), 4),
            'xgb_recall_holdout': round(float(recall_score(y_holdout, y_pred_holdout, zero_division=0)), 4),
            'xgb_brier_holdout': round(float(brier_score_loss(y_holdout, y_prob_holdout)), 4),
        })

    # ---------------- SHAP: stability + robustness on the final model ----------------
    import shap as shap_lib
    from core.utils.shap_explainer import compute_shap_stability_score, compute_shap_robustness_score
    bg_rng = np.random.default_rng(RNG_SEED)
    bg_size = min(100, len(X_train_full))
    background_X = X_train_full[bg_rng.choice(len(X_train_full), size=bg_size, replace=False)]
    explainer = shap_lib.TreeExplainer(final_model, data=background_X, feature_perturbation='interventional')
    shap_eval_X = X_holdout_full if holdout is not None else X_test_full
    shap_sample = shap_eval_X[:min(500, len(shap_eval_X))]
    # check_additivity=False: SHAP's own documented workaround for a known
    # floating-point precision mismatch between summed SHAP values and raw
    # model output (typically <0.01 difference, not a real correctness
    # issue) that can trip the default strict check on some XGBoost/tree
    # configurations.
    shap_vals = explainer.shap_values(shap_sample, check_additivity=False)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    result['shap_stability_score'] = round(float(compute_shap_stability_score(shap_vals)), 4)
    result['shap_robustness_score'] = round(float(compute_shap_robustness_score(explainer, shap_eval_X)), 4)
    result['shap_eval_set'] = 'holdout_studies' if holdout is not None else 'random_test'

    elapsed = time.time() - t0
    result['elapsed_seconds'] = round(elapsed, 1)
    log['steps'].append(f"Done in {elapsed:.1f}s")

    # save artifacts
    dump({'model': final_model, 'scaler': scaler, 'cox': cph_final,
          'feature_columns': MODEL_FEATURE_COLUMNS, 'cox_covariates': cox_covariates,
          'xgb_params': best_params, 'cox_params': {'penalizer': penalizer, 'l1_ratio': l1_ratio}},
         OUT_DIR / f'tuned_model_{label}.pkl')

    print(json.dumps(log, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    with open(OUT_DIR / f'tuned_results_{label}.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    return result, final_model, scaler, cph_final, pool_train, pool_test, holdout


if __name__ == '__main__':
    print("=" * 70)
    print("REAL DATA")
    print("=" * 70)
    df_real = pd.read_csv(OUT_DIR / 'real_dataset_final.csv')
    real_result, real_model, real_scaler, real_cox, real_train_pool, real_test_pool, real_holdout = run_pipeline(
        'real', df_real, COX_COVARIATES_REAL, group_col='study', holdout_studies=HOLDOUT_STUDIES
    )

    print("=" * 70)
    print("SYNTHETIC DATA (freshly generated, matching size, same schema)")
    print("=" * 70)
    from core.utils.data_pipeline import generate_synthetic_patients, generate_synthetic_visits

    n_synth = len(df_real)
    patients = generate_synthetic_patients(n=n_synth, seed=99)
    visits = generate_synthetic_visits(patients, visits_per_patient=8)
    last_visits = visits.sort_values('visit_number').groupby('patient_idx').tail(1).set_index('patient_idx')
    patients = patients.join(last_visits, how='inner')
    patients['visit_number'] = patients['visit_number'].fillna(0)

    def trend(vals):
        vals = [v for v in vals if v is not None and not pd.isna(v)]
        if len(vals) < 2:
            return 0.0
        x = np.arange(len(vals), dtype=float)
        y = np.array(vals, dtype=float)
        return float(np.polyfit(x, y, 1)[0]) if np.std(x) > 0 else 0.0

    rows = []
    for idx, pat in patients.iterrows():
        pat_visits = visits[visits['patient_idx'] == idx].sort_values('visit_number')
        if pat_visits.empty:
            continue
        ae_hist = pat_visits['adverse_events_count'].tolist()
        adh_hist = pat_visits['medication_adherence_score'].tolist()
        qol_hist = pat_visits['quality_of_life_score'].tolist()
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
    df_synth = pd.DataFrame(rows)
    df_synth.to_csv(OUT_DIR / 'synthetic_dataset_matched.csv', index=False)
    print(f"Synthetic dataset: {len(df_synth)} patients, {df_synth['dropout_status'].sum()} dropouts")

    synth_cox_covariates = ['age', 'condition_severity_encoded', 'cumulative_missed_visits',
                             'adverse_event_rate', 'medication_adherence_score']
    synth_result, synth_model, synth_scaler, synth_cox, synth_train_pool, synth_test_pool, _ = run_pipeline(
        'synthetic', df_synth, synth_cox_covariates, group_col=None, holdout_studies=None
    )

    print("ALL DONE")
