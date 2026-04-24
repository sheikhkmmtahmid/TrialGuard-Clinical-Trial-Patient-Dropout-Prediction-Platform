"""
XGBoost binary classifier for patient dropout prediction.
Includes Optuna hyperparameter tuning and full evaluation suite.
"""
import logging
import json
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger('core')

XGB_MODEL_PATH = Path(__file__).resolve().parents[2] / 'ml_models' / 'xgb_model.pkl'
EVAL_RESULTS_PATH = Path(__file__).resolve().parents[2] / 'evaluation_results.json'

from core.utils.data_pipeline import FEATURE_COLUMNS


def _objective(trial, X_train, X_val, y_train, y_val):
    import xgboost as xgb
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
        'use_label_encoder': False,
        'eval_metric': 'auc',
        'random_state': 42,
        'tree_method': 'hist',
    }
    model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


def train_xgboost_model(df: pd.DataFrame, n_optuna_trials: int = 50):
    """
    Fit XGBoost dropout classifier with Optuna tuning.
    Saves model + evaluation metrics.
    Returns (model, metrics_dict).
    """
    import xgboost as xgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X = df[FEATURE_COLUMNS].values
    y = df['dropout_status'].values

    if 'hazard_ratio' in df.columns:
        X = np.hstack([X, df[['hazard_ratio']].values])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    logger.info("Running Optuna hyperparameter search (%d trials)...", n_optuna_trials)
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: _objective(trial, X_train, X_val, y_train, y_val),
        n_trials=n_optuna_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_params.update({
        'use_label_encoder': False,
        'eval_metric': 'auc',
        'random_state': 42,
        'tree_method': 'hist',
        'early_stopping_rounds': 20,
    })

    logger.info("Best ROC-AUC (val): %.4f | Params: %s", study.best_value, best_params)

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        'xgb_roc_auc': round(roc_auc_score(y_test, y_prob), 4),
        'xgb_f1': round(f1_score(y_test, y_pred), 4),
        'xgb_precision': round(precision_score(y_test, y_pred), 4),
        'xgb_recall': round(recall_score(y_test, y_pred), 4),
        'calibration_brier_score': round(brier_score_loss(y_test, y_prob), 4),
        'optuna_best_val_auc': round(study.best_value, 4),
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'best_params': {k: v for k, v in best_params.items()
                        if k not in ('use_label_encoder', 'eval_metric', 'tree_method')},
    }

    logger.info(
        "XGBoost eval — ROC-AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
        metrics['xgb_roc_auc'], metrics['xgb_f1'],
        metrics['xgb_precision'], metrics['xgb_recall'],
    )

    dump(
        {'model': final_model, 'feature_columns': FEATURE_COLUMNS, 'metrics': metrics},
        XGB_MODEL_PATH
    )
    return final_model, metrics


def load_xgb_model():
    bundle = load(XGB_MODEL_PATH)
    return bundle['model']


def predict_dropout_probability(feature_vector: np.ndarray) -> float:
    """Single-row inference. feature_vector shape: (1, n_features)."""
    model = load_xgb_model()
    prob = model.predict_proba(feature_vector.reshape(1, -1))[0][1]
    return float(prob)


def predict_batch(X: np.ndarray) -> np.ndarray:
    """Batch inference. Returns probability array."""
    model = load_xgb_model()
    return model.predict_proba(X)[:, 1]


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#F0F4F5')

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', color='#0072CE',
            linewidth=2, label='XGBoost')
    ax.plot([0, 1], [0, 1], '--', color='#425563', alpha=0.5, label='Perfectly Calibrated')

    ax.set_xlabel('Mean Predicted Probability', color='#212B32', fontsize=11)
    ax.set_ylabel('Fraction of Positives', color='#212B32', fontsize=11)
    ax.set_title('Calibration Curve', color='#003087', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#425563')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#D8DDE0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#ffffff', edgecolor='#D8DDE0', labelcolor='#212B32')
    ax.grid(True, linestyle='--', alpha=0.35, color='#D8DDE0')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#ffffff')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def save_evaluation_results(metrics: dict, cox_concordance: float = None):
    if cox_concordance is not None:
        metrics['cox_concordance_index'] = round(cox_concordance, 4)
    EVAL_RESULTS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Evaluation results saved to %s", EVAL_RESULTS_PATH)
