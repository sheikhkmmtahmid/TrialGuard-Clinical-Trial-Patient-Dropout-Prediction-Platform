"""
SHAP explainability layer for TrialGuard XGBoost model.
Generates per-patient waterfall plots and global feature importance.
"""
import logging
import io
import base64
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump, load

logger = logging.getLogger('core')

SHAP_EXPLAINER_PATH = Path(__file__).resolve().parents[2] / 'ml_models' / 'shap_explainer.pkl'

from core.utils.data_pipeline import FEATURE_COLUMNS


def build_shap_explainer(xgb_model):
    """Create and persist a SHAP TreeExplainer."""
    import shap
    explainer = shap.TreeExplainer(xgb_model)
    dump(explainer, SHAP_EXPLAINER_PATH)
    logger.info("SHAP explainer saved to %s", SHAP_EXPLAINER_PATH)
    return explainer


def load_shap_explainer():
    return load(SHAP_EXPLAINER_PATH)


def compute_shap_values(X: np.ndarray):
    """Compute SHAP values for feature matrix X. Returns shap.Explanation or ndarray."""
    import shap
    explainer = load_shap_explainer()
    return explainer(X) if hasattr(explainer, '__call__') else explainer.shap_values(X)


def top_features_for_patient(shap_row: np.ndarray, feature_names: list = None, top_n: int = 5) -> list:
    """
    Return top N SHAP-driven features for one patient visit row.
    Returns list of dicts: {feature, shap_value, direction}.
    """
    if feature_names is None:
        feature_names = FEATURE_COLUMNS

    pairs = sorted(zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)
    result = []
    for feat, val in pairs[:top_n]:
        result.append({
            'feature': feat,
            'shap_value': round(float(val), 4),
            'direction': 'increases' if val > 0 else 'decreases',
        })
    return result


def shap_values_to_json(shap_row: np.ndarray, feature_names: list = None) -> dict:
    """Serialise per-patient SHAP values for storage in PredictionResult.shap_values_json."""
    top = top_features_for_patient(shap_row, feature_names)
    return {'top_features': top}


def compute_shap_stability_score(shap_values: np.ndarray) -> float:
    """
    Measure SHAP stability as mean pairwise cosine similarity of neighbouring rows.
    Returns score in [0, 1]; higher is more stable.
    """
    if len(shap_values) < 10:
        return 1.0
    norms = np.linalg.norm(shap_values, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normed = shap_values / norms
    sample = normed[:500]
    sim = np.dot(sample, sample.T)
    upper = sim[np.triu_indices_from(sim, k=1)]
    return round(float(np.mean(upper)), 4)


def plot_waterfall(shap_row: np.ndarray, base_value: float,
                   feature_names: list = None, title: str = 'SHAP Waterfall') -> str:
    """
    Generate a waterfall chart showing top 10 SHAP contributions.
    Returns base64-encoded PNG.
    """
    if feature_names is None:
        feature_names = FEATURE_COLUMNS

    pairs = sorted(zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)[:10]
    feats = [p[0].replace('_', ' ').title() for p in pairs]
    vals = [p[1] for p in pairs]

    colours = ['#CC0000' if v > 0 else '#009639' for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#F0F4F5')

    bars = ax.barh(feats[::-1], vals[::-1], color=colours[::-1], edgecolor='#D8DDE0',
                   linewidth=0.5, height=0.7)

    for bar, val in zip(bars, vals[::-1]):
        label = f'+{val:.3f}' if val > 0 else f'{val:.3f}'
        ax.text(
            val + (0.003 if val > 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            label,
            va='center',
            ha='left' if val > 0 else 'right',
            color='#212B32', fontsize=9
        )

    ax.axvline(0, color='#003087', linewidth=1.2, alpha=0.8)
    ax.set_xlabel('SHAP Value (Impact on Dropout Probability)', color='#212B32', fontsize=10)
    ax.set_title(title, color='#003087', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#425563', labelsize=9)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#D8DDE0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', linestyle='--', alpha=0.35, color='#D8DDE0')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=130, facecolor='#ffffff')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_global_summary(X: np.ndarray, shap_values: np.ndarray, feature_names: list = None) -> str:
    """
    Beeswarm-style global SHAP summary. Returns base64 PNG.
    """
    import shap
    if feature_names is None:
        feature_names = FEATURE_COLUMNS

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#F0F4F5')

    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False, max_display=15,
        color_bar_label='Feature Value',
    )

    plt.title('Global SHAP Feature Importance', color='#003087', fontsize=13, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#ffffff')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_dependence(X: np.ndarray, shap_values: np.ndarray, feature: str,
                    feature_names: list = None) -> str:
    """Dependence plot for a single feature. Returns base64 PNG."""
    import shap
    if feature_names is None:
        feature_names = FEATURE_COLUMNS

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#F0F4F5')

    shap.dependence_plot(feature, shap_values, X, feature_names=feature_names,
                         ax=ax, show=False)
    ax.set_title(f'SHAP Dependence: {feature}', color='#003087', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#425563')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#ffffff')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
