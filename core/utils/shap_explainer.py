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

from core.utils.data_pipeline import MODEL_FEATURE_COLUMNS


def build_shap_explainer(xgb_model, background_X: np.ndarray = None):
    """
    Create and persist a SHAP TreeExplainer.

    When features are correlated with each other, SHAP's default
    tree_path_dependent mode can split credit between them somewhat
    arbitrarily, since it estimates each feature's contribution by
    following the tree's own branching rather than asking "what would the
    prediction look like if this feature's value came from elsewhere in
    the population." Passing a background sample and using interventional
    mode instead asks that second question directly, which is the
    documented, standard way to get attributions that hold up better when
    features are correlated. It costs more compute (it evaluates the model
    against the background sample instead of just walking the tree), so
    background_X should stay a modest sample, not the full dataset.
    """
    import shap
    if background_X is not None:
        explainer = shap.TreeExplainer(
            xgb_model, data=background_X, feature_perturbation='interventional'
        )
    else:
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
        feature_names = MODEL_FEATURE_COLUMNS

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
    Despite the name, this measures how similar DIFFERENT patients' SHAP
    explanations are to each other (mean pairwise cosine similarity across
    a sample of rows), not whether any one patient's explanation is
    reliable. A high score here means most patients are being flagged for
    similar-looking reasons, which is what you would see if one dominant,
    overly clean signal was driving most predictions. A lower score can
    mean patients are being flagged for genuinely different, more
    individualised reasons, which is not necessarily worse. See
    compute_shap_robustness_score below for the textbook definition of
    explanation stability: does the SAME patient get a consistent
    explanation, not do different patients look alike.

    Returns score in [-1, 1]; kept for backward compatibility with
    evaluation_results.json history.
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


def compute_shap_robustness_score(
    explainer, X: np.ndarray, n_samples: int = 200, noise_std: float = 0.05, seed: int = 42,
) -> float:
    """
    Measure whether the SAME patient gets a consistent explanation when
    their recorded values are nudged by a small amount of realistic
    measurement noise, a slightly different adherence score, a slightly
    different adverse event count, the kind of natural variation you would
    see if the same patient's chart were recorded on two different days.
    This is the standard, widely used definition of explanation stability
    in the explainability literature: an explanation you can trust should
    not flip to a different story over a trivial change in the input.

    For each sampled patient, this computes their SHAP explanation twice,
    once as recorded and once with a small jitter added to every feature
    (scaled to that feature's own spread in the sample, so a feature with
    a wide natural range gets proportionally more jitter than one with a
    narrow range), and measures the cosine similarity between the two
    explanation vectors for that same patient. The average across many
    patients is the robustness score.

    Returns a value in [-1, 1], higher means explanations hold up better
    under small, realistic input noise.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    sample = X[idx]

    col_stds = sample.std(axis=0)
    jitter = rng.normal(0, noise_std, size=sample.shape) * col_stds
    jittered = sample + jitter

    # check_additivity=False: known floating-point precision mismatch
    # between summed SHAP values and raw model output on some tree
    # configurations, not a real correctness issue (see train_tuned_models.py).
    orig_shap = explainer.shap_values(sample, check_additivity=False)
    jit_shap = explainer.shap_values(jittered, check_additivity=False)
    if isinstance(orig_shap, list):
        orig_shap = orig_shap[1]
    if isinstance(jit_shap, list):
        jit_shap = jit_shap[1]

    orig_norms = np.linalg.norm(orig_shap, axis=1, keepdims=True)
    jit_norms = np.linalg.norm(jit_shap, axis=1, keepdims=True)
    orig_norms[orig_norms == 0] = 1e-9
    jit_norms[jit_norms == 0] = 1e-9

    orig_normed = orig_shap / orig_norms
    jit_normed = jit_shap / jit_norms

    sims = np.sum(orig_normed * jit_normed, axis=1)
    return round(float(np.mean(sims)), 4)


def plot_waterfall(shap_row: np.ndarray, base_value: float,
                   feature_names: list = None, title: str = 'SHAP Waterfall') -> str:
    """
    Generate a waterfall chart showing top 10 SHAP contributions.
    Returns base64-encoded PNG.
    """
    if feature_names is None:
        feature_names = MODEL_FEATURE_COLUMNS

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
        feature_names = MODEL_FEATURE_COLUMNS

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
        feature_names = MODEL_FEATURE_COLUMNS

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
