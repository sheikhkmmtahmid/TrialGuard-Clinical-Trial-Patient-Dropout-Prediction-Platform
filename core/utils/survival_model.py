"""
Cox Proportional Hazards survival analysis for dropout timeline modelling.
Outputs per-patient hazard ratios and survival probabilities at 30/60/90 days.
"""
import logging
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump, load

logger = logging.getLogger('core')


COX_MODEL_PATH = Path(__file__).resolve().parents[2] / 'ml_models' / 'cox_model.pkl'

COX_COVARIATES = [
    'age', 'condition_severity_encoded', 'distance_to_site_km',
    'cumulative_missed_visits', 'adverse_event_rate',
    'medication_adherence_score', 'prior_dropout_history',
]


def train_cox_model(df: pd.DataFrame):
    """Fit CoxPHFitter on the full dataset and persist the model."""
    try:
        from lifelines import CoxPHFitter
    except ImportError as e:
        raise ImportError("lifelines is required for survival analysis.") from e

    required = COX_COVARIATES + ['days_to_event', 'dropout_status']
    df_cox = df[required].dropna().copy()

    # Aggregate per-patient (take last visit values for visit-level features)
    if 'patient_id' in df.columns:
        agg = (
            df.groupby('patient_id')
            .agg({
                'age': 'first',
                'condition_severity_encoded': 'first',
                'distance_to_site_km': 'first',
                'prior_dropout_history': 'first',
                'cumulative_missed_visits': 'max',
                'adverse_event_rate': 'mean',
                'medication_adherence_score': 'mean',
                'days_to_event': 'max',
                'dropout_status': 'max',
            })
            .reset_index(drop=True)
        )
        df_cox = agg.copy()

    df_cox['days_to_event'] = df_cox['days_to_event'].clip(lower=1)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_cox, duration_col='days_to_event', event_col='dropout_status')

    c_index = cph.concordance_index_
    logger.info("Cox PH model fitted. Concordance index: %.4f", c_index)

    dump({'model': cph, 'concordance_index': c_index, 'covariates': COX_COVARIATES}, COX_MODEL_PATH)
    return cph, c_index


def load_cox_model():
    bundle = load(COX_MODEL_PATH)
    return bundle['model']


def predict_survival(patient_features: dict) -> dict:
    """
    Given a dict of covariate values, return hazard ratio and survival at 30/60/90 days.
    """
    cph = load_cox_model()

    row = {k: patient_features.get(k, 0.0) for k in COX_COVARIATES}
    df_input = pd.DataFrame([row])

    sf = cph.predict_survival_function(df_input)
    times = sf.index.values

    def surv_at(t):
        idx = np.searchsorted(times, t)
        if idx >= len(times):
            return float(sf.iloc[-1, 0])
        return float(sf.iloc[idx, 0])

    median_time = cph.predict_median(df_input)
    #hr = float(np.exp(cph.predict_log_partial_hazard(df_input).values[0]))
    #hr = float(np.exp(cph.predict_log_partial_hazard(df_input).iloc[0])) # Fix

    #return {
    #    'hazard_ratio': round(hr, 4),
    #    'survival_30d': round(surv_at(30), 4),
    #    'survival_60d': round(surv_at(60), 4),
    #    'survival_90d': round(surv_at(90), 4),
    #    #'median_survival_days': float(median_time.values[0]),
    #    'median_survival_days': float(median_time.iloc[0]), # Fix
    #}
    median_time = cph.predict_median(df_input)
    log_hr = cph.predict_log_partial_hazard(df_input)

    if hasattr(log_hr, "iloc"):
        log_hr_value = float(log_hr.iloc[0])
    else:
        log_hr_value = float(log_hr)

    if hasattr(median_time, "iloc"):
        median_time_value = float(median_time.iloc[0])
    else:
        median_time_value = float(median_time)

    hr = float(np.exp(log_hr_value))

    return {
        'hazard_ratio': round(hr, 4),
        'survival_30d': round(surv_at(30), 4),
        'survival_60d': round(surv_at(60), 4),
        'survival_90d': round(surv_at(90), 4),
        'median_survival_days': median_time_value,
    }


def plot_survival_curve(patient_features: dict, title: str = 'Patient Survival Curve') -> str:
    """
    Generate a Kaplan-Meier-style survival curve for a patient vs cohort baseline.
    Returns base64-encoded PNG.
    """
    cph = load_cox_model()
    row = {k: patient_features.get(k, 0.0) for k in COX_COVARIATES}
    df_input = pd.DataFrame([row])

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#F0F4F5')

    sf = cph.predict_survival_function(df_input)
    times = sf.index.values
    probs = sf.iloc[:, 0].values

    ax.plot(times, probs, color='#003087', linewidth=2.5, label='This Patient')
    ax.axvline(x=60, color='#CC0000', linestyle='--', linewidth=1.5, label='60-Day Mark', alpha=0.8)
    ax.fill_between(times, probs, alpha=0.12, color='#0072CE')

    ax.set_xlabel('Days Since Enrollment', color='#212B32', fontsize=11)
    ax.set_ylabel('Survival Probability (Retention)', color='#212B32', fontsize=11)
    ax.set_title(title, color='#003087', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#425563')
    ax.spines['bottom'].set_color('#D8DDE0')
    ax.spines['left'].set_color('#D8DDE0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor='#ffffff', edgecolor='#D8DDE0', labelcolor='#212B32')
    ax.grid(True, linestyle='--', alpha=0.35, color='#D8DDE0')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#ffffff')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_km_by_risk_tier(df: pd.DataFrame) -> str:
    """
    Plot Kaplan-Meier curves per risk tier. df must have columns:
    days_to_event, dropout_status, risk_tier.
    Returns base64 PNG.
    """
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return ''

    tier_colours = {
        'low': '#009639', 'medium': '#FFB81C',
        'high': '#E65C00', 'critical': '#CC0000',
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#F0F4F5')

    for tier, colour in tier_colours.items():
        sub = df[df['risk_tier'] == tier]
        if len(sub) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['days_to_event'], event_observed=sub['dropout_status'], label=tier.capitalize())
        kmf.plot_survival_function(ax=ax, color=colour, linewidth=2)

    ax.set_xlabel('Days Since Enrollment', color='#212B32', fontsize=11)
    ax.set_ylabel('Retention Probability', color='#212B32', fontsize=11)
    ax.set_title('Kaplan-Meier Survival by Risk Tier', color='#003087', fontsize=13, fontweight='bold')
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
