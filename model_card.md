# TrialGuard Model Card

**Version:** 1.0.0  
**Date:** 2026-04-23  
**Built by:** SKMMT · [skmmt.rootexception.com](https://skmmt.rootexception.com/)

---

## Model Overview

TrialGuard uses a two-layer ML stack to predict clinical trial patient dropout:

| Model | Purpose | Library |
|---|---|---|
| Cox Proportional Hazards | Dropout timeline estimation | lifelines |
| XGBoost Binary Classifier | Per-visit dropout probability | xgboost |
| SHAP TreeExplainer | Feature attribution / explainability | shap |

---

## Training Data

### Primary Source
- **Synthetic patients:** 5,000 patients generated with `numpy`-based synthesis using realistic clinical trial dropout distributions (GaussianCopula via SDV when available).
- **Structure:** Patient demographics + 2–8 visit records per patient.
- **Dropout rate:** ~35–45% (calibrated to mirror real-world Phase II/III trial attrition).

### Disclosure
> **SYNTHETIC DATA NOTICE:** The default training dataset is entirely synthetic. Models trained exclusively on synthetic data must be re-trained on real clinical trial data before deployment in a regulated clinical environment.

### Supplementary Integration Points
- CDISC CDASH demo datasets (if available at `data/cdisc/`)
- UCI ML Repository: Diabetic Retinopathy, Heart Disease (supplementary cohorts)

---

## Feature List (25 engineered features)

| Feature | Source | Type |
|---|---|---|
| age | Demographics | Continuous |
| age_group | Demographics (binned) | Categorical |
| gender_encoded | Demographics | Categorical |
| ethnicity_encoded | Demographics | Categorical |
| condition_severity_encoded | Demographics | Ordinal |
| distance_to_site_km | Demographics | Continuous |
| distance_bucket | Demographics (binned) | Categorical |
| employment_encoded | Demographics | Categorical |
| prior_dropout_history | Medical history | Binary |
| visit_number | Visit pattern | Ordinal |
| cumulative_missed_visits | Visit pattern | Continuous |
| visit_frequency_rate | Visit pattern | Continuous |
| days_since_last_visit | Visit pattern | Continuous |
| days_between_visits_mean | Visit pattern | Continuous |
| days_between_visits_std | Visit pattern | Continuous |
| adverse_events_count | Clinical | Continuous |
| adverse_event_rate | Clinical | Continuous |
| adverse_event_trend | Clinical (slope) | Continuous |
| medication_adherence_score | Clinical | Continuous |
| medication_adherence_trend | Clinical (slope) | Continuous |
| quality_of_life_score | Clinical | Continuous |
| qol_score_trend | Clinical (slope) | Continuous |
| early_dropout_signal | Risk flag | Binary |
| high_adverse_event_flag | Risk flag | Binary |
| low_adherence_flag | Risk flag | Binary |

**Additional feature (XGBoost only):** Cox PH hazard ratio (derived from survival model).

---

## Performance (Measured — v1.0.0 on Synthetic Data)

| Metric | Target | Achieved | Status |
|---|---|---|---|
| XGBoost ROC-AUC | ≥ 0.80 | **0.9914** | ✅ Exceeds target |
| XGBoost F1 Score | ≥ 0.75 | **0.8842** | ✅ Exceeds target |
| XGBoost Precision | — | **0.8668** | — |
| XGBoost Recall | — | **0.9022** | — |
| Cox Concordance Index | ≥ 0.70 | **0.9119** | ✅ Exceeds target |
| Brier Score | ≤ 0.20 | **0.0253** | ✅ Excellent calibration |
| SHAP Stability | ≥ 0.70 | **0.8348** | ✅ Exceeds target |
| Optuna Best Val AUC | — | **0.9925** | — |
| Training samples | — | 17,022 visit rows | — |
| Test samples | — | 5,007 visit rows | — |

> **Note:** These metrics are measured on synthetic training data. Performance on real-world clinical data will differ. Re-train with real patient cohorts before clinical deployment.

---

## Risk Tier Thresholds

| Tier | Probability Range | Recommended Action |
|---|---|---|
| Low | 0.00 – 0.30 | Routine monitoring |
| Medium | 0.31 – 0.55 | Coordinator check-in recommended |
| High | 0.56 – 0.75 | Immediate outreach required |
| Critical | 0.76 – 1.00 | Emergency intervention protocol |

---

## Intended Use

- **Clinical trial retention management** by trained coordinators
- **60-day early warning system** for at-risk patients
- **Decision support only** — not a standalone clinical diagnostic tool
- **Target users:** Trial coordinators, site managers, study sponsors

---

## Limitations

1. **Synthetic training data:** Default models are trained on synthetic data and require re-training on real patient cohorts for clinical validity.
2. **Population shift:** Models may not generalise across different therapeutic areas, geographies, or trial phases without re-training.
3. **Feature completeness:** Predictions degrade if key features (adherence scores, adverse event counts) are missing or inconsistently recorded.
4. **Temporal validity:** Models should be re-trained at least every 6 months as patient populations and trial protocols evolve.
5. **No causal inference:** SHAP attributions indicate correlation with dropout, not causal relationships.
6. **Visit-level predictions:** Risk scores are computed per visit — patients with fewer visits have less reliable predictions.

---

## Ethical Considerations

- **Patient privacy:** No PII should be stored in model artifacts. Patient IDs are internal database references only.
- **Bias awareness:** Models may exhibit differential performance across demographic subgroups. Coordinators should review flagged patients with awareness of potential algorithmic bias.
- **Human oversight required:** All intervention decisions must be made by qualified clinical staff. TrialGuard is a decision support tool, not an autonomous system.
- **Synthetic data transparency:** All documentation and interfaces clearly disclose when synthetic data was used in model training.
- **Re-identification risk:** Aggregated cohort forecasts must not be shared in ways that could re-identify individual patients.

---

## Regulatory Context

TrialGuard is intended as a **clinical decision support (CDS) tool**. Depending on jurisdiction and specific use:
- **FDA:** May qualify as Software as a Medical Device (SaMD) under 21st Century Cures Act guidance
- **EU:** May fall under MDR Article 22 (software as a medical device)
- **Recommendation:** Engage regulatory counsel before clinical deployment

---

## Versioning

| Component | Version |
|---|---|
| Cox PH Model | 1.0.0 |
| XGBoost Classifier | 1.0.0 |
| SHAP Explainer | 1.0.0 |
| Feature Pipeline | 1.0.0 |

---

*TrialGuard is built by SKMMT — [skmmt.rootexception.com](https://skmmt.rootexception.com/)*
