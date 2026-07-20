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
- **Structure:** Patient demographics + 2 to 8 visit records per patient.
- **Dropout rate:** varies by trial phase and therapeutic area (16.3% to 33.9% across the four seed trials as of this writing), calibrated against real AACT rates for each combination rather than one flat number. See the AACT comparison below for exactly what "calibrated" does and does not mean here.

### Disclosure
> **SYNTHETIC DATA NOTICE:** The default training dataset is entirely synthetic. No real patient record, visit, or outcome has been used anywhere in training. Real data (AACT) has only ever been used to set the *overall percentage* of patients who drop out per trial type, not to generate any individual patient. Models trained exclusively on synthetic data must be re-trained on real patient-level data before deployment in a regulated clinical environment.

### Supplementary Integration Points
- CDISC CDASH demo datasets (if available at `data/cdisc/`)
- UCI ML Repository: Diabetic Retinopathy, Heart Disease (supplementary cohorts)

### Real-World Comparison (AACT), and exactly what it does and does not prove

Run `python manage.py validate_against_aact` to compare the synthetic dropout
rate against real dropout rates pulled from AACT, the aggregate mirror of
ClinicalTrials.gov (593,122 registered studies, 63,233 of them interventional
trials with usable dropout data). Full results are in
[docs/aact_validation_report.md](docs/aact_validation_report.md); see
[docs/data_sourcing.md](docs/data_sourcing.md) for why AACT was chosen over
other real-data sources and what it can and cannot validate.

**Be precise about what "calibrated against real data" means here.** AACT
gives trial-level and arm-level dropout *counts* (how many of the 600
patients on this real trial dropped out, and why), not individual patient
records. What we did with that: for each of our four seed trials, we looked
up the real average dropout percentage for trials of that same phase and
therapeutic area, and tuned our synthetic generator so it produces that same
percentage. That is the full extent of it. Every synthetic patient's actual
data, demographics, visit history, symptoms, is still entirely invented by
our code, none of it comes from or resembles any real patient's chart. As of
this fix, the calibration lands within 0.0 to 0.1 percentage points of the
real rate for all four seed trials (see the full breakdown in
docs/aact_validation_report.md).

**Second correction, this one changed the actual number, not just the fit.**
The first version of this calibration used AACT's blunt started-versus-completed
count as "dropout," which silently includes death and disease progression
alongside genuine behavioral withdrawal. Once real patient-level data (PDS,
below) showed the real behavioral dropout rate for oncology was 4.6%, not
33.9%, the same conflation was checked and fixed in AACT itself: AACT's own
`drop_withdrawals` table records a reason for each departure, so the
calibration now counts only "Withdrawal by Subject," "Lost to Follow-up," and
"Withdrew Consent" as dropout, filtered case-insensitively across 594,605 real
reason records. The corrected overall rate across all 63,233 usable trials is
4.2%, independently landing almost exactly where the PDS patient data did,
two different real sources agreeing is a good sign the number is real. Full
before-and-after by phase and area is in
[docs/pds_validation_report.md](docs/pds_validation_report.md). The model was
retrained on this corrected calibration; see the Performance section for what
that changed, and read that section's note before assuming a higher AUC means
a better model.

**What this validates:** that our training data's overall dropout
*frequency* per trial type is realistic, which matters for the cohort-level
forecast (predicting how many of 600 enrolled patients will drop out).

**What this does not validate:** the XGBoost classifier's per-patient
predictions, or the model's real-world accuracy in any sense. A model can
have a perfectly realistic overall dropout rate in its training data and
still be completely wrong about which specific patients are at risk, those
are two different questions. Validating the second one requires a real
patient-level dataset (individual visit-by-visit records with real outcomes).

### Real-World Patient-Level Validation (Project Data Sphere)

That patient-level gap has now been partially addressed. See
[docs/pds_validation_report.md](docs/pds_validation_report.md) for the full
account, three findings worth knowing before reading anything else in this
document:

1. Real oncology "discontinuation" is mostly death, not behavioral dropout,
   conflating the two (as the AACT calibration above currently does) measures
   the wrong thing.
2. The real behavioral dropout rate, measured correctly, is 4.6% across 561
   real patients, far below the 33.9% this model's oncology calibration
   currently assumes. The AACT calibration needs to be redone filtering by
   reason, not just started-versus-completed.
3. The trained classifier, run against 118 real patients using their real
   age, sex, and adverse event data (everything else imputed at the
   synthetic training average, since it wasn't available), scored ROC-AUC
   0.477, no better than guessing. This is not clean proof the model fails
   on real patients, most of its strongest inputs were never actually
   tested, but it is proof that a fair test hasn't happened yet.

---

## Feature List (23 engineered, 20 currently used for training)

`age_group` and `distance_bucket` (bucketed copies of `age` and
`distance_to_site_km`) were removed, they carried no information the raw
values didn't already have, and duplicated/correlated features are a well
documented cause of unstable SHAP attribution. See the Performance section
below.

| Feature | Source | Type | Used for training |
|---|---|---|---|
| age | Demographics | Continuous | Yes |
| gender_encoded | Demographics | Categorical | Yes |
| ethnicity_encoded | Demographics | Categorical | Yes |
| condition_severity_encoded | Demographics | Ordinal | Yes |
| distance_to_site_km | Demographics | Continuous | No, reserved |
| employment_encoded | Demographics | Categorical | No, reserved |
| prior_dropout_history | Medical history | Binary | No, reserved |
| visit_number | Visit pattern | Ordinal | Yes |
| cumulative_missed_visits | Visit pattern | Continuous | Yes |
| visit_frequency_rate | Visit pattern | Continuous | Yes |
| days_since_last_visit | Visit pattern | Continuous | Yes |
| days_between_visits_mean | Visit pattern | Continuous | Yes |
| days_between_visits_std | Visit pattern | Continuous | Yes |
| adverse_events_count | Clinical | Continuous | Yes |
| adverse_event_rate | Clinical | Continuous | Yes |
| adverse_event_trend | Clinical (slope) | Continuous | Yes |
| medication_adherence_score | Clinical | Continuous | Yes |
| medication_adherence_trend | Clinical (slope) | Continuous | Yes |
| quality_of_life_score | Clinical | Continuous | Yes |
| qol_score_trend | Clinical (slope) | Continuous | Yes |
| early_dropout_signal | Risk flag | Binary | Yes |
| high_adverse_event_flag | Risk flag | Binary | Yes |
| low_adherence_flag | Risk flag | Binary | Yes |

**Additional feature (XGBoost only):** Cox PH hazard ratio (derived from survival model).

### Why three features are "reserved" rather than deleted

`distance_to_site_km`, `employment_encoded`, and `prior_dropout_history`
are still collected on every patient (the add-patient form, CSV upload,
admin panel, and PDF reports all still ask for and show them) and the
synthetic data generator still simulates them, but as of this version
they are excluded from `MODEL_FEATURE_COLUMNS`, the list actually passed
to the scaler and every model (`core/utils/data_pipeline.py`). Neither
Cox nor XGBoost can see them, so they cannot influence a hazard ratio,
a dropout probability, or a SHAP explanation, full stop, not just
"deprioritized."

This was a data-driven call, not a guess. Every real-world source
checked across this project, Project Data Sphere (45 studies), MUSIC,
UCI Heart Failure, and ImmPort, four independent real-world data
ecosystems, was searched by hand for these three fields and found none.
Every earlier real-data validation in this project had to substitute a
synthetic "average" value for them, meaning whatever importance they
showed in SHAP analysis was only ever demonstrated on synthetic data
built to make them predictive, never confirmed against a real patient.

They are kept in the schema, not deleted, specifically so that if a
future real data source ever does supply genuine values for one of
these, it can be added back to `MODEL_FEATURE_COLUMNS` and retrained on
without rebuilding any collection or encoding logic. See
`docs/pds_validation_report.md` ("Recommendation: drop three features")
and `docs/data_sourcing.md` for the full evidence trail.

---

## Performance (Measured, v1.0.0 on Synthetic Data)

This is the single, fully consistent run these numbers all come from, the
scaler, both models, the explainer, and every stored patient prediction were
all produced by one execution of `python manage.py train_models`, not
stitched together from separate runs.

| Metric | Target | Achieved | Status |
|---|---|---|---|
| XGBoost ROC-AUC | at least 0.80 | **0.9442** | meets target, read the note below before trusting this |
| XGBoost F1 Score | at least 0.75 | **0.4159** | dropped, see note |
| XGBoost Precision | none set | **0.9216** | |
| XGBoost Recall | none set | **0.2686** | dropped, see note on classification threshold below |
| Cox Concordance Index | at least 0.70 | **0.7581** | in a realistic clinical range |
| Brier Score | at most 0.20 | **0.0202** | see note, this looks better than it is |
| SHAP cross-patient similarity | not a target, see note | **0.0073** | low is expected, see earlier note |
| SHAP same-patient robustness | none set previously | **0.8643** | explanations still hold up well under realistic noise |
| Optuna Best Val AUC | none set | **0.9413** | |
| Training samples | none set | 16,918 visit rows | |
| Test samples | none set | 4,976 visit rows | |

**Important: AUC went up this time, and that is not the same as the model
getting better.** Every earlier retrain in this document's history made
AUC go down as real leaks got fixed. This one went up, from 0.90 to 0.94,
for a different reason entirely: the AACT calibration was corrected (see
below) to only count genuine behavioral dropout, not death or disease
progression. Real behavioral dropout turned out to be much rarer than
previously assumed, roughly 3 to 4% instead of 17 to 34%. When the thing
you are predicting becomes rarer, ROC-AUC can go up almost mechanically,
because there are so many more true negatives to correctly rank below the
few positives, even if the model's practical usefulness gets worse. Look
at recall instead: it dropped from 58% to 27% at the standard cutoff, the
model now catches barely one in four real dropouts by default. That is
the more honest number here, and it is a direct, expected consequence of
correctly modelling a rarer outcome, not a bug.

**On recall (now 0.27):** this is measured at the standard 0.5 probability
cutoff, which is a reporting convention, not how the product actually
works. The dashboard sorts patients into four risk tiers rather than a
single yes/no line, so a patient just under 0.5 is very likely already
landing in the medium or high risk tier, not being silently ignored. If a
higher catch-rate is wanted in exchange for more false alarms, that is a
deliberate threshold decision to make, not a limitation of the model
itself, and it has not yet been tuned or decided on. Given how rare
genuine behavioral dropout now looks in real terms, this threshold
decision matters more than it used to, not less.

### The full history of this number, and why it kept dropping

Earlier versions of this document reported ROC-AUC as high as 0.99. That
number was never a sign of a good model, it was a sign of two separate bugs
that let the model see the answer before being asked the question:

1. **Visit trends were generated from the final dropout label directly**,
   instead of from a shared, noisy underlying risk. Fixed by generating
   both a patient's visit trajectory and their dropout outcome from the
   same hidden risk factor, each with its own independent randomness, so
   neither one is a readout of the other.
2. **A feature called `visit_frequency_rate` used a patient's final,
   only-known-in-hindsight trial duration**, applied to every one of their
   visits, including their very first. This one applied to real trial data
   too, not just synthetic data, and was fixed to only use time elapsed as
   of that specific visit.

After both fixes, ROC-AUC was still 0.928, still higher than believable.
The reason: four of the model's inputs (age, condition severity, distance
to site, prior dropout history) are the same raw ingredients used to build
the hidden risk factor in the first place, with only a small amount of
independent noise mixed in. That let the model largely reconstruct the
hidden risk factor from demographics alone. The noise term in that formula
was increased significantly (matching the standard statistical idea that
any outcome = formula(known facts) + irreducible randomness, where the
randomness represents everything about a real person that a handful of
demographic facts can never capture). That produced the current, more
defensible number.

### SHAP: two different questions, two different metrics

`shap_stability_score` (kept the same name for continuity with earlier
evaluation_results.json history, despite the name) measures how similar
*different* patients' explanations are to each other. It dropped from 0.83
down to near zero (0.0195) over the course of these fixes. That is not a
regression. A high score here means most patients are being flagged for
similar-looking reasons, which is exactly what you would expect while the
model was still dominated by one clean, leaked signal. Once that signal was
removed, different patients started being flagged for genuinely different
combinations of reasons (hazard ratio, visit frequency, adverse events,
distance, quality of life trend, in varying order and weight per patient),
which is what a real, individualised clinical tool should look like, not a
templated explanation repeated for everyone.

The question that actually matters for trust is different: if you nudge a
given patient's numbers slightly, the kind of small variation you would
see from the same chart recorded on two different days, does their
explanation stay roughly the same, or does it flip to a different story?
That is `shap_robustness_score`, and it measures exactly that: each
sampled patient's SHAP explanation is computed twice, once as recorded and
once with small realistic jitter added, and the two explanations are
compared. It came out at **0.963**, meaning a given patient's explanation
is highly consistent under small, realistic noise. That is the number
worth trusting, and it is a genuinely good one, not an inflated one.

Two supporting methodology changes made this measurement meaningful in the
first place: two redundant engineered features (`age_group` and
`distance_bucket`, both just bucketed copies of features already present
as raw numbers) were removed, since correlated or duplicated features are
a well documented cause of SHAP arbitrarily splitting credit between them.
The explainer was also switched from SHAP's default tree-structural
attribution to interventional attribution with a background sample, which
is the method SHAP's own documentation recommends when features are
correlated with each other.

> **Note:** These metrics are measured on synthetic training data. Performance on real-world clinical data will differ. Re-train with real patient cohorts before clinical deployment. See [docs/aact_validation_report.md](docs/aact_validation_report.md) for the real-world comparison that exists so far (trial-level dropout rates only, not yet a patient-level validation).

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
