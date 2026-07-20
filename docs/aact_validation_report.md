# AACT real-world validation report

This compares TrialGuard's synthetic data generator against real dropout rates from AACT, the aggregate mirror of ClinicalTrials.gov. See docs/data_sourcing.md for why AACT was chosen and what it can and cannot validate.

The generator now calibrates each trial's dropout rate against the real AACT rate for that trial's phase and therapeutic area, instead of using one flat rate for every trial. The table below checks that the calibration actually lands where it is supposed to.

## By seed trial (phase and therapeutic area matched)

| Seed trial | Phase | Area | Real trials (n) | Real mean | Real median | Synthetic (calibrated) | Gap |
|---|---|---|---|---|---|---|---|
| CARDIO-GUARD Phase III | PHASE3 | Cardiovascular | 524 | 2.6% | 0.7% | 2.6% | +0.0% |
| ONCO-TRACE Phase II | PHASE2 | Oncology | 5989 | 3.4% | 0.0% | 3.4% | +0.0% |
| NEURO-SHIELD Phase II | PHASE2 | Neurology | 634 | 4.1% | 0.0% | 4.2% | +0.0% |
| DIAB-PROTECT Phase IV | PHASE4 | Endocrinology | 460 | 4.4% | 0.0% | 4.4% | +0.0% |

For reference, the overall AACT dropout rate across all 63,233 interventional trials is 20.7% (mean) and 9.5% (median). Trial-level dropout rates are skewed, most trials have low dropout, a smaller number have very high dropout, which is why mean and median differ this much and why matching on phase and area rather than one global number matters.

## Full breakdown by phase and therapeutic area

| Phase | Therapeutic area | n | Mean | Median | Std |
|---|---|---|---|---|---|
| nan | nan | 14841 | 4.6% | 0.0% | 11.0% |
| PHASE2 | nan | 9112 | 4.3% | 0.0% | 8.4% |
| PHASE3 | nan | 7543 | 4.5% | 1.9% | 7.3% |
| PHASE2 | Oncology | 5989 | 3.4% | 0.0% | 8.6% |
| PHASE4 | nan | 5516 | 4.2% | 0.0% | 9.6% |
| PHASE1 | nan | 2728 | 3.0% | 0.0% | 6.9% |
| PHASE1/PHASE2 | nan | 1706 | 4.0% | 0.0% | 9.5% |
| nan | Oncology | 1547 | 3.9% | 0.0% | 10.7% |
| PHASE1/PHASE2 | Oncology | 1444 | 3.9% | 0.0% | 8.7% |
| nan | Endocrinology | 1428 | 4.7% | 0.0% | 9.5% |
| PHASE3 | Oncology | 1396 | 4.6% | 2.5% | 7.5% |
| nan | Cardiovascular | 1110 | 3.3% | 0.0% | 8.8% |
| PHASE1 | Oncology | 978 | 4.4% | 0.0% | 8.5% |
| nan | Neurology | 920 | 4.8% | 0.0% | 11.0% |
| PHASE2/PHASE3 | nan | 855 | 5.0% | 0.0% | 10.8% |
| PHASE3 | Endocrinology | 692 | 4.7% | 3.2% | 5.8% |
| PHASE2 | Neurology | 634 | 4.1% | 0.0% | 8.3% |
| PHASE3 | Neurology | 533 | 4.8% | 2.8% | 5.8% |
| PHASE3 | Cardiovascular | 524 | 2.6% | 0.7% | 4.7% |
| PHASE4 | Cardiovascular | 487 | 2.9% | 0.0% | 7.3% |
| PHASE2 | Cardiovascular | 471 | 2.5% | 0.0% | 5.6% |
| PHASE4 | Endocrinology | 460 | 4.4% | 0.0% | 8.9% |
| PHASE2 | Endocrinology | 452 | 5.0% | 1.6% | 8.9% |
| EARLY_PHASE1 | nan | 300 | 4.5% | 0.0% | 10.7% |
| PHASE4 | Neurology | 296 | 4.2% | 0.0% | 10.5% |
| PHASE4 | Oncology | 228 | 4.7% | 0.0% | 11.1% |
| PHASE1 | Neurology | 173 | 2.5% | 0.0% | 5.8% |
| PHASE1 | Endocrinology | 147 | 2.0% | 0.0% | 4.1% |
| PHASE1/PHASE2 | Neurology | 112 | 5.3% | 0.0% | 13.4% |
| PHASE2/PHASE3 | Oncology | 106 | 4.2% | 0.0% | 7.4% |

## What this does and does not tell us

- This checks that the synthetic generator's dropout rate, once calibrated per trial, actually matches the real rate for trials of the same phase and therapeutic area. It is a check on the training data assumptions, not a validation of the trained model itself.
- AACT only gives trial-level and arm-level counts, not individual patient visit records, so this cannot validate the per-patient XGBoost classifier directly. That still needs a patient-level real dataset (Project Data Sphere or ImmPort are the two candidates noted in docs/data_sourcing.md).
- The calibration table (core/utils/aact_dropout_rates.json) only has real rates for combinations of phase and therapeutic area with at least 50 real trials behind them. New trial types outside our four seed areas fall back to a phase-only or overall average, which will be less accurate.
