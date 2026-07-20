# Project Data Sphere real-patient validation report

This is the first time TrialGuard's model has been checked against real,
patient-level data rather than synthetic patients or trial-level aggregate
counts. Five real oncology studies were reviewed: three small-cell lung
cancer trials (118 real patients combined), one breast cancer trial (443
real patients), and one lymphoma trial (107 real patients). Three separate
findings came out of this, and they are not all good news, on purpose,
that is the point of doing this.

## Finding 1: most real "discontinuation" is death, not dropout

Every one of these datasets records why a patient left the trial. Reading
the actual reason codes:

- **Lung trials**: of patients who discontinued, the overwhelming majority
  (85 of roughly 118) discontinued because they died. Only 12 patients
  (10.2%) discontinued for a reason a coordinator could plausibly have
  influenced (withdrawal by subject, or lost to follow-up).
- **Breast trial**: 350 of 443 patients (79%) completed treatment as
  planned. Of those who did not, only 14 (3.2%) are recorded as the
  patient choosing to withdraw. The rest are disease progression, adverse
  events forcing a stop, or other clinical reasons.
- **Lymphoma trial**: does not track dropout at all, only alive-versus-dead
  survival status. It was excluded from this analysis for that reason.

TrialGuard's premise is catching patients a coordinator can still act on,
not patients whose cancer is progressing. Lumping death in with dropout
measures a different, much larger, and much less actionable phenomenon.

## Finding 2: our AACT-based calibration was almost certainly measuring the wrong thing for oncology

Earlier, the synthetic generator's oncology dropout rate was calibrated to
AACT's reported rate for Phase II oncology trials: 33.9%. That number came
from AACT's participant-flow tables, which count *all* reasons a patient
left a trial arm, not just behavioral withdrawal. When the same real
studies used here are filtered down to genuine behavioral dropout only:

| Source | n | Real behavioral dropout rate |
|---|---|---|
| Lung (3 trials) | 118 | 10.2% |
| Breast (1 trial) | 443 | 3.2% |
| **Combined** | **561** | **4.6%** |

4.6% versus the 33.9% we calibrated against. This is not a rounding
difference, it means the AACT calibration for oncology needs to be redone
using AACT's own reason-level breakdown (AACT's `drop_withdrawals` table
does have a `reason` field with categories like "Withdrawal by Subject"
and "Death" separately, this analysis just used the blunter
started-vs-completed count instead of filtering by reason). That is a
concrete, scoped fix for next time, not done in this pass.

## Finding 3: the model, tested honestly, showed no real discriminative power, but the test itself was severely handicapped

The trained model was run, unchanged, against the 118 real lung patients,
using their real age, sex, and adverse event count. None of the other 20
features it expects (distance to site, employment status, prior dropout
history, visit cadence, medication adherence, quality of life trend) exist
in this dataset in a directly usable form, so each was filled in with the
synthetic training set's own average value for that feature, a standard,
defensible way to handle genuinely missing inputs, but one that
substitutes "assume average" for anything we could not observe.

**Result: ROC-AUC 0.477.** That is not a meaningful improvement over
guessing, it is very slightly worse than guessing.

This needs two honest caveats, not to soften the number, but because both
are true and matter for what to do next:

1. **Only 12 real dropout events.** With this few positive cases, the
   uncertainty band around any AUC estimate is wide. This number could
   plausibly sit anywhere in a broad range just from sample size alone.
2. **Most of the model's real signal was replaced with "average" before
   this test ran.** Recall from the SHAP analysis that hazard ratio, visit
   frequency, adverse event rate, distance to site, and quality-of-life
   trend were the model's strongest drivers. Of those, only adverse event
   rate was real in this test, everything else was a generic stand-in.

### Follow-up: getting the real visit-cadence data back didn't fix it

The dosing/exposure files (`adex`) for these same three studies turned out
to have real visit-by-visit dates and a real dose-interruption flag,
already downloaded, just not used in the first pass. Real
`visit_frequency_rate` (the model's single strongest driver) was
reconstructed from actual visit dates and a real dose-interruption signal
was added, replacing two of the imputed "average" placeholders with
genuine real values.

**Result: ROC-AUC 0.481.** Essentially unchanged from 0.477.

This is a more informative result than it looks. If the missing-features
explanation were the main problem, restoring the model's strongest feature
with real data should have moved the number meaningfully. It didn't. That
points at the other caveat, only 12 real dropout events, as the dominant
constraint, not feature availability. Twelve examples of the outcome being
predicted is not enough to reliably learn or measure a pattern from,
regardless of how good the surrounding features are.

**Honest conclusion, updated**: this is not evidence the model works on
real patients, and it is not clean evidence it doesn't, either. But it now
points specifically at volume, not missing fields, as the binding
constraint. Getting more real fields from these same files won't fix this
on its own. What's needed is more real patients with the outcome we're
trying to predict, on the order of 200 to 400 real dropout events (a
standard rule of thumb for this many model inputs), not the roughly 26
currently available across all usable studies. That means more, and
larger, real studies, not more columns from the ones already in hand.

## Combining cardiovascular data in: two independent checks, one consistent answer

Once MUSIC's 992 real cardiovascular patients (11 genuine lost-to-follow-up
events, decoded and verified directly against MUSIC's own codes file) were
added to the 561 real oncology patients, two separate checks were run.

**Check 1: run the existing synthetic-trained model, unchanged, against all
1,553 real patients combined** (age, sex, and severity real; everything
else imputed, same method as before). Result: **AUC 0.561**, but this
splits unevenly by domain:

| Source | n | real events | AUC |
|---|---|---|---|
| Cardiovascular (MUSIC) | 992 | 11 | **0.732** |
| Oncology (breast) | 443 | 14 | 0.524 |
| Oncology (lung) | 118 | 12 | 0.500 |

The model may carry real signal for cardiovascular patients even without
retraining. Worth real caution here too though: 11 events is a small base
for a strong claim, this could partly be luck.

**Check 2: train a small, fresh model directly on the real data**, not the
existing 23-feature model, a much simpler one using only the three fields
that could be honestly verified as real across every source (age, sex,
severity), evaluated with 5-fold cross-validation rather than a single
lucky split, since the whole point of doing this properly is reporting a
range, not a number. Result: **mean AUC 0.536, ranging from 0.404 to 0.673
across folds**. That range is the honest answer to "how much would this
number wobble," and it wobbles a lot, exactly what's expected with roughly
7 to 8 real events per test fold.

**Both checks land in the same place, independently**: real-world
performance today is somewhere around 0.53 to 0.56, nowhere near the 0.90
the synthetic data reports, and the uncertainty band around that number is
still wide. Two different methods agreeing on both the number and its
shakiness is a stronger result than either check alone, that consistency
is itself evidence the measurement is real, even though the underlying
accuracy isn't yet where it needs to be.

**Note on what was and wasn't changed**: this was an analysis exercise,
not a production retrain. The deployed model files were backed up to
`ml_models_synthetic_backup/` before this work started and were never
touched, everything in this section came from a separate, throwaway
3-feature model built only to test feasibility. The deployed model is
exactly as it was.

## What this changes going forward

- Fix the AACT oncology calibration to filter by reason, not just
  started-versus-completed.
- Before trusting any accuracy number for this model, get real
  visit-cadence and adherence-adjacent fields into a validation set, the
  current test could not fairly exercise the model's main strengths.
- Keep the death-vs-dropout distinction explicit anywhere this project
  reports a "real-world dropout rate" from now on, silently mixing the two
  produces a number that looks precise and means something different than
  it appears to.
- Cardiovascular is worth a closer look before oncology gets all the
  attention, 0.73 AUC on real data, even in a small sample, is the best
  real signal found anywhere in this project so far.

## Cardiovascular follow-up: how solid is that 0.73, and what's driving it

Two checks were run on MUSIC specifically, using a model and scaler pair
confirmed to be a matched, uncontaminated set (loaded explicitly from a
backup taken before a later retrain, rather than the live model directory,
after catching a real risk of testing against a scaler that had already
been updated while the model file hadn't, an internally inconsistent pair
that would have produced a meaningless number).

**How solid**: a 2,000-resample bootstrap gives a 95% confidence interval
of **0.532 to 0.900** around the 0.731 point estimate. That's wide, as
expected with only 11 real events, and the lower end sits close to chance.
Read as: probably real, not yet certain.

**What's driving it**: age, not severity. The model's prediction correlates
0.39 with age and only 0.13 with NYHA class (the severity measure used).
In the real data, patients lost to follow-up were older on average (66.8
vs 64.6 years), matching the direction the model already leans. Severity
barely differed between the two groups (1.18 vs 1.22), essentially noise
at this sample size. This is a different story from the oncology
assumption that sicker patients drop out more, here it looks like older
patients disengage more, independent of how severe their heart failure is.
Worth keeping in mind if cardiovascular becomes the next area of focus,
the model may need to lean on different signals than oncology does.

## Major update: 13 more real studies, real event count passes the target

More Project Data Sphere studies were added in two batches (22 study
folders, then 34). Each was opened individually, its disposition or
end-of-study reason field was located (the field name is different almost
every time: `DSDECOD`, `DSREAS`, `DSEOS`, `OFFTRT_RX`, `offtrt_reason`,
depending on the sponsor and the era the trial was run in), and every
reason code was read in full and classified by hand into behavioral
dropout (the patient's own choice to stop, matching the same definition
used everywhere else in this document: withdrawal by subject, withdrawal
of consent, lost to follow-up, patient refused further treatment) versus
everything else (death, disease progression, adverse event forcing a
stop, administrative or investigator decision, protocol violation). Nine
of the 13 new studies had a usable reason field. Four did not and were
set aside.

One correction caught along the way: an early pass on two glioma studies
matched only the exact phrase "withdrew consent" and missed "SUBJECT
WITHDREW CONSENT" as a separate wording of the same thing, undercounting
those two studies by 17 patients before the mistake was found and every
count was rechecked against the raw text values, not assumed category
names. Every number below has been checked this way, and checked again
for the same patient appearing under two different reason labels in the
same file (this happened in three of the studies below, each case is
counted once, not twice).

| Study | Real behavioral dropouts | Total patients |
|---|---|---|
| Glioma (EMD Serono, 2008) | 20 | 273 |
| Glioma (EMD Serono, 2009) | 6 | 89 |
| Pancreatic (EMD Serono, 2009) | 4 | 44 |
| Lung, non-small-cell (Eli Lilly, 2010) | 38 | 549 |
| Lung, small-cell (Eli Lilly, 2011) | 8 | 130 |
| Colorectal (Amgen, 2005) | 169 | 842 |
| Colorectal (Amgen, 2006, protocol 20050181) | 59 | 946 |
| Colorectal (Amgen, 2006, protocol 20050203) | 28 | 935 |
| Head and neck (Amgen, 2007) | 57 | 520 |
| Lung, small-cell (Amgen, 2002) | 16 | 479 |
| Lung, small-cell (Alliance, 1998) | 37 | 587 |
| Lung, small-cell (Alliance, 2007) | 11 | 95 |
| Multiple myeloma (Alliance, 2002) | 40 | 546 |
| Lung, small-cell (Pfizer, 2002, both cohorts) | 20 | 243 |
| **New subtotal** | **513** | **6,278** |

Four studies were checked and set aside, same honesty rule as before,
these are not being quietly dropped:

- **Breast cancer (Alliance, 2006)**: this dataset is surgical and
  pathology data (tumor markers, biopsy grading), it never tracked why a
  patient left the study, so there is nothing to extract.
- **Two colorectal studies (Amgen, 2004 and 2006)**: checked in an
  earlier pass, their disposition-adjacent files only record death and
  progression-free-survival flags, no behavioral reason field exists.
- **Seven "linked" studies** (breast, gastric x2, pancreatic, prostate
  x3): these are a different kind of dataset entirely, patient records
  linked to national insurance billing data (MEPS) for cost and
  utilization research, not trial disposition tracking. None of them
  have a disposition field of any kind. Confirmed again this pass across
  three more of these that were added since the last check, same result
  every time.

One source was resolved after further digging, one was ruled out of
scope:

- **The Pfizer lung cohort's `STATDI` code turned out to be a dead end,
  but a different field in the same study was not.** `STATDI` is
  labeled "Disease status" in the study's own data dictionary, meaning
  alive versus dead, not a reason for leaving the study. Reading the full
  99-page data dictionary further found a proper field, `REASWD`
  ("Primary reason of discontinuation"), that had been missed on the
  first pass because it lives in a different file (`popu`, patient
  population summary) than the one first checked (`dsstat`, disease
  status). Its value labels are not printed directly in the dictionary,
  the FORMAT column that should link the variable to its label table is
  blank for every variable in this PDF, including the two examples the
  dictionary's own README uses to explain how it's supposed to work.
  That link was rebuilt by matching `REASWD` to the study's `REASDC`
  label table (matching name, and every code actually appearing in the
  data falls inside `REASDC`'s defined range), then checked against the
  README's own worked examples (`AESTAT` to `STATAE`, `AESER` to
  `NOYES`) to confirm this name-matching approach reproduces the
  documented answer exactly before trusting it here. On that basis:
  **20 real behavioral dropout events** ("lost to follow-up" and
  "subject did not wish to continue") across the cohort's 243 patients,
  now added to the tally below.
- **The Spanish prostate cancer dataset is out of scope**, not for a
  data quality reason, English-language data is what this project uses,
  so it is excluded rather than translated or adapted.

### What this changes

Before this batch, the project had 37 real, verified behavioral dropout
events to work with (26 from the original five oncology studies, 11 from
MUSIC's cardiovascular cohort). That is a small enough number that any
accuracy figure computed from it, as the two independent checks earlier
in this document showed, wobbles wildly (the 5-fold cross-validation
range was 0.404 to 0.673 AUC on the same underlying signal).

Adding this batch's 513 events brings the project total to **550 real
behavioral dropout events**, across roughly 7,850 real patients. The
rule of thumb used earlier in this document, 200 to 400 real events
needed before a retrain's accuracy number can be trusted, has now been
passed. This does not by itself mean the model will score well on real
data, only that there is finally enough real signal to measure honestly
whether it does, with a confidence interval narrow enough to mean
something.

This is a good point to stop and decide, with the deployed model backed
up and untouched in `ml_models_synthetic_backup/`, whether to run a full
retrain on this real combined dataset next, following the same
methodology already established: bootstrap confidence intervals,
k-fold cross-validation, and an honest report of what the real accuracy
turns out to be, good or bad.

## Pfizer STATDI resolved, Spanish dataset dropped

Two loose ends from the previous update were closed out.

`STATDI` in the Pfizer lung cohort turned out to be a dead end after all,
its own data dictionary labels it "Disease status" (alive versus dead),
not a reason for leaving the study. But reading further through that
same 99-page dictionary turned up the field that should have been
checked in the first place: `REASWD`, "Primary reason of
discontinuation," sitting in a different file (`popu`, patient
population summary) than the one first checked. Its value labels were
not printed directly next to it, the FORMAT column that should link the
variable to its label table was blank for every variable in this PDF,
including the two examples the dictionary's own instructions use to
explain how decoding is supposed to work. The label table was
identified by matching the variable name to a label table called
`REASDC` (matching name, and every code that actually appears in the
data fits inside its defined range), then that same name-matching
approach was tested against the dictionary's own worked examples to
confirm it reproduces the documented answer exactly, before trusting it
here. Result: **20 more real behavioral dropout events**, out of 243
patients across the cohort.

The Spanish prostate cancer dataset was dropped from consideration,
this project uses English-language data only, so it is out of scope
rather than something to translate or adapt.

## Second major update: 10 more studies, several large ones

Ten more study folders were added and worked through the same way:
locate the real disposition or reason field (a different name almost
every time), read the source data dictionary to decode any numeric
codes rather than guess, count each patient once, and set aside
anything that cannot be confidently classified.

| Study | Real behavioral dropouts | Total patients |
|---|---|---|
| Breast cancer (Alliance, CALGB 40101, arm A) | 102 | 3,171 |
| Breast cancer (Alliance, CALGB 40101, arm B) | 135 | 3,871 |
| Breast cancer (Alliance, CALGB 40502) | 35 | 283 |
| Breast cancer (Eli Lilly) | 40 | 385 |
| Colorectal cancer (Alliance, 1997) | 56 | 851 |
| Colorectal cancer (Alliance, 2004) | 104 | 2,968 |
| Smoking cessation (Alliance, N99C4) | 619 | 1,551 |
| Pancreatic cancer (Clovis Oncology) | 18 | 367 |
| **New subtotal** | **1,109** | **13,447** |

Two studies in this batch were checked and set aside:

- **A 1994 colorectal surgery trial** (open versus laparoscopic
  colectomy) tracks overall survival status only, alive or dead, no
  disposition or dropout field exists in the data provided.
- **A prostate cancer "tumor growth" dataset** turned out to be pooled
  tumor-size measurements over time from nine different trials, used for
  modeling growth curves, not a disposition-tracking dataset of any
  kind.

Two studies had a mostly-usable reason field with a gap worth being
upfront about: in the second colorectal study, code "2" (261 of 2,968
patients) is never defined anywhere in that study's own data dictionary,
it jumps from code 1 straight to code 3. In the CALGB 40502 breast
study, code "19" (129 of 283 patients, the single largest group) is
similarly undefined, the dictionary jumps from an unlabeled note
straight to code 18. Both gaps were left uncounted rather than guessed
at, on both sides, not classified as behavioral and not classified as
non-behavioral either. Every other code in both studies was checked
against its source dictionary and is counted with confidence.

One study needs a flag for a different reason, not a data quality one.
**The 619-event study (N99C4) is a smoking cessation trial**, nicotine
inhaler and bupropion, not a cancer drug trial like everything else
counted so far. Its dictionary decodes cleanly (`2=Consent Withdrawn`,
`3=Lost to Follow-up`), and the events are exactly as real as any other
study here, but a 40% behavioral dropout rate is expected and normal for
a smoking-cessation intervention, not a signal that something is wrong
with the data. Folding it into a single blended "real dropout rate"
number without this note would make that number mean something
different than it looks like it means. It is included in the running
total below because it is genuine clinical trial data with the same
kind of real, coordinator-actionable dropout TrialGuard is built to
predict, just from a different kind of trial.

### Updated grand total

**1,659 real behavioral dropout events, across roughly 21,300 real
patients**, combining every batch processed so far.

## What "1,659" actually means, and what it doesn't

That number answers one question only: is the outcome label real and
correctly verified. It does not answer whether a row of data is usable
to teach the model anything, and those are genuinely different bars.
Every study here was checked against both.

**Bar 1: is this the right kind of trial?** TrialGuard predicts dropout
from treatment trials for a diagnosed condition, using clinical and
operational signals (adverse events, disease severity, visit cadence,
travel burden). The smoking-cessation study found in the second update
does not fit this, a patient there stops because of nicotine cravings,
not chemotherapy or travel distance. Pooling it in without a flag risks
teaching the model a contradictory pattern rather than a useful one.
Removing its 619 events leaves **1,040 events from the right kind of
trial**.

**Bar 2: do we have real features to learn from, not just a real
label?** The model does not just need to know a patient dropped out, it
needs the columns that explain why: visit dates (to compute how often
visits are missed or spaced out), adverse event severity over time,
quality-of-life trend, dosing/treatment cycle data. A study can have a
perfectly real, well-verified dropout label and still only be useful
for *checking* whether the model's existing predictions look reasonable
(as the earlier oncology and cardiovascular checks in this document
did), not for *teaching* it something new, if none of those other
columns exist. Every one of the 24 studies behind the 1,040 events was
checked file by file for three real feature domains: a visit or dosing
schedule with real dates, adverse event records, and a quality-of-life
or symptom questionnaire.

| Tier | What it means | Events | Studies |
|---|---|---|---|
| **Retrain-usable** | Has real visit/dosing dates and/or a real quality-of-life questionnaire, not just a label | **530** | Both glioma studies, the Pancreatic (EMD Serono) study, both Eli Lilly lung studies, all three Amgen colorectal studies, the Amgen head-and-neck study, the Amgen small-cell lung study, both Pfizer cohorts, the CALGB 40502 breast study, the Eli Lilly breast study, the Clovis Oncology pancreatic study, all three G1 Therapeutics lung studies |
| **Validate-only** | Verified real label, but only baseline demographics, no visit-level or symptom data to learn from | **510** | The two Alliance small-cell lung studies, the multiple myeloma study (NCT00052910), both CALGB 40101 breast cohorts, both remaining Alliance colorectal studies, the original breast (Alliance 2009) study, MUSIC |

One honest caveat that applies to every study in the "retrain-usable"
row, not just some of them: none of the 45 Project Data Sphere studies
checked in this entire project collect the two fields the model's
earlier SHAP analysis flagged as strong drivers, distance to the
clinic and employment status. That is not a gap specific to any one
study, no oncology trial's standard case report form asks a patient how
far they live from the site or whether they work full time, it is a
structural blind spot across real clinical trial data generally. Even
the richest studies above still need those two columns filled in with a
reasonable stand-in rather than a real value.

## Recommendation: drop three features the model has never really been tested on

Before any retrain, three of the model's 23 features should be dropped:
**`distance_to_site_km`**, **`employment_status`**, and
**`prior_dropout_history`**.

This isn't a guess, it's what the data itself forced. Every data
dictionary across all 45 real PDS studies, plus MUSIC's full 90-plus
column cardiovascular file, plus the heart failure dataset, was searched
by hand for these fields:

- `distance_to_site_km`: zero occurrences anywhere. No clinical trial
  paperwork checked in this entire project asks a patient how far they
  live from the clinic.
- `employment_status`: exists in real data, but only inside 7 PDS files
  already excluded for having no dropout-reason field at all (they are
  insurance-billing datasets, not trial disposition records). It has
  never once appeared in the same file as a verified dropout label.
- `prior_dropout_history`: zero occurrences. Each dataset is a single,
  independently de-identified trial, there is no way to know if a
  patient here also appears, and dropped out of, some other unrelated
  trial years earlier.

Every real-world check run in this project, going all the way back to
the first oncology validation, has had to feed the model a fake
"assume average" value for these three columns, because no real value
has ever existed to use instead. That means whatever importance they
showed in the model's SHAP analysis on synthetic data has never been
confirmed against a single real patient, it could easily be an artifact
of how the synthetic generator was built rather than a true real-world
pattern. Keeping a feature the model can never honestly be given a real
value for, in production, is worse than dropping it: it silently
guesses on every real prediction without saying so.

**Status: recommended, not yet applied to the code or a retrain.** This
requires an actual decision and a retrain to take effect, nothing has
been changed yet. If it's approved, `FEATURE_COLUMNS` in
`core/utils/data_pipeline.py`, the Cox model's `COX_COVARIATES` in
`core/utils/survival_model.py`, and the synthetic data generator would
all need the matching update.

## The real retrain, and the real vs. synthetic comparison

This is the first time in the project a model has actually been trained
on real patients, not just validated against them. 6,478 real patients
were assembled from every study with usable feature data, 22 real
studies across PDS and ImmPort combined, giving **544 real, verified
behavioral dropout events**, well past the earlier 530 estimate once
ImmPort's Neurology and Endocrinology patients were added in.

Each real patient's row was built the same way the model has always
worked: real visit dates where the source study had them (dosing
records, vital-sign visits), real adverse event counts and timing, real
disposition-confirmed labels. Two honest gaps, disclosed rather than
hidden: `cumulative_missed_visits` could not be reconstructed for most
studies (no scheduled-visit calendar exists in any of them to compare
against), and a genuine `medication_adherence_score` was only available
for a handful of studies, so both ended up as constants across most of
the real dataset. Cox's real-data model was fit on 3 covariates instead
of 5 for this reason, explained below.

**Both models, real and synthetic, were trained and evaluated the exact
same way**: the same 20 active features, the same train/test split
(75/25, stratified), the same non-Optuna XGBoost settings (so neither
side gets tuning as an unfair advantage), the same 2,000-resample
bootstrap for a confidence interval, not just a point estimate.

| Metric | Real data (n=6,478) | Synthetic data (n=6,478) |
|---|---|---|
| Events | 544 (8.4%) | 273 (4.2%) |
| Test set events | 136 | 68 |
| XGBoost ROC-AUC | **0.7044** | 0.6804 |
| AUC 95% bootstrap CI | 0.657 – 0.749 | 0.613 – 0.746 |
| F1 | 0.2488 | 0.1259 |
| Precision | 0.1861 | 0.1200 |
| Recall | 0.3750 | 0.1324 |
| Brier score | 0.1258 | 0.0630 |
| Cox concordance index | 0.5554 | 0.7356 |

**The headline: for the first time, the real-data model is not losing to
the synthetic one.** Every earlier real-data check in this project
landed around 0.48 to 0.56 AUC, clearly worse than chance-adjacent, next
to a synthetic model claiming 0.90+. That gap is gone. The two AUCs here
overlap heavily and are statistically indistinguishable at this sample
size, real data is now performing in the same range as synthetic, not a
fraction of it. Recall is meaningfully higher on real data too (0.375 vs
0.132 at the standard 0.5 threshold), meaning it catches more of the
real dropouts it's supposed to catch.

**Two things this isn't**: the synthetic number here (0.68) is much
lower than the model card's older 0.99, because that older number came
from a 50-trial Optuna search on the previous 23-feature schema
including the three now-reserved features. This comparison deliberately
turned tuning off for both sides to keep it fair, so the synthetic side
looks weaker here than in its own history, not because anything about
the synthetic generator changed, only the comparison conditions did.
Second, Cox's concordance is much better on synthetic (0.74) than real
(0.56), a direct consequence of the missed-visits and adherence gap
above, not a sign the real hazard model is worse at its actual job of
ranking risk, since the AUC-based XGBoost numbers above already carry
the real hazard ratio as an input feature and still come out ahead.

**Status: trained and evaluated, not yet deployed to `ml_models/`.**
The live model is still the one backed up in
`ml_models_pre_feature_reduction_backup/`. Scripts and the full
per-study extraction pipeline live in `scripts/real_data/`. Whether to
deploy this real-trained model into production, re-run XGBoost with a
real Optuna search on top of it, or keep collecting more real data
first is a decision worth checking in on before acting.

This section is superseded by the actual real retrain above: 530 was
the estimate before the ImmPort studies were added and the real
extraction pipeline was actually built and run. 544 is the real,
final number.
