# Real dataset v2: early-window ("truly prospective") rebuild

## Why this exists

Investigating the "is there a real cross-hospital pattern we're missing"
question (asked directly), found: `visit_number` and
`adverse_events_count` both show a consistent direction with dropout
across the large majority of independent real hospitals (22/27 and
21/26), surviving both a hospital-demeaning test and a per-hospital
direction-consistency check - a genuine, non-hospital-identity pattern.

**The catch, found before declaring victory:** those features are
currently computed from each patient's ENTIRE observed history,
including visits right up to the point they left. A patient who dropped
out early mechanically has fewer visits on file simply because they
weren't there long enough to accumulate more - that's not the same as
predicting dropout in advance. This v2 dataset exists to test whether
the pattern survives when features are rebuilt using ONLY information
that would have genuinely been available early, before the outcome was
known - a real test of prospective usefulness, not just a descriptive
correlation.

## Ground rules (fixed before touching any code, so this can't drift)

1. **60-day cutoff**, chosen to match this project's own stated purpose
   ("60-Day Early Warning"), not picked after seeing results.
2. **The label does not get windowed.** `dropout_status` and
   `days_to_event` still reflect the TRUE, full, eventual outcome -
   only the PREDICTOR features get truncated to the early window. This
   is not optional: if the label were windowed too, we'd just be
   measuring something else, not testing early-warning validity.
3. **Every feature that was built from a visit-day list, AE-day list,
   dosing-event list, or QoL-assessment list gets re-derived using only
   the events at day <= 60** (real day-offset field where the study has
   one; approximated from PPMI's age-at-visit where it doesn't - see
   PPMI note below).
4. **Correction made during implementation:** originally planned to use
   the EARLIEST real severity/QoL value rather than the last. Reconsidered
   this while implementing PPMI (which needed real date-based windowing,
   unlike the PDS studies which just needed a day-offset filter): the
   correct rule is not "always use the first value," it's "use the most
   RECENT value recorded at or before the cutoff day" - i.e. filter every
   value series to day <= cutoff_days first, exactly like visit days and
   AE counts already are, then take the last (most recent) point *within
   that filtered window*, not the global last. This is what a real
   prediction made on day 60 would actually have access to, and it's the
   same pattern every other windowed feature in this rebuild already
   uses (windowed visit_days -> take the resulting last day; windowed AE
   list -> count/trend over the resulting set). Applied consistently
   this way for every study, not just PPMI.
5. **Planned-visit schedules (for the 4 studies with real missed-visit
   reconstruction) get capped to the same 60-day window too** - counting
   "missed" a visit that was only ever scheduled for day 90 would be
   wrong and would inflate missed-visit counts for every patient.
6. **Population rule unchanged in spirit**: a patient needs >=1 real
   visit within the window to be included, same as the >=1-visit-ever
   rule v1 used, just scoped to the window.
7. Original extraction scripts are not being destroyed - each one gets
   a `cutoff_days` parameter (default `None` = old unlimited behavior,
   so v1 stays exactly reproducible), and a new value of `60` is used
   for v2. No silent behavior change to the existing pipeline.

## Known precision limits, disclosed up front

- **PPMI**: no per-visit calendar date was in the download, only
  age-at-visit (~0.1-year precision, ~36 days). Elapsed days
  approximated as `(age_at_visit - enroll_age) * 365.25`. A 60-day
  cutoff is close to this field's own precision floor - flagged as
  coarser than the day-level precision every PDS/ImmPort study has, not
  hidden.
- **MUSIC**: has exactly one baseline assessment per patient, no visit
  series at all. Already effectively "day-0 only" by construction -
  nothing to window, included as-is.

## Per-study progress (filled in as each script is converted)

- [ ] extract_amgen_group.py (4 studies)
- [ ] extract_pancrea_clovis.py
- [ ] extract_g1thera.py (3 studies)
- [ ] extract_breast_elililly.py
- [ ] extract_lungsm_amgen.py
- [ ] extract_immport.py (2 studies)
- [ ] extract_immport2.py (2 studies)
- [x] extract_glioma.py (2 studies + Pancrea_EMDSero_442, reused via direct
      `extract(base_dir='../../data/pds/Pancrea_EMDSero_2009_442/DATA',
      study_label='Pancrea_EMDSero_442')` call, same SDTM shape, no
      separate script file exists for it) - `count_missed()` already only
      counts planned visits up to the patient's own last ACTUAL day as
      "relevant", so passing it the windowed visit_days is correct without
      separately windowing the planned schedule. Verified: all 3
      populations (258/13, 85/4, 43/2) reproduce exactly with cutoff_days=None.
- [ ] extract_lungsm_elililly.py - has real planned-visit-schedule, same
- [ ] extract_breast_allianc.py
- [ ] extract_pfizer.py (2 cohorts)
- [ ] extract_lungno_elililly.py
- [x] extract_ppmi.py (4 cohorts) - age-based approximation for
      visit_number/spacing (no per-visit calendar date in this download);
      severity/QoL/AE use their own real INFODT dates, converted to
      elapsed days via each patient's real ENROLL_DATE.

      **CORRECTION #9, found while building this:** the original PPMI
      severity computation sorted patients' real UPDRS records by the
      raw MM/YYYY *string* INFODT field before taking the last value -
      string sort misorders dates (e.g. "3/2015" sorts before "12/2014"
      because '1' < '3' as the first character), which is a real,
      pre-existing bug, not something this rebuild introduced. Fixed by
      sorting on the actually-parsed date everywhere (both the tertile-
      cutoff calculation and the per-patient snapshot value use the same
      corrected sort now). This shifts the real severity distribution
      somewhat (0/1/2 tertile counts move from {1699/1680/1588} to
      {1860/1399/1708}) - a genuine, disclosed correction to v1, not
      preserved as-is just to avoid changing the number. Verified the fix
      is deterministic (re-ran with no code changes, identical result).
      Also fixed a related bug caught in the same pass: severity/QoL/AE
      were accidentally made to depend on having a valid ENROLL_DATE even
      in the unwindowed (v1) case, silently dropping real values for
      patients with a missing enrollment date - fixed so that dependency
      only applies when cutoff_days is actually set (needed to compute
      elapsed days for windowing).
- [ ] extract_music.py - no change needed (already single baseline row), just re-run for consistency

## Findings

Built `real_dataset_v2_final.csv`: 12,302 real patients, 27 studies (SDY1904
drops out entirely - genuine finding, its earliest recorded visit in this
data extract is real study day 161, no patient has anything within 60
days, confirmed by checking the raw VSDY values directly, not a bug),
1,135 real behavioral dropouts. Cleaning pass identical to every other
dataset in this project: zero duplicates, zero missing values after
imputation, all ranges sane.

Re-ran the exact same two tests used to find the original pattern
(hospital-demeaned correlation, per-hospital direction consistency),
this time on the early-window features:

| Feature | Full-history direction consistency | Early-window (60d) consistency | Full-history effect size (within-hospital r) | Early-window effect size |
|---|---|---|---|---|
| visit_number | 22/27 hospitals (82%) | 14/24 hospitals (58%) | -0.062 | -0.029 |
| adverse_events_count | 21/26 hospitals (81%) | 16/25 hospitals (64%) | -0.078 | -0.043 |
| days_between_visits_mean | 16/23 hospitals (70%) | 12/21 hospitals (57%) | +0.065 | +0.032 |

Average visit count, dropouts vs. retained: full-history 10.2 vs 13.2
(22% fewer); early-window 3.25 vs 4.47 (27% fewer) - the relative gap
holds up, but the absolute signal is much smaller and much less
consistent across independent hospitals once the model can't see a
patient's whole trajectory.

**Honest conclusion:** the pattern is partially real, not entirely an
artifact - there is a small, genuine early-warning signal in visit
frequency and early adverse-event reporting. But roughly half of what
looked like a strong, consistent cross-hospital pattern in the
full-history version was an artifact of measuring people's complete
observed history (which is mechanically shorter for early dropouts by
construction), not real prospective predictive power. The direction-
consistency dropped from ~80% of hospitals agreeing to ~60% - real
signal is still there, but far weaker than it first appeared, and this
is a big part of why the model's honest new-hospital AUC sits close to
chance (0.55): the strongest-looking pattern in the data turned out to
be mostly hindsight, not foresight.

This was worth verifying properly rather than either dismissing the
original finding or accepting it uncritically - both extremes would
have been wrong.
