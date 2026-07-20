# Why we picked AACT for real-world validation data

TrialGuard's models are currently trained and evaluated only on synthetic
patient data. Before we can claim the model works on anything real, we need
at least one dataset that comes from actual registered clinical trials, not
data we generated ourselves. This note explains which source we settled on
and why we passed on the others, so the reasoning doesn't get lost later.

## What we went with: AACT

AACT (Aggregate Analysis of ClinicalTrials.gov) is a database maintained by
the Clinical Trials Transformation Initiative, a partnership between Duke
University and the FDA. It mirrors every study registered on
ClinicalTrials.gov, updated daily, and for studies that have posted results
it includes the actual participant flow tables: how many patients started,
how many completed, how many dropped out, and the reason given for each
dropout (withdrawal by subject, adverse event, lost to follow up, death,
sponsor decision, and so on).

We chose it for a few plain reasons:

- It's real. This is not synthetic or simulated data, it's the same
  registry data that regulators, sponsors, and CROs already rely on.
- It's free and open. No account, no data use agreement, no committee
  review. We downloaded the full export directly.
- No identity verification of any kind is required. This mattered a lot
  once we found out the alternative sources now require it (more below).
- It's large. We downloaded the full export and checked the actual numbers
  ourselves rather than guessing. AACT has 593,122 registered studies in
  total. Of those, 47,259 have real per-arm dropout and withdrawal reason
  data, and 79,034 have participant flow (milestone) data. Of the 47,259
  with dropout data, 45,205 are interventional trials (actual clinical
  trials, not observational studies), 39,111 are completed, and they cover
  every phase from early phase 1 through phase 4. Broken down by the same
  therapeutic areas our synthetic trial seeds use: 2,079 studies in
  cardiovascular, 8,809 in oncology, 2,381 in neurology, and 2,910 in
  endocrinology, each with real dropout reason breakdowns attached.
- It's kept up to date. AACT is regenerated daily from live
  ClinicalTrials.gov data, so we are not working off a stale snapshot.

The honest limitation: AACT gives us trial-level and arm-level numbers, not
individual patient visit records. It tells us how many patients on a given
arm of a given trial dropped out and why, but not the week-by-week
adherence, adverse event, or visit history of any single patient. That
means AACT is the right tool to validate our cohort-level forecast (the
30/60/90 day dropout counts), but it cannot by itself validate the
per-patient XGBoost classifier, which needs patient-level visit sequences
to be tested properly. We will need a patient-level source for that piece
eventually, but AACT gets us a genuine, defensible real-world benchmark
today with zero friction.

## What we looked at and passed on

**Project Data Sphere.** Real, patient-level oncology trial data, and one
of the more open sources out there, free registration with no research
proposal needed. We didn't go with it right now mainly because of timing,
the registration review took over a week in our test, and the license,
while not explicitly banning commercial use, scopes access to "research"
and includes a clause where you waive the right to patent anything that
comes out of using the data. Worth revisiting once we're ready to invest
the wait, but not the right fit for getting a first real number quickly.

**YODA Project.** Real, multi-sponsor patient-level data, but their own
data request form makes you certify the data "will not be used in pursuit
of litigation or for commercial interests." That's a direct conflict with
where TrialGuard is headed. Even using it purely for a one-time validation
study gets awkward, since any product downstream would need to be built
without touching that data or anything derived from a model trained on it.
Also requires a full research proposal and an independent review board,
which takes weeks.

**Vivli.** Same shape of problem as YODA. Real data, but access is gated
behind a research proposal and a review process, and the usage terms tie
you to whatever was approved in that proposal rather than giving broad
rights. Their secure analysis environment is also only free for the first
year, after that there's a daily fee if you're still using it.

**BioLINCC (NHLBI).** This one actually had the best real data on paper,
including a real completed randomized trial (the Digitalis Investigation
Group trial) and the Framingham Heart Study cohort, both offered through a
lighter "teaching dataset" track that used to skip the heavier data use
agreement. We ruled it out after checking their current login page: NIH
rolled out a new policy (notice NOT-OD-25-083) that now requires signing in
through Login.gov or ID.me, which means verifying your identity with a
government ID, and on top of that the requester has to hold a permanent,
senior research position at an institution, postdocs and grad students are
explicitly excluded, let alone an independent developer. This isn't a slow
process we could wait out, it's a policy that structurally excludes anyone
who isn't a tenured or tenure-track researcher with institutional backing.

**SEER.** The enhanced "Research Plus" tier has the exact same Login.gov
and institutional sponsor requirement as BioLINCC, so that's out for the
same reason. The basic SEER Research Data tier loosened its email
requirements in 2025 and might still be reachable, but SEER is a cancer
registry tracking survival and vital status, not clinical trial visits, so
even if we got access the fit for dropout prediction specifically is weak.
Not worth chasing right now.

**ImmPort.** Real NIAID-funded trial data, and registration looked lighter
than the NIH biomedical repositories above, several studies are posted as
fully public with no agreement at all. We didn't rule this out for good,
it's a reasonable second source to check later, we just didn't need it once
AACT gave us enough to work with.

**Update: checked properly, this is the lead for Neurology and
Endocrinology.** ImmPort's public search API needs no account at all, and
querying it directly turned up real, named interventional trials in both
of the two areas the project has zero real data for:

- Multiple sclerosis (Neurology): 349 matching studies, including real
  trials like SDY471 (44 patients), SDY547 "STAyCIS" (83 patients),
  SDY3285 "ACCLAIM" (65 patients), SDY549 "HALT-MS" (25 patients).
- Type 1 diabetes (Endocrinology): 1,005 matching studies, including
  SDY1904 (136 patients), SDY1178 islet transplantation (48 patients),
  SDY524 "AbATE" (83 patients), SDY797 (49 patients).

The public search only returns study-level summaries (title, enrollment,
objectives), not participant records. Pulling the actual data hit a flat
401 Unauthorized, participant-level data needs a registered account and
access token. Registration itself, per ImmPort's own documentation, is
self-service ("simple and quick"), no data use agreement, no committee
review, no institutional sponsor required, unlike PPMI or BioLINCC. Still
requires a real identity to sign up, so it needs the project owner to
create the account, same handoff pattern as Project Data Sphere.

Honest caveat: these are small (9 to 136 patients per study) Immune
Tolerance Network mechanistic trials, not large registrational drug
trials. A good haul here will meaningfully help close the Neurology and
Endocrinology gap but won't rival PDS in scale. Whether the disposition
tracking inside these studies is as clean as PDS's is not yet known,
that requires actual account access to check.

**Update: account created, data pulled and checked.** Registered, got an
API key, and downloaded real participant-level disposition files for all
8 candidate studies using ImmPort's Aspera-based transfer API. Each
study's actual case report form or data dictionary was read to decode
numeric reason codes properly, the same standard applied to every other
source in this project, nothing was guessed from a similar-looking
study elsewhere.

| Study | Area | Real behavioral dropouts | Total patients |
|---|---|---|---|
| SDY471, Copaxone/Albuterol for MS | Neurology | 7 | 44 |
| SDY3285, ACCLAIM (Abatacept in MS) | Neurology | 12 | 65 |
| SDY524, AbATE (antibody therapy, new T1D) | Endocrinology | 9 | 83 |
| SDY797, T1DAL (Alefacept, new T1D) | Endocrinology | 5 | 49 |
| SDY1904, Tocilizumab in new-onset T1D | Endocrinology | 9 | 136 |
| **Subtotal** | | **42** | **377** |

Behavioral dropout was defined the same way as everywhere else in this
project: the patient's own choice to leave (withdrew consent, refused
further participation, lost to follow-up), not death, disease
progression, or an adverse event forcing a stop.

Three studies were checked and could not be used:

- **SDY549 (HALT-MS)**: the files ImmPort has for this study are T-cell
  receptor sequencing and immune repertoire data only, no disposition
  file of any kind was shared for this study.
- **SDY1178 (CIT-07 islet transplantation)**: dozens of real lab-result
  files but no disposition or termination-reason file anywhere in the
  file list.
- **SDY547 (STAyCIS)**: has a `DSREAS` reason-code column, but no data
  dictionary or annotated CRF was available for this specific study to
  decode what the numbers mean. Guessing based on another study's
  scheme (even one from the same trial network) was ruled out after the
  Pfizer STATDI lesson earlier in this project, wrong once is enough.
  Left uncounted rather than guessed at.

**The three chronically-missing fields (`distance_to_site_km`,
`employment_status`, `prior_dropout_history`) are absent here too.**
Checked two ways: a text search across all 17 downloaded case report
forms, data dictionaries, and glossary files (zero hits), and a direct
inspection of the richest demographics file's 44 actual columns (also
zero hits). This is not a PDS-specific gap. It now holds across every
real source checked in this project: PDS, MUSIC, UCI Heart Failure, and
ImmPort. Four independent real-world data ecosystems, the same three
fields missing from all of them.

**What this changes**: before this, the project had zero real data for
Neurology and Endocrinology. It now has 42 real, verified behavioral
dropout events across both, small compared to PDS's oncology numbers,
but the first real signal in either of these two areas, and it further
confirms the case for dropping the three fields, a fourth independent
data source with the same absence is no longer a coincidence worth
re-litigating.

**CDISC Pilot 01.** Originally treated as real trial data, but on closer
look it's the leftover dataset from a 2008 to 2010 industry pilot project
run with the FDA, and CDISC's own documentation says it accepted both
de-identified real data and "dummy data" built to look realistic. We can't
verify which parts are actually real patient outcomes, so we're keeping it
only as a reference for testing our data pipeline's SDTM format handling,
not as a source for any real-world accuracy claim.

**UCI Diabetes 130-Hospitals.** Genuinely real hospital data, no login
needed at all, but it's about diabetic patients being readmitted to a
hospital, not clinical trial patients dropping out of a trial. The domain
gap is too wide to use as a validation source. We're keeping this only to
stress-test whether our feature engineering code holds up against messy,
real-world data that wasn't generated to fit our schema.

## Bottom line

AACT is the dataset doing the real work right now: real, free, fast, no
identity checks, and large enough to matter. It validates the cohort
forecast side of the product today. The patient-level classifier still
needs its own real-world validation eventually, and when we're ready for
that, Project Data Sphere or ImmPort are the two worth trying first, in
that order.

## Update: first comparison run

We ran the comparison (`python manage.py validate_against_aact`, results in
[aact_validation_report.md](aact_validation_report.md)). Our synthetic
dropout rate turned out to be 14.7%, not the 35 to 45% the model card had
claimed, that number was never actually checked against the generator's
output until now. Against real AACT data across 63,233 interventional
trials, our synthetic rate lines up reasonably well for cardiovascular and
endocrinology trials, but is far too low for oncology (real trials average
33.9% dropout, ours assumes 14.7%). The generator also uses one flat rate
for every trial regardless of phase or therapeutic area, which the real
data says is not a safe assumption. Worth fixing before we lean on the
synthetic data for anything else.

## Real candidates found for neurology and endocrinology, not pursued yet

Both of these are real, genuinely useful, and confirmed to have some form
of dropout or discontinuation tracking, but both require registering under
your own name, not something that can be done without you. Saved here so
the research doesn't have to be redone later.

### Neurology: PPMI (Parkinson's Progression Markers Initiative)

- **Link**: [ppmi-info.org](https://www.ppmi-info.org/) , data access page:
  [ppmi-info.org/access-data-specimens/download-data](https://www.ppmi-info.org/access-data-specimens/download-data)
- Real, large: over 4,000 real Parkinson's patients tracked since 2010,
  original cohort of 423 untreated PD patients plus healthy controls,
  expanding to include prodromal participants.
- Dropout is explicitly tracked and reported: under 5% when fully enrolled.
- **Access**: sign a Data Use Agreement, submit an online application,
  reviewed by the Data and Publications Committee within about a week. Free.
  Not instant, needs your registration.

### Endocrinology: T1DiabetesGranada — downloaded, NOT usable for dropout_status, excluded

- **Link**: hosted on Zenodo, described in
  [this paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10733323/) (open
  access, free to read).
- Real: 736 type 1 diabetes patients, followed 4 years (Jan 2018 to Mar
  2022), over 22.6 million real continuous glucose readings, plus lab
  results and diagnosis codes. Downloaded and present at
  `data/T1DiabetesGranada/` (4 CSVs: `Patient_info.csv`, `Diagnostics.csv`,
  `Biochemical_parameters.csv`, `Glucose_measurements.csv`).
- **Earlier ambiguity, now resolved**: the paper's "no participant withdrew
  from the study" refers to a formal administrative act (unregistering from
  the system) that never happened. Separately, in "seldom cases" data
  collection ended early for one of four reasons: device allergy, death,
  transfer to another clinical unit, or the patient's personal decision to
  stop wearing the glucose monitor. Not actually contradictory — just two
  different things ("formal withdrawal" vs. "data collection stopped
  early").
- **The real, blocking problem, found once resolving the above**: only one
  of those four reasons ("patient's personal decision") is a genuine
  behavioral dropout in the sense used everywhere else in this project.
  Death, device allergy, and hospital transfer are not. The paper only
  describes the four reasons in aggregate narrative text — there is no
  per-patient reason code anywhere in the data. Checked all 4 raw CSV
  headers directly to be sure the paper's summary hadn't omitted a field:
  `Patient_info.csv` has 11 columns (ID, sex, birth year, measurement/lab
  date ranges and counts, diagnosis count) and no reason field;
  `Diagnostics.csv` is just ICD-9 codes; `Biochemical_parameters.csv` is
  lab values; `Glucose_measurements.csv` is date/time/reading. Unlike
  MUSIC's exit codes or PPMI's date bug, there is no second real field here
  to cross-reference against to recover which reason applies to which
  patient.
- **Decision**: using "data collection ended early" as a proxy for
  behavioral dropout would silently mix in deaths and medical/admin
  reasons — exactly the contamination this project's other sources
  explicitly filter out via real `DSREAS`/`DCSREAS`-style fields. Since
  that can't be done here, this source is excluded from the real dropout
  dataset (`real_dataset_final.csv`). Not read further (diagnostics/
  biochemical/glucose files were structurally verified but not extracted).
  Could still be used later for something else (e.g. glucose-variability
  research unrelated to dropout), but not for this project's label.
- **Access**: authenticate on Zenodo, accept a Data Usage Agreement, submit
  a request with your name, email, and a stated reason for wanting the
  data. Reviewed by the University of Granada's Department of Computer
  Engineering, Automatics, and Robotics. Free, no stated timeline, not
  instant.

## Cardiovascular: a second real dataset, now verified by download

[UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
(299 real patients, Chicco and Jurman 2020, Faisalabad Institute of
Cardiology and Allied Hospital). Downloaded directly from UCI (not the
Kaggle mirror, which also hosts a suspicious 5,000-patient version of this
same dataset that looks like a synthetic expansion, not the real one) and
checked: no duplicates, no missing values, no impossible values, 299 rows
exactly matching the documented 13 fields. Real and clean. Limitation: only
tracks `DEATH_EVENT`, no separate lost-to-follow-up or withdrawal category,
so it's useful for general survival-model checks but not for validating
dropout prediction specifically. Sits alongside MUSIC (992 real patients,
11 of them genuinely lost to follow-up) as the two real cardiovascular
sources currently in hand.
