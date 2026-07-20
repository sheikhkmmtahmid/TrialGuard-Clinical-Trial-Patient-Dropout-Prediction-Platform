# Harmonized real dataset build log

Purpose: build one combined, properly-understood, harmonized real
dataset from every file in `D:\Trial Guard\data\` and its subfolders
(pds, immport, music, heart_failure, aact), read and verified file by
file, not skimmed. This log exists so nothing gets lost or re-guessed
partway through a task this large. Updated as work proceeds, not just
at the end.

## Ground rules for this pass

1. Every file gets opened and classified: **DATA** (contains patient
   records) or **DOCS** (data dictionary, CRF, protocol, codebook,
   glossary, curation notes) or **N/A** (irrelevant to any of our 20
   model features, e.g. a pure lab-chemistry panel with no bearing on
   dropout, demographics, visits, AE, adherence, or QoL).
2. For every field that ends up used in the combined dataset, the real,
   source-verified meaning is written down here before it's used, not
   assumed from a prior pass. Where a prior session already verified a
   field (many have been), that verification is re-confirmed and cited
   here rather than silently trusted.
3. **Harmonization is the actual point of this pass.** A field is only
   pooled across studies if it genuinely measures the same thing the
   same way. Where it doesn't (different QoL instruments, different
   visit cadences, different severity scales), that gets recorded
   explicitly and handled deliberately, not pooled blindly.

## Top-level inventory of D:\Trial Guard\data\

| Folder | Status |
|---|---|
| pds/ | in progress, see per-study table below |
| immport/ | not started |
| music/ | not started |
| heart_failure/ | not started |
| aact/ | not started (trial-level aggregate, not patient-level, likely N/A for the combined dataset itself, confirm) |

## Harmonization decisions (filled in as determined)

| Attribute | Decision | Reasoning |
|---|---|---|
| (to be filled in) | | |

## Per-study file inventory and classification

(to be filled in study by study)

## CORRECTION #3: file format does not equal file role

The user caught this directly: I was classifying files as DATA vs DOCS
by file extension (.sas7bdat/.csv/.xlsx = data, .pdf/.docx/.doc = docs).
That's wrong. 33 `.xlsx` files across the PDS folders are actually
documentation, not patient records, and were run through the data
summarizer (which dumped value-counts as if they were patient rows)
before this was caught:

- 7x "Dataset Contents and Variable Crosswalk.xlsx"
- 11x "Descriptive_Stats_*.xlsx"
- "Data_Dictionary.xlsx", "C9732_Dictionary.xlsx"
- "DDT_408_v2.xlsx", "DDT_203_v2.xlsx" (Data Definition Table)
- 3x "MEPS Variable Crosswalk*.xlsx"
- 3x "PDS_DATA_PROFILE_CREATED_*.xlsx"
- "DataDescription_v1.xlsx"
- 3x "Data descriptors for SCLC 0X.xlsx" (to verify, likely docs)

All being re-read properly now as explanatory text/tables, not treated
as patient rows. Standing rule going forward: **file role is determined
by opening and reading the content, never assumed from the file
extension or folder position.** A csv, xlsx, sas7bdat, or even a pdf
could in principle hold either patient data or an explanation, each
file gets checked on its own.

## Standing rule: unclear field meaning

If any field's meaning is not immediately obvious from its name and
value pattern, the correct move is to look for a data dictionary, DDT,
codebook, crosswalk, or CRF/protocol document **in that same study's
folder** and read it before deciding whether the field is usable, not
guess from the field name or from how a similarly-named field worked in
a different study. Logged per-field below as this proceeds.

## Harmonization decisions (final, after reading every file)

### Severity
Real ECOG Performance Status (the standard 0-4 oncology severity scale)
confirmed present, by column, in: Colorec_Amgen_2005_262 (B_ECOG),
Colorec_Amgen_2006_263 (B_ECOG), Colorec_Amgen_2006_264 (B_ECOG),
HeadNe_Amgen_2007_265 (B_ECOG), LungSm_G1Thera x3 (ECOG/ECOGBL),
LungNo_EliLill_2010_272 (ECOGBL), Pancrea_ClovisO_2010_186 (ECOG). These
9 studies already use real ECOG in the existing extraction, mapped
0->mild, 1->moderate, 2+->severe, matching the model's existing
`_SEVERITY_MAP`. Every other study has no real severity signal and is
marked with a `has_real_severity` flag rather than silently defaulted;
`condition_severity_encoded` stays at the imputed moderate default for
those, honestly flagged, not fabricated as real.

### Quality of life
Not one shared instrument. Three real, but genuinely different, scales
found: LungSm_Amgen_2002_266 uses a 0-100 VAS (visual analogue scale),
Pancrea_ClovisO_2010_186 uses the EQ-5D "your own health state today"
VAS (also 0-100, a standard, comparable instrument), Glioma x2 and the
G1 Therapeutics/Pfizer studies use their own multi-item questionnaires
rescaled by hand earlier this session. Decision: keep `has_real_qol` as
the honesty flag it already is, and additionally record which studies'
QoL values come from a directly-comparable 0-100 VAS
(LungSm_Amgen_2002_266, Pancrea_ClovisO_2010_186) versus an
instrument-rescaled approximation (everyone else with `has_real_qol`
True). Not pooled as if identical, the difference is preserved in the
data itself via this marker.

### Medication adherence
Computed three different but conceptually consistent ways depending on
what each study recorded (percentage of real dosing events without a
recorded delay/reduction/modification). Directionally comparable across
studies (higher always means fewer recorded dosing problems), kept as
is, difference in exact computation documented per study in the
extraction scripts' comments already in place.

### Visit cadence
`visit_frequency_rate` and `days_between_visits_mean/std` are already
expressed as a rate per 30 days, not a raw visit count, so studies with
weekly monitoring, 21-day chemo cycles, or an assumed monthly schedule
are already on the same time-normalized footing. What differs
underneath (what actually counts as "a visit") is a real difference in
what's being measured, not a units mismatch, kept as documented per
study rather than treated as a problem to solve away.

### Ethnicity / race
Previously defaulted to "unknown" for nearly the entire real dataset,
this was a real gap, not because the data doesn't exist, several
studies do have a real RACE/RACECAT/RACEGR1 field, just never
extracted. Being added in the rebuild: RACE-family fields, mapped to the
same buckets the model already uses (white/black/hispanic/asian/other/
unknown), for every study where a real field exists, verified per study
before use rather than assumed to mean the same coding across studies.

### Age
Already real for nearly every study in the existing extraction; no
harmonization issue found. One apparent outlier group checked and
confirmed real, not an error: ImmPort_SDY797 (T1DAL, Alefacept trial
for new-onset Type 1 Diabetes) has ages 12-16, which is normal and
expected for a new-onset-T1D pediatric/adolescent immunotherapy trial,
confirmed against `docs/data_sourcing.md`'s study description before
accepting rather than assuming it was a data error.

## Ethnicity extraction added (this pass)
Real RACE/RACECAT/RACEGR1 fields extracted and mapped into the model's
existing ethnicity codes for 8 studies previously defaulted to
"unknown": `extract_amgen_group.py` (4 studies, RACCAT/RACE),
`extract_pancrea_clovis.py` (RACE), `extract_g1thera.py` (3 studies,
RACEGR1, binary Caucasian/Non-Caucasian only), `extract_breast_elililly.py`
(RACEGR1, binary White/Non-White only). Verified against each study's
actual field values before mapping, not assumed. All 4 scripts re-run,
patient/event counts confirmed unchanged (no regression from the
ethnicity change itself).

## CORRECTION #4: real duplicate-patient bug found during cleaning pass

Found while running the required post-harmonization cleaning pass
(duplicate check, missing-value check, range sanity check, per-study
summary), not something previously flagged:

`extract_pfizer.py` (LungSm_Pfizer_2002_419, both cohorts) matched
patients on `SUBJID` alone. Verified directly against the source
files: `SUBJID` is only unique **within a site** in this multi-site
study (every site restarts its own numbering, e.g. every site has its
own "subject 1"). Real patient count = unique (SITEID, SUBJID) pairs:
204 in cohort2 (not 44), 39 in cohort1 (not 11). Matching on bare
SUBJID silently pooled AE, dosing, and QoL records from *different real
patients at different sites* who happened to share a subject number,
one merged row showed 713 adverse events for a single patient, which
is implausible and was the tell. Fixed by keying every join in
`extract_pfizer.py` on the real composite identifier (SITEID, SUBJID).
Re-ran: cohort1 39->38 patients, cohort2 204->203 patients (one
patient per cohort had zero real dosing records once correctly
isolated, correctly dropped rather than kept with fabricated data).
Zero duplicate (study, usubjid) pairs and zero fully-duplicate rows in
the rebuilt dataset after this fix (previously 210 fully-duplicate
rows existed silently in the pooled data).

Checked all other studies with an equivalent "duplicate usubjid" signal
for the same root cause before accepting the fix as complete:
- Glioma_2008_441 / Glioma_2009_440: apparent duplicate USUBJIDs (e.g.
  "1", "2", "3") turned out to be two *separate* SDTM-compliant studies
  that each independently number their own patients starting at 1; the
  `study` label already disambiguates them, USUBJID is genuinely unique
  within each study. Not a bug, confirmed by checking there was zero
  overlap once split by `study`.
- Breast_Allianc_2008_158, LungSm_Amgen_2002_266, Pancrea_EMDSero_442:
  zero duplicate usubjid within their own files; the only "duplicates"
  seen were the same MASK_ID/SUBJID number coincidentally reused across
  *different, unrelated studies* (e.g. patient "57" exists in both
  Breast_Allianc and Pfizer), which is expected and harmless since
  `study` + `usubjid` together are what identify a real patient in the
  combined dataset, confirmed unique.

## CORRECTION #5: real gender/sex bug found while auditing every column

While building an exact empty/unusable/usable count for every column (at
the user's request), `gender_encoded` showed 4,224 of 6,476 patients
(65%) as "unknown" - implausible, real trials essentially always record
sex. Traced to real bugs, not real missingness, in the extraction code:
- `extract_amgen_group.py`, `extract_lungsm_amgen.py`: source SEX field
  is spelled out ("Male"/"Female"), verified against the real files; the
  code only matched single letters "M"/"F" and silently fell back to
  "unknown" for every single patient in these 5 studies.
- `extract_pfizer.py`: source SEX field is a real numeric code (1./2.),
  verified against the real file, not a letter at all; same silent
  fallback for both cohorts.
- `extract_immport2.py` (SDY524): a real `Sex` field (Male/Female) exists
  in `ADSTAND.txt` and was never read at all, hardcoded to "unknown"
  unconditionally.
Fixed all 4 scripts to recognize the real values actually present
(verified the exact spelling/coding per source file first, not
assumed). Checked SDY1904 and SDY3285 for an equivalent fix and found
none: no gender/sex field exists in any downloaded file for either
study, confirmed a real gap, not a bug, 198 patients honestly remain
"unknown." Re-ran all 5 scripts: patient/event counts unchanged (no
regression). `gender_encoded` unusable rate dropped from 4,224 (65%) to
198 (3%) after the fix.

## CORRECTION #6: real age-range bug in the 3 G1 Therapeutics studies

Same column audit found 118 patients (all of LungSm_G1Thera_2015_433,
2015_434, 2017_435) with AGE stored as a real 5-year band ("50-54",
"<45", ">=80"), verified against the source file, not a raw number.
Previously treated as missing and silently replaced with the whole real
cohort's overall median (60) - throwing away real, usable information.
Fixed `extract_g1thera.py` to convert each real band to its numeric
midpoint (open-ended bands `<45`/`>=80` use a midpoint 2.5 years beyond
the stated bound, consistent with every other band's 5-year width).
Raw missing age count dropped from 847 to 729 after this fix (118
patients recovered).

## Final harmonized + cleaned dataset (this pass)

`real_dataset_final.csv`: 6,476 real patients, 23 real studies, 544
real behavioral dropout events (8.4%). Post-cleaning checks all pass:
zero duplicate (study, usubjid) pairs, zero fully-duplicate rows, zero
missing values in any of the 20 model feature columns after imputation,
all categorical codes (gender, ethnicity, severity, dropout_status)
fall within their defined valid sets, no negative or impossible values
in visit counts / days-to-event / adherence scores, age range (12-89)
fully accounted for. This is the harmonized, cleaned deliverable the
user asked for before naming the next phase.
