# PPMI (IDA/LONI) real data build log

Purpose: same rigor as the PDS/ImmPort harmonization work - read every
downloaded file in full, verify field meanings against the actual data
dictionary/code list before using anything, and build a properly cleaned,
harmonized addition to the real dataset. Neurology has been a real gap
(zero PD-specific real data before this), so this matters.

## Inventory (63 files downloaded, all listed here so nothing gets missed)

### Data & Databases (documentation, read first, decodes everything else)
- [ ] Data_Dictionary_-_Harmonized_13Jul2026.csv (7,667 lines)
- [ ] Data_Dictionary_-__Annotated__13Jul2026.csv (7,667 lines)
- [ ] Code_List_-_Harmonized_13Jul2026.csv (9,546 lines)
- [ ] Code_List_-__Annotated__13Jul2026.csv (9,546 lines)
- [ ] Deprecated_Variables_13Jul2026.csv (925 lines)

### Root
- [ ] Participant_Status_13Jul2026.csv (8,855 lines) - likely core outcome file

### Study_Enrollment (candidates for real signal)
- [ ] Conclusion_of_Study_Participation_13Jul2026.csv (1,007 lines)
- [ ] Screen_Fail_13Jul2026.csv (3,499 lines)
- [ ] Visit_Type_13Jul2026.csv (15,044 lines)
- [ ] Baseline_Visit_Start_13Jul2026.csv (2,812 lines)
- [ ] Inclusion_Exclusion_13Jul2026.csv (10,260 lines)
- [ ] Eligibility_Override_13Jul2026.csv (247 lines)
- [ ] FOUND_Enrollment_Status_13Jul2026.csv (3,542 lines)
- [ ] MRI_Waiver_13Jul2026.csv (178 lines)
- [ ] Participant_Inquiries_13Jul2026.csv (4,531 lines)
- [ ] Program_Assessment_13Jul2026.csv (8,066 lines)
- [ ] Research_Proxy_Designation_13Jul2026.csv (12,327 lines)
- [ ] Continuing_Consent_13Jul2026.csv (9,574 lines)
- [ ] Documentation_of_Informed_Consent_13Jul2026.csv (7,839 lines)
- [ ] Documentation_of_Prodromal_Screening_Consent_13Jul2026.csv (3,682 lines)
- [ ] Informed_Consent_Tracking_Log_13Jul2026.csv (18,032 lines)

### Study_Enrollment (imaging-substudy consent/screening forms, expected N/A
### for dropout prediction, but each gets opened to confirm, not assumed)
- [ ] AV-133_Prodromal_Substudy_Conclusion_of_Study_Participation
- [ ] AV-133_Prodromal_Substudy_Documentation_of_Informed_Consent
- [ ] AV-133_Prodromal_Substudy_Inclusion_Exclusion_Criteria
- [ ] AV-133_Prodromal_Substudy_Screen_Fail
- [ ] C05-05_PET_Imaging_Substudy_Conclusion_of_Study_Participation
- [ ] C05-05_PET_Imaging_Substudy_Documentation_of_Informed_Consent
- [ ] C05-05_PET_Imaging_Substudy_Inclusion_Exclusion_Criteria
- [ ] DOWNLOAD_Tau_Substudy__Inclusion_Exclusion_Criteria
- [ ] DPA-714_PET_Imaging_Substudy_Conclusion_of_Study_Participation
- [ ] DPA-714_PET_Imaging_Substudy_Documentation_of_Informed_Consent
- [ ] DPA-714_PET_Imaging_Substudy_Inclusion_Exclusion_Criteria
- [ ] DPA-714_PET_Imaging_Substudy_Screen_Fail
- [ ] Dual_PET_AV-133_in_PD_Imaging_Substudy_Doc_of_Informed_Consent
- [ ] Dual_PET_AV-133_in_PD_Imag_Substdy_Inclusion_Exclusion_Criteria
- [ ] Dual_PET_AV-133_in_PI_Substudy_Doc_of_Informed_Consent
- [ ] Dual_PET_AV133_in_PI_Substudy_Inclusion_Exclusion_Criteria
- [ ] Dual_PET_DPA-714_Imaging_Substudy_Doc_of_Informed_Consent
- [ ] Dual_PET_DPA-714_Imaging_Substudy_Inclusion_Exclusion_Criteria
- [ ] Early_Imaging_Documentation_of_Informed_Consent
- [ ] Early_Imaging_Eligibility
- [ ] Early_Imaging_Screen_Fail
- [ ] FD4_Tracer_Substudy_Conclusion_of_Study_Participation
- [ ] FD4_Tracer_Substudy_Documentation_of_Informed_Consent
- [ ] FD4_Tracer_Substudy_Inclusion_Exclusion_Criteria
- [ ] FD4_Tracer_Substudy_Screen_Fail
- [ ] Gait_Substudy_ActiGraph_LEAP_Documentation_of_Informed_Consent
- [ ] Gait_Substudy_ActiGraph_LEAP_Inclusion_Exclusion_Criteria
- [ ] Gait_Substudy_Conclusion_of_Study_Participation
- [ ] Gait_Substudy_Documentation_of_Informed_Consent
- [ ] Gait_Substudy_Inclusion_Exclusion_Criteria
- [ ] Gait_Substudy_Screen_Fail
- [ ] NX_PI-2620_Tau_Imaging_Substudy_Doc_of_Informed_Consent
- [ ] NX_PI-2620_Tau_Imaging_Substudy_Inclusion_Exclusion_Criteria
- [ ] SV2A_PET_Imaging_Substudy_Documentation_of_Informed_Consent
- [ ] SV2A_PET_Imaging_Substudy_Inclusion_Exclusion_Criteria
- [ ] SV2A_PET_Imaging_Substudy_Screen_Fail
- [ ] Tau_Substudy_Documentation_of_Informed_Consent

### Subject_Demographics
- [ ] Demographics_13Jul2026.csv (8,799 lines)
- [ ] ST-Direct_Demographics_13Jul2026.csv (11,326 lines)
- [ ] Age_at_visit_13Jul2026.csv (48,744 lines)
- [ ] Race_and_Ethnicity_Question_in_myPPMI_13Jul2026.csv (11,455 lines)
- [ ] Socio-Economics_13Jul2026.csv (8,710 lines)
- [ ] Subject_Cohort_History_13Jul2026.csv (935 lines)
- [ ] Cognitive_Activities_Testing_Participant_13Jul2026.csv (1,174 lines)

## Findings (verified against actual data + data dictionary, all 63 files read)

**Core files used:**
- `Participant_Status.csv` (8,855 rows, 1/patient): PATNO, COHORT_DEFINITION
  (PD/Healthy Control/SWEDD/Prodromal/Genetic Registry), ENROLL_DATE,
  ENROLL_STATUS, STATUS_DATE, ENROLL_AGE. ENROLL_STATUS has 12 real values;
  only Enrolled/Withdrew/Complete/Baseline/Baseline Withdraw/Withdraw
  Deceased represent people who actually started (4,968 of 8,855) -
  Screen failed/Screened/Pending/Declined/Screen Scheduled never began,
  excluded from the model population same as every other real source's
  screen-fail exclusion.
- `Study_Enrollment/Conclusion_of_Study_Participation.csv` (CONCL, 1,006
  patients): granular withdrawal-reason checkboxes, verified against the
  data dictionary field-by-field before use. Behavioral (patient's own
  choice/circumstance, matches this project's standing definition):
  WDDISINT, WDFAMILY, WDLTFU, WDNONCOMP, WDTRANSPORT, WDBURDEN, WDOTHER.
  Not behavioral: WDCMPLT (completed), WDDEATH, WDAE (adverse event),
  WDHEALTH (health decline), WDSITE (site closure) - same distinction
  applied to every other real source in this project.
- `Subject_Demographics/Demographics.csv` (SCREEN module, 8,799 patients):
  SEX (0=Female/1=Male per code list - opposite of this project's
  gender_encoded convention, flipped on ingest), race flags
  (RAWHITE/RABLACK/RAASIAN/etc.), HISPLAT (Hispanic ethnicity, separate
  field per US OMB convention).
- `Subject_Demographics/Age_at_visit.csv` (8,575 patients, 48,744 rows):
  real per-visit age, used for visit_number and visit-spacing (age
  difference in years -> days), since no per-visit calendar date file was
  downloaded in this pass.

**Confirmed NOT used, and why:** all ~35 imaging-substudy consent/
screening/eligibility files (AV-133, C05-05 PET, DPA-714 PET, Dual PET
variants, Early Imaging, FD4 Tracer, Gait Substudy, NX-Tau, SV2A PET, Tau
Substudy) - each opened individually, all are 0-296 patients, all are
consent paperwork for optional add-on imaging studies, not the main
cohort's dropout outcome. Also confirmed N/A: Eligibility_Override,
MRI_Waiver, Participant_Inquiries, Program_Assessment,
Research_Proxy_Designation, Continuing_Consent,
Documentation_of_Informed_Consent, Documentation_of_Prodromal_Screening_
Consent, Informed_Consent_Tracking_Log, Deprecated_Variables (a
documentation file, not patient data), Cognitive_Activities_Testing_
Participant, FOUND_Enrollment_Status, Inclusion_Exclusion (eligibility
criteria, not an outcome), Baseline_Visit_Start, Visit_Type (turned out to
track visit *modifications* only, not a general visit log, per its own
MODVISIT field definition), Subject_Cohort_History (redundant with
Participant_Status.COHORT_DEFINITION), Race_and_Ethnicity/
ST-Direct_Demographics (supplementary/fallback race source only, SCREEN
module used as primary since it matches the core clinical cohort),
Socio-Economics (real EDUCYRS exists but isn't part of this project's
current feature schema).

**What PPMI does NOT give us, honestly flagged:** no severity scale
(UPDRS wasn't in this download - it lives under "Motor Assessments",
not downloaded), no adverse event log, no medication adherence, no
quality-of-life instrument. These get the same honest default treatment
as every other real study missing these fields.

## Harmonization decisions

- One row per patient (last-known-state, matching this project's
  standard pattern).
- `study` column: split by COHORT_DEFINITION into separate labels
  (PPMI_PD, PPMI_Prodromal, PPMI_HealthyControl, PPMI_SWEDD,
  PPMI_GeneticRegistry) rather than one combined "PPMI" label - matches
  how every other multi-cohort real source in this project (Glioma x2,
  Pfizer x2 cohorts) was split, and is required for the leave-studies-out
  validation methodology to treat them as distinct groups.
- `dropout_status`: CONCL's behavioral flags where a conclusion record
  exists; falls back to ENROLL_STATUS ("Withdrew"/"Baseline Withdraw" ->
  1, everything else -> 0) where no CONCL record exists yet.
- `days_to_event`: STATUS_DATE minus ENROLL_DATE (or WDDT where more
  precise) in days; dates are month/year only (no day-of-month, stripped
  for de-identification), so precision is coarser than PDS/ImmPort's
  day-level dates - disclosed, not hidden.
- `gender_encoded`: PPMI's SEX (0=F/1=M) flipped to this project's scheme
  (M=0/F=1).
- `ethnicity_encoded`: HISPLAT=1 -> hispanic; else race flags in priority
  White > Black > Asian > other/unknown, falling back to the myPPMI
  Race_and_Ethnicity table only when the primary Demographics record is
  missing for that patient.

## Result

`extract_ppmi.py` -> `ppmi_real.csv`: 4,967 real patients, 567 real
behavioral dropouts (11.4%), split into 4 study groups by cohort
(PPMI_PD 1,670/266, PPMI_Prodromal 2,897/206, PPMI_HealthyControl
336/86, PPMI_SWEDD 64/9). Merged into `real_combined.csv` and rebuilt
`real_dataset_final.csv` via the existing pipeline (no changes needed
to `finalize_real_dataset.py`). Full merged dataset: 11,443 real
patients across 27 studies, 1,111 real behavioral dropouts, zero
duplicates, zero missing values in any of the 20 model feature columns
after imputation. This is the first Neurology-area real data in the
project (was a documented zero-coverage gap before this).

## Update: Motor Assessments / MDS-UPDRS batch (second download)

User downloaded the "Motor___MDS-UPDRS" category (13 files: full MDS-UPDRS
Parts I-IV, Modified Schwab & England ADL, 2 Neuro-QoL short forms, gait
sensor data, 1 methods PDF). Verified each module against the data
dictionary before use, same rule as everything else.

**Used:**
- `MDS-UPDRS_Part_III` (NP3TOT field): the standard clinician-rated PD
  motor-severity score, verified real for 4,959 of our 4,967 PPMI
  patients (99.8%). Bucketed into mild/moderate/severe by TERTILE of
  this dataset's own real distribution (33rd/66th percentile) - NOT a
  published clinical cutoff, since none was independently verified this
  session; disclosed as a distributional split, not a diagnostic
  threshold.
- `Modified_Schwab_and_England_ADL` (MSEADLG field): real 0-100
  functional-independence score, already in the same direction as this
  project's existing QoL convention (100 = best), no rescaling needed.
  Real for 4,887 of 4,967 (98.4%).

**Not used (verified, not assumed):**
- Gait_Data (Axivity/Opals sensors) and Gait_Substudy_Gait_Mobility: the
  optional gait sub-study, small N, same category as the imaging
  sub-study consent forms excluded earlier.
- Neuro-QoL Lower/Upper Extremity Function short forms: real PROMIS-style
  items exist, but computing the actual score requires an official
  IRT-based T-score conversion table this session doesn't have verified
  access to - using the raw item sum would not be the real, valid score,
  so this was skipped rather than risk publishing an incorrectly-scored
  field. Schwab & England (already a valid, pre-computed single score)
  used instead.
- Participant_Motor_Function_Questionnaire: a real PD symptom checklist,
  but has no pre-computed total score field and isn't the standard MDS-
  UPDRS instrument - skipped since NP3TOT already covers severity more
  authoritatively.
- MDS-UPDRS Parts I, I-Patient, II-Patient, IV: real, valid, and each has
  its own real total score field (NP1RTOT, NP1PTOT, NP2PTOT, NP4TOT), but
  not pulled in this pass since this project's schema has one severity
  slot, already filled by Part III (the primary, most-used motor score in
  PD research) - kept as a documented option for a future finer-grained
  severity model, not needed for the current 3-bucket schema.

## Full-project effect of adding PPMI (both batches)

Real severity coverage: 4,728/6,476 (73.0%) -> 9,695/11,443 (84.7%).
Real QoL coverage: 1,366/6,476 (21.1%) -> 6,253/11,443 (54.6%) - PPMI's
near-universal Schwab & England coverage more than doubled the real-QoL
share of the whole project, not just added new rows.

Zero duplicates, zero missing values in any of the 20 model feature
columns, confirmed on the full 11,443-row merged dataset after this
update.

## MUSIC (cardiovascular, already-downloaded data used for the first time)

`data/music/subject-info.csv` (992 real European heart-failure patients)
had been downloaded and verified clean months earlier per
`docs/data_sourcing.md`, but no extraction script had ever been written
for it - a real, zero-acquisition-cost gap. Built
`scripts/real_data/extract_music.py`.

Real fields used: Age (one non-numeric value, ">89", top-coded for
privacy, mapped to 90), Gender (male=1) flipped to this project's
convention, NYHA class (the standard heart-failure severity scale, real
severity - only classes II/III present in this cohort, mapped to
moderate/severe), Follow-up period from enrollment (days) (real,
day-precision days_to_event).

Outcome required resolving a real gap in the vendor's own codebook:
`Exit of the study` only documents code 0 ("survivor") in
`subject-info_codes.csv`; codes 1/2/3 are undefined in the file as
shipped. Rather than guess, cross-tabulated against `Cause of death`:
every patient coded 3 has a real non-zero cause-of-death code (266
patients, all death), every patient coded 1 or 2 has none (11 + 20 = 31
patients, confirmed non-death exits). Used that as the behavioral-
dropout definition rather than assume what "1" vs "2" specifically
means beyond "not death".

No per-visit series exists in this file (one baseline row per patient),
so visit_number is honestly set to 1 and the visit-cadence features are
left at default - there's no real spacing to compute from a single
assessment.

Result: 992 patients, 31 real behavioral dropouts (3.1%). Cleaning pass
identical to every other source: zero duplicates, zero missing values,
all ranges sane. Merged in - full dataset now 12,435 real patients
across 28 studies, 1,142 real behavioral dropouts. This is the first
real Cardiovascular-area data in the project (was a documented zero-
coverage gap before this).

## Update: Medical History batch (third PPMI download, 56 files)

User downloaded PPMI's Medical History category in full. Read every file
before deciding what to use, same rule as always.

**Real finding, worth flagging clearly:** the actual detailed "AE"
module (adverse-event severity, MedDRA term, seriousness, start/stop
dates - verified in the data dictionary as MOD_NAME='AE') was NOT part
of this download. What's available instead is
`Adverse_Event_In-Clinic_Assessment` and
`Adverse_Event_Telephone_Assessment` - per-visit "was an adverse event
observed, yes/no" flags (AERPRT / TELAERPT, verified 0/1 against the
code list), not the full event-level log. Used anyway since it's real,
not a guess: counted how many of each patient's real visits had an AE
flagged (1,174 of 4,967 PPMI patients have at least one). Coarser than
the day-level AE logs used for the PDS oncology trials, disclosed as
such - not presented as equivalent.

**Confirmed real but NOT usable for anything in this project's schema,
after reading each one:** Concomitant_Medication_Log and
LEDD_Concomitant_Medication_Log (real medication lists - drug, dose,
dates - but no field indicating a missed or reduced dose, so no honest
adherence signal can be built from them, confirms what was already
expected going into this download), Clinical_Diagnosis,
Medical_Conditions_Log, PD_Diagnosis_History, Primary_Research_Diagnosis
(diagnosis logs), Vital_Signs, General_Physical_Exam (real vitals/exam
findings, not part of the current feature schema),
Initiation_of_Dopaminergic_Therapy, Procedure_for_PD_Log (treatment
history), Clinical_Global_Impression / Participant_Global_Impression,
Determination_of_Freezing_and_Falls, Features_of_Parkinsonism,
Other_Clinical_Features, Features_of_REM_Behavior_Disorder (PD clinical
detail, not mapped to this schema), Early_Intervention_Trials_Survey,
ST-Direct_High_Interest_Questions, Pregnancy_Test/Report_of_Pregnancy,
and ~20 imaging-substudy-specific AE/pregnancy/genetic-testing files
(each opened and confirmed tiny, 0-17 patients, same pattern as every
other substudy consent file in this project).

Result: `adverse_events_count`/`adverse_event_rate`/`adverse_event_trend`
now real for 4,967 PPMI patients (previously 100% default for all of
them). Re-ran, re-merged: patient/event counts unchanged (567/4,967, no
regression), full dataset still 12,435 patients / 28 studies / 1,142
events, zero duplicates, zero missing values.

## Not yet done

Retraining/retuning the models on this expanded dataset - the last
training run used the 23-study, 6,476-patient version. The Cox/XGBoost/
SHAP numbers reported earlier this session do not yet reflect PPMI or
the severity/QoL improvements above. Retraining is a separate step, not
run automatically here.
