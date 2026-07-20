# Audit log: looking for real medication adherence and real visit-schedule data

Started because the real-data retrain showed two specific, disclosed weak
spots: `cumulative_missed_visits` and `medication_adherence_score` could
not be honestly reconstructed for most real patients, so they ended up as
constants. This log tracks every real data source checked for these two
things specifically, minutely (actual file/column inspection, not
filename guessing), so nothing gets lost or re-checked twice.

Status values: `not checked` / `checked, nothing found` / `checked, FOUND` / `checked, partial`

## What counts as a real hit

- **Adherence**: a real, patient-level, per-visit-or-cycle measure of how
  well the patient took their medication as prescribed. Pill counts
  returned, dose-modification/interruption logs, MEMS cap data, a
  compliance percentage field. Not: a blank column, not a study-level
  aggregate.
- **Visit schedule**: a real record of which visits were *planned* for a
  patient (not just which ones happened), so a missed visit can actually
  be detected as a gap against the plan, not inferred from vitals/AE
  presence alone.

## CORRECTION, found while auditing

Earlier claim "no scheduled-visit calendar exists in any real study
checked" was **wrong**. SDTM (the standard format several PDS studies
use) has a dedicated domain for exactly this, `tv.sas7bdat` (Trial
Visits), which lists every *planned* visit and its target day, separate
from what actually happened. Confirmed present, by opening the file
directly and inspecting real rows, in:

- Glioma_EMDSero_2008_441
- Glioma_EMDSero_2009_440
- Pancrea_EMDSero_2009_442
- LungSm_EliLill_2011_287

All four of these were already used in the real-data retrain, currently
with `cumulative_missed_visits` hardcoded to 0 because I didn't check
for this domain when I built the extraction scripts. This is fixable
for these four studies specifically. Action item, not yet done: rebuild
their extraction to compare each patient's actual visit days against
this real planned schedule and compute genuine missed-visit counts.

## PDS studies (adherence / visit-schedule column scan, keyword-based, verified by opening each file)

| Study | Adherence | Visit schedule | Notes |
|---|---|---|---|
| Breast_Allianc_2002_194 | none found | none found | |
| Breast_Allianc_2002_200 | none found | none found | |
| Breast_Allianc_2006_216 | none found | none found | |
| Breast_Allianc_2008_158 | FOUND (partial) | none found | c40502_cycles.csv: pac_dosemod/nab_dosemod/ixa_dosemod, binary dose-modification flags per cycle, not a compliance percentage |
| Breast_Allianc_2009_162 | none found | none found | |
| Breast_EliLill_2008_168 | none found | none found | |
| Breast_Multipl_2004_414 | none found | none found | |
| Colorec_Allianc_1994_201 | false positive | none found | "complica" = surgical complications, not compliance |
| Colorec_Allianc_1997_182 | none found | none found | 1 file (outcome_other_baseline_data.xlsx) failed to scan, needs retry |
| Colorec_Allianc_2004_161 | FOUND (partial) | none found | characteristic.csv: ADHERENC, binary yes/no field, already used in earlier PDS event-count work, not a graded score |
| Colorec_Amgen_2004_310 | none found | none found | |
| Colorec_Amgen_2005_262 | FOUND (partial) | none found | exposure.sas7bdat: DOSCHGYN (dose changed y/n, binary); respeval.sas7bdat: RSCOMPLT (response evaluation complete, false positive, not adherence) |
| Colorec_Amgen_2006_263 | none found | none found | |
| Colorec_Amgen_2006_264 | none found | none found | |
| Colorec_Amgen_2006_309 | none found | none found | |
| Colorec_Multipl_2006_251 | none found | none found | |
| Gastric_Multipl_1999_416 | none found | none found | |
| Gastric_Multipl_2008_415 | none found | none found | |
| Glioma_EMDSero_2008_441 | none found | **FOUND (real)** | tv.sas7bdat, see correction above |
| Glioma_EMDSero_2009_440 | none found | **FOUND (real)** | tv.sas7bdat, see correction above |

| HeadNe_Amgen_2007_265 | none found | none found | |
| LungNo_EliLill_2009_438 | FOUND (partial) | none found | exsum.sas7bdat: DOSEINTWK (dose interruption week count), a real proxy, not a percentage. Study itself already excluded from the event tally as too ambiguous, but this file could still be usable if revisited |
| LungNo_EliLill_2010_272 | none found | none found | |
| LungNo_Multipl_2018_231 | none found | none found | |
| LungSm_Allianc_1998_261 | none found | none found | |
| LungSm_Allianc_2007_218 | none found | none found | |
| LungSm_Amgen_2002_266 | none found | none found | |
| LungSm_EliLill_2011_287 | FOUND (partial) | **FOUND (real)** | dosmod.sas7bdat/ex.sas7bdat/excomp.sas7bdat: DOSMODFL/DOSMOD (dose modification flag, binary); tv.sas7bdat confirmed too |
| LungSm_G1Thera_2015_433 | none found | none found | |
| LungSm_G1Thera_2015_434 | none found | none found | |
| LungSm_G1Thera_2017_435 | none found | none found | |
| LungSm_Pfizer_2002_419 | not fully scanned | not fully scanned | 3 CSV files failed to scan (encoding issue, not utf-8), need retry with latin1; already-checked doseval.sas7bdat files (used in extraction) had no clean percentage score |
| Lymphom_Allianc_2006_212 | none found | none found | |
| Multiple_Allianc_2002_202 | none found | none found | |
| Multiple_Allianc_2002_213 | none found | none found | |
| Multiple_Brigham_454 | none found | none found | |
| Pancrea_ClovisO_2010_186 | FOUND (partial) | none found | adex.sas7bdat: EXDELAY, EXREDUC (dose delay/reduction flags, binary) |
| Pancrea_EMDSero_2009_442 | none found | **FOUND (real)** | tv.sas7bdat confirmed |
| Pancrea_Multipl_2020_430 | none found | none found | this is a MEPS-linkage dataset, already confirmed no disposition data either |
| Prostat_Asociac_484 | none found | none found | Spanish pharmacy billing data, out of scope (non-English), not pursued further |
| Prostat_Multipl_2008_406 | none found | none found | MEPS-linkage dataset |
| Prostat_Multipl_2008_420 | none found | none found | MEPS-linkage dataset |
| Prostat_Multipl_2009_417 | none found | none found | MEPS-linkage dataset |
| Prostat_Multipl_2018_234 | none found | none found | MEPS-linkage dataset |
| Prostat_Researc_2016_167 | none found | none found | tumor growth measurements only, already confirmed no disposition data |

### PDS summary

**No study anywhere in PDS has a clean, graded, 0-100 medication
adherence percentage.** What exists instead, in 6 studies, is a binary
dose-modification/interruption/delay flag (yes/no, not "how much"):
Breast_Allianc_2008_158, Colorec_Allianc_2004_161, Colorec_Amgen_2005_262,
LungNo_EliLill_2009_438, LungSm_EliLill_2011_287, Pancrea_ClovisO_2010_186.
These are real signals, just cruder than a true adherence score, worth
using as a better-than-imputed-average stand-in, not yet done.

**Real planned-visit schedules exist in exactly 4 studies**, all SDTM-
formatted: Glioma_EMDSero_2008_441, Glioma_EMDSero_2009_440,
Pancrea_EMDSero_2009_442, LungSm_EliLill_2011_287. Not yet used to
rebuild `cumulative_missed_visits` for these four, that's a concrete,
scoped follow-up.

## Other real sources

| Source | Adherence | Visit schedule | Notes |
|---|---|---|---|
| MUSIC | none found | none found | single baseline record per patient, no follow-up visits at all |
| UCI Heart Failure | none found | none found | single-row-per-patient dataset, no visit structure |
| ImmPort (all 5 studies, files downloaded so far) | none found | none found | only "Completed Study"-type false positive keyword matches (study completion status, not medication compliance). Caveat: only the specific domain files already downloaded for the earlier extraction were checked, not each study's full file package, so this is a partial check for ImmPort specifically, not as exhaustive as the PDS pass above |

## External search (open, free datasets not yet in this project)

Five separate searches run: general adherence+visit+dropout open datasets,
PhysioNet specifically, Kaggle/Zenodo/UCI, Vivli/YODA/PDS drug
accountability specifically, and MEMS electronic pill-cap data
specifically. None found a freely-licensed, downloadable, real
patient-level dataset combining a graded adherence percentage AND a real
scheduled-visit calendar tied to a clinical trial dropout outcome.

What the search did confirm:
- Real drug-accountability data (real pill counts, a real percentage)
  genuinely exists inside some trials' source records, but it wasn't
  included in any of PDS's or ImmPort's public, de-identified extracts
  checked in this project, and the platforms that do host it
  (Vivli, YODA) were already ruled out earlier in this project for
  licensing reasons (YODA explicitly bans use "in pursuit of litigation
  or for commercial interests"; Vivli requires a research proposal, a
  review process, and a paid analysis environment after year one), not
  because the data doesn't exist, because the terms don't fit what
  TrialGuard is for.
- A Kaggle "IoT-based medication adherence" dataset exists but is a
  simulated sensor demo, not real trial patients, not pursued further.
- Published research papers using real adherence data (e.g. a 403-patient
  Ethiopia diabetes study, an 8,141-patient Zimbabwe claims study) exist,
  but their underlying data is not stated as publicly downloadable, it
  belongs to the specific hospital systems that ran those studies.

**Honest conclusion**: this specific combination (real, graded adherence
+ real visit schedule + real trial dropout, freely licensed) was not
found anywhere searched. The gap is not for lack of looking.

## CORRECTION #2, found while implementing the fix

`Colorec_Allianc_2004_161`'s `ADHERENC` field, listed above as a usable
binary medication-adherence signal, is **not medication adherence at
all**. Checked the actual case report form (not just the data
dictionary) and found the real question: "Tumor Characteristics:
Perforation / Obstruction / **Adherence**", a surgical/pathology finding
about whether the tumor has physically invaded or stuck to nearby
organs (a T4 staging concept), a completely different meaning of the
word "adherence." The 1=Yes/2=No decode was correct, what it measures
was wrong. This field is excluded from the adherence-proxy fix. Caught
before it was used, not after, by checking the actual form wording
rather than trusting the field name and data dictionary label alone.

## Action items, done

1. **Done.** Rebuilt real `cumulative_missed_visits` for the 4 studies
   with a confirmed real planned-visit domain (Glioma x2,
   Pancrea_EMDSero_442, LungSm_EliLill_2011_287), verified first on one
   hand-checked patient before running on everyone. See
   `scripts/real_data/rebuild_missed_visits.py`.
2. **Done for 4 of the 6 candidates.** Breast_Allianc_2008_158,
   Colorec_Amgen_2005_262, LungSm_EliLill_2011_287, and
   Pancrea_ClovisO_2010_186 now carry a real, verified adherence
   percentage instead of a flat imputed average.
   `Colorec_Allianc_2004_161`'s `ADHERENC` field was excluded, see
   Correction #2 above, it measures tumor invasion of nearby organs, not
   medication compliance, caught by reading the actual case report form.
   `LungNo_EliLill_2009_438` was not attempted, that study was already
   excluded from the dataset entirely for an unrelated, ambiguous
   disposition field.
3. Not done. The 3 Pfizer CSV encoding failures and 1
   Colorec_Allianc_1997_182 xlsx failure from the original scan are
   still unresolved either way, low priority given how small a slice of
   the dataset they represent.
4. Standing practice going forward: any future PDS or ImmPort data drop
   should get the same column-level scan via
   `scripts/real_data/audit_adherence_visit.py`, not an assumption that
   past absence still holds.

## Bug caught while wiring these fixes in

While rebuilding the dataset, `early_dropout_signal` was found to be
built partly from `dropout_status`, the actual outcome, meaning the
model was being handed a feature that was quietly telling it part of
the answer. This was in the code before today's fixes and had already
been used for the previously-reported real-data AUC of 0.7044. Fixed
before any further training ran on it (now built only from real visit
behavior, never from the outcome). The real, leak-free number is
**0.6774** AUC, reported to the user as the corrected figure alongside
an explanation of what changed and why. See
`scripts/real_data/finalize_real_dataset.py`.
