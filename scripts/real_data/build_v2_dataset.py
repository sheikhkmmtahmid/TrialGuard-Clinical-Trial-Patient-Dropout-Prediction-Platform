"""
Builds real_dataset_v2_final.csv: the same 28 real studies as
real_dataset_final.csv, but every predictive feature is computed using
ONLY events at day <= CUTOFF_DAYS (a genuinely prospective, "what would
we have known 60 days in" view) - dropout_status and days_to_event are
never windowed, they still reflect the true, eventual, real outcome.

See docs/real_dataset_v2_early_window_log.md for why this exists (testing
whether the cross-hospital pattern found in visit_number/adverse_events_count
survives when features can't "see" a patient's full trajectory, only their
early one) and the exact windowing logic per source.
"""
import sys
import pandas as pd

sys.path.insert(0, '.')

CUTOFF_DAYS = 60

results = []

# ---- Amgen group (4 studies) ----
from extract_amgen_group import extract as extract_amgen
for base_dir, label, disp_file, reason_col, day_col in [
    ('../../data/pds/Colorec_Amgen_2005_262/SAS dataset - 20040249', 'Colorec_Amgen_2005_262',
     'disposit.sas7bdat', 'DSEOS', 'LASTOSDY'),
    ('../../data/pds/Colorec_Amgen_2006_263', 'Colorec_Amgen_2006_263',
     'disposit.sas7bdat', 'DSREAS', 'DSDY'),
    ('../../data/pds/Colorec_Amgen_2006_264/SAS dataset - 20050203', 'Colorec_Amgen_2006_264',
     'disposit.sas7bdat', 'DSREAS', 'DSDY'),
    ('../../data/pds/HeadNe_Amgen_2007_265', 'HeadNe_Amgen_2007_265',
     'disposit.sas7bdat', 'DSEOS', 'DSDY'),
]:
    df = extract_amgen(base_dir, label, disp_file, reason_col, day_col, cutoff_days=CUTOFF_DAYS)
    print(f"{label}: {len(df)} patients, {df['dropout_status'].sum()} behavioral")
    results.append(df)

# ---- Pancrea_ClovisO ----
from extract_pancrea_clovis import extract as extract_pancrea_clovis
df = extract_pancrea_clovis(cutoff_days=CUTOFF_DAYS)
print(f"Pancrea_ClovisO_2010_186: {len(df)} patients, {df['dropout_status'].sum()} behavioral")
results.append(df)

# ---- G1 Therapeutics (3 studies) ----
from extract_g1thera import extract as extract_g1thera
import pandas as pd

def get_behavioral_and_days(base_dir):
    BEHAVIORAL_TERMS = {'withdrawal by subject', 'lost to follow-up'}
    adsl = pd.read_sas(f"{base_dir}/adsl.sas7bdat", format='sas7bdat', encoding='latin1')
    adsl['_norm'] = adsl['DCSREAS'].astype(str).str.strip().str.lower()
    beh = set(adsl[adsl['_norm'].isin(BEHAVIORAL_TERMS)]['USUBJID'])
    days = adsl.set_index('USUBJID')['EOSDY'].to_dict()
    return beh, days

for d, label in [
    ('../../data/pds/LungSm_G1Thera_2015_433', 'G1Thera_2015_433'),
    ('../../data/pds/LungSm_G1Thera_2015_434', 'G1Thera_2015_434'),
    ('../../data/pds/LungSm_G1Thera_2017_435', 'G1Thera_2017_435'),
]:
    beh, days = get_behavioral_and_days(d)
    df = extract_g1thera(d, label, beh, days, cutoff_days=CUTOFF_DAYS)
    print(f"{label}: {len(df)} patients, {df['dropout_status'].sum()} behavioral")
    results.append(df)

# ---- Breast_EliLill ----
from extract_breast_elililly import extract as extract_breast_elililly
df = extract_breast_elililly(cutoff_days=CUTOFF_DAYS)
print(f"Breast_EliLill_2008_168: {len(df)} patients, {df['dropout_status'].sum()} behavioral")
results.append(df)

# ---- LungSm_Amgen ----
from extract_lungsm_amgen import extract as extract_lungsm_amgen
df = extract_lungsm_amgen(cutoff_days=CUTOFF_DAYS)
print(f"LungSm_Amgen_2002_266: {len(df)} patients, {df['dropout_status'].sum()} behavioral")
results.append(df)

# ---- ImmPort (5 studies across 2 scripts) ----
from extract_immport import extract as extract_immport1
df = extract_immport1(cutoff_days=CUTOFF_DAYS)
results.append(df)

from extract_immport2 import extract as extract_immport2
df = extract_immport2(cutoff_days=CUTOFF_DAYS)
results.append(df)

# ---- Glioma (2 studies) + Pancrea_EMDSero (reused, same SDTM shape) ----
from extract_glioma import extract as extract_glioma
df1 = extract_glioma('../../data/pds/Glioma_EMDSero_2008_441/DATA_SAS', 'Glioma_2008_441', cutoff_days=CUTOFF_DAYS)
df2 = extract_glioma('../../data/pds/Glioma_EMDSero_2009_440/DATA', 'Glioma_2009_440', cutoff_days=CUTOFF_DAYS)
df3 = extract_glioma('../../data/pds/Pancrea_EMDSero_2009_442/DATA', 'Pancrea_EMDSero_442', cutoff_days=CUTOFF_DAYS)
print(f"Glioma_2008_441: {len(df1)} patients, {df1['dropout_status'].sum()} behavioral")
print(f"Glioma_2009_440: {len(df2)} patients, {df2['dropout_status'].sum()} behavioral")
print(f"Pancrea_EMDSero_442: {len(df3)} patients, {df3['dropout_status'].sum()} behavioral")
results.extend([df1, df2, df3])

# ---- LungSm_EliLill ----
from extract_lungsm_elililly import extract as extract_lungsm_elililly
df = extract_lungsm_elililly(cutoff_days=CUTOFF_DAYS)
print(f"LungSm_EliLill_2011_287: {len(df)} patients, {df['dropout_status'].sum()} behavioral")
results.append(df)

# ---- Breast_Allianc ----
from extract_breast_allianc import extract as extract_breast_allianc
df = extract_breast_allianc(cutoff_days=CUTOFF_DAYS)
print(f"Breast_Allianc_2008_158: {len(df)} patients, {df['dropout_status'].sum()} behavioral")
results.append(df)

# ---- Pfizer (2 cohorts) ----
from extract_pfizer import extract as extract_pfizer
df1 = extract_pfizer('cohort1', cutoff_days=CUTOFF_DAYS)
df2 = extract_pfizer('cohort2', cutoff_days=CUTOFF_DAYS)
print(f"LungSm_Pfizer_2002_419_cohort1: {len(df1)} patients, {df1['dropout_status'].sum()} behavioral")
print(f"LungSm_Pfizer_2002_419_cohort2: {len(df2)} patients, {df2['dropout_status'].sum()} behavioral")
results.extend([df1, df2])

# ---- LungNo_EliLill ----
from extract_lungno_elililly import extract as extract_lungno_elililly
df = extract_lungno_elililly(cutoff_days=CUTOFF_DAYS)
print(f"LungNo_EliLill_2010_272: {len(df)} patients, {df['dropout_status'].sum()} behavioral")
results.append(df)

# ---- PPMI (4 cohorts) ----
from extract_ppmi import extract as extract_ppmi
df = extract_ppmi(cutoff_days=CUTOFF_DAYS)
for study, sub in df.groupby('study'):
    print(f"{study}: {len(sub)} patients, {sub['dropout_status'].sum()} behavioral")
results.append(df)

# ---- MUSIC (no windowing possible, single baseline row per patient) ----
from extract_music import extract as extract_music
df = extract_music(cutoff_days=CUTOFF_DAYS)
print(f"MUSIC: {len(df)} patients, {df['dropout_status'].sum()} behavioral")
results.append(df)

combined = pd.concat(results, ignore_index=True)
print(f"\nTOTAL (early-window, cutoff={CUTOFF_DAYS}d): {len(combined)} patients, "
      f"{combined['dropout_status'].sum()} behavioral, {combined['study'].nunique()} studies")
print("dup (study,usubjid):", combined.duplicated(subset=['study', 'usubjid']).sum())
combined.to_csv('real_combined_v2.csv', index=False)
print("saved real_combined_v2.csv")
