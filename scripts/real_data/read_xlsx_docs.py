"""
Properly read xlsx files that are actually documentation (data
dictionaries, crosswalks, DDTs, descriptive-stats reports, data
profiles), not patient data, as full sheet content, not value-counts.
Corrects the earlier mistake of running these through the data
summarizer.
"""
import pandas as pd
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

OUT = Path(r'D:\Trial Guard\docs\xlsx_docs_read.txt')

FILES = [
    r"Breast_Multipl_2004_414\Dataset Contents and Variable Crosswalk.xlsx",
    r"Breast_Multipl_2004_414\Descriptive_Stats_Linked_Breast_SanofiU_2004_135.xlsx",
    r"Colorec_Allianc_1997_182\Data_Dictionary.xlsx",
    r"Colorec_Amgen_2004_310\DDT_408_v2.xlsx",
    r"Colorec_Amgen_2006_309\DDT_203_v2.xlsx",
    r"Colorec_Multipl_2006_251\MEPS Variable Crosswalk.xlsx",
    r"Gastric_Multipl_1999_416\Dataset Contents and Variable Crosswalk.xlsx",
    r"Gastric_Multipl_1999_416\Descriptive_Stats_Linked_Gastric_SanofiU_1999_143.xlsx",
    r"Gastric_Multipl_2008_415\Dataset Contents and Variable Crosswalk.xlsx",
    r"Gastric_Multipl_2008_415\Descriptive_Stats_Linked_Gastric_MerckKG_2008_130.xlsx",
    r"Gastric_Multipl_2008_415\PDS_DATA_PROFILE_CREATED_130.xlsx",
    r"Glioma_EMDSero_2008_441\Descriptive_Stats_EMD121974_011.xlsx",
    r"Glioma_EMDSero_2009_440\Descriptive_Stats_EMD_121974_012.xlsx",
    r"LungNo_EliLill_2009_438\Descriptive_Stats_H3E_US_S130.xlsx",
    r"LungNo_Multipl_2018_231\MEPS Variable Crosswalk.xlsx",
    r"LungSm_Allianc_1998_261\C9732_Dictionary.xlsx",
    r"LungSm_G1Thera_2015_433\Data descriptors for SCLC 01.xlsx",
    r"LungSm_G1Thera_2015_433\Descriptive_Stats_G1T_NCT02499770.xlsx",
    r"LungSm_G1Thera_2015_434\Data descriptors for SCLC 02.xlsx",
    r"LungSm_G1Thera_2015_434\Descriptive_Stats_G1T_NCT02514447.xlsx",
    r"LungSm_G1Thera_2017_435\Data descriptors for SCLC 03.xlsx",
    r"LungSm_G1Thera_2017_435\Descriptive_Stats_G1T_NCT03041311.xlsx",
    r"LungSm_Pfizer_2002_419\PDS_DATA_PROFILE_CREATED_419.xlsx",
    r"LungSm_Pfizer_2002_419\XRP4174D-3001_cohort1_CSVfiles\Descriptive_Stats_NCT00143455.xlsx",
    r"Multiple_Brigham_454\PDS_DATA_PROFILE_CREATED_VITAL_trial_NEJM_2022.xlsx",
    r"Pancrea_EMDSero_2009_442\Descriptive_Stats_EMR 200066-003.xlsx",
    r"Pancrea_Multipl_2020_430\Dataset Contents and Variable Crosswalk.xlsx",
    r"Pancrea_Multipl_2020_430\Descriptive_Stats_Linked_Pancrea_ClovisO_2010_186.xlsx",
    r"Prostat_Asociac_484\DataDescription_v1.xlsx",
    r"Prostat_Multipl_2008_406\Dataset Contents and Variable Crosswalk.xlsx",
    r"Prostat_Multipl_2008_406\Descriptive_Stats_Linked_Prostat_AstraZe_2008_103.xlsx",
    r"Prostat_Multipl_2008_420\Dataset Contents and Variable Crosswalk.xlsx",
    r"Prostat_Multipl_2008_420\Descriptive_Stats_Linked_Prostat_AstraZe_2008_104.xlsx",
    r"Prostat_Multipl_2009_417\Dataset Contents and Variable Crosswalk.xlsx",
    r"Prostat_Multipl_2009_417\Descriptive_Stats_Linked_Prostat_AstraZe_2009_144.xlsx",
    r"Prostat_Multipl_2018_234\MEPS Variable Crosswalk_20171109.xlsx",
]

ROOT = Path(r'D:\Trial Guard\data\pds')

with open(OUT, 'w', encoding='utf-8') as out:
    for rel in FILES:
        f = ROOT / rel
        out.write(f"\n\n===== {f} =====\n")
        if not f.exists():
            out.write("FILE NOT FOUND\n")
            continue
        try:
            xl = pd.ExcelFile(f)
            for sh in xl.sheet_names:
                df = xl.parse(sh, header=None)
                out.write(f"--- sheet: {sh} (shape {df.shape}) ---\n")
                # Print full content for small sheets, first 100 rows for large ones
                to_print = df.head(150)
                out.write(to_print.to_string(max_colwidth=80))
                out.write("\n")
        except Exception as e:
            out.write(f"ERROR: {e}\n")

print("done, wrote", OUT)
