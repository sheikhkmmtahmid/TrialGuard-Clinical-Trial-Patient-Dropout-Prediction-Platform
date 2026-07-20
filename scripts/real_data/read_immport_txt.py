"""
Read every .txt file in data/immport, classifying and logging each as
DATA (tab-separated patient records) or DOCS (glossary, curation notes,
dictionary), based on actual content, not filename.
"""
import pandas as pd
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(r'D:\Trial Guard\data\immport')
OUT = Path(r'D:\Trial Guard\docs\immport_txt_read.txt')

DOCS_FILES = {
    'Data_Curation_Notes.txt', 'StudyGlossary.txt', 'ITN_Diabetes_Study_data_dictionary.txt',
}

files = sorted(ROOT.rglob('*.txt'))
with open(OUT, 'w', encoding='utf-8') as out:
    for f in files:
        out.write(f"\n\n===== {f} =====\n")
        if f.name in DOCS_FILES:
            out.write("[classified: DOCS, reading full text]\n")
            try:
                with open(f, encoding='utf-8', errors='ignore') as fh:
                    out.write(fh.read())
            except Exception as e:
                out.write(f"ERROR: {e}\n")
        else:
            out.write("[classified: DATA, reading as tab-separated table]\n")
            try:
                df = pd.read_csv(f, sep='\t', low_memory=False)
                out.write(f"shape: {df.shape}\n")
                for col in df.columns:
                    try:
                        nun = df[col].nunique(dropna=True)
                        if nun == 0:
                            out.write(f"  {col}: all null\n")
                        elif nun <= 25:
                            out.write(f"  {col}: {df[col].value_counts(dropna=False).to_dict()}\n")
                        else:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                out.write(f"  {col}: numeric, min={df[col].min()}, max={df[col].max()}, "
                                          f"mean={df[col].mean():.2f}, nulls={df[col].isna().sum()}\n")
                            else:
                                out.write(f"  {col}: {nun} unique values, e.g. "
                                          f"{df[col].dropna().unique()[:5].tolist()}\n")
                    except Exception as e:
                        out.write(f"  {col}: ERROR {e}\n")
            except Exception as e:
                out.write(f"ERROR reading as data: {e}\n")

print("done, wrote", OUT)
