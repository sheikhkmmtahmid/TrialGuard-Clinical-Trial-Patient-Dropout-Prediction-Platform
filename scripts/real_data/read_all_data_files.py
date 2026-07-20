"""
Open and read the actual content of EVERY data file (sas7bdat, csv, xlsx)
across all of data/pds, data/immport, data/music, data/heart_failure,
data/aact. For each file: full column list, row count, and for every
column either value_counts (if few unique values, i.e. categorical/coded)
or a numeric summary (min/max/mean) if continuous. This is a full content
read, not a filename guess. Output goes to a text file for review.
"""
import pandas as pd
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(r'D:\Trial Guard\data')
OUT = Path(r'D:\Trial Guard\docs\all_data_files_read.txt')


def summarize_df(df, f, out):
    out.write(f"\n----- {f} -----\n")
    out.write(f"shape: {df.shape}\n")
    for col in df.columns:
        try:
            nun = df[col].nunique(dropna=True)
            if nun == 0:
                out.write(f"  {col}: all null\n")
            elif nun <= 25:
                vc = df[col].value_counts(dropna=False).to_dict()
                out.write(f"  {col}: {vc}\n")
            else:
                if pd.api.types.is_numeric_dtype(df[col]):
                    out.write(f"  {col}: numeric, min={df[col].min()}, max={df[col].max()}, "
                               f"mean={df[col].mean():.2f}, nulls={df[col].isna().sum()}\n")
                else:
                    out.write(f"  {col}: {nun} unique text values, e.g. {df[col].dropna().unique()[:5].tolist()}\n")
        except Exception as e:
            out.write(f"  {col}: ERROR reading column: {e}\n")


def process_file(f, out):
    suffix = f.suffix.lower()
    try:
        if suffix == '.sas7bdat':
            df = pd.read_sas(f, format='sas7bdat', encoding='latin1')
            summarize_df(df, f, out)
        elif suffix == '.csv':
            try:
                df = pd.read_csv(f, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(f, low_memory=False, encoding='latin1')
            summarize_df(df, f, out)
        elif suffix == '.xlsx':
            xl = pd.ExcelFile(f)
            for sh in xl.sheet_names:
                try:
                    df = xl.parse(sh)
                    summarize_df(df, f"{f} [sheet: {sh}]", out)
                except Exception as e:
                    out.write(f"\n----- {f} [sheet: {sh}] -----\nERROR: {e}\n")
        elif suffix == '.xpt':
            df = pd.read_sas(f, format='xport', encoding='latin1')
            summarize_df(df, f, out)
        else:
            return False
    except Exception as e:
        out.write(f"\n----- {f} -----\nERROR opening file: {e}\n")
    return True


if __name__ == '__main__':
    import sys
    subfolder = sys.argv[1] if len(sys.argv) > 1 else ''
    target = ROOT / subfolder if subfolder else ROOT
    files = sorted(target.rglob('*'))
    data_files = [f for f in files if f.is_file() and f.suffix.lower() in
                  ('.sas7bdat', '.csv', '.xlsx', '.xpt')]
    print(f"Processing {len(data_files)} data files under {target}")
    with open(OUT, 'a', encoding='utf-8') as out:
        out.write(f"\n\n========== BATCH: {subfolder or 'ALL'} ({len(data_files)} files) ==========\n")
        for i, f in enumerate(data_files):
            process_file(f, out)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(data_files)} done")
    print("done, appended to", OUT)
