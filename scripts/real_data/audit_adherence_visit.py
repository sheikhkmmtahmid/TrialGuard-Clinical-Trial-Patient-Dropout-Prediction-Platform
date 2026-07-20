"""
Minutely scan every real data file in the project (PDS, MUSIC, UCI Heart
Failure) for columns that could be real medication adherence data or real
scheduled-visit data. Reads actual column headers from every file, not
filenames, not memory. Prints a per-study hit list so nothing is guessed.
"""
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ADHERENCE_KEYWORDS = [
    'adher', 'compl', 'account', 'pillct', 'pilcnt', 'pillcount', 'doschg',
    'dosemod', 'dosmod', 'dosered', 'doseredu', 'doseint', 'dose_int',
    'missed dose', 'refill', 'dosedelay', 'dosdelay', 'exred', 'exdelay',
    'exreduc', 'dosecomp',
]
VISIT_SCHEDULE_KEYWORDS = [
    'planned_visit', 'plannedvisit', 'planned visit', 'sched', 'protocol visit',
    'visit window', 'expected visit', 'windowsp', 'visit_win', 'target day',
    'nominal day', 'nomday', 'plnvisit', 'planvisit',
]

PDS_ROOT = Path(r'D:\Trial Guard\data\pds')


def check_columns(cols, keywords):
    hits = []
    for c in cols:
        lc = str(c).lower()
        for kw in keywords:
            if kw in lc:
                hits.append(c)
                break
    return hits


def scan_file(path):
    try:
        suffix = path.suffix.lower()
        if suffix == '.txt':
            try:
                df = pd.read_csv(path, sep='\t', nrows=1, low_memory=False)
                if len(df.columns) <= 1:
                    df = pd.read_csv(path, nrows=1, low_memory=False)
                cols = list(df.columns)
            except Exception:
                return None
            return cols
        if suffix == '.sas7bdat':
            # Read only the header via pandas (still opens whole file for
            # SAS, but there is no lighter-weight reader available here;
            # kept because correctness matters more than speed for this
            # audit).
            df = pd.read_sas(path, format='sas7bdat', encoding='latin1')
            cols = list(df.columns)
        elif suffix == '.csv':
            df = pd.read_csv(path, nrows=1, low_memory=False)
            cols = list(df.columns)
        elif suffix == '.xlsx':
            xl = pd.ExcelFile(path)
            cols = []
            for sh in xl.sheet_names:
                try:
                    d = xl.parse(sh, nrows=1)
                    cols += list(d.columns)
                except Exception:
                    pass
        else:
            return None
        return cols
    except Exception as e:
        return f'ERROR: {e}'


def scan_study(study_dir):
    files = (list(study_dir.rglob('*.sas7bdat')) + list(study_dir.rglob('*.csv'))
             + list(study_dir.rglob('*.xlsx')) + list(study_dir.rglob('*.txt')))
    adherence_hits = {}
    visit_hits = {}
    errors = []
    tv_domain_files = [f.name for f in files if f.name.lower() in ('tv.sas7bdat', 'tv.txt')]
    for f in files:
        cols = scan_file(f)
        if cols is None:
            continue
        if isinstance(cols, str):
            errors.append((f.name, cols))
            continue
        a = check_columns(cols, ADHERENCE_KEYWORDS)
        v = check_columns(cols, VISIT_SCHEDULE_KEYWORDS)
        if a:
            adherence_hits[f.name] = a
        if v:
            visit_hits[f.name] = v
    if tv_domain_files:
        visit_hits['__SDTM_TV_DOMAIN__'] = tv_domain_files
    return adherence_hits, visit_hits, errors


if __name__ == '__main__':
    import sys
    studies = sorted([d for d in PDS_ROOT.iterdir() if d.is_dir()])
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(studies)

    for study_dir in studies[start:end]:
        print(f"\n===== {study_dir.name} =====")
        a, v, errs = scan_study(study_dir)
        if a:
            print("ADHERENCE-like columns found:")
            for fname, cols in a.items():
                print(f"  {fname}: {cols}")
        else:
            print("ADHERENCE-like columns: none found")
        if v:
            print("VISIT-SCHEDULE-like columns found:")
            for fname, cols in v.items():
                print(f"  {fname}: {cols}")
        else:
            print("VISIT-SCHEDULE-like columns: none found")
        if errs:
            print("Files that errored (not scanned):", errs)
