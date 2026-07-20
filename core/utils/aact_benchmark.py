"""
Loads real clinical trial data from AACT (the aggregate ClinicalTrials.gov
mirror maintained by CTTI/Duke) and computes real-world dropout benchmarks.

This is used to check TrialGuard's synthetic training data and model output
against actual reported trial outcomes. See docs/data_sourcing.md for why
AACT was chosen over the other sources that were considered.

Data files are not committed to the repo (too large). Download instructions
are in docs/data_sourcing.md. Expected location: data/aact/*.txt
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger('core')

AACT_DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'aact'

REQUIRED_TABLES = ['studies', 'drop_withdrawals', 'milestones', 'conditions']

# Same therapeutic area keywords used when we scoped AACT's coverage during
# the sourcing decision, kept here so the benchmark buckets line up with
# TrialGuard's four synthetic trial seeds (see generate_synthetic_data.py).
THERAPEUTIC_AREA_KEYWORDS = {
    'Cardiovascular': ['heart', 'cardiac', 'cardiovascular', 'coronary', 'hypertension'],
    'Oncology': ['cancer', 'tumor', 'oncology', 'carcinoma', 'leukemia', 'lymphoma'],
    'Neurology': ['neuro', 'alzheimer', 'parkinson', 'epilepsy', 'stroke', 'multiple sclerosis'],
    'Endocrinology': ['diabetes', 'endocrine', 'thyroid', 'obesity'],
}

# Real dropout reasons in AACT that represent a patient choosing to leave or
# quietly disappearing, the kind of thing a coordinator could plausibly
# still act on. Everything else (death, adverse event, physician decision,
# disease progression, sponsor decision, protocol violation, and so on) is
# a real reason a patient stopped, just not one retention outreach can fix.
# This is the same distinction the PDS oncology data forced us to make by
# hand (see docs/pds_validation_report.md), applied here to AACT's full
# 63,000-trial breakdown instead of 561 real patients.
#
# Matched case-insensitively since AACT's own reason field is inconsistently
# capitalised in the raw data ("Disease Progression" and "Disease progression"
# both appear as separate literal strings).
BEHAVIORAL_DROPOUT_REASONS = {
    'withdrawal by subject',
    'lost to follow-up',
    'lost to follow up',
    'withdrew consent',
}


def _normalize_reason(reason: str) -> str:
    return str(reason).strip().lower()


def load_aact_tables(data_dir: Path = None) -> dict:
    """Read the AACT pipe-delimited flat files we need into DataFrames."""
    data_dir = data_dir or AACT_DATA_DIR
    tables = {}
    for name in REQUIRED_TABLES:
        path = data_dir / f'{name}.txt'
        if not path.exists():
            raise FileNotFoundError(
                f"AACT table not found: {path}\n"
                f"Download the AACT static export and place the extracted "
                f".txt files in {data_dir}, see docs/data_sourcing.md."
            )
        tables[name] = pd.read_csv(path, sep='|', low_memory=False)
        logger.info("Loaded AACT table %s: %d rows", name, len(tables[name]))
    return tables


def _match_therapeutic_area(condition_name: str):
    name = str(condition_name).lower()
    for area, keywords in THERAPEUTIC_AREA_KEYWORDS.items():
        if any(k in name for k in keywords):
            return area
    return None


def compute_study_level_dropout(tables: dict) -> pd.DataFrame:
    """
    One row per study, with real started/completed/dropped counts and a
    dropout rate, built from the milestones table's STARTED/COMPLETED rows
    for the "Overall Study" period. This is the standard AACT participant
    flow structure, the same numbers ClinicalTrials.gov shows on a study's
    results page.
    """
    ms = tables['milestones']
    overall = ms[ms['period'] == 'Overall Study']

    started = (
        overall[overall['title'] == 'STARTED']
        .groupby('nct_id')['count'].sum()
        .rename('started')
    )
    completed = (
        overall[overall['title'] == 'COMPLETED']
        .groupby('nct_id')['count'].sum()
        .rename('completed')
    )

    df = pd.concat([started, completed], axis=1).dropna()
    df = df[df['started'] > 0]
    df['dropped'] = (df['started'] - df['completed']).clip(lower=0)
    df['dropout_rate'] = df['dropped'] / df['started']
    return df


def compute_behavioral_dropout_rate(tables: dict, started: pd.Series) -> pd.Series:
    """
    Real behavioral dropout count per study (Withdrawal by Subject, Lost to
    Follow-up, Withdrew Consent), divided by the same "started" denominator
    used for the all-cause rate, so the two numbers are directly comparable.
    Studies with no drop_withdrawals rows at all get 0, not missing, since
    the milestones table already establishes the study reported disposition
    data.
    """
    dw = tables['drop_withdrawals'].copy()
    dw['reason_norm'] = dw['reason'].apply(_normalize_reason)
    behavioral = dw[dw['reason_norm'].isin(BEHAVIORAL_DROPOUT_REASONS)]
    behavioral_count = behavioral.groupby('nct_id')['count'].sum()

    rate = (behavioral_count / started).reindex(started.index).fillna(0.0)
    return rate.rename('behavioral_dropout_rate')


def attach_study_metadata(dropout_df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """Attach phase, study type, and a best-guess therapeutic area to each study."""
    studies = tables['studies'][['nct_id', 'phase', 'study_type', 'overall_status', 'enrollment']]
    df = dropout_df.merge(studies, on='nct_id', how='left')
    df = df[df['study_type'] == 'INTERVENTIONAL']

    conditions = tables['conditions'][['nct_id', 'name']].copy()
    conditions['therapeutic_area'] = conditions['name'].apply(_match_therapeutic_area)
    area_map = (
        conditions.dropna(subset=['therapeutic_area'])
        .drop_duplicates(subset=['nct_id', 'therapeutic_area'])
        .groupby('nct_id')['therapeutic_area'].first()
    )
    df = df.merge(area_map.rename('therapeutic_area'), on='nct_id', how='left')
    return df


def real_dropout_benchmark(data_dir: Path = None) -> pd.DataFrame:
    """
    Main entry point. Returns a study-level DataFrame of real dropout rates
    for interventional trials, restricted to sane values (0 to 1), with
    phase and therapeutic area attached where we could match one.

    Two rates are included: `dropout_rate` (all-cause, the original
    started-versus-completed count) and `behavioral_dropout_rate` (only
    Withdrawal by Subject / Lost to Follow-up / Withdrew Consent). The
    behavioral rate is the one that should be used to calibrate the
    synthetic generator, the all-cause rate mixes in death and disease
    progression, which a retention tool cannot influence. Both are kept so
    the gap between them stays visible rather than silently discarded.
    """
    tables = load_aact_tables(data_dir)
    dropout_df = compute_study_level_dropout(tables)
    dropout_df['behavioral_dropout_rate'] = compute_behavioral_dropout_rate(tables, dropout_df['started'])
    dropout_df = dropout_df.reset_index()
    df = attach_study_metadata(dropout_df, tables)
    df = df[(df['dropout_rate'] >= 0) & (df['dropout_rate'] <= 1)]
    df = df[(df['behavioral_dropout_rate'] >= 0) & (df['behavioral_dropout_rate'] <= 1)]
    return df


def summarize_by_phase_and_area(df: pd.DataFrame, rate_col: str = 'behavioral_dropout_rate') -> pd.DataFrame:
    summary = (
        df.groupby(['phase', 'therapeutic_area'], dropna=False)[rate_col]
        .agg(n='count', mean_rate='mean', median_rate='median', std_rate='std')
        .reset_index()
        .sort_values('n', ascending=False)
    )
    return summary
