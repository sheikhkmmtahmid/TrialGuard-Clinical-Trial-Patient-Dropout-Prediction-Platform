"""
Rebuild real cumulative_missed_visits for the 4 studies with a confirmed
real planned-visit schedule (tv.sas7bdat), by comparing each patient's
actual visit days (already extracted, from ex.sas7bdat) against their
arm's real planned schedule. A planned visit counts as missed if no
actual visit falls within +/- 7 days of it, and only planned visits up to
the patient's own last real visit day are counted (a visit they simply
hadn't reached yet by the time their data ends is not "missed").

Verified once on a single real patient before running on everyone, to
catch logic errors before they propagate silently. See
docs/adherence_visit_data_audit.md for how these 4 studies were found.
"""
import pandas as pd
import numpy as np
from pathlib import Path

WINDOW_DAYS = 7


def planned_days_for_arm(tv, armcd):
    sub = tv[tv['ARMCD'].astype(str).str.strip().str.lower() == str(armcd).strip().lower()]
    days = sorted(sub['VISITDY'].dropna().unique().tolist())
    return days


def count_missed(actual_days, planned_days, window=WINDOW_DAYS):
    if not actual_days or not planned_days:
        return 0
    last_actual = actual_days[-1]
    relevant_planned = [d for d in planned_days if d <= last_actual and d > 0]
    missed = 0
    for pd_day in relevant_planned:
        if not any(abs(a - pd_day) <= window for a in actual_days):
            missed += 1
    return missed


# ---- sanity check on one real patient before trusting this on everyone ----
if __name__ == '__main__':
    import sys
    tv = pd.read_sas(r'D:\Trial Guard\data\pds\Glioma_EMDSero_2008_441\DATA_SAS\tv.sas7bdat',
                      format='sas7bdat', encoding='latin1')
    ex = pd.read_sas(r'D:\Trial Guard\data\pds\Glioma_EMDSero_2008_441\DATA_SAS\ex.sas7bdat',
                      format='sas7bdat', encoding='latin1')
    dm = pd.read_sas(r'D:\Trial Guard\data\pds\Glioma_EMDSero_2008_441\DATA_SAS\dm.sas7bdat',
                      format='sas7bdat', encoding='latin1')

    test_subj = dm['USUBJID'].iloc[0]
    armcd = dm[dm['USUBJID'] == test_subj]['ARMCD'].iloc[0]
    planned = planned_days_for_arm(tv, armcd)
    actual = sorted(ex[ex['USUBJID'] == test_subj]['EXSTDY'].dropna().unique().tolist())

    print(f"Test patient: {test_subj}, arm: {armcd}")
    print(f"Real planned visit days (first 15): {planned[:15]}")
    print(f"Real actual visit days: {actual}")
    missed = count_missed(actual, planned)
    print(f"Computed missed visits: {missed}")
    print(f"\nManual check: planned days up to their last actual visit "
          f"({actual[-1] if actual else 'N/A'}):")
    if actual:
        relevant = [d for d in planned if d <= actual[-1] and d > 0]
        for d in relevant:
            hit = any(abs(a - d) <= WINDOW_DAYS for a in actual)
            print(f"  planned day {d}: {'HIT' if hit else 'MISSED'}")
