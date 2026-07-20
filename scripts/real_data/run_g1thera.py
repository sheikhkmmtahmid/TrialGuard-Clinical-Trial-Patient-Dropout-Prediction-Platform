import pandas as pd
from extract_g1thera import extract

BEHAVIORAL_TERMS = {'withdrawal by subject', 'lost to follow-up'}


def get_behavioral_and_days(base_dir):
    adsl = pd.read_sas(f"{base_dir}/adsl.sas7bdat", format='sas7bdat', encoding='latin1')
    adsl['_norm'] = adsl['DCSREAS'].astype(str).str.strip().str.lower()
    beh = set(adsl[adsl['_norm'].isin(BEHAVIORAL_TERMS)]['USUBJID'])
    days = adsl.set_index('USUBJID')['EOSDY'].to_dict()
    return beh, days


results = []
for d, label in [
    ('../../data/pds/LungSm_G1Thera_2015_433', 'G1Thera_2015_433'),
    ('../../data/pds/LungSm_G1Thera_2015_434', 'G1Thera_2015_434'),
    ('../../data/pds/LungSm_G1Thera_2017_435', 'G1Thera_2017_435'),
]:
    beh, days = get_behavioral_and_days(d)
    df = extract(d, label, beh, days)
    print(f"{label}: {len(df)} patients with real visit data, {df['dropout_status'].sum()} behavioral")
    results.append(df)

combined = pd.concat(results, ignore_index=True)
combined.to_csv('g1thera_real.csv', index=False)
print("saved g1thera_real.csv, total rows:", len(combined))
