"""
Management command: python manage.py validate_against_aact

Compares TrialGuard's synthetic training data against real dropout rates
pulled from AACT (the aggregate ClinicalTrials.gov mirror). Generates each
seed trial's synthetic population the same way generate_synthetic_data does,
phase and therapeutic area aware, and checks the resulting dropout rate
against the real AACT rate for trials of that same phase and area.

This does not touch the database and does not require trained models. It
only needs the synthetic data generator (pure numpy/pandas) and the AACT
flat files in data/aact/. See docs/data_sourcing.md for background on why
AACT was chosen and what its limits are (trial-level, not patient-level).

Writes a report to docs/aact_validation_report.md.
"""
from pathlib import Path

from django.core.management.base import BaseCommand

from core.utils.data_pipeline import generate_synthetic_patients
from core.utils.aact_benchmark import real_dropout_benchmark, summarize_by_phase_and_area

# Matches the four seed trials in generate_synthetic_data.py. Phase is given
# in the Trial model's own codes (I, II, III, IV), the same ones
# generate_synthetic_patients expects, aact_phase is only used to look up
# the matching row in the AACT breakdown table for this report.
SEED_TRIAL_PROFILES = [
    ('CARDIO-GUARD Phase III', 'III', 'PHASE3', 'Cardiovascular'),
    ('ONCO-TRACE Phase II', 'II', 'PHASE2', 'Oncology'),
    ('NEURO-SHIELD Phase II', 'II', 'PHASE2', 'Neurology'),
    ('DIAB-PROTECT Phase IV', 'IV', 'PHASE4', 'Endocrinology'),
]

REPORT_PATH = Path(__file__).resolve().parents[3] / 'docs' / 'aact_validation_report.md'


class Command(BaseCommand):
    help = 'Compare synthetic training data dropout rate against real AACT dropout rates'

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING('\nAACT real-world validation\n'))

        self.stdout.write('  Loading AACT real trial data (data/aact/*.txt)...')
        try:
            real_df = real_dropout_benchmark()
        except FileNotFoundError as e:
            self.stdout.write(self.style.ERROR(f'  {e}'))
            return
        self.stdout.write(self.style.SUCCESS(
            f'  {len(real_df):,} interventional trials with real dropout data loaded\n'
        ))

        summary = summarize_by_phase_and_area(real_df)

        self.stdout.write('  Generating each seed trial\'s synthetic population, phase and area aware...')
        rows = []
        for i, (trial_name, phase, aact_phase, area) in enumerate(SEED_TRIAL_PROFILES):
            synth = generate_synthetic_patients(2000, phase=phase, therapeutic_area=area, seed=1000 + i)
            synth_rate = synth['dropout_status'].mean()

            match = summary[(summary['phase'] == aact_phase) & (summary['therapeutic_area'] == area)]
            if not match.empty:
                n = int(match.iloc[0]['n'])
                mean_rate = match.iloc[0]['mean_rate']
                median_rate = match.iloc[0]['median_rate']
                gap = synth_rate - mean_rate
                self.stdout.write(
                    f'    {trial_name:28s} ({aact_phase}, {area:15s}) real trials n={n:5d}  '
                    f'real mean={mean_rate:.1%}  synthetic={synth_rate:.1%}  gap={gap:+.1%}'
                )
                rows.append((trial_name, aact_phase, area, n, mean_rate, median_rate, synth_rate))
            else:
                self.stdout.write(f'    {trial_name:28s} ({aact_phase}, {area}): no matching real trials found')
                rows.append((trial_name, aact_phase, area, 0, None, None, synth_rate))

        self._write_report(real_df, summary, rows)
        self.stdout.write(self.style.SUCCESS(f'\n  Report written to {REPORT_PATH}\n'))

    def _write_report(self, real_df, summary, seed_rows):
        overall_real_mean = real_df['dropout_rate'].mean()
        overall_real_median = real_df['dropout_rate'].median()

        lines = []
        lines.append('# AACT real-world validation report')
        lines.append('')
        lines.append(
            'This compares TrialGuard\'s synthetic data generator against real '
            'dropout rates from AACT, the aggregate mirror of ClinicalTrials.gov. '
            'See docs/data_sourcing.md for why AACT was chosen and what it can '
            'and cannot validate.'
        )
        lines.append('')
        lines.append(
            'The generator now calibrates each trial\'s dropout rate against the '
            'real AACT rate for that trial\'s phase and therapeutic area, instead '
            'of using one flat rate for every trial. The table below checks that '
            'the calibration actually lands where it is supposed to.'
        )
        lines.append('')
        lines.append('## By seed trial (phase and therapeutic area matched)')
        lines.append('')
        lines.append('| Seed trial | Phase | Area | Real trials (n) | Real mean | Real median | Synthetic (calibrated) | Gap |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for trial_name, phase, area, n, mean_rate, median_rate, synth_rate in seed_rows:
            if n:
                gap = synth_rate - mean_rate
                lines.append(
                    f'| {trial_name} | {phase} | {area} | {n} | {mean_rate:.1%} | '
                    f'{median_rate:.1%} | {synth_rate:.1%} | {gap:+.1%} |'
                )
            else:
                lines.append(f'| {trial_name} | {phase} | {area} | 0 | - | - | {synth_rate:.1%} | - |')
        lines.append('')
        lines.append(
            'For reference, the overall AACT dropout rate across all '
            f'{len(real_df):,} interventional trials is {overall_real_mean:.1%} '
            f'(mean) and {overall_real_median:.1%} (median). Trial-level dropout '
            'rates are skewed, most trials have low dropout, a smaller number '
            'have very high dropout, which is why mean and median differ this '
            'much and why matching on phase and area rather than one global '
            'number matters.'
        )
        lines.append('')
        lines.append('## Full breakdown by phase and therapeutic area')
        lines.append('')
        lines.append('| Phase | Therapeutic area | n | Mean | Median | Std |')
        lines.append('|---|---|---|---|---|---|')
        for _, row in summary.head(30).iterrows():
            area = row['therapeutic_area'] if row['therapeutic_area'] else '(unmatched)'
            std = f"{row['std_rate']:.1%}" if row['std_rate'] == row['std_rate'] else '-'
            lines.append(
                f"| {row['phase']} | {area} | {int(row['n'])} | "
                f"{row['mean_rate']:.1%} | {row['median_rate']:.1%} | {std} |"
            )
        lines.append('')
        lines.append('## What this does and does not tell us')
        lines.append('')
        lines.append(
            '- This checks that the synthetic generator\'s dropout rate, once '
            'calibrated per trial, actually matches the real rate for trials of '
            'the same phase and therapeutic area. It is a check on the training '
            'data assumptions, not a validation of the trained model itself.'
        )
        lines.append(
            '- AACT only gives trial-level and arm-level counts, not individual '
            'patient visit records, so this cannot validate the per-patient '
            'XGBoost classifier directly. That still needs a patient-level real '
            'dataset (Project Data Sphere or ImmPort are the two candidates '
            'noted in docs/data_sourcing.md).'
        )
        lines.append(
            '- The calibration table (core/utils/aact_dropout_rates.json) only '
            'has real rates for combinations of phase and therapeutic area with '
            'at least 50 real trials behind them. New trial types outside our '
            'four seed areas fall back to a phase-only or overall average, '
            'which will be less accurate.'
        )
        lines.append('')

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')
