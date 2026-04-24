"""
Management command: python manage.py generate_synthetic_data

Generates 5,000 synthetic patients with realistic clinical trial dropout patterns
and persists them to the database. Uses numpy-based synthesis (falls back gracefully
if SDV is not installed).
"""
import random
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import Trial, Patient, Visit
from core.utils.data_pipeline import generate_synthetic_patients, generate_synthetic_visits

logger = logging.getLogger('core')

TRIAL_SEEDS = [
    {
        'name': 'CARDIO-GUARD Phase III',
        'sponsor': 'HeartBridge Pharmaceuticals',
        'phase': 'III',
        'therapeutic_area': 'Cardiovascular',
        'target_enrollment': 1200,
    },
    {
        'name': 'ONCO-TRACE Phase II',
        'sponsor': 'NovaCure Oncology',
        'phase': 'II',
        'therapeutic_area': 'Oncology',
        'target_enrollment': 600,
    },
    {
        'name': 'NEURO-SHIELD Phase II',
        'sponsor': 'AxisNeuro Sciences',
        'phase': 'II',
        'therapeutic_area': 'Neurology',
        'target_enrollment': 450,
    },
    {
        'name': 'DIAB-PROTECT Phase IV',
        'sponsor': 'GlycoClear Therapeutics',
        'phase': 'IV',
        'therapeutic_area': 'Endocrinology',
        'target_enrollment': 2000,
    },
]


class Command(BaseCommand):
    help = 'Generate 5,000 synthetic patients and persist to database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--n', type=int, default=5000,
            help='Number of synthetic patients to generate (default: 5000)'
        )
        parser.add_argument(
            '--clear', action='store_true',
            help='Clear existing synthetic data before generating'
        )

    def handle(self, *args, **options):
        n_patients = options['n']
        self.stdout.write(self.style.MIGRATE_HEADING(
            f'\n🧬 TrialGuard Synthetic Data Generator — {n_patients:,} patients\n'
        ))

        if options['clear']:
            self.stdout.write('  Clearing existing data...')
            Visit.objects.all().delete()
            Patient.objects.all().delete()
            Trial.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('  ✓ Cleared\n'))

        # ── Create trials ──────────────────────────────────────────────────
        self.stdout.write('  Creating trial records...')
        trials = []
        today = date.today()
        for seed in TRIAL_SEEDS:
            trial, created = Trial.objects.get_or_create(
                name=seed['name'],
                defaults={
                    **seed,
                    'start_date': today - timedelta(days=random.randint(180, 730)),
                    'end_date': today + timedelta(days=random.randint(180, 730)),
                }
            )
            trials.append(trial)
            if created:
                self.stdout.write(f'    + {trial.name}')

        # ── Generate patient feature matrix ───────────────────────────────
        self.stdout.write(f'\n  Generating {n_patients:,} synthetic patient records...')
        patient_df = generate_synthetic_patients(n_patients)
        visit_df = generate_synthetic_visits(patient_df, visits_per_patient=8)

        # ── Persist patients ──────────────────────────────────────────────
        self.stdout.write('  Persisting patients to database...')
        rng = np.random.default_rng(42)

        GENDERS = ['M', 'F', 'O']
        GENDER_PROBS = [0.48, 0.48, 0.04]
        ETHNICITIES = ['white', 'black', 'hispanic', 'asian', 'other', 'unknown']
        ETHNICITIES_PROBS = [0.60, 0.13, 0.18, 0.06, 0.02, 0.01]
        SEVERITIES = ['mild', 'moderate', 'severe']
        EMPLOYMENTS = ['employed', 'unemployed', 'retired', 'student', 'other']

        SEVERITY_FROM_CODE = {0: 'mild', 1: 'moderate', 2: 'severe'}
        GENDER_FROM_CODE = {0: 'M', 1: 'F', 2: 'O'}

        batch_size = 500
        patient_objs = []
        patient_index_map = {}  # df_index -> Patient pk (set after bulk_create)

        for idx, row in patient_df.iterrows():
            trial = random.choice(trials)
            enrollment_days_ago = random.randint(30, 730)
            enrollment_date = today - timedelta(days=enrollment_days_ago)

            dropout_status = bool(row['dropout_status'])
            dropout_date = None
            if dropout_status:
                dropout_days = min(int(row['days_to_event']), enrollment_days_ago)
                dropout_date = enrollment_date + timedelta(days=max(dropout_days, 1))

            patient_objs.append(Patient(
                trial=trial,
                age=int(row['age']),
                gender=GENDER_FROM_CODE.get(int(row['gender_encoded']), 'U'),
                ethnicity=rng.choice(ETHNICITIES, p=ETHNICITIES_PROBS),
                condition_severity=SEVERITY_FROM_CODE.get(int(row['condition_severity_encoded']), 'moderate'),
                distance_to_site_km=float(row['distance_to_site_km']),
                employment_status=rng.choice(EMPLOYMENTS),
                prior_dropout_history=bool(row['prior_dropout_history']),
                enrollment_date=enrollment_date,
                dropout_date=dropout_date,
                dropout_status=dropout_status,
            ))

        #with transaction.atomic():
        #    created_patients = Patient.objects.bulk_create(patient_objs, batch_size=batch_size)
        #    #Fix
        #    # Map dataframe index -> actual DB patient IDs
        #    patient_id_map = {
        #        idx: patient.id
        #        for idx, patient in enumerate(created_patients)
        #    }
        #    #Fix end

        with transaction.atomic():
            Patient.objects.bulk_create(patient_objs, batch_size=batch_size)

        created_patients = list(
            Patient.objects.order_by('-id')[:len(patient_objs)]
        )
        created_patients.reverse()

        patient_id_map = {
            idx: patient.id
            for idx, patient in enumerate(created_patients)
        }

        self.stdout.write(self.style.SUCCESS(f'  ✓ {len(created_patients):,} patients created'))

        # ── Persist visits ────────────────────────────────────────────────
        self.stdout.write('  Persisting visits...')
        visit_objs = []

        for (pat_idx, group) in visit_df.groupby('patient_idx'):
            if pat_idx >= len(created_patients):
                continue
            #patient = created_patients[int(pat_idx)]
            #enroll = patient.enrollment_date
            #Fix
            patient_obj = created_patients[int(pat_idx)]
            patient_id = patient_id_map[int(pat_idx)]
            enroll = patient_obj.enrollment_date
            #Fix end
            visit_date = enroll

            for _, vrow in group.iterrows():
                visit_date = visit_date + timedelta(days=int(vrow['days_since_last_visit']) or 30)
                visit_objs.append(Visit(
                    #patient=patient,
                    patient_id=patient_id, #Fix
                    visit_number=int(vrow['visit_number']),
                    visit_date=visit_date,
                    adverse_events_count=int(vrow['adverse_events_count']),
                    missed_visits_to_date=int(vrow['missed_visits_to_date']),
                    medication_adherence_score=float(vrow['medication_adherence_score']),
                    quality_of_life_score=float(vrow['quality_of_life_score']),
                    days_since_last_visit=int(vrow['days_since_last_visit']),
                ))

        with transaction.atomic():
            #Visit.objects.bulk_create(visit_objs, batch_size=1000, ignore_conflicts=True)
            Visit.objects.bulk_create(visit_objs, batch_size=1000) #Fix

        #self.stdout.write(self.style.SUCCESS(f'  ✓ {len(visit_objs):,} visits created'))
        self.stdout.write(self.style.SUCCESS(f'  ✓ {Visit.objects.count():,} total visits now in database')) #Fix
        self.stdout.write(self.style.SUCCESS('\n✅ Synthetic data generation complete.\n'))
        self.stdout.write(
            '   Next step: python manage.py train_models\n'
        )
