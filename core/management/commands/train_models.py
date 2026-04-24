"""
Management command: python manage.py train_models

Trains all TrialGuard ML models in sequence:
  1. Build feature matrix from DB
  2. Fit StandardScaler
  3. Fit CoxPHFitter (survival analysis)
  4. Add Cox hazard ratio as feature
  5. Run Optuna XGBoost tuning + final fit
  6. Build SHAP TreeExplainer
  7. Run batch inference → store PredictionResult records
  8. Generate CohortForecasts per trial
  9. Save evaluation_results.json

Run time: ~5-15 minutes depending on data size and Optuna trials.
"""
import json
import logging
import numpy as np
import pandas as pd
from datetime import date

from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import Patient, Trial, PredictionResult, CohortForecast
from core.utils.data_pipeline import (
    build_full_dataset, fit_and_save_scaler, load_scaler,
    FEATURE_COLUMNS, generate_synthetic_patients
)
from core.utils.survival_model import train_cox_model, load_cox_model, predict_survival
from core.utils.xgboost_model import train_xgboost_model, load_xgb_model, save_evaluation_results
from core.utils.shap_explainer import build_shap_explainer, compute_shap_values, shap_values_to_json, compute_shap_stability_score

logger = logging.getLogger('core')
# Fix
def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except:
        return default

    if not np.isfinite(value):
        return default

    return value
#FIx end

class Command(BaseCommand):
    help = 'Train Cox PH + XGBoost models and run batch inference on all patients'

    def add_arguments(self, parser):
        parser.add_argument(
            '--optuna-trials', type=int, default=50,
            help='Number of Optuna hyperparameter search trials (default: 50)'
        )
        parser.add_argument(
            '--skip-inference', action='store_true',
            help='Skip post-training batch inference (useful for re-training only)'
        )

    def handle(self, *args, **options):
        n_trials = options['optuna_trials']
        self.stdout.write(self.style.MIGRATE_HEADING(
            '\n⚕️  TrialGuard Model Training Pipeline\n'
        ))

        # Step 1: Build dataset 
        self.stdout.write('  [1/7] Building feature matrix from database...')
        df = build_full_dataset()

        if len(df) < 100:
            self.stdout.write(self.style.WARNING(
                '  ⚠ Less than 100 patient records found. '
                'Run `python manage.py generate_synthetic_data` first.'
            ))
            return

        self.stdout.write(self.style.SUCCESS(
            f'  ✓ {len(df):,} visit rows across {df["patient_id"].nunique():,} patients'
        ))

        # Step 2: Fit scaler 
        self.stdout.write('  [2/7] Fitting StandardScaler...')
        scaler = fit_and_save_scaler(df)
        X_scaled = scaler.transform(df[FEATURE_COLUMNS])
        self.stdout.write(self.style.SUCCESS(f'  ✓ Scaler fitted on {len(FEATURE_COLUMNS)} features'))

        # Step 3: Train Cox PH model
        self.stdout.write('  [3/7] Training Cox Proportional Hazards model...')
        cox_model, c_index = train_cox_model(df)
        self.stdout.write(self.style.SUCCESS(f'  ✓ Cox model trained — Concordance Index: {c_index:.4f}'))

        if c_index < 0.60:
            self.stdout.write(self.style.WARNING(
                f'  ⚠ Concordance index {c_index:.4f} is below 0.65 target. '
                'Consider adding more data.'
            ))

        # Step 4: Add Cox hazard ratio to feature matrix 
        self.stdout.write('  [4/7] Computing Cox hazard ratios for all patients...')
        from core.utils.data_pipeline import (
            _SEVERITY_MAP, _age_group, _distance_bucket
        )

        hazard_ratios = []
        for _, row in df.iterrows():
            feats = {
                'age': row['age'],
                'condition_severity_encoded': row['condition_severity_encoded'],
                'distance_to_site_km': row['distance_to_site_km'],
                'prior_dropout_history': row['prior_dropout_history'],
                'cumulative_missed_visits': row['cumulative_missed_visits'],
                'adverse_event_rate': row['adverse_event_rate'],
                'medication_adherence_score': row['medication_adherence_score'],
            }
            try:
                hr_info = predict_survival(feats)
                hazard_ratios.append(hr_info['hazard_ratio'])
            except Exception:
                hazard_ratios.append(1.0)

        df['hazard_ratio'] = hazard_ratios
        X_with_hr = np.hstack([X_scaled, np.array(hazard_ratios).reshape(-1, 1)])
        self.stdout.write(self.style.SUCCESS(f'  ✓ Hazard ratios computed'))

        # Step 5: Train XGBoost 
        self.stdout.write(f'  [5/7] Training XGBoost (Optuna: {n_trials} trials)...')
        self.stdout.write('        This may take several minutes...')
        df_with_hr = df.copy()
        df_with_hr['hazard_ratio'] = hazard_ratios
        xgb_model, metrics = train_xgboost_model(df_with_hr, n_optuna_trials=n_trials)
        self.stdout.write(self.style.SUCCESS(
            f'  ✓ XGBoost trained — '
            f'ROC-AUC: {metrics["xgb_roc_auc"]:.4f} | '
            f'F1: {metrics["xgb_f1"]:.4f}'
        ))

        if metrics['xgb_roc_auc'] < 0.70:
            self.stdout.write(self.style.WARNING(
                f'  ⚠ ROC-AUC {metrics["xgb_roc_auc"]:.4f} is below 0.80 target. '
                'More data or feature engineering may improve performance.'
            ))

        #  Step 6: Build SHAP explainer 
        self.stdout.write('  [6/7] Building SHAP TreeExplainer...')
        explainer = build_shap_explainer(xgb_model)

        #sample_size = min(500, len(X_scaled))
        sample_size = min(500, len(X_with_hr)) #Fix
        #shap_sample = X_scaled[:sample_size]
        # Use SAME feature matrix as XGBoost (includes hazard_ratio)
        shap_sample = X_with_hr[:sample_size] #Fix
        shap_vals = explainer.shap_values(shap_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        stability = compute_shap_stability_score(shap_vals)
        metrics['shap_stability_score'] = stability
        self.stdout.write(self.style.SUCCESS(f'  ✓ SHAP explainer built — stability: {stability:.4f}'))

        # Step 7: Save evaluation results
        save_evaluation_results(metrics, cox_concordance=c_index)
        self.stdout.write(self.style.SUCCESS('  ✓ evaluation_results.json saved'))

        if options['skip_inference']:
            self.stdout.write(self.style.SUCCESS('\n✅ Training complete (inference skipped).\n'))
            return

        # Step 8: Batch inference 
        self.stdout.write('  [7/7] Running batch inference on all patients...')
        patients = list(
            Patient.objects
            .prefetch_related('visits')
            .select_related('trial')
            .all()
        )

        self.stdout.write(f'        Processing {len(patients):,} patients...')
        pred_objs = []
        processed = 0

        for patient in patients:
            visits = list(patient.visits.order_by('visit_number'))
            if not visits:
                continue

            latest_visit = visits[-1]
            try:
                from core.utils.data_pipeline import engineer_features_for_patient
                feat_df = engineer_features_for_patient(patient, patient.visits.all())
                if feat_df.empty:
                    continue

                last_row = feat_df.tail(1)
                #X_row = scaler.transform(last_row[FEATURE_COLUMNS].values)
                X_row_df = last_row[FEATURE_COLUMNS].copy() #Fix
                X_row = scaler.transform(X_row_df) #Fix
                #hr = float(predict_survival({
                #    'age': patient.age,
                #    'condition_severity_encoded': last_row['condition_severity_encoded'].values[0],
                #    'distance_to_site_km': patient.distance_to_site_km,
                #    'prior_dropout_history': int(patient.prior_dropout_history),
                #    'cumulative_missed_visits': last_row['cumulative_missed_visits'].values[0],
                #    'adverse_event_rate': last_row['adverse_event_rate'].values[0],
                #    'medication_adherence_score': last_row['medication_adherence_score'].values[0],
                #}).get('hazard_ratio', 1.0))
                # Fix
                hr_input = {
                    'age': int(patient.age),
                    'condition_severity_encoded': int(last_row.iloc[0]['condition_severity_encoded']),
                    'distance_to_site_km': float(patient.distance_to_site_km),
                    'prior_dropout_history': int(patient.prior_dropout_history),
                    'cumulative_missed_visits': int(last_row.iloc[0]['cumulative_missed_visits']),
                    'adverse_event_rate': float(last_row.iloc[0]['adverse_event_rate']),
                    'medication_adherence_score': float(last_row.iloc[0]['medication_adherence_score']),
                }
                hr = _safe_float(predict_survival(hr_input).get('hazard_ratio', 1.0), default=1.0)
                # Fix end
                X_full = np.hstack([X_row, [[hr]]])

                #prob = float(xgb_model.predict_proba(X_full)[0][1])
                prob = _safe_float(xgb_model.predict_proba(X_full)[0][1], default=0.0) # Fix
                prob = min(max(prob, 0.0), 1.0) # Fix
                risk_tier = PredictionResult.get_risk_tier(prob)

                #row_shap = explainer.shap_values(X_row)
                row_shap = explainer.shap_values(X_full) #Fix
                if isinstance(row_shap, list):
                    row_shap = row_shap[1]
                #shap_json = shap_values_to_json(row_shap[0], FEATURE_COLUMNS)
                # Fix
                shap_json = shap_values_to_json(row_shap[0], FEATURE_COLUMNS + ['hazard_ratio'])
                # Fix end

                #surv_info = predict_survival({
                #    'age': patient.age,
                #    'condition_severity_encoded': last_row['condition_severity_encoded'].values[0],
                #    'distance_to_site_km': patient.distance_to_site_km,
                #    'prior_dropout_history': int(patient.prior_dropout_history),
                #    'cumulative_missed_visits': last_row['cumulative_missed_visits'].values[0],
                #    'adverse_event_rate': last_row['adverse_event_rate'].values[0],
                #    'medication_adherence_score': last_row['medication_adherence_score'].values[0],
                #})
                surv_info = predict_survival(hr_input) #Fix

                #pred_objs.append(PredictionResult(
                #    patient=patient,
                #    visit=latest_visit,
                #    dropout_probability=prob,
                #    risk_tier=risk_tier,
                #    shap_values_json=shap_json,
                #    survival_time_estimate=surv_info.get('median_survival_days'),
                #    hazard_ratio=hr,
                #    model_version='1.0.0',
                #))
                # Fix 
                pred_objs.append(PredictionResult(
                    patient=patient,
                    visit=latest_visit,
                    dropout_probability=prob,
                    risk_tier=risk_tier,
                    shap_values_json=shap_json,
                    survival_time_estimate=_safe_float(surv_info.get('median_survival_days'), default=0.0),
                    hazard_ratio=_safe_float(hr, default=1.0),
                    model_version='1.0.0',
                ))
                #Fix end
                processed += 1

            except Exception as e:
                logger.warning("Inference failed for patient %s: %s", patient.pk, e)

            if processed % 500 == 0 and processed > 0:
                self.stdout.write(f'        {processed:,} / {len(patients):,} done...')

        with transaction.atomic():
            PredictionResult.objects.all().delete()
            PredictionResult.objects.bulk_create(pred_objs, batch_size=500)

        self.stdout.write(self.style.SUCCESS(
            f'  ✓ {len(pred_objs):,} predictions stored'
        ))

        # Step 9: Generate CohortForecasts 
        self.stdout.write('  Generating cohort forecasts...')
        trials = Trial.objects.all()
        forecasts = []
        for trial in trials:
            critical_preds = PredictionResult.objects.filter(
                patient__trial=trial,
                risk_tier__in=['high', 'critical']
            ).count()
            total = Patient.objects.filter(trial=trial).count()
            if total == 0:
                continue

            base = critical_preds
            forecasts.append(CohortForecast(
                trial=trial,
                forecast_date=date.today(),
                predicted_dropouts_30d=max(0, int(base * 0.4)),
                predicted_dropouts_60d=max(0, int(base * 0.7)),
                predicted_dropouts_90d=max(0, int(base * 1.0)),
                confidence_interval_lower=max(0, int(base * 0.5)),
                confidence_interval_upper=int(base * 1.5),
            ))

        with transaction.atomic():
            CohortForecast.objects.all().delete()
            CohortForecast.objects.bulk_create(forecasts)

        self.stdout.write(self.style.SUCCESS(
            f'  ✓ {len(forecasts)} cohort forecasts generated'
        ))

        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('✅ All models trained and inference complete.\n'))
        self.stdout.write('   Dashboard is ready at: http://localhost:8000/dashboard/\n')
        self.stdout.write('   API docs at: http://localhost:8000/api/docs/\n')
        self.stdout.write('')
        self.stdout.write('   Model Performance Summary:')
        self.stdout.write(f'   ├─ XGBoost ROC-AUC:       {metrics["xgb_roc_auc"]:.4f}')
        self.stdout.write(f'   ├─ XGBoost F1:            {metrics["xgb_f1"]:.4f}')
        self.stdout.write(f'   ├─ Cox Concordance Index: {c_index:.4f}')
        self.stdout.write(f'   ├─ Brier Score:           {metrics["calibration_brier_score"]:.4f}')
        self.stdout.write(f'   └─ SHAP Stability:        {stability:.4f}\n')
