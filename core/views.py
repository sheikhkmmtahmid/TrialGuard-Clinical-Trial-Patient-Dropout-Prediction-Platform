"""
TrialGuard views — web UI + DRF REST API.
"""
import io
import base64
import logging
import csv
import json
import threading
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Count, Q, Avg
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from django.views.decorators.cache import never_cache

from rest_framework import viewsets, generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Trial, Patient, Visit, PredictionResult, CohortForecast, CoordinatorAction
from .forms import CSVUploadForm, CoordinatorActionForm, AddPatientForm
from .serializers import (
    TrialSerializer, PatientListSerializer, PatientDetailSerializer,
    PredictionResultSerializer, CohortForecastSerializer,
    PatientUploadSerializer, VisitUploadSerializer,
)

logger = logging.getLogger('core')
ML_MODELS_DIR = Path(__file__).resolve().parents[1] / 'ml_models'


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _models_trained():
    """Check whether trained ML model artifacts exist."""
    return (ML_MODELS_DIR / 'xgb_model.pkl').exists()


def _get_patient_features_dict(patient, visit):
    """Build the covariate dict needed by the survival model."""
    from core.utils.data_pipeline import (
        _SEVERITY_MAP, _age_group, _distance_bucket
    )
    return {
        'age': patient.age,
        'condition_severity_encoded': _SEVERITY_MAP.get(patient.condition_severity, 1),
        'distance_to_site_km': patient.distance_to_site_km,
        'prior_dropout_history': int(patient.prior_dropout_history),
        'cumulative_missed_visits': visit.missed_visits_to_date,
        'adverse_event_rate': visit.adverse_events_count,
        'medication_adherence_score': visit.medication_adherence_score,
    }


def _run_prediction_for_patient(patient, visit):
    """
    Run XGBoost + Cox models for a single patient/visit and persist result.
    Returns PredictionResult or None.
    """
    if not _models_trained():
        return None

    try:
        from core.utils.data_pipeline import engineer_features_for_patient, load_scaler, FEATURE_COLUMNS
        from core.utils.xgboost_model import load_xgb_model
        from core.utils.survival_model import predict_survival
        from core.utils.shap_explainer import load_shap_explainer, shap_values_to_json

        df = engineer_features_for_patient(patient, patient.visits.all())
        if df.empty:
            return None

        scaler = load_scaler()
        row = df[df['visit_id'] == visit.pk]
        if row.empty:
            row = df.tail(1)

        X_raw = row[FEATURE_COLUMNS].values
        X_scaled = scaler.transform(X_raw)

        xgb_model = load_xgb_model()
        prob = float(xgb_model.predict_proba(X_scaled)[0][1])
        risk_tier = PredictionResult.get_risk_tier(prob)

        features_dict = _get_patient_features_dict(patient, visit)
        survival_info = predict_survival(features_dict)

        explainer = load_shap_explainer()
        shap_vals = explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_json = shap_values_to_json(shap_vals[0], FEATURE_COLUMNS)

        result = PredictionResult.objects.update_or_create(
            patient=patient,
            visit=visit,
            defaults={
                'dropout_probability': prob,
                'risk_tier': risk_tier,
                'shap_values_json': shap_json,
                'survival_time_estimate': survival_info.get('median_survival_days'),
                'hazard_ratio': survival_info.get('hazard_ratio'),
                'model_version': '1.0.0',
            }
        )[0]
        return result

    except Exception as e:
        logger.error("Prediction failed for patient %s: %s", patient.pk, e)
        return None


def _build_risk_timeline_b64(predictions_qs):
    """Build a line chart of dropout probability over visits. Returns base64 PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    preds = list(predictions_qs.select_related('visit').order_by('visit__visit_number'))
    if not preds:
        return ''

    xs = [p.visit.visit_number for p in preds]
    ys = [p.dropout_probability for p in preds]
    tiers = [p.risk_tier for p in preds]

    tier_colour_map = {
        'low': '#009639', 'medium': '#FFB81C',
        'high': '#E65C00', 'critical': '#CC0000',
    }
    pt_colours = [tier_colour_map.get(t, '#425563') for t in tiers]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#F0F4F5')

    ax.plot(xs, ys, color='#003087', linewidth=2, zorder=1)
    ax.scatter(xs, ys, c=pt_colours, s=60, zorder=2, edgecolors='#ffffff', linewidth=0.8)

    ax.axhline(0.55, color='#FFB81C', linestyle='--', linewidth=1.5, alpha=0.9, label='High Risk Threshold (55%)')
    ax.axhline(0.75, color='#CC0000', linestyle='-',  linewidth=1.5, alpha=0.9, label='Critical Threshold (75%)')
    ax.fill_between(xs, ys, alpha=0.10, color='#0072CE')

    ax.set_xlabel('Visit Number', color='#212B32', fontsize=10)
    ax.set_ylabel('Dropout Probability', color='#212B32', fontsize=10)
    ax.set_title('Dropout Risk Over Time', color='#003087', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors='#425563', labelsize=9)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#D8DDE0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#ffffff', edgecolor='#D8DDE0', labelcolor='#212B32', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.35, color='#D8DDE0')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#ffffff')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── WEB VIEWS ─────────────────────────────────────────────────────────────────

def index(request):
    """Public landing page with stats and feature overview."""
    stats = {
        'trials': Trial.objects.count(),
        'patients': Patient.objects.count(),
        'high_risk': PredictionResult.objects.filter(
            risk_tier__in=['high', 'critical']
        ).values('patient').distinct().count(),
    }
    return render(request, 'index.html', {'stats': stats})

@never_cache
def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            next_url = request.GET.get('next', 'dashboard')
            return redirect(next_url)
        messages.error(request, 'Invalid credentials. Please try again.')

    return render(request, 'login.html')


_FILTER_LABELS = {
    'all':        'All Patients',
    'active':     'Active Patients',
    'high_risk':  'High & Critical Risk',
    'critical':   'Critical Risk Only',
    'horizon_30': 'At-Risk Within 30 Days',
    'horizon_60': 'At-Risk Within 60 Days',
    'horizon_90': 'At-Risk Within 90 Days',
}


@login_required
def dashboard(request):
    """Main coordinator dashboard with KPIs, charts, and patient table."""
    trials = Trial.objects.all().order_by('name')
    selected_trial_id = request.GET.get('trial')
    filter_by = request.GET.get('filter', 'all')
    if filter_by not in _FILTER_LABELS:
        filter_by = 'all'

    if selected_trial_id:
        trial = get_object_or_404(Trial, pk=selected_trial_id)
    elif trials.exists():
        trial = trials.first()
    else:
        trial = None

    patients = []
    kpis = {}
    risk_dist = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
    forecast_data = []
    latest_forecast = None

    if trial:
        base_qs = (
            Patient.objects
            .filter(trial=trial)
            .prefetch_related('predictions')
            .order_by('-created_at')
        )

        total = base_qs.count()
        active = base_qs.filter(dropout_status=False).count()
        retention_rate = round((active / total * 100), 1) if total else 0

        for p in base_qs:
            pred = p.predictions.order_by('-prediction_timestamp').first()
            if pred:
                risk_dist[pred.risk_tier] = risk_dist.get(pred.risk_tier, 0) + 1

        high_risk = risk_dist['high'] + risk_dist['critical']
        critical = risk_dist['critical']

        latest_forecast = (
            CohortForecast.objects
            .filter(trial=trial)
            .order_by('-forecast_date')
            .first()
        )

        kpis = {
            'total': total,
            'active': active,
            'high_risk': high_risk,
            'critical': critical,
            'retention_rate': retention_rate,
            'predicted_60d': latest_forecast.predicted_dropouts_60d if latest_forecast else 'N/A',
        }

        # Apply table filter
        from django.db.models import Subquery, OuterRef, Q
        patient_qs = base_qs
        if filter_by == 'active':
            patient_qs = patient_qs.filter(dropout_status=False)
        elif filter_by in ('high_risk', 'critical', 'horizon_30', 'horizon_60', 'horizon_90'):
            pred_base = PredictionResult.objects.filter(patient_id=OuterRef('pk')).order_by('-prediction_timestamp')
            patient_qs = patient_qs.annotate(
                latest_risk=Subquery(pred_base.values('risk_tier')[:1]),
                latest_survival=Subquery(pred_base.values('survival_time_estimate')[:1]),
            )
            if filter_by == 'high_risk':
                patient_qs = patient_qs.filter(latest_risk__in=['high', 'critical'])
            elif filter_by == 'critical':
                patient_qs = patient_qs.filter(latest_risk='critical')
            elif filter_by == 'horizon_30':
                patient_qs = patient_qs.filter(
                    Q(latest_survival__lte=30) |
                    Q(latest_survival__isnull=True, latest_risk='critical')
                ).order_by('latest_survival')
            elif filter_by == 'horizon_60':
                patient_qs = patient_qs.filter(
                    Q(latest_survival__lte=60) |
                    Q(latest_survival__isnull=True, latest_risk__in=['high', 'critical'])
                ).order_by('latest_survival')
            elif filter_by == 'horizon_90':
                patient_qs = patient_qs.filter(
                    Q(latest_survival__lte=90) |
                    Q(latest_survival__isnull=True, latest_risk__in=['medium', 'high', 'critical'])
                ).order_by('latest_survival')

        # Paginated patient table
        paginator = Paginator(patient_qs.select_related('trial'), 25)
        page_num = request.GET.get('page', 1)
        patients = paginator.get_page(page_num)

        # Forecast line chart data (last 12 forecasts)
        forecasts = (
            CohortForecast.objects
            .filter(trial=trial)
            .order_by('forecast_date')
            .values('forecast_date', 'predicted_dropouts_60d',
                    'confidence_interval_lower', 'confidence_interval_upper')
        )[:12]
        forecast_data = [
            {
                'date': str(f['forecast_date']),
                'predicted': f['predicted_dropouts_60d'],
                'ci_lower': f['confidence_interval_lower'],
                'ci_upper': f['confidence_interval_upper'],
            }
            for f in forecasts
        ]

    return render(request, 'dashboard.html', {
        'trials': trials,
        'selected_trial': trial,
        'patients': patients,
        'kpis': kpis,
        'risk_dist': risk_dist,
        'forecast_data': json.dumps(forecast_data),
        'latest_forecast': latest_forecast,
        'models_trained': _models_trained(),
        'filter_by': filter_by,
        'filter_label': _FILTER_LABELS.get(filter_by, 'All Patients'),
        'is_horizon': filter_by.startswith('horizon_'),
    })


@login_required
def patient_detail(request, patient_id):
    """Individual patient risk view with charts, SHAP plot, and action log."""
    patient = get_object_or_404(
        Patient.objects.select_related('trial').prefetch_related(
            'visits', 'predictions__visit', 'coordinator_actions__coordinator'
        ),
        pk=patient_id
    )

    all_predictions = patient.predictions.select_related('visit').order_by('visit__visit_number')
    latest_pred = all_predictions.last()

    # Action form
    action_form = CoordinatorActionForm()
    if request.method == 'POST':
        action_form = CoordinatorActionForm(request.POST)
        if action_form.is_valid():
            action = action_form.save(commit=False)
            action.patient = patient
            action.coordinator = request.user
            action.save()
            messages.success(request, 'Intervention logged successfully.')
            return redirect('patient_detail', patient_id=patient_id)

    # Charts
    risk_timeline_b64 = _build_risk_timeline_b64(all_predictions)
    shap_b64 = ''
    survival_b64 = ''

    if latest_pred and _models_trained():
        try:
            from core.utils.shap_explainer import plot_waterfall
            top = latest_pred.shap_top_features()
            if top:
                shap_row = np.array([f['shap_value'] for f in top])
                feature_names = [f['feature'] for f in top]
                shap_b64 = plot_waterfall(
                    shap_row, 0.5,
                    feature_names=feature_names,
                    title=f'SHAP — Patient #{patient.pk}'
                )
        except Exception as e:
            logger.warning("SHAP plot failed: %s", e)

        try:
            latest_visit = patient.visits.order_by('-visit_number').first()
            if latest_visit:
                from core.utils.survival_model import plot_survival_curve
                features = _get_patient_features_dict(patient, latest_visit)
                survival_b64 = plot_survival_curve(features, title=f'Survival — Patient #{patient.pk}')
        except Exception as e:
            logger.warning("Survival plot failed: %s", e)

    coordinator_actions = patient.coordinator_actions.order_by('-action_date')

    return render(request, 'patient_detail.html', {
        'patient': patient,
        'latest_pred': latest_pred,
        'all_predictions': all_predictions,
        'risk_timeline_b64': risk_timeline_b64,
        'shap_b64': shap_b64,
        'survival_b64': survival_b64,
        'coordinator_actions': coordinator_actions,
        'action_form': action_form,
        'models_trained': _models_trained(),
    })


@login_required
def cohort_view(request, trial_id):
    """Cohort-level dropout forecast and Kaplan-Meier curves."""
    trial = get_object_or_404(Trial, pk=trial_id)

    latest_forecast = (
        CohortForecast.objects.filter(trial=trial).order_by('-forecast_date').first()
    )

    # KM curves from prediction data
    km_b64 = ''
    if _models_trained():
        try:
            from core.utils.survival_model import plot_km_by_risk_tier
            pred_qs = PredictionResult.objects.filter(patient__trial=trial).select_related('patient')
            if pred_qs.exists():
                rows = []
                seen = set()
                for p in pred_qs.order_by('patient_id', '-prediction_timestamp'):
                    if p.patient_id in seen:
                        continue
                    seen.add(p.patient_id)
                    rows.append({
                        'days_to_event': p.patient.days_to_event(),
                        'dropout_status': int(p.patient.dropout_status),
                        'risk_tier': p.risk_tier,
                    })
                if rows:
                    df = pd.DataFrame(rows)
                    km_b64 = plot_km_by_risk_tier(df)
        except Exception as e:
            logger.warning("KM plot failed: %s", e)

    forecasts = (
        CohortForecast.objects
        .filter(trial=trial)
        .order_by('forecast_date')
        .values('forecast_date', 'predicted_dropouts_30d', 'predicted_dropouts_60d',
                'predicted_dropouts_90d', 'confidence_interval_lower', 'confidence_interval_upper')
    )

    # MySQL-compatible: fetch ordered predictions, pick latest per patient in Python
    tier_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
    latest_by_patient = {}
    for pred in (PredictionResult.objects
                 .filter(patient__trial=trial)
                 .order_by('patient_id', '-prediction_timestamp')
                 .values('patient_id', 'risk_tier')):
        pid = pred['patient_id']
        if pid not in latest_by_patient:
            latest_by_patient[pid] = pred['risk_tier']
    for tier in latest_by_patient.values():
        if tier in tier_counts:
            tier_counts[tier] += 1

    return render(request, 'cohort.html', {
        'trial': trial,
        'latest_forecast': latest_forecast,
        'forecasts': json.dumps([
            {
                'date': str(f['forecast_date']),
                '30d': f['predicted_dropouts_30d'],
                '60d': f['predicted_dropouts_60d'],
                '90d': f['predicted_dropouts_90d'],
                'ci_lower': f['confidence_interval_lower'],
                'ci_upper': f['confidence_interval_upper'],
            }
            for f in forecasts
        ]),
        'km_b64': km_b64,
        'tier_counts': tier_counts,
        'models_trained': _models_trained(),
    })


@login_required
def upload_view(request):
    """CSV data upload with field validation and re-prediction trigger."""
    form = CSVUploadForm()
    upload_result = None

    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload_type = form.cleaned_data['upload_type']
            csv_file = form.cleaned_data['csv_file']
            trial = form.cleaned_data['trial']

            try:
                decoded = csv_file.read().decode('utf-8-sig')
                reader = csv.DictReader(io.StringIO(decoded))
                rows = list(reader)

                if upload_type == 'patients':
                    upload_result = _process_patient_csv(rows, trial)
                else:
                    upload_result = _process_visit_csv(rows, trial)

                messages.success(
                    request,
                    f"Upload complete: {upload_result['created']} created, "
                    f"{upload_result['skipped']} skipped, {upload_result['errors']} errors."
                )
            except Exception as e:
                messages.error(request, f'Upload failed: {e}')
                logger.error("CSV upload error: %s", e)

    return render(request, 'upload.html', {
        'form': form,
        'upload_result': upload_result,
        'add_form': AddPatientForm(),
        'active_tab': 'csv',
        'models_trained': _models_trained(),
    })


def _process_patient_csv(rows, trial):
    created = skipped = errors = 0
    required = {'age', 'gender', 'condition_severity', 'distance_to_site_km', 'enrollment_date'}
    for row in rows:
        if not required.issubset(row.keys()):
            errors += 1
            continue
        try:
            Patient.objects.create(
                trial=trial,
                age=int(row['age']),
                gender=row['gender'].upper()[:1],
                ethnicity=row.get('ethnicity', 'unknown').lower(),
                condition_severity=row['condition_severity'].lower(),
                distance_to_site_km=float(row['distance_to_site_km']),
                employment_status=row.get('employment_status', 'other').lower(),
                prior_dropout_history=row.get('prior_dropout_history', '0') in ('1', 'true', 'True'),
                enrollment_date=row['enrollment_date'],
                dropout_status=row.get('dropout_status', '0') in ('1', 'true', 'True'),
            )
            created += 1
        except Exception:
            errors += 1
    return {'created': created, 'skipped': skipped, 'errors': errors}


def _process_visit_csv(rows, trial):
    created = skipped = errors = 0
    required = {'patient_id', 'visit_number', 'visit_date', 'medication_adherence_score',
                'quality_of_life_score'}
    for row in rows:
        if not required.issubset(row.keys()):
            errors += 1
            continue
        try:
            patient = Patient.objects.get(pk=int(row['patient_id']), trial=trial)
            visit, _ = Visit.objects.get_or_create(
                patient=patient,
                visit_number=int(row['visit_number']),
                defaults={
                    'visit_date': row['visit_date'],
                    'adverse_events_count': int(row.get('adverse_events_count', 0)),
                    'missed_visits_to_date': int(row.get('missed_visits_to_date', 0)),
                    'medication_adherence_score': float(row['medication_adherence_score']),
                    'quality_of_life_score': float(row['quality_of_life_score']),
                    'days_since_last_visit': int(row.get('days_since_last_visit', 0)),
                }
            )
            _run_prediction_for_patient(patient, visit)
            created += 1
        except Patient.DoesNotExist:
            skipped += 1
        except Exception:
            errors += 1
    return {'created': created, 'skipped': skipped, 'errors': errors}


@login_required
@require_http_methods(['POST'])
def add_patient_view(request):
    """Manually add a single patient record (AJAX POST)."""
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    form = AddPatientForm(request.POST)
    if form.is_valid():
        patient = form.save()
        if is_ajax:
            return JsonResponse({
                'ok': True,
                'patient_id': patient.pk,
                'message': f'Patient #{patient.pk} added successfully.',
            })
        messages.success(request, f'Patient #{patient.pk} added successfully.')
        return redirect('upload')
    if is_ajax:
        return JsonResponse({'ok': False, 'errors': form.errors}, status=400)
    return render(request, 'upload.html', {
        'form': CSVUploadForm(),
        'add_form': form,
        'active_tab': 'add',
        'models_trained': _models_trained(),
    })


@login_required
def download_report(request, patient_id):
    """Generate and stream a PDF patient risk report."""
    patient = get_object_or_404(
        Patient.objects.select_related('trial').prefetch_related(
            'visits', 'predictions__visit', 'coordinator_actions'
        ),
        pk=patient_id
    )

    latest_pred = patient.predictions.order_by('-prediction_timestamp').first()
    actions = list(patient.coordinator_actions.order_by('-action_date'))

    risk_timeline_b64 = _build_risk_timeline_b64(
        patient.predictions.select_related('visit')
    )

    shap_b64 = survival_b64 = ''
    if latest_pred and _models_trained():
        try:
            from core.utils.shap_explainer import plot_waterfall
            top = latest_pred.shap_top_features()
            if top:
                shap_row = np.array([f['shap_value'] for f in top])
                feature_names = [f['feature'] for f in top]
                shap_b64 = plot_waterfall(shap_row, 0.5, feature_names=feature_names,
                                          title=f'SHAP — Patient #{patient.pk}')
        except Exception:
            pass

        try:
            latest_visit = patient.visits.order_by('-visit_number').first()
            if latest_visit:
                from core.utils.survival_model import plot_survival_curve
                features = _get_patient_features_dict(patient, latest_visit)
                survival_b64 = plot_survival_curve(features)
        except Exception:
            pass

    try:
        from core.utils.report_generator import generate_patient_report
        filepath = generate_patient_report(
            patient, latest_pred, actions,
            survival_b64=survival_b64,
            shap_b64=shap_b64,
            risk_timeline_b64=risk_timeline_b64,
        )
        with open(filepath, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/pdf')
            response['Content-Disposition'] = (
                f'attachment; filename="trialguard_patient_{patient_id}.pdf"'
            )
            return response
    except Exception as e:
        messages.error(request, f'Report generation failed: {e}')
        return redirect('patient_detail', patient_id=patient_id)


# ── DRF API VIEWS ──────────────────────────────────────────────────────────────

class TrialViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Trial.objects.all().order_by('-created_at')
    serializer_class = TrialSerializer
    permission_classes = [IsAuthenticated]
    search_fields = ['name', 'sponsor', 'therapeutic_area']
    ordering_fields = ['name', 'start_date', 'phase']


class TrialPatientsView(generics.ListAPIView):
    serializer_class = PatientListSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return (
            Patient.objects
            .filter(trial_id=self.kwargs['trial_id'])
            .select_related('trial')
            .prefetch_related('predictions')
            .order_by('-created_at')
        )


class PatientPredictionsView(generics.ListAPIView):
    serializer_class = PredictionResultSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return (
            PredictionResult.objects
            .filter(patient_id=self.kwargs['patient_id'])
            .select_related('visit')
            .order_by('-prediction_timestamp')
        )


class PatientSurvivalView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, patient_id):
        patient = get_object_or_404(Patient, pk=patient_id)
        if not _models_trained():
            return Response({'error': 'Models not yet trained.'}, status=503)

        latest_visit = patient.visits.order_by('-visit_number').first()
        if not latest_visit:
            return Response({'error': 'No visit data available.'}, status=404)

        try:
            from core.utils.survival_model import predict_survival
            features = _get_patient_features_dict(patient, latest_visit)
            return Response(predict_survival(features))
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class UploadPatientsView(generics.CreateAPIView):
    serializer_class = PatientUploadSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        many = isinstance(request.data, list)
        serializer = self.get_serializer(data=request.data, many=many)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class UploadVisitsView(generics.CreateAPIView):
    serializer_class = VisitUploadSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        many = isinstance(request.data, list)
        serializer = self.get_serializer(data=request.data, many=many)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class CohortForecastAPIView(generics.RetrieveAPIView):
    serializer_class = CohortForecastSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return get_object_or_404(
            CohortForecast.objects.filter(trial_id=self.kwargs['trial_id'])
            .order_by('-forecast_date')
        )


class HealthCheckView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        from django.db import connection
        db_ok = False
        try:
            connection.ensure_connection()
            db_ok = True
        except Exception:
            pass

        return Response({
            'status': 'healthy' if db_ok else 'degraded',
            'database': 'connected' if db_ok else 'disconnected',
            'models_trained': _models_trained(),
            'version': '1.0.0',
            'service': 'TrialGuard',
        })


# ── MODEL TRAINING ────────────────────────────────────────────────────────────

_TRAINING_LOCK = threading.Lock()
TRAINING_STATUS_PATH = ML_MODELS_DIR / 'training_status.json'


def _get_training_status():
    if TRAINING_STATUS_PATH.exists():
        try:
            return json.loads(TRAINING_STATUS_PATH.read_text())
        except Exception:
            pass
    return {'running': False, 'last_result': None}


def _run_training_background(optuna_trials: int):
    from django.core.management import call_command
    status = {'running': True, 'last_result': None}
    TRAINING_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_STATUS_PATH.write_text(json.dumps(status))
    try:
        call_command('train_models', optuna_trials=optuna_trials)
        status = {'running': False, 'last_result': 'success'}
    except Exception as e:
        logger.error("Background training failed: %s", e)
        status = {'running': False, 'last_result': 'error', 'error': str(e)}
    finally:
        TRAINING_STATUS_PATH.write_text(json.dumps(status))
        _TRAINING_LOCK.release()


@login_required
@require_http_methods(['POST', 'GET'])
def train_models_view(request):
    if not request.user.is_staff:
        return JsonResponse({'error': 'Staff access required.'}, status=403)

    if request.method == 'GET':
        return JsonResponse(_get_training_status())

    # POST — kick off training
    if not _TRAINING_LOCK.acquire(blocking=False):
        return JsonResponse({'error': 'Training already in progress.'}, status=409)

    optuna_trials = int(request.POST.get('optuna_trials', 20))
    t = threading.Thread(target=_run_training_background, args=(optuna_trials,), daemon=True)
    t.start()
    return JsonResponse({'status': 'started'})
