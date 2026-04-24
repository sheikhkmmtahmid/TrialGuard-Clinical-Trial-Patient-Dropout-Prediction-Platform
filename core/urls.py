from django.urls import path, include
from django.contrib.auth import views as auth_views
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from . import views

router = DefaultRouter()
router.register(r'trials', views.TrialViewSet, basename='trial')

urlpatterns = [
    # Public
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),

    # Authenticated web views
    path('dashboard/', views.dashboard, name='dashboard'),
    path('patient/<int:patient_id>/', views.patient_detail, name='patient_detail'),
    path('patient/<int:patient_id>/report/', views.download_report, name='download_report'),
    path('cohort/<int:trial_id>/', views.cohort_view, name='cohort'),
    path('upload/', views.upload_view, name='upload'),
    path('upload/add-patient/', views.add_patient_view, name='add_patient'),
    path('admin/train/', views.train_models_view, name='train_models'),

    # REST API
    path('api/', include(router.urls)),
    path('api/trials/<int:trial_id>/patients/', views.TrialPatientsView.as_view(), name='api-trial-patients'),
    path('api/patients/<int:patient_id>/predictions/', views.PatientPredictionsView.as_view(), name='api-patient-predictions'),
    path('api/patients/<int:patient_id>/survival/', views.PatientSurvivalView.as_view(), name='api-patient-survival'),
    path('api/upload/patients/', views.UploadPatientsView.as_view(), name='api-upload-patients'),
    path('api/upload/visits/', views.UploadVisitsView.as_view(), name='api-upload-visits'),
    path('api/cohort/<int:trial_id>/forecast/', views.CohortForecastAPIView.as_view(), name='api-cohort-forecast'),
    path('api/health/', views.HealthCheckView.as_view(), name='api-health'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
