from django.db import models
from django.contrib.auth.models import User


class Trial(models.Model):
    PHASE_CHOICES = [
        ('I', 'Phase I'), ('II', 'Phase II'),
        ('III', 'Phase III'), ('IV', 'Phase IV'), ('N/A', 'N/A'),
    ]

    name = models.CharField(max_length=255)
    sponsor = models.CharField(max_length=255)
    phase = models.CharField(max_length=5, choices=PHASE_CHOICES)
    therapeutic_area = models.CharField(max_length=100)
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    target_enrollment = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'trials'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.phase})"

    def active_patient_count(self):
        return self.patients.filter(dropout_status=False).count()

    def retention_rate(self):
        total = self.patients.count()
        if total == 0:
            return 100.0
        active = self.patients.filter(dropout_status=False).count()
        return round((active / total) * 100, 1)


class Patient(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'), ('F', 'Female'), ('O', 'Other'), ('U', 'Unknown'),
    ]
    ETHNICITY_CHOICES = [
        ('white', 'White'), ('black', 'Black or African American'),
        ('hispanic', 'Hispanic or Latino'), ('asian', 'Asian'),
        ('other', 'Other'), ('unknown', 'Unknown'),
    ]
    SEVERITY_CHOICES = [
        ('mild', 'Mild'), ('moderate', 'Moderate'), ('severe', 'Severe'),
    ]
    EMPLOYMENT_CHOICES = [
        ('employed', 'Employed'), ('unemployed', 'Unemployed'),
        ('retired', 'Retired'), ('student', 'Student'), ('other', 'Other'),
    ]

    trial = models.ForeignKey(Trial, on_delete=models.CASCADE, related_name='patients')
    age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    ethnicity = models.CharField(max_length=20, choices=ETHNICITY_CHOICES, default='unknown')
    condition_severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    distance_to_site_km = models.FloatField()
    employment_status = models.CharField(max_length=15, choices=EMPLOYMENT_CHOICES, default='other')
    prior_dropout_history = models.BooleanField(default=False)
    enrollment_date = models.DateField()
    dropout_date = models.DateField(null=True, blank=True)
    dropout_status = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'patients'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['trial', 'dropout_status']),
            models.Index(fields=['enrollment_date']),
        ]

    def __str__(self):
        return f"Patient #{self.pk} — {self.trial.name}"

    def days_to_event(self):
        from django.utils import timezone
        end_date = self.dropout_date or timezone.now().date()
        return max((end_date - self.enrollment_date).days, 1)

    def latest_prediction(self):
        return self.predictions.select_related('visit').order_by('-prediction_timestamp').first()


class Visit(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='visits')
    visit_number = models.IntegerField()
    visit_date = models.DateField()
    adverse_events_count = models.IntegerField(default=0)
    missed_visits_to_date = models.IntegerField(default=0)
    medication_adherence_score = models.FloatField()
    quality_of_life_score = models.FloatField()
    days_since_last_visit = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'visits'
        ordering = ['patient', 'visit_number']
        unique_together = [['patient', 'visit_number']]
        indexes = [
            models.Index(fields=['patient', 'visit_date']),
        ]

    def __str__(self):
        return f"Visit #{self.visit_number} — Patient #{self.patient_id}"


class PredictionResult(models.Model):
    RISK_TIER_CHOICES = [
        ('low', 'Low'), ('medium', 'Medium'),
        ('high', 'High'), ('critical', 'Critical'),
    ]

    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='predictions')
    visit = models.ForeignKey(Visit, on_delete=models.CASCADE, related_name='predictions')
    dropout_probability = models.FloatField()
    risk_tier = models.CharField(max_length=10, choices=RISK_TIER_CHOICES)
    shap_values_json = models.JSONField(default=dict)
    survival_time_estimate = models.FloatField(null=True, blank=True)
    hazard_ratio = models.FloatField(null=True, blank=True)
    prediction_timestamp = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50, default='1.0.0')

    class Meta:
        db_table = 'prediction_results'
        ordering = ['-prediction_timestamp']
        indexes = [
            models.Index(fields=['patient', 'prediction_timestamp']),
            models.Index(fields=['risk_tier']),
        ]

    def __str__(self):
        return f"Prediction: Patient #{self.patient_id} — {self.risk_tier} ({self.dropout_probability:.2%})"

    @staticmethod
    def get_risk_tier(probability: float) -> str:
        if probability <= 0.30:
            return 'low'
        elif probability <= 0.55:
            return 'medium'
        elif probability <= 0.75:
            return 'high'
        return 'critical'

    def shap_top_features(self):
        if not self.shap_values_json:
            return []
        features = self.shap_values_json.get('top_features', [])
        return features[:5]

    def plain_english_explanation(self):
        tier_label = self.risk_tier.upper()
        top = self.shap_top_features()
        if not top:
            return f"This patient's dropout risk is {tier_label} based on recent clinical data."
        drivers = ', '.join(f['feature'].replace('_', ' ') for f in top[:3])
        return (
            f"This patient's dropout risk is {tier_label} primarily due to: {drivers}. "
            f"Dropout probability: {self.dropout_probability:.1%}."
        )


class CohortForecast(models.Model):
    trial = models.ForeignKey(Trial, on_delete=models.CASCADE, related_name='forecasts')
    forecast_date = models.DateField()
    predicted_dropouts_30d = models.IntegerField()
    predicted_dropouts_60d = models.IntegerField()
    predicted_dropouts_90d = models.IntegerField()
    confidence_interval_lower = models.FloatField()
    confidence_interval_upper = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'cohort_forecasts'
        ordering = ['-forecast_date']

    def __str__(self):
        return f"Forecast: {self.trial.name} on {self.forecast_date}"


class CoordinatorAction(models.Model):
    ACTION_TYPE_CHOICES = [
        ('phone_call', 'Phone Call'),
        ('email', 'Email'),
        ('visit_reminder', 'Visit Reminder'),
        ('transportation_arranged', 'Transportation Arranged'),
        ('counseling_referral', 'Counseling Referral'),
        ('dose_adjustment', 'Dose Adjustment'),
        ('other', 'Other'),
    ]
    OUTCOME_CHOICES = [
        ('retained', 'Patient Retained'),
        ('dropped_out', 'Patient Dropped Out'),
        ('pending', 'Pending'),
        ('no_response', 'No Response'),
    ]

    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='coordinator_actions')
    coordinator = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='actions'
    )
    action_type = models.CharField(max_length=30, choices=ACTION_TYPE_CHOICES)
    notes = models.TextField(blank=True)
    action_date = models.DateField()
    outcome = models.CharField(max_length=20, choices=OUTCOME_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'coordinator_actions'
        ordering = ['-action_date']

    def __str__(self):
        return f"Action: Patient #{self.patient_id} — {self.action_type} on {self.action_date}"
