from rest_framework import serializers
from .models import Trial, Patient, Visit, PredictionResult, CohortForecast, CoordinatorAction


class TrialSerializer(serializers.ModelSerializer):
    retention_rate = serializers.SerializerMethodField()
    active_patients = serializers.SerializerMethodField()

    class Meta:
        model = Trial
        fields = '__all__'

    def get_retention_rate(self, obj):
        return obj.retention_rate()

    def get_active_patients(self, obj):
        return obj.active_patient_count()


class VisitSerializer(serializers.ModelSerializer):
    class Meta:
        model = Visit
        exclude = ['created_at']


class PredictionResultSerializer(serializers.ModelSerializer):
    top_features = serializers.SerializerMethodField()
    explanation = serializers.SerializerMethodField()

    class Meta:
        model = PredictionResult
        fields = '__all__'

    def get_top_features(self, obj):
        return obj.shap_top_features()

    def get_explanation(self, obj):
        return obj.plain_english_explanation()


class PatientListSerializer(serializers.ModelSerializer):
    latest_risk_tier = serializers.SerializerMethodField()
    latest_probability = serializers.SerializerMethodField()
    trial_name = serializers.CharField(source='trial.name', read_only=True)

    class Meta:
        model = Patient
        fields = '__all__'

    def get_latest_risk_tier(self, obj):
        pred = obj.predictions.order_by('-prediction_timestamp').first()
        return pred.risk_tier if pred else None

    def get_latest_probability(self, obj):
        pred = obj.predictions.order_by('-prediction_timestamp').first()
        return pred.dropout_probability if pred else None


class PatientDetailSerializer(serializers.ModelSerializer):
    visits = VisitSerializer(many=True, read_only=True)
    predictions = PredictionResultSerializer(many=True, read_only=True)
    trial = TrialSerializer(read_only=True)

    class Meta:
        model = Patient
        fields = '__all__'


class PatientUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        exclude = ['created_at']


class VisitUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Visit
        exclude = ['created_at']


class CohortForecastSerializer(serializers.ModelSerializer):
    trial_name = serializers.CharField(source='trial.name', read_only=True)

    class Meta:
        model = CohortForecast
        fields = '__all__'


class CoordinatorActionSerializer(serializers.ModelSerializer):
    coordinator_name = serializers.SerializerMethodField()

    class Meta:
        model = CoordinatorAction
        fields = '__all__'

    def get_coordinator_name(self, obj):
        if obj.coordinator:
            return obj.coordinator.get_full_name() or obj.coordinator.username
        return 'System'
