from django.contrib import admin
from django.utils.html import format_html
from .models import Trial, Patient, Visit, PredictionResult, CohortForecast, CoordinatorAction

RISK_COLOURS = {
    'low': '#009639', 'medium': '#FFB81C',
    'high': '#E65C00', 'critical': '#CC0000',
}


@admin.register(Trial)
class TrialAdmin(admin.ModelAdmin):
    list_display = ('name', 'sponsor', 'phase', 'therapeutic_area', 'target_enrollment',
                    'start_date', 'end_date', 'patient_count', 'retention')
    list_filter = ('phase', 'therapeutic_area')
    search_fields = ('name', 'sponsor', 'therapeutic_area')
    ordering = ('-created_at',)
    readonly_fields = ('created_at',)

    def patient_count(self, obj):
        return obj.patients.count()
    patient_count.short_description = 'Patients'

    def retention(self, obj):
        rate = obj.retention_rate()
        colour = '#009639' if rate >= 80 else ('#FFB81C' if rate >= 60 else '#E65C00')
        return format_html('<span style="color: {}; font-weight: bold;">{:.1f}%</span>', colour, rate)
    retention.short_description = 'Retention Rate'


class VisitInline(admin.TabularInline):
    model = Visit
    extra = 0
    fields = ('visit_number', 'visit_date', 'adverse_events_count',
              'missed_visits_to_date', 'medication_adherence_score', 'quality_of_life_score')
    readonly_fields = ('visit_number',)
    ordering = ('visit_number',)


class PredictionInline(admin.TabularInline):
    model = PredictionResult
    extra = 0
    fields = ('visit', 'risk_tier_display', 'dropout_probability', 'prediction_timestamp')
    readonly_fields = ('visit', 'risk_tier_display', 'dropout_probability', 'prediction_timestamp')
    ordering = ('-prediction_timestamp',)
    max_num = 5

    def risk_tier_display(self, obj):
        colour = RISK_COLOURS.get(obj.risk_tier, '#888')
        return format_html(
            '<span style="color: {}; font-weight: bold; text-transform: uppercase;">{}</span>',
            colour, obj.risk_tier
        )
    risk_tier_display.short_description = 'Risk Tier'


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('id', 'trial', 'age', 'gender', 'condition_severity',
                    'dropout_status', 'risk_tier_display', 'enrollment_date')
    list_filter = ('trial', 'gender', 'condition_severity', 'dropout_status',
                   'ethnicity', 'employment_status', 'prior_dropout_history')
    search_fields = ('id', 'trial__name')
    ordering = ('-created_at',)
    readonly_fields = ('created_at',)
    inlines = [VisitInline, PredictionInline]
    list_select_related = ('trial',)

    def risk_tier_display(self, obj):
        pred = obj.predictions.order_by('-prediction_timestamp').first()
        if not pred:
            return format_html('<span style="color: #888;">—</span>')
        colour = RISK_COLOURS.get(pred.risk_tier, '#888')
        return format_html(
            '<span style="color: {}; font-weight: bold; text-transform: uppercase;">{}</span>',
            colour, pred.risk_tier
        )
    risk_tier_display.short_description = 'Latest Risk'


@admin.register(Visit)
class VisitAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'visit_number', 'visit_date', 'adverse_events_count',
                    'missed_visits_to_date', 'medication_adherence_score', 'quality_of_life_score')
    list_filter = ('visit_date',)
    search_fields = ('patient__id',)
    ordering = ('patient', 'visit_number')
    list_select_related = ('patient', 'patient__trial')


@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'visit', 'risk_tier_coloured', 'probability_bar',
                    'hazard_ratio', 'survival_time_estimate', 'model_version', 'prediction_timestamp')
    list_filter = ('risk_tier', 'model_version', 'prediction_timestamp')
    search_fields = ('patient__id',)
    ordering = ('-prediction_timestamp',)
    readonly_fields = ('prediction_timestamp',)
    list_select_related = ('patient', 'visit', 'patient__trial')

    def risk_tier_coloured(self, obj):
        colour = RISK_COLOURS.get(obj.risk_tier, '#888')
        return format_html(
            '<span style="color: {}; font-weight: bold; text-transform: uppercase;">{}</span>',
            colour, obj.risk_tier
        )
    risk_tier_coloured.short_description = 'Risk Tier'

    def probability_bar(self, obj):
        pct = int(obj.dropout_probability * 100)
        colour = RISK_COLOURS.get(obj.risk_tier, '#888')
        return format_html(
            '<div style="background:#F0F4F5;border-radius:4px;width:120px;">'
            '<div style="background:{};width:{}%;padding:2px 4px;border-radius:4px;'
            'color:#ffffff;font-size:11px;text-align:center;">{:.1f}%</div></div>',
            colour, pct, obj.dropout_probability * 100
        )
    probability_bar.short_description = 'Probability'


@admin.register(CohortForecast)
class CohortForecastAdmin(admin.ModelAdmin):
    list_display = ('trial', 'forecast_date', 'predicted_dropouts_30d',
                    'predicted_dropouts_60d', 'predicted_dropouts_90d',
                    'confidence_interval_lower', 'confidence_interval_upper', 'created_at')
    list_filter = ('trial', 'forecast_date')
    search_fields = ('trial__name',)
    ordering = ('-forecast_date',)
    list_select_related = ('trial',)


@admin.register(CoordinatorAction)
class CoordinatorActionAdmin(admin.ModelAdmin):
    list_display = ('patient', 'action_type', 'action_date', 'outcome', 'coordinator', 'created_at')
    list_filter = ('action_type', 'outcome', 'action_date')
    search_fields = ('patient__id', 'notes')
    ordering = ('-action_date',)
    readonly_fields = ('created_at',)
    list_select_related = ('patient', 'patient__trial', 'coordinator')
