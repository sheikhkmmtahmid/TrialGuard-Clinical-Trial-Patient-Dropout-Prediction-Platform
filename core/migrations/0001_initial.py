import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Trial',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=255)),
                ('sponsor', models.CharField(max_length=255)),
                ('phase', models.CharField(choices=[('I','Phase I'),('II','Phase II'),('III','Phase III'),('IV','Phase IV'),('N/A','N/A')], max_length=5)),
                ('therapeutic_area', models.CharField(max_length=100)),
                ('start_date', models.DateField()),
                ('end_date', models.DateField(blank=True, null=True)),
                ('target_enrollment', models.IntegerField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={'db_table': 'trials', 'ordering': ['-created_at']},
        ),
        migrations.CreateModel(
            name='Patient',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ('age', models.IntegerField()),
                ('gender', models.CharField(choices=[('M','Male'),('F','Female'),('O','Other'),('U','Unknown')], max_length=1)),
                ('ethnicity', models.CharField(choices=[('white','White'),('black','Black or African American'),('hispanic','Hispanic or Latino'),('asian','Asian'),('other','Other'),('unknown','Unknown')], default='unknown', max_length=20)),
                ('condition_severity', models.CharField(choices=[('mild','Mild'),('moderate','Moderate'),('severe','Severe')], max_length=10)),
                ('distance_to_site_km', models.FloatField()),
                ('employment_status', models.CharField(choices=[('employed','Employed'),('unemployed','Unemployed'),('retired','Retired'),('student','Student'),('other','Other')], default='other', max_length=15)),
                ('prior_dropout_history', models.BooleanField(default=False)),
                ('enrollment_date', models.DateField()),
                ('dropout_date', models.DateField(blank=True, null=True)),
                ('dropout_status', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('trial', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='patients', to='core.trial')),
            ],
            options={'db_table': 'patients', 'ordering': ['-created_at']},
        ),
        migrations.CreateModel(
            name='Visit',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ('visit_number', models.IntegerField()),
                ('visit_date', models.DateField()),
                ('adverse_events_count', models.IntegerField(default=0)),
                ('missed_visits_to_date', models.IntegerField(default=0)),
                ('medication_adherence_score', models.FloatField()),
                ('quality_of_life_score', models.FloatField()),
                ('days_since_last_visit', models.IntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='visits', to='core.patient')),
            ],
            options={'db_table': 'visits', 'ordering': ['patient', 'visit_number']},
        ),
        migrations.CreateModel(
            name='PredictionResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ('dropout_probability', models.FloatField()),
                ('risk_tier', models.CharField(choices=[('low','Low'),('medium','Medium'),('high','High'),('critical','Critical')], max_length=10)),
                ('shap_values_json', models.JSONField(default=dict)),
                ('survival_time_estimate', models.FloatField(blank=True, null=True)),
                ('hazard_ratio', models.FloatField(blank=True, null=True)),
                ('prediction_timestamp', models.DateTimeField(auto_now_add=True)),
                ('model_version', models.CharField(default='1.0.0', max_length=50)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='predictions', to='core.patient')),
                ('visit', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='predictions', to='core.visit')),
            ],
            options={'db_table': 'prediction_results', 'ordering': ['-prediction_timestamp']},
        ),
        migrations.CreateModel(
            name='CohortForecast',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ('forecast_date', models.DateField()),
                ('predicted_dropouts_30d', models.IntegerField()),
                ('predicted_dropouts_60d', models.IntegerField()),
                ('predicted_dropouts_90d', models.IntegerField()),
                ('confidence_interval_lower', models.FloatField()),
                ('confidence_interval_upper', models.FloatField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('trial', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='forecasts', to='core.trial')),
            ],
            options={'db_table': 'cohort_forecasts', 'ordering': ['-forecast_date']},
        ),
        migrations.CreateModel(
            name='CoordinatorAction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ('action_type', models.CharField(choices=[('phone_call','Phone Call'),('email','Email'),('visit_reminder','Visit Reminder'),('transportation_arranged','Transportation Arranged'),('counseling_referral','Counseling Referral'),('dose_adjustment','Dose Adjustment'),('other','Other')], max_length=30)),
                ('notes', models.TextField(blank=True)),
                ('action_date', models.DateField()),
                ('outcome', models.CharField(choices=[('retained','Patient Retained'),('dropped_out','Patient Dropped Out'),('pending','Pending'),('no_response','No Response')], default='pending', max_length=20)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='coordinator_actions', to='core.patient')),
                ('coordinator', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='actions', to=settings.AUTH_USER_MODEL)),
            ],
            options={'db_table': 'coordinator_actions', 'ordering': ['-action_date']},
        ),
        migrations.AddIndex(
            model_name='patient',
            index=models.Index(fields=['trial', 'dropout_status'], name='patients_trial_dropout_idx'),
        ),
        migrations.AddIndex(
            model_name='patient',
            index=models.Index(fields=['enrollment_date'], name='patients_enrollment_idx'),
        ),
        migrations.AlterUniqueTogether(
            name='visit',
            unique_together={('patient', 'visit_number')},
        ),
        migrations.AddIndex(
            model_name='visit',
            index=models.Index(fields=['patient', 'visit_date'], name='visits_patient_date_idx'),
        ),
        migrations.AddIndex(
            model_name='predictionresult',
            index=models.Index(fields=['patient', 'prediction_timestamp'], name='pred_patient_ts_idx'),
        ),
        migrations.AddIndex(
            model_name='predictionresult',
            index=models.Index(fields=['risk_tier'], name='pred_risk_tier_idx'),
        ),
    ]
