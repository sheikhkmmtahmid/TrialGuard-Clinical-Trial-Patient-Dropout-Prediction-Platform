from django import forms
from django.utils import timezone
from .models import CoordinatorAction, Patient, Trial


class CSVUploadForm(forms.Form):
    upload_type = forms.ChoiceField(
        choices=[('patients', 'Patient Data'), ('visits', 'Visit Data')],
        widget=forms.Select(attrs={'class': 'form-select'}),
    )
    csv_file = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'accept': '.csv', 'class': 'form-file-input'}),
        help_text='CSV file. Max 10 MB. Required columns depend on upload type.',
    )
    trial = forms.ModelChoiceField(
        queryset=Trial.objects.all(),
        empty_label='— Select Trial —',
        widget=forms.Select(attrs={'class': 'form-select'}),
    )

    def clean_csv_file(self):
        f = self.cleaned_data['csv_file']
        if not f.name.endswith('.csv'):
            raise forms.ValidationError('Only .csv files are accepted.')
        if f.size > 10 * 1024 * 1024:
            raise forms.ValidationError('File size must not exceed 10 MB.')
        return f


class AddPatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = [
            'trial', 'age', 'gender', 'ethnicity', 'condition_severity',
            'distance_to_site_km', 'employment_status', 'prior_dropout_history',
            'enrollment_date',
        ]
        widgets = {
            'trial': forms.Select(attrs={'class': 'form-select'}),
            'age': forms.NumberInput(attrs={'class': 'form-input', 'min': 18, 'max': 110}),
            'gender': forms.Select(attrs={'class': 'form-select'}),
            'ethnicity': forms.Select(attrs={'class': 'form-select'}),
            'condition_severity': forms.Select(attrs={'class': 'form-select'}),
            'distance_to_site_km': forms.NumberInput(attrs={
                'class': 'form-input', 'step': '0.1', 'min': '0', 'placeholder': 'km',
            }),
            'employment_status': forms.Select(attrs={'class': 'form-select'}),
            'prior_dropout_history': forms.CheckboxInput(attrs={'class': 'form-checkbox'}),
            'enrollment_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-input'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['enrollment_date'].initial = timezone.now().date()
        self.fields['prior_dropout_history'].required = False


class CoordinatorActionForm(forms.ModelForm):
    class Meta:
        model = CoordinatorAction
        fields = ['action_type', 'notes', 'action_date', 'outcome']
        widgets = {
            'action_type': forms.Select(attrs={'class': 'form-select'}),
            'notes': forms.Textarea(attrs={'class': 'form-textarea', 'rows': 3,
                                           'placeholder': 'Describe the intervention taken...'}),
            'action_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-input'}),
            'outcome': forms.Select(attrs={'class': 'form-select'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['action_date'].initial = timezone.now().date()
