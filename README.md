---
title: TrialGuard
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
short_description: Clinical Trial Dropout Prediction Platform
---

# TrialGuard — Clinical Trial Dropout Prediction

> A three-stage ML pipeline (Cox survival analysis → XGBoost → SHAP) that predicts which
> clinical trial patients are at risk of dropping out, built end-to-end — synthetic data,
> real-world validation, and all — as a portfolio project.

TrialGuard gives trial coordinators a per-patient dropout risk score, a survival timeline,
and a plain-English explanation of *why* the model flagged that patient. It's a Django web
app backed by Cox Proportional Hazards (lifelines), XGBoost (Optuna-tuned), and SHAP.

> Built by [SKMMT](https://skmmt.rootexception.com/).
> [Live demo](https://sheikhkmmtahmid-trialguard.hf.space/) · [How it was built & validated](/methodology/)

---

## Read this before the metrics

I trained this two ways: on synthetic data I generated myself, and on 12,435 real,
de-identified patients pulled from 28 real clinical trials and studies (Project Data
Sphere, ImmPort, PPMI, MUSIC). The synthetic model performs well. The real-data model,
tested honestly — held out on hospitals it never trained on, not just a random shuffle —
performs close to chance.

I'm leading with that instead of hiding it because how a model's limits get found and
explained is the point of this project, not a footnote. I've since more than doubled the
real dataset and retrained everything specifically to test whether more real data would
close that gap — it didn't (see Results below). The full breakdown, including the
bugs I hit while combining 28 differently-structured real datasets, is at
[`/methodology/`](/methodology/) once the app is running, or in
[`docs/model_training_log.md`](docs/model_training_log.md) if you're reading the code
directly.

---

## Results

Both models below were tuned with the same effort (Optuna, 50 trials, cross-validated),
so this is an even comparison — not a case of one getting more attention than the other.

| Metric | Real Data | Synthetic Data | Note |
|---|---|---|---|
| Cox concordance index | 0.551 (on hospitals never seen in training) | 0.734 | 50 = coin flip, 100 = perfect ranking |
| XGBoost ROC-AUC | 0.550 (on hospitals never seen in training) | 0.732 | same scale |
| XGBoost ROC-AUC (random split) | 0.755 | — | shown for comparison — this is the flattering number that hides the generalization gap |
| Brier score (calibration) | 0.114 | 0.042 | lower is better |
| SHAP explanation robustness | 0.903 | 0.975 | does the explanation survive small, realistic noise in the input? max 1.0 |
| Cox risk factors statistically significant | 1 of 3 (p = 0.997, 0.136, 0.0004) | 3 of 5 (all p < 0.01) | is the pattern real or just noise? |

The real-data drop from 0.755 (random split) to 0.550 (new-hospital holdout) is the actual
finding of this project: with real trial data, a model can partly "cheat" by learning which
hospital a patient came from, since each hospital has its own baseline dropout rate. Remove
that shortcut and there isn't enough real signal yet to transfer to a hospital the model
hasn't seen — a known, still-unsolved problem in clinical ML generally, not something unique
to this implementation. I retrained on more than double the real data (up from 6,476
patients / 23 studies) specifically to test whether more data would close this gap; it
didn't move the new-hospital number, though it did make the model's reasoning measurably
steadier and turned one more Cox risk factor statistically real. Full explanation at
[`/methodology/`](/methodology/).

The live demo currently serves the **synthetic-trained model**, since it's the one with
clean signal to actually demonstrate the pipeline end to end.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TrialGuard Platform                      │
├─────────────────┬───────────────────┬───────────────────────┤
│  Django Web UI  │   REST API (DRF)  │   Admin Panel         │
│  ─────────────  │  ─────────────    │  ─────────────        │
│  Landing Page   │  JWT Auth         │  Full ORM admin       │
│  Methodology    │  /api/trials/     │  Risk tier colouring  │
│  Dashboard      │  /api/patients/   │  Inline predictions   │
│  Patient Detail │  /api/health/     │                       │
│  Cohort Forecast│  /api/docs/       │                       │
│  CSV Upload     │                   │                       │
└────────┬────────┴────────┬──────────┴───────────────────────┘
         │                 │
         ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML Pipeline                              │
├──────────────┬──────────────┬───────────────────────────────┤
│ Cox PH Model │ XGBoost      │ SHAP TreeExplainer            │
│ (lifelines)  │ (+ Optuna)   │ (per-patient waterfall)       │
│              │              │                               │
│ hazard_ratio │ dropout_prob │ top_5_drivers JSON            │
│ survival_30d │ risk_tier    │ plain-english explanation     │
│ survival_60d │ 0–1 score    │ beeswarm global summary       │
│ survival_90d │              │                               │
└──────────────┴──────────────┴───────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  MySQL-compatible Database (Django ORM, tested against TiDB)│
│  trials · patients · visits · prediction_results            │
│  cohort_forecasts · coordinator_actions                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- MySQL 8.0+ or a MySQL-compatible database (this project was run against TiDB)
- A virtual environment

### 2. Installation

```bash
git clone <repo-url>
cd "Trial Guard"

# Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup

```bash
mysql -u root -p
CREATE DATABASE trialguard_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'trialguard_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON trialguard_db.* TO 'trialguard_user'@'localhost';
FLUSH PRIVILEGES;
```

### 4. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your DB credentials and SECRET_KEY
```

### 5. Django Setup

```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py collectstatic --no-input
```

### 6. Generate Synthetic Data & Train Models

```bash
python manage.py generate_synthetic_data --n 5000
python manage.py train_models --optuna-trials 50
```

The real-data training pipeline (data harmonization, grouped cross-validation, and the
leave-studies-out validation behind the numbers above) lives in `scripts/real_data/` and
isn't wired into a management command — it was a one-time investigation, not something the
app re-runs on demand. See `docs/model_training_log.md` for the exact steps.

### 7. Run Development Server

```bash
python manage.py runserver
```

Open [http://localhost:8000](http://localhost:8000), and [/methodology/](http://localhost:8000/methodology/)
for the write-up on how this was built and validated.

### 8. Generate Favicons

```bash
python static/img/make_icons_stdlib.py
```

---

## Production Deployment (Gunicorn + WhiteNoise)

```bash
# Set DEBUG=False, ALLOWED_HOSTS in .env
gunicorn trialguard.wsgi:application \
  --workers 4 \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile -
```

---

## API Documentation

Full OpenAPI schema available at `/api/docs/` (Swagger UI) and `/api/redoc/`.

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/trials/` | List all clinical trials |
| GET | `/api/trials/{id}/patients/` | Patients for a trial |
| GET | `/api/patients/{id}/predictions/` | Prediction history |
| GET | `/api/patients/{id}/survival/` | Cox survival estimates |
| POST | `/api/upload/patients/` | Bulk patient upload |
| POST | `/api/upload/visits/` | Bulk visit upload |
| GET | `/api/cohort/{id}/forecast/` | 30/60/90d forecast |
| GET | `/api/health/` | Health check |
| POST | `/api/token/` | Obtain JWT token |
| POST | `/api/token/refresh/` | Refresh JWT token |

### Authentication

```bash
curl -X POST http://localhost:8000/api/token/ \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "yourpassword"}'

curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/trials/
```

---

## Data Sources

| Source | Type | Usage |
|---|---|---|
| Custom synthetic generator (`core/utils/data_pipeline.py`) | Synthetic | Powers the live demo — 12,435+ generated patients with intentional, known dropout relationships |
| [Project Data Sphere](https://www.projectdatasphere.org/) | Real (public) | 18 real oncology trials, used for honest validation, not training the live model |
| [ImmPort](https://www.immport.org/) | Real (public) | 5 real NIH/NIAID-funded trials (MS + type 1 diabetes), same use |
| [PPMI](https://www.ppmi-info.org/) (via IDA/LONI) | Real (public) | 4 real cohorts (Parkinson's, at-risk/prodromal, healthy control, SWEDD), same use |
| MUSIC (via PhysioNet) | Real (public) | 1 real heart-failure cohort, same use |
| AACT (ClinicalTrials.gov mirror) | Real (public) | Used only to sanity-check synthetic dropout rates against real aggregate statistics, not patient-level |

28 real trials/studies, 12,435 real patients total, all de-identified and pulled from
public repositories. Broken down by sponsor:

| Sponsor / Program | Studies | Area | Source |
|---|---|---|---|
| Amgen | 5 | 3 colorectal, 1 head & neck, 1 small-cell lung | Project Data Sphere |
| Eli Lilly | 3 | breast, non-small-cell lung, small-cell lung | Project Data Sphere |
| EMD Serono | 3 | 2 glioma, 1 pancreatic | Project Data Sphere |
| G1 Therapeutics | 3 | small-cell lung | Project Data Sphere |
| Pfizer | 1 trial, 2 cohorts | small-cell lung | Project Data Sphere |
| Clovis Oncology | 1 | pancreatic | Project Data Sphere |
| Alliance for Clinical Trials in Oncology | 1 | breast (CALGB 40502) | Project Data Sphere |
| NIH / NIAID, incl. Immune Tolerance Network | 5 | 2 multiple sclerosis, 3 new-onset type 1 diabetes | ImmPort |
| Michael J. Fox Foundation (PPMI) | 4 cohorts | Parkinson's, at-risk/prodromal, healthy control, SWEDD | IDA / LONI (USC) |
| MUSIC study group | 1 | heart failure | PhysioNet |

Every file — patient data and documentation alike — was read and its field meanings
verified before use; details in `docs/harmonized_dataset_build_log.md`,
`docs/ppmi_data_build_log.md`, and `docs/data_sourcing.md`.

---

## Tech Stack

![Django](https://img.shields.io/badge/Django-4.2-092E20?style=flat&logo=django)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-FF6600?style=flat)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=flat&logo=mysql)
![DRF](https://img.shields.io/badge/DRF-3.15-red?style=flat)

| Component | Technology |
|---|---|
| Web Framework | Django 4.2 |
| REST API | Django REST Framework + SimpleJWT |
| ML — Classification | XGBoost 2.1 + Optuna |
| ML — Survival Analysis | lifelines CoxPHFitter |
| ML — Explainability | SHAP TreeExplainer |
| Synthetic Data | Custom generator (`core/utils/data_pipeline.py`) |
| PDF Reports | ReportLab |
| Database | MySQL-compatible (tested against TiDB) |
| Static Files | WhiteNoise |
| Production Server | Gunicorn |
| Frontend Charts | Chart.js 4.4 |
| Typography | Fira Sans + Lato (Google Fonts) |

---

## Project Structure

```
Trial Guard/
├── trialguard/          # Django project settings
│   ├── settings.py      # Environment-variable driven config
│   ├── urls.py          # Root URL routing
│   └── wsgi.py / asgi.py
├── core/                # Main application
│   ├── models.py        # Trial, Patient, Visit, PredictionResult, etc.
│   ├── views.py         # Web views + DRF API views (incl. /methodology/)
│   ├── admin.py         # Admin interface with risk colour coding
│   ├── forms.py         # CSV upload + coordinator action forms
│   ├── serializers.py   # DRF serializers
│   ├── urls.py          # App URL patterns
│   ├── utils/
│   │   ├── data_pipeline.py    # Feature engineering + synthetic generator
│   │   ├── survival_model.py   # Cox PH model
│   │   ├── xgboost_model.py    # XGBoost + Optuna
│   │   ├── shap_explainer.py   # SHAP TreeExplainer
│   │   └── report_generator.py # PDF reports (ReportLab)
│   ├── management/commands/
│   │   ├── generate_synthetic_data.py
│   │   ├── train_models.py
│   │   └── validate_against_aact.py
│   └── migrations/
├── ml_models/            # Trained model artifacts actually served by the app (.pkl)
│   ├── cox_model.pkl
│   ├── xgb_model.pkl
│   ├── shap_explainer.pkl
│   └── scaler.pkl
├── scripts/real_data/    # Real-data acquisition, harmonization, and validation work
│                         # (not wired into the app — a one-time investigation, see
│                         #  docs/model_training_log.md)
├── data/                 # Real, de-identified patient data (Project Data Sphere, ImmPort,
│                         #  PPMI, MUSIC)
├── docs/                 # Build logs, validation methodology, findings
├── templates/            # Django HTML templates
│   ├── base.html         # Navbar + footer
│   ├── index.html        # Public landing page
│   ├── methodology.html  # How this was built & validated (public)
│   ├── dashboard.html    # Coordinator dashboard
│   ├── patient_detail.html
│   ├── cohort.html       # Cohort forecast view
│   ├── upload.html       # CSV data import
│   └── login.html
├── static/
│   ├── css/main.css      # Design system (CSS custom properties)
│   ├── js/dashboard.js   # Navigation + Chart.js helpers
│   └── img/
│       ├── logo.svg
│       ├── favicon.ico
│       └── apple-touch-icon.png
├── media/reports/        # Generated PDF reports
├── evaluation_results.json   # Live model's metrics (regenerated on each train)
├── model_card.md
├── requirements.txt
└── .env.example
```

---

## Colour Palette (NHS-Inspired)

| Token | Hex | Usage |
|---|---|---|
| `--primary` | `#003087` | Primary brand, navbar, headers |
| `--accent` | `#0072CE` | Links, buttons, active states |
| `--accent-light` | `#41B6E6` | Secondary accents, icons |

---

## Risk Tier System

| Tier | Colour | Meaning |
|---|---|---|
| Low | `#009639` | Routine monitoring |
| Medium | `#FFB81C` | Check-in recommended |
| High | `#E65C00` | Immediate outreach |
| Critical | `#CC0000` | Emergency intervention |

---

## Model Card

See [model_card.md](model_card.md) for model documentation — note it predates the real-data
validation work described above and in [`/methodology/`](/methodology/), which is the more
current source on what the model can and can't actually do.

---

## License

Proprietary. All rights reserved.

---

**Built by [SKMMT](https://skmmt.rootexception.com/)** · TrialGuard © 2026
