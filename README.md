---
title: TrialGuard
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
short_description: Clinical Trial Dropout Prediction Platform
---

# TrialGuard — 60-Day Early Warning for Clinical Trial Retention

> **Predict patient dropout before it happens. Protect your trial. Save $40K per dropout.**

TrialGuard is a production-ready clinical AI platform that gives trial coordinators
a 60-day early warning when patients are at risk of dropping out. Powered by
XGBoost gradient boosting, Cox Proportional Hazards survival analysis, and
SHAP explainability — deployed as a Django web service.

---

## Key Metrics

| Metric | Achieved | Technology |
|---|---|---|
| XGBoost ROC-AUC | **0.9914** | XGBoost + Optuna (50 trials) |
| XGBoost F1 Score | **0.8842** | Balanced F1 on held-out test |
| Cox Concordance Index | **0.9119** | lifelines CoxPHFitter |
| Brier Score | **0.0253** | Excellent probability calibration |
| SHAP Stability | **0.8348** | TreeSHAP pairwise cosine similarity |
| Prediction Latency | < 100ms | Django + joblib (cached models) |
| Early Warning Window | **60 days** | Cox survival analysis |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TrialGuard Platform                      │
├─────────────────┬───────────────────┬───────────────────────┤
│  Django Web UI  │   REST API (DRF)  │   Admin Panel         │
│  ─────────────  │  ─────────────    │  ─────────────        │
│  Landing Page   │  JWT Auth         │  Full ORM admin       │
│  Dashboard      │  /api/trials/     │  Risk tier colouring  │
│  Patient Detail │  /api/patients/   │  Inline predictions   │
│  Cohort Forecast│  /api/health/     │                       │
│  CSV Upload     │  /api/docs/       │                       │
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
│  MySQL Database (Django ORM)                                │
│  trials · patients · visits · prediction_results            │
│  cohort_forecasts · coordinator_actions                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- MySQL 8.0+
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
# Create MySQL database
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
# Generate 5,000 synthetic patients
python manage.py generate_synthetic_data --n 5000

# Train Cox PH + XGBoost (50 Optuna trials) + run batch inference
python manage.py train_models --optuna-trials 50
```

### 7. Run Development Server

```bash
python manage.py runserver
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

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
# Get JWT token
curl -X POST http://localhost:8000/api/token/ \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "yourpassword"}'

# Use token
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/trials/
```

---

## Data Sources

| Source | Type | Usage |
|---|---|---|
| SDV GaussianCopula synthesis | Synthetic | Primary training data (5,000 patients) |
| CDISC CDASH Demo Datasets | Real (public) | Supplementary validation |
| UCI Heart Disease | Real (public) | Supplementary cohort |
| UCI Diabetic Retinopathy | Real (public) | Supplementary cohort |

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
| Synthetic Data | SDV GaussianCopula |
| PDF Reports | ReportLab |
| Database | MySQL 8.0 (mysqlclient) |
| Static Files | WhiteNoise |
| Production Server | Gunicorn |
| Frontend Charts | Chart.js 4.4 |
| Typography | Cinzel + Lato (Google Fonts) |

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
│   ├── views.py         # Web views + DRF API views
│   ├── admin.py         # Admin interface with risk colour coding
│   ├── forms.py         # CSV upload + coordinator action forms
│   ├── serializers.py   # DRF serializers
│   ├── urls.py          # App URL patterns
│   ├── utils/
│   │   ├── data_pipeline.py    # Feature engineering
│   │   ├── survival_model.py   # Cox PH model
│   │   ├── xgboost_model.py    # XGBoost + Optuna
│   │   ├── shap_explainer.py   # SHAP TreeExplainer
│   │   └── report_generator.py # PDF reports (ReportLab)
│   ├── management/commands/
│   │   ├── generate_synthetic_data.py
│   │   └── train_models.py
│   └── migrations/
├── ml_models/           # Trained model artifacts (.pkl)
│   ├── cox_model.pkl
│   ├── xgb_model.pkl
│   ├── shap_explainer.pkl
│   └── scaler.pkl
├── templates/           # Django HTML templates
│   ├── base.html        # Navbar + footer (Cinzel/Lato, Gryffindor palette)
│   ├── index.html       # Public landing page
│   ├── dashboard.html   # Coordinator dashboard
│   ├── patient_detail.html
│   ├── cohort.html      # Cohort forecast view
│   ├── upload.html      # CSV data import
│   └── login.html
├── static/
│   ├── css/main.css     # Full design system (CSS custom properties)
│   ├── js/dashboard.js  # Navigation + Chart.js helpers
│   └── img/
│       ├── logo.svg     # TrialGuard SVG logo (shield + DNA helix)
│       ├── favicon.ico
│       └── apple-touch-icon.png
├── media/reports/       # Generated PDF reports
├── evaluation_results.json   # Model metrics (generated after training)
├── model_card.md
├── requirements.txt
└── .env.example
```

---

## Colour Palette (Gryffindor-Inspired)

| Token | Hex | Usage |
|---|---|---|
| `--crimson` | `#740001` | Primary brand, headers, critical risk |
| `--gold` | `#D3A625` | Accents, links, charts, logo text |
| `--dark-bg` | `#1A0A00` | Page background |
| `--card-bg` | `#2C1A0E` | Cards and panels |
| `--text-primary` | `#F5E6C8` | Body text (warm parchment) |
| `--accent` | `#FF6B35` | Hover states |

---

## Risk Tier System

| Tier | Probability | Colour | Action |
|---|---|---|---|
| Low | 0–30% | `#4CAF50` | Routine monitoring |
| Medium | 31–55% | `#FFC107` | Check-in recommended |
| High | 56–75% | `#FF5722` | Immediate outreach |
| Critical | 76–100% | `#740001` | Emergency intervention (pulsing UI) |

---

## Model Card

See [model_card.md](model_card.md) for full model documentation including features,
performance targets, limitations, and ethical considerations.

---

## License

Proprietary. All rights reserved.

---

**Built by [SKMMT](https://skmmt.rootexception.com/)** · TrialGuard © 2026 · Powered by XGBoost & Survival Analysis
