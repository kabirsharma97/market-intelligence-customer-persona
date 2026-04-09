# Market Intelligence Driven by Customer Persona

**TwinSim v5.0** — End-to-end OTT customer persona intelligence platform.

Built with [Reflex](https://reflex.dev) · EXL branding · Python 3.11+

---

## Pipeline Overview

| Step | Screen | What it does |
|------|--------|-------------|
| 01 | Upload Dataset | Upload 3 source CSV files |
| 02 | Ingestion & Feature Check | Enrich sessions · validate 81 features across 7 categories |
| 03 | Health & Quality Check | 239 automated checks (A1–A19 · B15–B24 · C25–C26) · 0 FAIL gate |
| 04 | Feature Engineering | 50 ML-ready features · 12-check quality assessment · clustering ready |

## Input Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `session_events.csv` | ~30K | 36 | User watch events & quality signals |
| `user_profiles.csv` | ~5K | 38 | Subscription, device & engagement attributes |
| `content_catalogue.csv` | 20 | 9 | Content type, genre & availability metadata |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
reflex run
```

App will be available at **http://localhost:3000**

## Deployment

### Reflex Cloud
```bash
reflex deploy
```

### Railway / Render
Push this repo — the `Procfile` handles startup.
Set the `PORT` environment variable on your host.

## Project Structure

```
twinsim_ui/
├── twinsim_ui/           # Reflex app package
│   ├── twinsim_ui.py     # App entry point & routing
│   ├── state.py          # Global state & pipeline orchestration
│   ├── styles.py         # EXL brand design tokens
│   ├── components/
│   │   ├── sidebar.py    # Dark sidebar navigation
│   │   └── topbar.py     # Top bar with page title
│   └── pages/
│       ├── upload.py     # Screen 1 — Upload
│       ├── ingest.py     # Screen 2 — Ingestion
│       ├── health.py     # Screen 3 — Health checks
│       └── features.py   # Screen 4 — Feature engineering
├── pipeline/             # Data pipeline scripts
│   ├── enrich_sessions.py
│   ├── health_report.py
│   ├── feature_store.py
│   ├── feature_store_assessment.py
│   └── data/
│       ├── data_dictionary_session_events.json
│       └── data_dictionary_user_profiles.json
├── uploaded_files/       # Runtime uploads (git-ignored)
├── assets/               # Static assets
├── rxconfig.py
├── requirements.txt
└── Procfile
```

---

*EXL Service · Customer Analytics · TwinSim v5.0*
