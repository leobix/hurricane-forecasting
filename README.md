# Hurricane Forecasting: ML Pipeline + Streamlit Dashboard

This repository is a complete, end-to-end machine learning project for **24-hour hurricane intensity forecasting**.

It has two connected parts:
1. A **training pipeline** that prepares features, trains models, evaluates them, and saves artifacts.
2. A **Streamlit dashboard** that loads those saved artifacts and lets you interact with predictions.

If you are opening this repo for the first time as a student, this README walks you through what each piece does and how to run everything on your own computer.

## What You Are Looking At

This is not just a notebook. It is a production-style workflow split into modules:
- Data loading + validation
- Feature engineering
- Model training + model selection
- Artifact/report generation
- Interactive app for inference and analysis

The old notebook-export script is kept only as reference:
- `tutorial_1_hurricane_forecasting_and_boosted_trees_v1.py`

## Learning Goals

By running this project, you will practice:
- Building a structured ML pipeline (instead of one large notebook)
- Time-aware model evaluation for forecasting tasks
- Comparing baseline vs. tree-based models
- Persisting trained artifacts for reproducible inference
- Serving model outputs in an interactive decision dashboard

## End-to-End Flow (Big Picture)

1. `scripts/train_models.py` starts training.
2. `src/hurricane_forecasting/data.py` ensures data files exist, downloads if needed, and validates schema.
3. `src/hurricane_forecasting/features.py` engineers features and creates train/test split.
4. `src/hurricane_forecasting/models.py` trains candidate models (`dummy`, `cart`, `rf`, `lgbm`).
5. `src/hurricane_forecasting/train.py` evaluates models, picks the best by RMSE, and saves artifacts.
6. `app/streamlit_app.py` loads the saved artifacts and serves the interactive dashboard.

## Repository Structure

- `app/streamlit_app.py`
  Streamlit interface for single, manual, and batch predictions, metrics, SHAP, and visual analytics.

- `scripts/train_models.py`
  Command-line entry point for training.

- `src/hurricane_forecasting/config.py`
  Central configuration (paths, model settings, thresholds, required schema).

- `src/hurricane_forecasting/data.py`
  Data download/cache + schema validation.

- `src/hurricane_forecasting/features.py`
  Feature transforms (cyclical date and geo transforms) and split logic.

- `src/hurricane_forecasting/models.py`
  Model training and hyperparameter search.

- `src/hurricane_forecasting/evaluate.py`
  RMSE/MAE/R² metrics helpers.

- `src/hurricane_forecasting/train.py`
  Orchestration that writes trained model files and reports.

- `data/raw/`
  Input CSVs (auto-downloaded if missing).

- `models/`
  Saved trained models + manifest + preprocessing metadata.

- `reports/`
  Metrics report and feature-importance outputs.

## First-Time Setup (Local Computer)

### Prerequisites
- Python **3.11** recommended (matches `runtime.txt`)
- `pip`
- Git

### 1) Clone and enter the repo

```bash
git clone <YOUR_REPO_URL>
cd hurricane-forecasting
```

### 2) Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Verify key install

```bash
python -c "import streamlit, sklearn, pandas; print('Environment OK')"
```

## Run Paths

### Option A: Fastest demo (use existing trained artifacts)

If `models/model_manifest.json` and model files are already present, launch the app directly:

```bash
streamlit run app/streamlit_app.py
```

### Option B: Full student workflow (recommended)

Run full training first, then start dashboard:

```bash
python scripts/train_models.py
streamlit run app/streamlit_app.py
```

Optional training flags:

```bash
python scripts/train_models.py --data-dir data/raw --model all --seed 42 --n-jobs 1
```

`--model` values: `all`, `dummy`, `cart`, `rf`, `lgbm`.

## What Training Produces

After `python scripts/train_models.py`, you should see:

- `models/best_model.joblib`
- `models/model_manifest.json`
- `models/preprocessing_metadata.json`
- `models/candidates/*.joblib`
- `reports/metrics.json`
- `reports/feature_importance.csv` (if supported by selected best model)
- `reports/figures/feature_importance_top15.png`

The Streamlit app depends on these artifacts. If they are missing, the app will ask you to run training first.

## Streamlit Dashboard: What Each Area Does

When you run `streamlit run app/streamlit_app.py`, the dashboard opens with tabs:

- `Intro`
  Plain-language overview of problem, features, and workflow.

- `Forecast`
  Three prediction workflows:
  - Load example storm (existing row)
  - Manual input (editable controls)
  - Batch CSV upload (multiple rows)

- `Metrics`
  Model comparison table/charts plus SHAP explainability (for non-dummy models).

- `Hurricane Atlas`
  Global storm track visualization.

- `Descriptive Analytics`
  Distribution, relationship, composition, and advanced EDA plots.

## Data Notes

- Raw datasets are expected in `data/raw/`:
  - `tropical_cyclones.csv`
  - `targets_tropical_cyclones.csv`
- If missing, they are downloaded automatically from configured URLs in `src/hurricane_forecasting/config.py`.
- Schema checks require key columns (e.g., `YEAR_0`, `MONTH_0`, `DAY_0`, and required prefix groups like `LATITUDE_`, `LONGITUDE_`, `WMO_WIND_`, etc.).

## Reproducibility

Defaults are set for reproducible classroom runs:
- Random seed: `42`
- Time-aware CV (`TimeSeriesSplit`) for tuned models
- Fixed feature order persisted in manifest and reused at inference time

## Troubleshooting

- `Model manifest not found`
  Run: `python scripts/train_models.py`

- `lightgbm` import error
  Reinstall dependencies: `pip install -r requirements.txt`

- Dataset download failure
  Check internet access and retry training (data is pulled on demand).

- Feature mismatch during batch upload
  Ensure uploaded CSV matches the raw training schema expected by the feature pipeline.

- Streamlit command not found
  Use: `python -m streamlit run app/streamlit_app.py`

## Typical Student Session

1. Activate your virtual environment.
2. Run `python scripts/train_models.py`.
3. Run `streamlit run app/streamlit_app.py`.
4. Open the app in your browser (local URL shown in terminal).
5. Try Forecast modes, then inspect Metrics and SHAP.
6. Export prediction reports from the app.
