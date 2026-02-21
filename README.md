# Hurricane Forecasting (Python Pipeline + Streamlit Decision Cockpit)

This project provides a Python-first hurricane intensity forecasting workflow:

1. Download and validate raw data.
2. Engineer cyclical and geospatial features.
3. Train and compare tree-based regressors with time-aware CV.
4. Persist trained artifacts and evaluation reports.
5. Serve a user-friendly Streamlit decision cockpit (inference only, no retraining).

The legacy notebook-export file is retained as reference:
- `/Users/leobix/hurricane-forecasting/tutorial_1_hurricane_forecasting_and_boosted_trees_v1.py`

## Project Layout

- `/Users/leobix/hurricane-forecasting/src/hurricane_forecasting/config.py`
- `/Users/leobix/hurricane-forecasting/src/hurricane_forecasting/data.py`
- `/Users/leobix/hurricane-forecasting/src/hurricane_forecasting/features.py`
- `/Users/leobix/hurricane-forecasting/src/hurricane_forecasting/models.py`
- `/Users/leobix/hurricane-forecasting/src/hurricane_forecasting/evaluate.py`
- `/Users/leobix/hurricane-forecasting/src/hurricane_forecasting/train.py`
- `/Users/leobix/hurricane-forecasting/scripts/train_models.py`
- `/Users/leobix/hurricane-forecasting/app/streamlit_app.py`
- `/Users/leobix/hurricane-forecasting/models/`
- `/Users/leobix/hurricane-forecasting/reports/`
- `/Users/leobix/hurricane-forecasting/data/raw/`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Models

```bash
python scripts/train_models.py
```

Optional flags:

```bash
python scripts/train_models.py --data-dir data/raw --model all --seed 42 --n-jobs 1
```

`--model` options: `all`, `dummy`, `cart`, `rf`, `lgbm`.

## Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard loads only persisted artifacts from `/Users/leobix/hurricane-forecasting/models/`.

## Artifacts

Training creates the following key outputs:

- `/Users/leobix/hurricane-forecasting/models/best_model.joblib`
- `/Users/leobix/hurricane-forecasting/models/model_manifest.json`
- `/Users/leobix/hurricane-forecasting/models/preprocessing_metadata.json`
- `/Users/leobix/hurricane-forecasting/models/candidates/*.joblib`
- `/Users/leobix/hurricane-forecasting/reports/metrics.json`
- `/Users/leobix/hurricane-forecasting/reports/feature_importance.csv` (when available)
- `/Users/leobix/hurricane-forecasting/reports/figures/feature_importance_top15.png`

## End-to-End Flow

1. `scripts/train_models.py` builds `TrainingConfig`.
2. `data.py` downloads cached raw CSV files and validates schema.
3. `features.py` applies feature engineering and temporal split.
4. `models.py` trains/tunes candidate models.
5. `train.py` evaluates, selects best RMSE model, and persists outputs.
6. `app/streamlit_app.py` reads manifest + model and serves predictions.

## Troubleshooting

- **`Model manifest not found`**:
  Run `python scripts/train_models.py` first.

- **Missing dependency (e.g., `lightgbm`)**:
  Reinstall requirements with `pip install -r requirements.txt`.

- **Dataset download issues**:
  Check internet access and Dropbox URL availability; files are cached in `/Users/leobix/hurricane-forecasting/data/raw/`.

- **Feature mismatch at inference**:
  Ensure uploaded CSV follows the training raw schema and includes required columns.
