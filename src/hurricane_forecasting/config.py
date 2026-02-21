from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ModelName = Literal["all", "dummy", "cart", "rf", "lgbm"]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


DEFAULT_DATA_URLS = {
    "tropical_cyclones.csv": "https://www.dropbox.com/s/sukxezag91rt70u/tropical_cyclones.csv?dl=1",
    "targets_tropical_cyclones.csv": "https://www.dropbox.com/s/3jffi9ygb18teh2/targets_tropical_cyclones.csv?dl=1",
}


@dataclass(slots=True)
class TrainingConfig:
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    reports_dir: Path = REPORTS_DIR
    figures_dir: Path = FIGURES_DIR
    dataset_urls: dict[str, str] = field(default_factory=lambda: DEFAULT_DATA_URLS.copy())

    train_size: float = 0.7
    random_seed: int = 42
    cv_splits: int = 2
    n_jobs: int = 1

    primary_metric: str = "rmse"
    app_compat_version: str = "1.0.0"

    # Balanced runtime defaults
    lgbm_random_search_iter: int = 3


REQUIRED_EXACT_COLUMNS = ["YEAR_0", "MONTH_0", "DAY_0"]
REQUIRED_PREFIX_GROUPS = [
    "basin_",
    "LATITUDE_",
    "LONGITUDE_",
    "WMO_WIND_",
    "WMO_PRESSURE_",
    "DISTANCE_TO_LAND_",
    "STORM_TRANSLATION_SPEED_",
]


SAFFIR_SIMPSON_THRESHOLDS = [
    (0, 63, "Tropical Storm", "#4f8df7"),
    (64, 82, "Category 1", "#28a745"),
    (83, 95, "Category 2", "#ffc107"),
    (96, 112, "Category 3", "#fd7e14"),
    (113, 136, "Category 4", "#dc3545"),
    (137, 500, "Category 5", "#6f42c1"),
]
