from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .config import REQUIRED_EXACT_COLUMNS, REQUIRED_PREFIX_GROUPS, TrainingConfig


class DataValidationError(ValueError):
    pass


def _download_file(url: str, destination: Path, timeout_seconds: int = 60) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def ensure_data_files(cfg: TrainingConfig, force_download: bool = False) -> dict[str, Path]:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    resolved_paths: dict[str, Path] = {}
    for filename, url in cfg.dataset_urls.items():
        file_path = cfg.data_dir / filename
        if force_download or not file_path.exists():
            _download_file(url=url, destination=file_path)
        resolved_paths[filename] = file_path

    return resolved_paths


def validate_feature_schema(df_features: pd.DataFrame) -> None:
    missing_exact = [col for col in REQUIRED_EXACT_COLUMNS if col not in df_features.columns]
    if missing_exact:
        raise DataValidationError(
            f"Missing required feature columns: {missing_exact}. "
            "Please verify the source dataset schema."
        )

    missing_prefix_groups = [
        prefix
        for prefix in REQUIRED_PREFIX_GROUPS
        if not any(col.startswith(prefix) for col in df_features.columns)
    ]
    if missing_prefix_groups:
        raise DataValidationError(
            f"Missing required feature groups by prefix: {missing_prefix_groups}."
        )


def validate_target_schema(df_target: pd.DataFrame) -> None:
    if df_target.shape[1] != 1:
        raise DataValidationError(
            f"Expected a single target column, got {df_target.shape[1]} columns."
        )


def load_raw_data(data_dir: Path | None = None, cfg: TrainingConfig | None = None) -> tuple[pd.DataFrame, pd.Series]:
    cfg = cfg or TrainingConfig()
    if data_dir is not None:
        cfg.data_dir = Path(data_dir)

    paths = ensure_data_files(cfg)

    features_path = paths["tropical_cyclones.csv"]
    target_path = paths["targets_tropical_cyclones.csv"]

    df_features = pd.read_csv(features_path, index_col=0)
    df_target = pd.read_csv(target_path, index_col=0)

    validate_feature_schema(df_features)
    validate_target_schema(df_target)

    if len(df_features) != len(df_target):
        raise DataValidationError(
            "Feature and target row counts do not match: "
            f"{len(df_features)} vs {len(df_target)}"
        )

    target_series = df_target.iloc[:, 0]
    target_series.name = df_target.columns[0]

    return df_features, target_series
