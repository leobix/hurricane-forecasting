from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import TrainingConfig


@dataclass(slots=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


@dataclass(slots=True)
class FeatureMetadata:
    latitude_columns: list[str]
    longitude_columns: list[str]
    dropped_columns: list[str]


def _sorted_suffix_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [col for col in df.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def _encode_day_of_year(df: pd.DataFrame) -> pd.DataFrame:
    required = ["YEAR_0", "MONTH_0", "DAY_0"]
    if not all(col in df.columns for col in required):
        return df

    parsed_date = pd.to_datetime(
        {
            "year": df["YEAR_0"].astype(int),
            "month": df["MONTH_0"].astype(int),
            "day": df["DAY_0"].astype(int),
        },
        errors="coerce",
    )
    if parsed_date.isna().any():
        bad_rows = parsed_date.index[parsed_date.isna()].tolist()[:10]
        raise ValueError(
            "Invalid YEAR_0/MONTH_0/DAY_0 combination detected while encoding dates. "
            f"Example bad row indices: {bad_rows}"
        )

    day_number = parsed_date.dt.dayofyear
    df = df.copy()
    df["COS_DATE"] = np.cos(2 * np.pi * day_number / 365.0)
    df["SIN_DATE"] = np.sin(2 * np.pi * day_number / 365.0)
    return df


def transform_raw_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, FeatureMetadata]:
    """Apply notebook-equivalent feature engineering in a deterministic way."""
    df = df_raw.copy()

    latitude_columns = _sorted_suffix_columns(df, "LATITUDE_")
    longitude_columns = _sorted_suffix_columns(df, "LONGITUDE_")

    for col in latitude_columns:
        df[f"COS_{col}"] = np.cos(2 * np.pi * df[col] / 90.0)
        df[f"SIN_{col}"] = np.sin(2 * np.pi * df[col] / 90.0)

    for col in longitude_columns:
        df[f"COS_{col}"] = np.cos(2 * np.pi * df[col] / 180.0)
        df[f"SIN_{col}"] = np.sin(2 * np.pi * df[col] / 180.0)

    df = _encode_day_of_year(df)

    dropped_columns = latitude_columns + longitude_columns + ["YEAR_0", "MONTH_0", "DAY_0"]
    dropped_columns = [col for col in dropped_columns if col in df.columns]

    X = df.drop(columns=dropped_columns)

    return X, FeatureMetadata(
        latitude_columns=latitude_columns,
        longitude_columns=longitude_columns,
        dropped_columns=dropped_columns,
    )


def build_features(
    df_features: pd.DataFrame,
    target: pd.Series,
    cfg: TrainingConfig,
) -> tuple[DataSplit, FeatureMetadata]:
    X, metadata = transform_raw_features(df_features)

    if not isinstance(target, pd.Series):
        target = pd.Series(target)

    n_rows = len(X)
    if n_rows < 10:
        raise ValueError("Dataset is too small to split for training/testing.")

    train_rows = int(n_rows * cfg.train_size)
    train_rows = max(1, min(train_rows, n_rows - 1))

    X_train = X.iloc[:train_rows].copy()
    X_test = X.iloc[train_rows:].copy()
    y_train = target.iloc[:train_rows].copy()
    y_test = target.iloc[train_rows:].copy()

    split = DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
    )
    return split, metadata


def prepare_inference_features(raw_df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    engineered, _ = transform_raw_features(raw_df)

    missing = [col for col in feature_order if col not in engineered.columns]
    if missing:
        raise ValueError(
            "Input data is missing required engineered features. "
            f"Missing columns: {missing[:10]}"
        )

    extra = [col for col in engineered.columns if col not in feature_order]
    if extra:
        engineered = engineered.drop(columns=extra)

    ordered = engineered[feature_order].copy()

    if ordered.isna().any().any():
        bad_cols = ordered.columns[ordered.isna().any()].tolist()
        raise ValueError(
            "Input data produced NaN values after feature engineering. "
            f"Columns with NaN: {bad_cols[:10]}"
        )

    return ordered


def make_serializable_metadata(metadata: FeatureMetadata) -> dict[str, Any]:
    return {
        "latitude_columns": metadata.latitude_columns,
        "longitude_columns": metadata.longitude_columns,
        "dropped_columns": metadata.dropped_columns,
    }
