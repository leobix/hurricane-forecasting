from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return {
        "train_rmse": float(metrics.root_mean_squared_error(y_train, y_train_pred)),
        "test_rmse": float(metrics.root_mean_squared_error(y_test, y_test_pred)),
        "train_mae": float(metrics.mean_absolute_error(y_train, y_train_pred)),
        "test_mae": float(metrics.mean_absolute_error(y_test, y_test_pred)),
        "train_r2": float(metrics.r2_score(y_train, y_train_pred)),
        "test_r2": float(metrics.r2_score(y_test, y_test_pred)),
    }


def extract_best_cv_metrics(search: Any) -> dict[str, float]:
    idx = int(search.best_index_)
    cv = search.cv_results_

    # sklearn stores RMSE/MAE scorers as negated values for maximization.
    rmse = -float(cv["mean_test_rmse"][idx])
    mae = -float(cv["mean_test_mae"][idx])
    r2 = float(cv["mean_test_r2"][idx])

    return {
        "cv_rmse": rmse,
        "cv_mae": mae,
        "cv_r2": r2,
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value
