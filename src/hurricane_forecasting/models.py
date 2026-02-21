from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from .config import TrainingConfig
from .evaluate import extract_best_cv_metrics
from .features import DataSplit


@dataclass(slots=True)
class ModelRunResult:
    name: str
    estimator: Any
    best_params: dict[str, Any]
    cv_metrics: dict[str, float]
    fit_seconds: float


def _scoring() -> dict[str, str]:
    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }


def _timeseries_cv(cfg: TrainingConfig) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=cfg.cv_splits)


def train_dummy(split: DataSplit) -> ModelRunResult:
    start = perf_counter()
    model = DummyRegressor(strategy="mean")
    model.fit(split.X_train, split.y_train)
    duration = perf_counter() - start

    return ModelRunResult(
        name="dummy",
        estimator=model,
        best_params={"strategy": "mean"},
        cv_metrics={"cv_rmse": np.nan, "cv_mae": np.nan, "cv_r2": np.nan},
        fit_seconds=duration,
    )


def train_decision_tree(split: DataSplit, cfg: TrainingConfig) -> ModelRunResult:
    grid = {
        "max_depth": [4, 6],
        "min_samples_leaf": [1, 5],
    }

    search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=cfg.random_seed),
        param_grid=grid,
        scoring=_scoring(),
        refit="rmse",
        cv=_timeseries_cv(cfg),
        n_jobs=cfg.n_jobs,
    )

    start = perf_counter()
    search.fit(split.X_train, split.y_train)
    duration = perf_counter() - start

    return ModelRunResult(
        name="cart",
        estimator=search.best_estimator_,
        best_params=search.best_params_,
        cv_metrics=extract_best_cv_metrics(search),
        fit_seconds=duration,
    )


def train_random_forest(split: DataSplit, cfg: TrainingConfig) -> ModelRunResult:
    grid = {
        "n_estimators": [150],
        "max_depth": [8, None],
        "min_samples_leaf": [1, 4],
        "max_features": ["sqrt"],
    }

    search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=cfg.random_seed, n_jobs=cfg.n_jobs),
        param_grid=grid,
        scoring=_scoring(),
        refit="rmse",
        cv=_timeseries_cv(cfg),
        n_jobs=cfg.n_jobs,
    )

    start = perf_counter()
    search.fit(split.X_train, split.y_train)
    duration = perf_counter() - start

    return ModelRunResult(
        name="rf",
        estimator=search.best_estimator_,
        best_params=search.best_params_,
        cv_metrics=extract_best_cv_metrics(search),
        fit_seconds=duration,
    )


def _import_lgbm() -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for LGBM training. Install dependencies from requirements.txt."
        ) from exc
    return lgb


def _focused_values(value: float, low: float, high: float, step: float) -> list[float]:
    values = [value - step, value, value + step]
    clipped = [min(high, max(low, v)) for v in values]
    return sorted(set(round(v, 6) for v in clipped))


def train_lightgbm(split: DataSplit, cfg: TrainingConfig) -> ModelRunResult:
    lgb = _import_lgbm()

    base = lgb.LGBMRegressor(
        random_state=cfg.random_seed,
        objective="regression",
        n_jobs=cfg.n_jobs,
        verbose=-1,
    )

    random_space = {
        "n_estimators": [180, 240, 320, 420, 520],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "max_depth": [3, 4, 5, 6, -1],
        "num_leaves": [15, 31, 63, 95],
        "min_child_samples": [10, 20, 40, 60],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0],
    }

    stage1 = RandomizedSearchCV(
        estimator=base,
        param_distributions=random_space,
        n_iter=cfg.lgbm_random_search_iter,
        random_state=cfg.random_seed,
        scoring=_scoring(),
        refit="rmse",
        cv=_timeseries_cv(cfg),
        n_jobs=cfg.n_jobs,
    )

    start = perf_counter()
    stage1.fit(split.X_train, split.y_train)

    p = stage1.best_params_
    # Keep stage-2 refinement bounded for a practical "balanced" runtime.
    focused_grid = {
        "n_estimators": [p["n_estimators"]],
        "learning_rate": sorted(set([round(p["learning_rate"], 6), round(min(0.2, p["learning_rate"] + 0.02), 6)])),
        "max_depth": [p["max_depth"]],
        "num_leaves": sorted(set([p["num_leaves"], p["num_leaves"] + 12])),
        "min_child_samples": [p["min_child_samples"]],
        # Hold weaker-impact regularization/subsampling params fixed from stage-1 winner.
        "subsample": [p["subsample"]],
        "colsample_bytree": [p["colsample_bytree"]],
        "reg_alpha": [p["reg_alpha"]],
        "reg_lambda": [p["reg_lambda"]],
    }

    stage2 = GridSearchCV(
        estimator=base,
        param_grid=focused_grid,
        scoring=_scoring(),
        refit="rmse",
        cv=_timeseries_cv(cfg),
        n_jobs=cfg.n_jobs,
    )
    stage2.fit(split.X_train, split.y_train)
    duration = perf_counter() - start

    return ModelRunResult(
        name="lgbm",
        estimator=stage2.best_estimator_,
        best_params=stage2.best_params_,
        cv_metrics=extract_best_cv_metrics(stage2),
        fit_seconds=duration,
    )


def train_all_models(
    split: DataSplit,
    cfg: TrainingConfig,
    selected_model: str = "all",
) -> dict[str, ModelRunResult]:
    selected_model = selected_model.lower().strip()

    runs: dict[str, ModelRunResult] = {}
    # Baseline always included for sanity checks.
    runs["dummy"] = train_dummy(split)

    if selected_model in {"all", "cart"}:
        runs["cart"] = train_decision_tree(split, cfg)

    if selected_model in {"all", "rf"}:
        runs["rf"] = train_random_forest(split, cfg)

    if selected_model in {"all", "lgbm"}:
        runs["lgbm"] = train_lightgbm(split, cfg)

    if selected_model not in {"all", "dummy", "cart", "rf", "lgbm"}:
        raise ValueError(
            f"Unsupported model '{selected_model}'. "
            "Expected one of: all, dummy, cart, rf, lgbm"
        )

    return runs
