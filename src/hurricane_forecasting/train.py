from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .config import PROJECT_ROOT, TrainingConfig
from .data import load_raw_data
from .evaluate import evaluate_model, to_jsonable
from .features import build_features, make_serializable_metadata
from .models import ModelRunResult, train_all_models


@dataclass(slots=True)
class BestModelBundle:
    name: str
    estimator: Any
    metrics: dict[str, float]


@dataclass(slots=True)
class TrainingOutput:
    best_model_name: str
    best_model_path: Path
    manifest_path: Path
    metrics_path: Path


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _select_best_model(
    runs: dict[str, ModelRunResult],
    run_metrics: dict[str, dict[str, float]],
    primary_metric: str = "rmse",
) -> BestModelBundle:
    comparable_names = [name for name in runs if name != "dummy"] or list(runs.keys())

    if primary_metric == "r2":
        best_name = max(comparable_names, key=lambda name: run_metrics[name]["test_r2"])
    elif primary_metric == "mae":
        best_name = min(comparable_names, key=lambda name: run_metrics[name]["test_mae"])
    else:
        best_name = min(comparable_names, key=lambda name: run_metrics[name]["test_rmse"])

    return BestModelBundle(
        name=best_name,
        estimator=runs[best_name].estimator,
        metrics=run_metrics[best_name],
    )


def _save_feature_importance(best_model: Any, feature_names: list[str], reports_dir: Path, figures_dir: Path) -> Path | None:
    if not hasattr(best_model, "feature_importances_"):
        return None

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "feature_importance.csv"
    importance.to_csv(csv_path, index=False)

    top_n = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_n["feature"], top_n["importance"])
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()

    fig_path = figures_dir / "feature_importance_top15.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return csv_path


def run_training_pipeline(cfg: TrainingConfig, selected_model: str = "all") -> TrainingOutput:
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    (cfg.models_dir / "candidates").mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    raw_features, target = load_raw_data(cfg=cfg)
    split, metadata = build_features(raw_features, target, cfg)

    runs = train_all_models(split=split, cfg=cfg, selected_model=selected_model)

    all_metrics: dict[str, dict[str, float]] = {}
    candidate_model_paths: dict[str, str] = {}

    for name, run in runs.items():
        metrics = evaluate_model(
            run.estimator,
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
        )
        metrics.update(run.cv_metrics)
        metrics["fit_seconds"] = run.fit_seconds

        all_metrics[name] = metrics

        candidate_path = cfg.models_dir / "candidates" / f"{name}.joblib"
        joblib.dump(run.estimator, candidate_path)
        candidate_model_paths[name] = _portable_path(candidate_path)

    best = _select_best_model(runs, all_metrics, cfg.primary_metric)

    best_model_path = cfg.models_dir / "best_model.joblib"
    joblib.dump(best.estimator, best_model_path)

    metrics_path = cfg.reports_dir / "metrics.json"
    _save_json(metrics_path, all_metrics)

    preprocessing_metadata_path = cfg.models_dir / "preprocessing_metadata.json"
    _save_json(
        preprocessing_metadata_path,
        {
            "train_size": cfg.train_size,
            "feature_order": split.feature_names,
            "feature_engineering": make_serializable_metadata(metadata),
        },
    )

    feature_importance_csv = _save_feature_importance(
        best_model=best.estimator,
        feature_names=split.feature_names,
        reports_dir=cfg.reports_dir,
        figures_dir=cfg.figures_dir,
    )

    best_beats_dummy = None
    if "dummy" in all_metrics and best.name in all_metrics:
        best_beats_dummy = all_metrics[best.name]["test_rmse"] < all_metrics["dummy"]["test_rmse"]

    manifest_path = cfg.models_dir / "model_manifest.json"
    _save_json(
        manifest_path,
        {
            "model_name": best.name,
            "feature_order": split.feature_names,
            "train_timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": all_metrics,
            "artifact_paths": {
                "best_model": _portable_path(best_model_path),
                "candidate_models": candidate_model_paths,
                "metrics_report": _portable_path(metrics_path),
                "preprocessing_metadata": _portable_path(preprocessing_metadata_path),
                "feature_importance_csv": _portable_path(feature_importance_csv) if feature_importance_csv else None,
                "figures_dir": _portable_path(cfg.figures_dir),
            },
            "app_compat_version": cfg.app_compat_version,
            "primary_metric": cfg.primary_metric,
            "best_beats_dummy_rmse": best_beats_dummy,
        },
    )

    return TrainingOutput(
        best_model_name=best.name,
        best_model_path=best_model_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
    )
