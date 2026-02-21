#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hurricane_forecasting.config import TrainingConfig
from hurricane_forecasting.train import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hurricane forecasting models.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory for raw CSV datasets.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model to train: all, dummy, cart, rf, lgbm",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for model training.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for model search; use 1 for stable/log-clean runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainingConfig(
        data_dir=args.data_dir,
        random_seed=args.seed,
        n_jobs=args.n_jobs,
    )

    output = run_training_pipeline(cfg=cfg, selected_model=args.model)

    print("Training complete")
    print(f"Best model: {output.best_model_name}")
    print(f"Best model path: {output.best_model_path}")
    print(f"Manifest path: {output.manifest_path}")
    print(f"Metrics path: {output.metrics_path}")


if __name__ == "__main__":
    main()
