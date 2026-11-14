#!/usr/bin/env python3
"""Memory-conscious linear baseline for ventilator pressure prediction.

Quick run (first 5,000 complete breaths):
    python3 baseline.py

Full training data:
    python3 baseline.py --max-breaths 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["R", "C", "time_step", "u_in", "u_out"]
POD_DATA = Path("/home/jjqian/ventilator-pressure-prediction/train.csv")
REPO_DATA = Path(__file__).parent / "data" / "ventilator-pressure-prediction" / "train.csv"
DEFAULT_DATA = POD_DATA if POD_DATA.is_file() else REPO_DATA
DEFAULT_SCORE = Path(__file__).parent / "baseline_score.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help=f"training CSV (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--max-breaths", type=int, default=5000,
        help="read this many complete breaths from the start (0 means all rows)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-out", type=Path, default=DEFAULT_SCORE)
    args = parser.parse_args()
    if args.max_breaths < 0:
        parser.error("--max-breaths must be zero or positive")

    # Each breath occupies 80 contiguous rows in the provided file. Read only
    # requested columns and, for quick runs, only complete leading breaths.
    nrows = args.max_breaths * 80 if args.max_breaths else None
    df = pd.read_csv(args.data, usecols=["breath_id", *FEATURES, "pressure"], nrows=nrows)
    breath_ids = df["breath_id"].drop_duplicates().to_numpy()
    rng = np.random.RandomState(args.seed)
    shuffled = rng.permutation(breath_ids)
    n_valid_breaths = int(np.ceil(0.1 * len(breath_ids)))
    valid_breaths = shuffled[:n_valid_breaths]
    is_valid = df["breath_id"].isin(valid_breaths).to_numpy()
    train_idx = np.flatnonzero(~is_valid)
    valid_idx = np.flatnonzero(is_valid)

    # Ordinary least squares with an intercept, using only NumPy. The data
    # matrix is small enough to fit comfortably while avoiding extra packages.
    x_train = df.iloc[train_idx][FEATURES].to_numpy(dtype=np.float64)
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    y_train = df.iloc[train_idx]["pressure"].to_numpy(dtype=np.float64)
    coefficients = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    valid = df.iloc[valid_idx]
    inspiratory = valid["u_out"].to_numpy() == 0
    x_valid = valid.loc[inspiratory, FEATURES].to_numpy(dtype=np.float64)
    prediction = np.column_stack((np.ones(len(x_valid)), x_valid)) @ coefficients
    mae = np.mean(np.abs(valid.loc[inspiratory, "pressure"].to_numpy() - prediction))

    result = {
        "model": "ordinary_least_squares_with_intercept",
        "features": FEATURES,
        "seed": args.seed,
        "validation_fraction_by_breath": 0.1,
        "max_breaths": args.max_breaths or None,
        "rows": int(len(df)),
        "breaths": int(df["breath_id"].nunique()),
        "train_breaths": int(df.iloc[train_idx]["breath_id"].nunique()),
        "validation_breaths": int(df.iloc[valid_idx]["breath_id"].nunique()),
        "validation_inspiratory_rows": int(inspiratory.sum()),
        "validation_inspiratory_mae": float(mae),
    }
    args.score_out.parent.mkdir(parents=True, exist_ok=True)
    args.score_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Score saved to {args.score_out}")


if __name__ == "__main__":
    main()
