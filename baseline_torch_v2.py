#!/usr/bin/env python3
"""PyTorch baseline with per-breath control history features.

Quick run (same 5,000-breath/10-epoch settings as baseline_torch.py):
    python3 baseline_torch_v2.py

Full training CSV:
    python3 baseline_torch_v2.py --max-breaths 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_FEATURES = ["R", "C", "time_step", "u_in", "u_out"]
HISTORY_FEATURES = [
    "u_in_cum", "u_in_prev1", "u_in_prev2", "u_out_prev1", "u_out_cum",
]
FEATURES = BASE_FEATURES + HISTORY_FEATURES
POD_DATA = Path("/home/jjqian/ventilator-pressure-prediction/train.csv")
REPO_DATA = Path(__file__).parent / "data" / "ventilator-pressure-prediction" / "train.csv"
DEFAULT_DATA = POD_DATA if POD_DATA.is_file() else REPO_DATA
DEFAULT_SCORE = Path(__file__).parent / "baseline_torch_v2_score.json"


class PressureMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(len(FEATURES), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def add_breath_history_features(df):
    """Add causal control features, restarting at every breath boundary."""
    by_breath = df.groupby("breath_id", sort=False)
    df["u_in_cum"] = by_breath["u_in"].cumsum()
    df["u_in_prev1"] = by_breath["u_in"].shift(1, fill_value=0)
    df["u_in_prev2"] = by_breath["u_in"].shift(2, fill_value=0)
    df["u_out_prev1"] = by_breath["u_out"].shift(1, fill_value=0)
    df["u_out_cum"] = by_breath["u_out"].cumsum()
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-breaths", type=int, default=5000,
                        help="read leading complete breaths; 0 reads all rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--score-out", type=Path, default=DEFAULT_SCORE)
    args = parser.parse_args()
    if args.max_breaths < 0 or args.epochs < 1 or args.batch_size < 1:
        parser.error("max-breaths must be >= 0; epochs and batch-size must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Each breath is 80 contiguous rows in this dataset. Read only the needed
    # columns and, for the quick run, a whole-breath prefix.
    nrows = args.max_breaths * 80 if args.max_breaths else None
    df = pd.read_csv(
        args.data,
        usecols=["breath_id", *BASE_FEATURES, "pressure"],
        nrows=nrows,
    )
    df = add_breath_history_features(df)

    # Match the original baseline exactly: same ordered unique breath IDs,
    # RandomState seed, 10% rounded up, and group membership for every row.
    breath_ids = df["breath_id"].drop_duplicates().to_numpy()
    valid_breaths = np.random.RandomState(args.seed).permutation(breath_ids)
    valid_breaths = valid_breaths[:int(np.ceil(0.1 * len(breath_ids)))]
    is_valid = df["breath_id"].isin(valid_breaths).to_numpy()
    is_train = ~is_valid

    x = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["pressure"].to_numpy(dtype=np.float32)
    mean = x[is_train].mean(axis=0)
    scale = x[is_train].std(axis=0)
    scale[scale == 0] = 1.0
    x = (x - mean) / scale

    train_x = torch.from_numpy(np.ascontiguousarray(x[is_train]))
    train_y = torch.from_numpy(np.ascontiguousarray(y[is_train]))
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        pin_memory=(device.type == "cuda"),
    )

    model = PressureMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(args.epochs):
        loss_total = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(batch_y)
            seen += len(batch_y)
        print(f"epoch {epoch + 1}/{args.epochs} train_mse={loss_total / seen:.6f}")

    valid_idx = np.flatnonzero(is_valid & (df["u_out"].to_numpy() == 0))
    valid_x = torch.from_numpy(np.ascontiguousarray(x[valid_idx]))
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in valid_x.split(args.batch_size):
            predictions.append(model(batch.to(device)).cpu().numpy())
    mae = float(np.mean(np.abs(y[valid_idx] - np.concatenate(predictions))))

    result = {
        "model": "pytorch_mlp_10_64_64_1_relu_with_breath_history",
        "features": FEATURES,
        "history_features": HISTORY_FEATURES,
        "seed": args.seed,
        "validation_fraction_by_breath": 0.1,
        "max_breaths": args.max_breaths or None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "rows": int(len(df)),
        "breaths": int(len(breath_ids)),
        "train_breaths": int(np.unique(df.loc[is_train, "breath_id"]).size),
        "validation_breaths": int(np.unique(df.loc[is_valid, "breath_id"]).size),
        "validation_inspiratory_rows": int(len(valid_idx)),
        "validation_inspiratory_mae": mae,
    }
    args.score_out.parent.mkdir(parents=True, exist_ok=True)
    args.score_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Score saved to {args.score_out}")


if __name__ == "__main__":
    main()
