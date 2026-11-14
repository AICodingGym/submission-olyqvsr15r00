#!/usr/bin/env python3
"""Experiment suite for improving the ventilator pressure baseline.

Quick pod run:
    python baseline_torch_lab.py

Full training set:
    python baseline_torch_lab.py --max-breaths 0

The split is always the seed-42 90/10 split by complete breath_id groups.
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
    "u_in_cum", "u_out_cum", "u_in_prev1", "u_in_prev2", "u_in_prev3",
    "u_in_prev4", "u_in_prev8", "u_out_prev1", "u_in_roll3_mean",
    "u_in_roll8_mean",
]
PHASE_FEATURES = ["step_in_breath", "breath_progress", "time_since_start", "time_since_uout_change"]
FEATURES = BASE_FEATURES + HISTORY_FEATURES + PHASE_FEATURES
POD_DATA = Path("/home/jjqian/ventilator-pressure-prediction/train.csv")
REPO_DATA = Path(__file__).parent / "data" / "ventilator-pressure-prediction" / "train.csv"
DEFAULT_DATA = POD_DATA if POD_DATA.is_file() else REPO_DATA
DEFAULT_RESULTS = Path(__file__).parent / "baseline_torch_lab_results.json"


class RowMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BreathCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_features, 64, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(64, 64, 5, padding=2), nn.ReLU())
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2)).transpose(1, 2)).squeeze(-1)


class BreathGRU(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.gru = nn.GRU(n_features, 64, num_layers=2, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.gru(x)[0]).squeeze(-1)


def engineer_features(df):
    """Create causal control history and phase features within each breath."""
    g = df.groupby("breath_id", sort=False)
    step = g.cumcount()
    df["step_in_breath"] = step.astype(np.float32)
    # 80 rows per breath in the supplied files.
    df["breath_progress"] = step.to_numpy(dtype=np.float32) / 79.0
    df["time_since_start"] = (df["time_step"] - g["time_step"].transform("first")).astype(np.float32)
    for col in ("u_in", "u_out"):
        df[f"{col}_cum"] = g[col].cumsum().astype(np.float32)
    for lag in (1, 2, 3, 4, 8):
        df[f"u_in_prev{lag}"] = g["u_in"].shift(lag, fill_value=0).astype(np.float32)
    df["u_out_prev1"] = g["u_out"].shift(1, fill_value=0).astype(np.float32)
    for window in (3, 8):
        df[f"u_in_roll{window}_mean"] = (
            g["u_in"].rolling(window, min_periods=1).mean()
            .reset_index(level=0, drop=True).astype(np.float32)
        )
    # Track the last control transition's row index, restarting each breath.
    prev_out = g["u_out"].shift(1, fill_value=0)
    transition = df["u_out"].ne(prev_out).to_numpy()
    last_change = np.zeros(len(df), dtype=np.float32)
    ids = df["breath_id"].to_numpy()
    start = 0
    while start < len(df):
        end = start + 1
        while end < len(df) and ids[end] == ids[start]:
            end += 1
        local = np.flatnonzero(transition[start:end])
        idx = np.arange(end - start, dtype=np.float32)
        last_change[start:end] = np.maximum.accumulate(np.where(transition[start:end], idx, 0))
        start = end
    df["time_since_uout_change"] = (step.to_numpy(dtype=np.float32) - last_change)
    return df


def make_model(name, n_features):
    if name == "mlp":
        return RowMLP(n_features)
    if name == "cnn":
        return BreathCNN(n_features)
    if name == "gru":
        return BreathGRU(n_features)
    raise ValueError(name)


def fit_torch(name, x, y, train_breath_mask, valid_breath_mask, valid_insp, seed,
              epochs, batch_size, lr, weight_decay, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if name == "mlp":
        n_breath, n_step, n_feat = x.shape
        xtrain = x[train_breath_mask].reshape(-1, n_feat)
        ytrain = y[train_breath_mask].reshape(-1)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(xtrain), torch.from_numpy(ytrain)),
            batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda",
        )
    else:
        xtrain = x[train_breath_mask]
        ytrain = y[train_breath_mask]
        loader = DataLoader(
            TensorDataset(torch.from_numpy(xtrain), torch.from_numpy(ytrain)),
            batch_size=max(1, batch_size // x.shape[1]), shuffle=True,
            generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda",
        )
    model = make_model(name, x.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    if name == "mlp":
        xv = torch.from_numpy(x[valid_breath_mask].reshape(-1, x.shape[-1]))
    else:
        xv = torch.from_numpy(x[valid_breath_mask])
    yv_insp = y[valid_breath_mask][valid_insp]
    model.train()
    for epoch in range(epochs):
        total, seen = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            count = by.numel()
            total += loss.item() * count
            seen += count
        model.eval()
        with torch.no_grad():
            if name == "mlp":
                pflat = torch.cat([model(z.to(device)).cpu() for z in xv.split(batch_size)]).numpy()
                val_pred = pflat.reshape(-1, x.shape[1])[valid_insp]
            else:
                seq_batch = max(1, batch_size // x.shape[1])
                vp = torch.cat([model(z.to(device)).cpu() for z in xv.split(seq_batch)]).numpy()
                val_pred = vp[valid_insp]
        val_mae = float(np.mean(np.abs(yv_insp - val_pred)))
        print(f"{name} seed={seed} lr={lr:g} wd={weight_decay:g} epoch={epoch + 1}/{epochs} "
              f"train_mse={total / seen:.6f} validation_inspiratory_mae={val_mae:.8f}")
        model.train()
    model.eval()
    with torch.no_grad():
        if name == "mlp":
            pv = torch.cat([model(z.to(device)).cpu() for z in xv.split(batch_size)]).numpy()
            pv = pv.reshape(-1, x.shape[1])
        else:
            seq_batch = max(1, batch_size // x.shape[1])
            pv = torch.cat([model(z.to(device)).cpu() for z in xv.split(seq_batch)]).numpy()
    pred = pv[valid_insp]
    truth = y[valid_breath_mask][valid_insp]
    return pred, truth


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--max-breaths", type=int, default=5000, help="0 reads all rows")
    p.add_argument("--seeds", default="42,43,44", help="training seeds; split remains seed 42")
    p.add_argument("--models", default="mlp,cnn,gru,tree")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--tune", action="store_true", help="try lr/weight decay grid (3x2)")
    p.add_argument("--tree-sample-rows", type=int, default=250000)
    p.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    a = p.parse_args()
    if a.max_breaths < 0 or a.epochs < 1 or a.batch_size < 1:
        p.error("max-breaths must be >= 0; epochs and batch-size must be positive")
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    seeds = [int(s) for s in a.seeds.split(",")]
    allowed = {"mlp", "cnn", "gru", "tree"}
    if not models or set(models) - allowed:
        p.error(f"models must be selected from {sorted(allowed)}")

    split_seed = 42
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nrows = a.max_breaths * 80 if a.max_breaths else None
    df = pd.read_csv(a.data, usecols=["breath_id", *BASE_FEATURES, "pressure"], nrows=nrows)
    df = engineer_features(df)
    breath_ids = df["breath_id"].drop_duplicates().to_numpy()
    if len(df) != len(breath_ids) * 80:
        raise ValueError("Expected complete contiguous 80-row breaths")
    # Match baseline_torch.py's fixed split construction exactly.
    valid_ids = np.random.RandomState(split_seed).permutation(breath_ids)
    valid_ids = valid_ids[:int(np.ceil(0.1 * len(breath_ids)))]
    is_valid_row = df["breath_id"].isin(valid_ids).to_numpy()
    nbreath, nstep = len(breath_ids), 80
    valid_breath = is_valid_row.reshape(nbreath, nstep)[:, 0]
    train_breath = ~valid_breath

    x = df[FEATURES].to_numpy(dtype=np.float32).reshape(nbreath, nstep, len(FEATURES))
    y = df["pressure"].to_numpy(dtype=np.float32).reshape(nbreath, nstep)
    uout = df["u_out"].to_numpy().reshape(nbreath, nstep)
    # Fit normalization only on training rows, then apply it to both partitions.
    train_rows = train_breath.repeat(nstep)
    flat = x.reshape(-1, len(FEATURES))
    mean = flat[train_rows].mean(axis=0)
    scale = flat[train_rows].std(axis=0)
    scale[scale == 0] = 1.0
    x = ((flat - mean) / scale).reshape(nbreath, nstep, len(FEATURES)).astype(np.float32)
    valid_insp = uout[valid_breath] == 0

    settings = [(1e-3, 0.0)]
    if a.tune:
        settings = [(lr, wd) for lr in (5e-4, 1e-3, 2e-3) for wd in (0.0, 1e-5)]

    results = []
    for name in models:
        if name == "tree":
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
            except ImportError:
                print("Skipping tree: scikit-learn is unavailable")
                continue
            rng = np.random.RandomState(split_seed)
            tr = flat[train_rows]
            ty = y[train_breath].reshape(-1)
            n_sample = min(a.tree_sample_rows, len(ty))
            chosen = rng.choice(len(ty), size=n_sample, replace=False)
            va_rows = np.flatnonzero(is_valid_row & (df["u_out"].to_numpy() == 0))
            tree = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=31,
                                                 learning_rate=0.08, l2_regularization=1.0,
                                                 random_state=split_seed)
            tree.fit(tr[chosen], ty[chosen])
            pred = tree.predict(flat[va_rows])
            truth = y.reshape(-1)[va_rows]
            results.append({"model": "hist_gradient_boosting", "seed": split_seed,
                            "mae": float(np.mean(np.abs(truth - pred))),
                            "validation_inspiratory_rows": int(len(va_rows)),
                            "tree_fit_rows": int(n_sample)})
            continue

        for lr, wd in settings:
            for seed in seeds:
                pred, truth = fit_torch(name, x, y, train_breath, valid_breath,
                                        valid_insp, seed, a.epochs, a.batch_size,
                                        lr, wd, device)
                mae = float(np.mean(np.abs(truth - pred)))
                results.append({"model": name, "seed": seed, "learning_rate": lr,
                                "weight_decay": wd, "epochs": a.epochs,
                                "batch_size": a.batch_size, "mae": mae,
                                "validation_inspiratory_rows": int(len(truth))})
                print(f"RESULT {name} seed={seed} lr={lr:g} wd={wd:g} inspiratory_MAE={mae:.8f}")

    # Per lung-setting diagnostics for the best validation prediction model
    # require retaining predictions; report group counts for targeted follow-up
    # and write per-combination MAE from a final default MLP prediction.
    # Refit a single seed-42 MLP to obtain consistent diagnostic residuals.
    diagnostics = {"by_R_C": [], "by_breath": []}
    if "mlp" in models:
        pred, truth = fit_torch("mlp", x, y, train_breath, valid_breath, valid_insp,
                                42, a.epochs, a.batch_size, 1e-3, 0.0, device)
        val_df = df.loc[is_valid_row & (df["u_out"].to_numpy() == 0), ["R", "C"]].copy()
        val_df["abs_error"] = np.abs(truth - pred)
        diagnostics["by_R_C"] = (val_df.groupby(["R", "C"], as_index=False)
                                 .agg(inspiratory_rows=("abs_error", "size"), mae=("abs_error", "mean"))
                                 .to_dict(orient="records"))
        val_breath_ids = df.loc[is_valid_row & (df["u_out"].to_numpy() == 0), "breath_id"].to_numpy()
        val_df["breath_id"] = val_breath_ids
        diagnostics["by_breath"] = (val_df.groupby("breath_id", as_index=False)
                                     .agg(inspiratory_rows=("abs_error", "size"), mae=("abs_error", "mean"))
                                     .to_dict(orient="records"))

    summary = {
        "split_seed": split_seed,
        "split_method": "RandomState(42) permutation of ordered unique breath_id; first ceil(10%) validation",
        "validation_metric": "MAE on rows where u_out == 0",
        "features": FEATURES,
        "rows": int(len(df)),
        "breaths": int(nbreath),
        "train_breaths": int(train_breath.sum()),
        "validation_breaths": int(valid_breath.sum()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "results": results,
        "mlp_seed42_mae_by_R_C": diagnostics,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved experiment results to {a.output}")


if __name__ == "__main__":
    main()
