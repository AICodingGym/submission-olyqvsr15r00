"""Controlled A/B/C feature and architecture screen; no test predictions/submission."""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_torch_lab import FEATURES, engineer_features
from pattern_features import ControlPatterns, breath_features


class PatternLSTM(nn.Module):
    def __init__(self, features, large=False):
        super().__init__()
        self.large = large
        if not large:
            self.layers = nn.LSTM(features, 56, num_layers=2, batch_first=True, bidirectional=True)
            self.head = nn.Linear(112, 1)
        else:
            self.layers = nn.ModuleList()
            size = features
            for width in [256, 128, 64]:
                self.layers.append(nn.LSTM(size, width, batch_first=True, bidirectional=True))
                size += 2 * width
            self.head = nn.Sequential(nn.Linear(size, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x):
        if self.large:
            for layer in self.layers:
                x = torch.cat([x, layer(x)[0]], dim=-1)
        else:
            x = self.layers(x)[0]
        return self.head(x).squeeze(-1)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    a.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data)
    grouped_ids = df.breath_id.to_numpy().reshape(-1, 80)
    assert (grouped_ids == grouped_ids[:, :1]).all()
    ids = grouped_ids[:, 0]
    val_ids = np.random.RandomState(42).permutation(ids[:5000])[:500]
    val = np.isin(ids, val_ids)
    train = ~val
    if a.smoke:
        train[:] = False
        train[np.flatnonzero(~val)[:256]] = True
    reference = np.load(a.reference)
    # Align to the exact existing validation order, not just an equivalent set.
    vi = np.array([np.flatnonzero(ids == b)[0] for b in reference['breath_ids']])
    assert np.array_equal(np.sort(ids[val]), np.sort(reference['breath_ids']))
    y = df.pressure.to_numpy(dtype=np.float32).reshape(-1, 80)
    mask = df.u_out.to_numpy().reshape(-1, 80) == 0
    assert np.array_equal(y[vi], reference['truth'])
    assert np.array_equal(mask[vi], reference['inspiratory'])
    inputs = df.drop(columns=['pressure', 'id'])
    del df
    pattern = ControlPatterns().fit(inputs.loc[np.repeat(train, 80)])
    assert not np.intersect1d(pattern.training_breath_ids, ids[val]).size
    joblib.dump(pattern, a.output / 'pattern_statistics.joblib')
    patterns = pattern.transform(inputs)
    rich = breath_features(inputs)
    base = engineer_features(inputs)[FEATURES]
    names = list(base.columns) + list(patterns.columns) + list(rich.columns)
    all_x = np.concatenate([base.to_numpy(), patterns.to_numpy(), rich.to_numpy()], axis=1).astype(np.float32).reshape(len(ids), 80, -1)
    assert np.isfinite(all_x).all()
    audit = dict(train_breaths=int(train.sum()), validation_breaths=int(val.sum()),
                 train_only_pattern_statistics=True, pressure_excluded_from_inputs=True,
                 unseen_validation_fraction=float(patterns.pattern_unseen.to_numpy().reshape(-1, 80)[vi].mean()))
    write_json(a.output / 'audit.json', audit)
    del inputs, base, patterns, rich
    gc.collect()
    configuration = dict(epochs=a.epochs, seeds=a.seeds, features=names, early_stopping_patience=10,
                         plateau_patience=3, batch_size=128, lr=.001, loss='inspiratory row MAE',
                         large_widths=[256, 128, 64], from_scratch=True, smoke=a.smoke)
    write_json(a.output / 'config.json', configuration)
    results = []
    for variant, width in [('control', 19), ('A', 27), ('AB', len(names)), ('ABC', len(names))]:
        raw = all_x[:, :, :width]
        mean = raw[train].mean(axis=(0, 1)); scale = raw[train].std(axis=(0, 1))
        # An unseen-group flag is constant zero in training but can be one at
        # inference. Do not amplify such constant features by a million.
        scale = np.where(scale < 1e-6, 1.0, scale)
        normalized = (raw - mean) / scale
        assert np.isfinite(normalized).all()
        ds = TensorDataset(torch.from_numpy(normalized[train]), torch.from_numpy(y[train]), torch.from_numpy(mask[train]))
        vx = torch.from_numpy(normalized[vi]); del normalized
        members = []
        for seed in a.seeds:
            dest = a.output / variant / str(seed); dest.mkdir(parents=True, exist_ok=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = PatternLSTM(width, variant == 'ABC').cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3, min_lr=1e-5)
            gen = torch.Generator().manual_seed(seed)
            loader = DataLoader(ds, batch_size=128, shuffle=True, generator=gen, pin_memory=True)
            history = []; best = float('inf'); best_epoch = 0; start = time.perf_counter()
            def predict():
                model.eval()
                with torch.no_grad():
                    return torch.cat([model(b.cuda()) .cpu() for b in vx.split(128)]).numpy()
            for epoch in range(1, (1 if a.smoke else a.epochs) + 1):
                epoch_start = time.perf_counter(); model.train(); total = 0.; count = 0
                for bx, by, bm in loader:
                    bx, by, bm = [b.cuda(non_blocking=True) for b in (bx, by, bm)]
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(bx); loss = (pred[bm] - by[bm]).abs().mean()
                    assert torch.isfinite(loss)
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                    total += loss.item() * int(bm.sum()); count += int(bm.sum())
                vp = predict(); score = float(np.abs(vp[mask[vi]] - y[vi][mask[vi]]).mean())
                assert np.isfinite(score)
                scheduler.step(score)
                row = dict(variant=variant, seed=seed, epoch=epoch, validation_mae=score,
                           train_mae=total/count, lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-epoch_start)
                history.append(row); print(json.dumps(row), flush=True)
                if score < best:
                    best = score; best_epoch = epoch
                    torch.save(dict(model=model.state_dict(), mean=mean, scale=scale, features=names[:width],
                                    variant=variant, seed=seed, epoch=epoch, validation_mae=best,
                                    train_breath_ids=ids[train], validation_breath_ids=ids[vi], config=configuration), dest / 'best.pt')
                    np.savez_compressed(dest / 'validation.npz', prediction=vp, truth=y[vi], inspiratory=mask[vi], breath_ids=ids[vi])
                write_json(dest / 'history.json', history)
                if epoch - best_epoch >= 10:
                    break
            members.append(np.load(dest / 'validation.npz')['prediction'])
            results.append(dict(variant=variant, seed=seed, mae=best, best_epoch=best_epoch, epochs=len(history),
                                parameters=sum(t.numel() for t in model.parameters()), seconds=time.perf_counter()-start))
            write_json(a.output / 'results.json', results)
            del model, optimizer, scheduler, loader
            torch.cuda.empty_cache()
        ensemble = np.mean(members, axis=0)
        np.savez_compressed(a.output / variant / 'ensemble.npz', prediction=ensemble, truth=y[vi], inspiratory=mask[vi], breath_ids=ids[vi])
        print('VARIANT_COMPLETE', variant, float(np.abs(ensemble[mask[vi]]-y[vi][mask[vi]]).mean()), flush=True)
        del ds, vx
        gc.collect()
    print('PATTERN_EXPERIMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
