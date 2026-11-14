"""Extend the existing GRU with fresh recorded seeds and recoverable checkpoints."""
import argparse
import json
import secrets
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from baseline_torch_lab import BASE_FEATURES, FEATURES, BreathGRU, engineer_features


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def save_checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--runs', type=int, default=3)
    a = p.parse_args()
    if min(a.epochs, a.patience, a.runs) < 1:
        p.error('epochs, patience and runs must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing accidental CPU training')
    torch.set_num_threads(8)
    a.output.mkdir(parents=True, exist_ok=True)
    config_path = a.output / 'config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        for key, value in [('epochs', a.epochs), ('patience', a.patience), ('runs', a.runs), ('data', str(a.data.resolve()))]:
            if config[key] != value:
                raise ValueError(f'Resume configuration differs: {key}')
    else:
        seeds = []
        while len(seeds) < a.runs:
            seed = secrets.randbelow(2**31)
            if seed not in seeds:
                seeds.append(seed)
        config = dict(seeds=seeds, epochs=a.epochs, patience=a.patience, runs=a.runs,
                      data=str(a.data.resolve()), max_breaths=5000, split_seed=42,
                      features=FEATURES, learning_rate=.001, loss='all-row MSE',
                      batch_size_breaths=51, gpu=torch.cuda.get_device_name(0),
                      torch_version=str(torch.__version__))
        save_json(config_path, config)
    print(json.dumps(config), flush=True)
    t0 = time.perf_counter()
    df = pd.read_csv(a.data, usecols=['breath_id', *BASE_FEATURES, 'pressure'], nrows=400000)
    ids = df.breath_id.to_numpy().reshape(-1, 80)
    if len(ids) != 5000 or not (ids == ids[:, :1]).all() or len(np.unique(ids[:, 0])) != len(ids):
        raise ValueError('Expected 5000 complete contiguous breaths')
    valid_ids = np.random.RandomState(42).permutation(ids[:, 0])[:500]
    valid = np.isin(ids[:, 0], valid_ids)
    df = engineer_features(df)
    flat = df[FEATURES].to_numpy(dtype=np.float32)
    mean, scale = flat[(~valid).repeat(80)].mean(0), flat[(~valid).repeat(80)].std(0)
    scale[scale == 0] = 1
    x = ((flat-mean)/scale).reshape(5000, 80, -1)
    y = df.pressure.to_numpy(dtype=np.float32).reshape(5000, 80)
    mask = df.u_out.to_numpy().reshape(5000, 80)[valid] == 0
    dataset = TensorDataset(torch.from_numpy(x[~valid]), torch.from_numpy(y[~valid]))
    xv = torch.from_numpy(x[valid])
    truth = y[valid][mask]
    print(f'Preprocessing seconds={time.perf_counter()-t0:.2f}; validation rows={len(truth)}', flush=True)
    results = []
    for seed in config['seeds']:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=51, shuffle=True, generator=generator, pin_memory=True)
        model = BreathGRU(len(FEATURES)).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        last_path = a.output / f'seed_{seed}_last.pt'
        best_path = a.output / f'seed_{seed}_best.pt'
        history, best, best_epoch, stale, elapsed = [], float('inf'), 0, 0, 0.
        if last_path.exists():
            ck = torch.load(last_path, map_location='cpu')
            model.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            generator.set_state(ck['loader_rng'])
            torch.set_rng_state(ck['torch_rng'])
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
            history, best, best_epoch, stale, elapsed = (ck[k] for k in ['history','best_mae','best_epoch','stale','elapsed_seconds'])
        for epoch in range(len(history)+1, a.epochs+1):
            if epoch > 10 and stale >= a.patience:
                break
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.train()
            total, count = 0., 0
            for bx, by in loader:
                bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(bx), by)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                total += loss.item()*by.numel()
                count += by.numel()
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(b.cuda()).cpu() for b in xv.split(51)]).numpy()
            mae = float(np.abs(pred[mask]-truth).mean())
            if not np.isfinite(mae):
                raise RuntimeError('Nonfinite validation MAE')
            torch.cuda.synchronize()
            duration = time.perf_counter()-start
            elapsed += duration
            improved = mae < best
            if improved:
                best, best_epoch, stale = mae, epoch, 0
            else:
                stale += 1
            row = dict(seed=seed, epoch=epoch, train_mse=total/count, validation_mae=mae,
                       epoch_seconds=duration, elapsed_seconds=elapsed)
            history.append(row)
            ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                      loader_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all(), history=history, best_mae=best,
                      best_epoch=best_epoch, stale=stale, elapsed_seconds=elapsed,
                      mean=mean, scale=scale, validation_breath_ids=valid_ids, config=config, seed=seed)
            save_checkpoint(last_path, ck)
            if improved:
                save_checkpoint(best_path, ck)
            save_json(a.output / f'seed_{seed}_history.json', history)
            print(json.dumps(row), flush=True)
        results.append(dict(seed=seed, best_mae=best, best_epoch=best_epoch,
                            epoch10_mae=history[9]['validation_mae'] if len(history)>=10 else None,
                            epochs_completed=len(history), elapsed_seconds=elapsed))
        save_json(a.output / 'results.json', results)
    print('COMPLETE ' + json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
