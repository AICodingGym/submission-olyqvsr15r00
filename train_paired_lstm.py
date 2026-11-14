"""Mirror-paired input experiment with matched, warm-started continuation controls."""
import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from baseline_torch_lab import FEATURES, engineer_features
from pattern_features import breath_features
from train_pattern_lstm import PatternLSTM, write_json


class PairedLSTM(nn.Module):
    def __init__(self, features=61, paired=True):
        super().__init__()
        self.paired = paired
        self.base = PatternLSTM(features * (2 if paired else 1), large=True)

    def forward(self, x):
        # Reverse only within each breath, never across breaths or features.
        return self.base(torch.cat([x, x.flip(1)], dim=-1) if self.paired else x)

    def initialize(self, source, features=61):
        state = self.base.state_dict()
        for name, old in source.items():
            if state[name].shape == old.shape:
                state[name] = old.clone()
            else:
                # Every dense skip input begins with the raw feature channels.
                assert self.paired and old.ndim == 2
                assert state[name].shape == (old.shape[0], old.shape[1] + features)
                new = torch.zeros_like(state[name])
                new[:, :features] = old[:, :features]
                new[:, 2*features:] = old[:, features:]
                state[name] = new
        self.base.load_state_dict(state)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--parents', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=40)
    a = p.parse_args()
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    a.output.mkdir(parents=True, exist_ok=True)
    parent_seeds = [527193, 1804369]
    training_seeds = [931741, 2186703]
    df = pd.read_csv(a.data)
    ids = df.breath_id.to_numpy().reshape(-1, 80)[:, 0]
    y = df.pressure.to_numpy(dtype=np.float32).reshape(-1, 80)
    mask = df.u_out.to_numpy().reshape(-1, 80) == 0
    ref = np.load(a.parents / 'ABC/ensemble.npz')
    vi = np.array([np.flatnonzero(ids == b)[0] for b in ref['breath_ids']])
    train = ~np.isin(ids, ref['breath_ids'])
    assert train.sum() == 67405 and len(vi) == 500
    assert np.array_equal(y[vi], ref['truth']) and np.array_equal(mask[vi], ref['inspiratory'])
    inputs = df.drop(columns=['pressure', 'id']); del df
    patterns = joblib.load(a.parents / 'pattern_statistics.joblib')
    assert np.array_equal(np.sort(ids[train]), patterns.training_breath_ids)
    f = patterns.transform(inputs); b = breath_features(inputs); base = engineer_features(inputs)[FEATURES]
    names = list(base) + list(f) + list(b)
    raw = np.concatenate([base.to_numpy(), f.to_numpy(), b.to_numpy()], axis=1).astype(np.float32).reshape(-1, 80, len(names))
    del inputs, f, b, base
    gc.collect()
    parent0 = torch.load(a.parents / 'ABC' / str(parent_seeds[0]) / 'best.pt', map_location='cpu')
    assert parent0['features'] == names
    normal = (raw - parent0['mean']) / parent0['scale']; del raw
    # Keep inputs on device to remove host transfer/loader gaps from every batch.
    tx = torch.from_numpy(normal[train]).cuda(); vx = torch.from_numpy(normal[vi]).cuda(); del normal
    ty = torch.from_numpy(y[train]).cuda(); tm = torch.from_numpy(mask[train]).cuda()
    vy = torch.from_numpy(y[vi]).cuda(); vm = torch.from_numpy(mask[vi]).cuda()

    def predict(model):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(z) for z in vx.split(128)])

    # Warm-start equivalence is essential: pairing must add no initial advantage.
    ordinary = PatternLSTM(len(names), large=True).cuda().eval()
    ordinary.load_state_dict(parent0['model'])
    paired = PairedLSTM(len(names), True).cuda().eval(); paired.initialize(parent0['model'])
    with torch.no_grad():
        delta = (ordinary(vx[:16]) - paired(vx[:16])).abs().max().item()
        assert delta < 1e-4
    del ordinary, paired
    torch.cuda.empty_cache()

    # Timed optimizer steps on disposable models; no benchmark weights are used.
    benchmarks = []
    for batch in [128, 256, 512, 1024]:
        model = PairedLSTM(len(names), True).cuda(); model.initialize(parent0['model'])
        opt = torch.optim.Adam(model.parameters(), lr=.0003)
        torch.cuda.reset_peak_memory_stats()
        try:
            for iteration in range(8):
                if iteration == 2:
                    torch.cuda.synchronize(); start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                pred = model(tx[:batch]); loss = (pred - ty[:batch]).abs().masked_select(tm[:batch]).mean()
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            torch.cuda.synchronize()
            benchmarks.append(dict(batch=batch, breaths_per_second=6*batch/(time.perf_counter()-start),
                                   peak_memory_bytes=torch.cuda.max_memory_allocated()))
        except torch.cuda.OutOfMemoryError:
            benchmarks.append(dict(batch=batch, out_of_memory=True))
        del model, opt
        gc.collect(); torch.cuda.empty_cache()
    valid_bench = [b for b in benchmarks if 'breaths_per_second' in b]
    fastest = max(b['breaths_per_second'] for b in valid_bench)
    # Prefer the smaller batch if throughput is within 5% of the best.
    batch = min(b['batch'] for b in valid_bench if b['breaths_per_second'] >= .95*fastest)
    config = dict(epochs=a.epochs, parent_seeds=parent_seeds, training_seeds=training_seeds, batch=batch,
                  learning_rate=.0003, patience=10, plateau_patience=3, features=names,
                  pairing='concat x[t] with x[79-t]; outputs remain chronological',
                  benchmark=benchmarks, initial_equivalence_max_difference=delta,
                  training_breaths=int(train.sum()), validation_breaths=len(vi), gpu=torch.cuda.get_device_name())
    write_json(a.output/'config.json', config); print('BENCHMARK', json.dumps(config), flush=True)
    results = []
    for variant in ['control', 'paired']:
        members = []
        for source_seed, seed in zip(parent_seeds, training_seeds):
            dest = a.output/variant/str(seed); dest.mkdir(parents=True, exist_ok=True)
            source = torch.load(a.parents/'ABC'/str(source_seed)/'best.pt', map_location='cpu')
            assert np.array_equal(source['mean'], parent0['mean']) and np.array_equal(source['scale'], parent0['scale'])
            assert np.array_equal(source['train_breath_ids'], ids[train])
            assert np.array_equal(source['validation_breath_ids'], ids[vi])
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = PairedLSTM(len(names), variant == 'paired').cuda(); model.initialize(source['model'])
            optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3, min_lr=1e-6)
            gen = torch.Generator(device='cuda').manual_seed(seed)
            initial = predict(model)
            saved_parent = np.load(a.parents/'ABC'/str(source_seed)/'validation.npz')['prediction']
            assert np.allclose(initial.cpu().numpy(), saved_parent, atol=1e-4)
            best = float((initial[vm]-vy[vm]).abs().mean()); best_epoch = 0; history = []
            def save_best(pred):
                torch.save(dict(model=model.state_dict(), variant=variant, seed=seed, parent_seed=source_seed,
                                mean=source['mean'], scale=source['scale'], features=names, config=config,
                                epoch=best_epoch, validation_mae=best, train_breath_ids=ids[train],
                                validation_breath_ids=ids[vi]), dest/'best.pt')
                np.savez_compressed(dest/'validation.npz', prediction=pred.detach().cpu().numpy(),
                                    truth=y[vi], inspiratory=mask[vi], breath_ids=ids[vi])
            save_best(initial)
            start_run = time.perf_counter()
            for epoch in range(1, a.epochs+1):
                start = time.perf_counter(); model.train()
                permutation = torch.randperm(len(tx), device='cuda', generator=gen)
                total = torch.zeros((), device='cuda'); count = torch.zeros((), device='cuda')
                for index in permutation.split(batch):
                    bx, by, bm = tx[index], ty[index], tm[index]
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(bx); ae = (pred-by).abs()
                    n = bm.sum(); loss = (ae*bm).sum()/n
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                    total += loss.detach()*n; count += n
                vp = predict(model); score = float((vp[vm]-vy[vm]).abs().mean())
                assert np.isfinite(score)
                scheduler.step(score)
                if score < best:
                    best, best_epoch = score, epoch; save_best(vp)
                row = dict(variant=variant, seed=seed, epoch=epoch, validation_mae=score,
                           best_mae=best, train_mae=float(total/count), lr=optimizer.param_groups[0]['lr'],
                           seconds=time.perf_counter()-start)
                history.append(row); write_json(dest/'history.json', history)
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                generator=gen.get_state(), epoch=epoch, config=config), dest/'last.pt')
                print(json.dumps(row), flush=True)
                if epoch-best_epoch >= 10:
                    break
            members.append(np.load(dest/'validation.npz')['prediction'])
            results.append(dict(variant=variant, seed=seed, parent_seed=source_seed, mae=best,
                                best_epoch=best_epoch, epochs=len(history), seconds=time.perf_counter()-start_run))
            write_json(a.output/'results.json', results)
            del model, optimizer, scheduler
            gc.collect(); torch.cuda.empty_cache()
        ensemble = np.mean(members, axis=0)
        np.savez_compressed(a.output/variant/'ensemble.npz', prediction=ensemble,
                            truth=y[vi], inspiratory=mask[vi], breath_ids=ids[vi])
        print('VARIANT_COMPLETE', variant, float(np.abs(ensemble[mask[vi]]-y[vi][mask[vi]]).mean()), flush=True)
    print('PAIRED_EXPERIMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
