"""Generate an ABC or mirror-paired two-seed submission with frozen preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from baseline_torch_lab import FEATURES, engineer_features
from pattern_features import breath_features
from train_pattern_lstm import PatternLSTM


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--sample', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--paired', action='store_true')
    p.add_argument('--statistics', type=Path, help='Frozen pattern statistics; required for paired checkpoints')
    a = p.parse_args()
    if a.paired and a.statistics is None:
        p.error('--paired requires --statistics from the original ABC training run')
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    df = pd.read_csv(a.data)
    sample = pd.read_csv(a.sample)
    assert list(sample.columns) == ['id', 'pressure']
    assert df.id.is_unique and np.array_equal(df.id, sample.id)
    ids = df.breath_id.to_numpy().reshape(-1, 80)
    assert (ids == ids[:, :1]).all() and len(np.unique(ids[:, 0])) == len(ids)
    patterns_path = a.statistics or a.results / 'pattern_statistics.joblib'
    patterns = joblib.load(patterns_path)
    assert not np.intersect1d(patterns.training_breath_ids, ids[:, 0]).size
    pattern = patterns.transform(df)
    rich = breath_features(df)
    base = engineer_features(df)[FEATURES]
    names = list(base.columns) + list(pattern.columns) + list(rich.columns)
    raw = np.concatenate([base.to_numpy(), pattern.to_numpy(), rich.to_numpy()], axis=1).astype(np.float32).reshape(-1, 80, len(names))
    assert np.isfinite(raw).all()
    predictions = []
    manifest = []
    variant = 'paired' if a.paired else 'ABC'
    seeds = [931741, 2186703] if a.paired else [527193, 1804369]
    for seed in seeds:
        path = a.results / variant / str(seed) / 'best.pt'
        checkpoint = torch.load(path, map_location='cpu')
        assert checkpoint['features'] == names and checkpoint['variant'] == variant
        assert np.array_equal(np.sort(checkpoint['train_breath_ids']), patterns.training_breath_ids)
        assert not np.intersect1d(checkpoint['validation_breath_ids'], ids[:, 0]).size
        if a.paired:
            from train_paired_lstm import PairedLSTM
            model = PairedLSTM(len(names), paired=True).cuda().eval()
        else:
            model = PatternLSTM(len(names), large=True).cuda().eval()
        model.load_state_dict(checkpoint['model'])
        x = torch.from_numpy((raw - checkpoint['mean']) / checkpoint['scale'])
        with torch.no_grad():
            prediction = torch.cat([model(b.cuda()).cpu() for b in x.split(128)]).numpy()
        assert np.isfinite(prediction).all()
        predictions.append(prediction)
        manifest.append(dict(seed=seed, checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print('GPU inference complete', seed, torch.cuda.get_device_name(), flush=True)
    sample['pressure'] = np.mean(predictions, axis=0).reshape(-1)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(a.output, index=False)
    saved = pd.read_csv(a.output)
    assert list(saved.columns) == ['id', 'pressure'] and np.array_equal(saved.id, df.id)
    assert len(saved) == len(df) and saved.id.is_unique and np.isfinite(saved.pressure).all()
    record = dict(rows=len(saved), variant=variant,
                  validation_mae=0.27142059803009033 if a.paired else 0.32928264141082764,
                  statistics_sha256=hashlib.sha256(patterns_path.read_bytes()).hexdigest(),
                  device=torch.cuda.get_device_name(), checkpoints=manifest,
                  csv_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    a.output.with_suffix('.manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print('SUBMISSION_VERIFIED', json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
