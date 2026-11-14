"""Evaluate fixed mean/median ensembles and predeclared control-pattern groups."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from baseline_torch_lab import BASE_FEATURES, FEATURES, BreathGRU, engineer_features
from gru_long import BidirectionalGRU, save_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--forward', type=Path, required=True)
    p.add_argument('--bidirectional', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    df = pd.read_csv(a.data, nrows=400000)
    ids = df.breath_id.to_numpy().reshape(-1,80)
    assert ids.shape == (5000,80) and (ids == ids[:,:1]).all()
    valid_ids = np.random.RandomState(42).permutation(ids[:,0])[:500]
    valid = np.isin(ids[:,0], valid_ids)
    row_valid = valid.repeat(80)
    # Define abrupt changes using only training inspiratory rows, before inspecting errors.
    delta = df.groupby('breath_id', sort=False).u_in.diff().abs().fillna(0)
    cutoff = float(delta[(~row_valid) & df.u_out.eq(0)].quantile(.95))
    vdf = df.loc[row_valid].copy()
    truth = vdf.pressure.to_numpy().reshape(-1,80)
    insp = vdf.u_out.to_numpy().reshape(-1,80) == 0
    groups = {'all_inspiratory': insp,
              'valve_saturated': insp & vdf.u_in.isin([0,100]).to_numpy().reshape(-1,80),
              'valve_unsaturated': insp & (~vdf.u_in.isin([0,100])).to_numpy().reshape(-1,80),
              'abrupt_change': insp & (delta[row_valid].to_numpy().reshape(-1,80) > cutoff),
              'non_abrupt_change': insp & (delta[row_valid].to_numpy().reshape(-1,80) <= cutoff)}
    for r,c in sorted(set(zip(vdf.R,vdf.C))):
        groups[f'R={r},C={c}'] = insp & ((vdf.R==r)&(vdf.C==c)).to_numpy().reshape(-1,80)
    flat = engineer_features(df)[FEATURES].to_numpy(dtype=np.float32)
    predictions, metadata = {}, {}
    for label, folder, bi in [('forward',a.forward,False),('bidirectional',a.bidirectional,True)]:
        config = json.loads((folder/'config.json').read_text())
        members=[]
        for seed in config['seeds']:
            ck=torch.load(folder/f'seed_{seed}_best.pt',map_location='cpu')
            assert np.array_equal(ck['validation_breath_ids'],valid_ids)
            assert ck['config']['features']==FEATURES
            model=(BidirectionalGRU if bi else BreathGRU)(len(FEATURES)).cuda()
            model.load_state_dict(ck['model']); model.eval()
            xv=torch.from_numpy(((flat-ck['mean'])/ck['scale']).reshape(5000,80,-1)[valid])
            torch.cuda.synchronize(); start=time.perf_counter()
            with torch.no_grad():
                pred=torch.cat([model(z.cuda()).cpu() for z in xv.split(51)]).numpy()
            torch.cuda.synchronize(); seconds=time.perf_counter()-start
            assert np.isfinite(pred).all()
            mae=float(np.abs(pred[insp]-truth[insp]).mean())
            assert abs(mae-ck['best_mae'])<1e-5, (mae,ck['best_mae'])
            key=f'{label}_{seed}'; predictions[key]=pred; members.append(pred)
            metadata[key]=dict(best_epoch=ck['best_epoch'], inference_seconds=seconds,
                               parameters=sum(t.numel() for t in model.parameters()))
        predictions[f'{label}_mean']=np.mean(members,axis=0)
        predictions[f'{label}_median']=np.median(members,axis=0)
    report={'abrupt_change_threshold':cutoff,'threshold_definition':'training inspiratory 95th percentile of absolute u_in step change',
            'validation_breaths':int(valid.sum()),'models':{}}
    for name,pred in predictions.items():
        report['models'][name]={'metadata':metadata.get(name,{}),'groups':{}}
        for group,mask in groups.items():
            report['models'][name]['groups'][group]={'rows':int(mask.sum()),'mae':float(np.abs(pred[mask]-truth[mask]).mean()) if mask.any() else None}
    a.output.mkdir(parents=True,exist_ok=True)
    save_json(a.output/'comparison.json',report)
    np.savez_compressed(a.output/'validation_predictions.npz',**predictions,truth=truth,inspiratory=insp,breath_ids=ids[valid,0])
    for name in predictions:
        print(name,report['models'][name]['groups']['all_inspiratory'],flush=True)
    print('EVALUATION COMPLETE',flush=True)


if __name__=='__main__':
    main()
