"""Predict all test rows from explicit model checkpoint directories."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sequence_compare import make_model, FEATURES, engineer_features


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--sample',type=Path,required=True)
    p.add_argument('--folders',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); torch.set_num_threads(8)
    df=pd.read_csv(a.data); sample=pd.read_csv(a.sample)
    assert list(sample.columns)==['id','pressure']
    assert df.id.is_unique and np.array_equal(df.id.to_numpy(),sample.id.to_numpy())
    ids=df.breath_id.to_numpy().reshape(-1,80)
    assert (ids==ids[:,:1]).all() and len(np.unique(ids[:,0]))==len(ids)
    flat=engineer_features(df)[FEATURES].to_numpy(dtype=np.float32)
    ensemble=np.zeros((len(ids),80),dtype=np.float64); manifest=[]
    for folder in a.folders:
        config=json.loads((folder/'config.json').read_text())
        seeds=config['seeds']; per_model=np.zeros_like(ensemble)
        for seed in seeds:
            path=folder/f'seed_{seed}_best.pt'
            ck=torch.load(path,map_location='cpu'); assert ck['config']['features']==FEATURES
            model=make_model(config['model'],len(FEATURES)).cuda()
            model.load_state_dict(ck['model']); model.eval()
            x=torch.from_numpy(((flat-ck['mean'])/ck['scale']).reshape(len(ids),80,-1))
            with torch.no_grad():
                pred=torch.cat([model(b.cuda()).cpu() for b in x.split(128)]).numpy()
            assert np.isfinite(pred).all()
            per_model+=pred/len(seeds)
            manifest.append(dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=seed))
        ensemble+=per_model/len(a.folders)
    output=pd.DataFrame({'id':df.id,'pressure':ensemble.reshape(-1)})
    assert np.isfinite(output.pressure).all() and len(output)==len(sample)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    output.to_csv(a.output,index=False)
    loaded=pd.read_csv(a.output)
    assert list(loaded.columns)==['id','pressure'] and np.array_equal(loaded.id,sample.id)
    assert np.isfinite(loaded.pressure).all() and loaded.id.is_unique
    a.output.with_suffix('.manifest.json').write_text(json.dumps({'rows':len(output),'checkpoints':manifest,'csv_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()},indent=2)+'\n')
    print('PREDICTIONS VERIFIED',len(output),str(a.output),flush=True)


if __name__=='__main__':
    main()
