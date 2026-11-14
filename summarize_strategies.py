"""Evaluate predeclared equal-weight combinations on the unchanged holdout."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    a=p.parse_args(); reference=None; ensembles={}; folders={}; runs={}
    entries=[('parent_lstm',a.parents/'lstm'),('parent_tcn',a.parents/'tcn')]
    entries += [(s+'_'+m,a.results/s/m) for s,m in [('mse_control','lstm'),('mae','lstm'),('mae','tcn'),('delta','lstm'),('categorical','lstm'),('cnn','lstm')]]
    for name,folder in entries:
        config=json.loads((folder/'config.json').read_text()); members=[]
        for seed in config['seeds']:
            d=np.load(folder/f'seed_{seed}_validation.npz')
            if reference is None: reference={k:d[k] for k in ['truth','inspiratory','breath_ids']}
            for key in reference: assert np.array_equal(d[key],reference[key]),(name,key)
            assert np.isfinite(d['predictions']).all(); members.append(d['predictions'])
        ensembles[name]=np.mean(members,axis=0); folders[name]=str(folder.resolve())
        runs[name]=json.loads((folder/'results.json').read_text())
    candidates={'reference':['parent_lstm','parent_tcn']}
    for s in ['mse_control','mae','delta','categorical','cnn']:
        for t in ['parent_tcn','mae_tcn']:
            candidates[s+'_'+t]=[s+'_lstm',t]
    mask=reference['inspiratory']; truth=reference['truth']
    score=lambda pred: float(np.abs(pred[mask]-truth[mask]).mean())
    singles={k:score(v) for k,v in ensembles.items()}
    metrics={k:score(np.mean([ensembles[x] for x in names],axis=0)) for k,names in candidates.items()}
    winner=min(metrics,key=metrics.get)
    report=dict(individual_ensembles=singles,combinations=metrics,selected=winner,
                selected_folders=[folders[k] for k in candidates[winner]],runs=runs,
                validation_breaths=len(truth),validation_rows=int(mask.sum()))
    (a.results/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.results/'ensemble_predictions.npz',**ensembles,**reference)
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2),flush=True)
    print('STRATEGIES_COMPLETE',flush=True)


if __name__=='__main__':
    main()
