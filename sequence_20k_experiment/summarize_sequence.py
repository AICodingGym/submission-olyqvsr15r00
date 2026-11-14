"""Compare saved per-model predictions, including fixed equal-weight ensembles."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('results')
    predictions={}; rows=[]; reference=None
    for name in ('gru','lstm','tcn'):
        folder=root/name
        config=json.loads((folder/'config.json').read_text())
        scores=json.loads((folder/'results.json').read_text())
        members=[]
        for seed in config['seeds']:
            data=np.load(folder/f'seed_{seed}_validation.npz')
            if reference is None:
                reference={k:data[k] for k in ('truth','inspiratory','breath_ids')}
            for key in reference:
                assert np.array_equal(reference[key],data[key]), key
            pred=data['predictions']; members.append(pred)
            predictions[f'{name}_{seed}']=pred
        predictions[f'{name}_mean']=np.mean(members,axis=0)
        predictions[f'{name}_median']=np.median(members,axis=0)
        rows.append({'model':name,'runs':scores,'training_validation_seconds':sum(x['elapsed_seconds'] for x in scores)})
    predictions['gru_lstm_mean']=(predictions['gru_mean']+predictions['lstm_mean'])/2
    predictions['gru_tcn_mean']=(predictions['gru_mean']+predictions['tcn_mean'])/2
    predictions['lstm_tcn_mean']=(predictions['lstm_mean']+predictions['tcn_mean'])/2
    predictions['all_mean']=np.mean([predictions[f'{n}_mean'] for n in ('gru','lstm','tcn')],axis=0)
    mask=reference['inspiratory']; truth=reference['truth']
    metrics={name:float(np.abs(pred[mask]-truth[mask]).mean()) for name,pred in predictions.items()}
    report={'validation_breaths':len(truth),'validation_inspiratory_rows':int(mask.sum()),'metrics':metrics,'runs':rows}
    (root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(root/'ensemble_predictions.npz',**predictions,**reference)
    print(json.dumps(report,indent=2),flush=True)
    print('SUITE COMPLETE',flush=True)


if __name__=='__main__':
    main()
