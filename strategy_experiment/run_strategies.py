"""Full-data paired five-epoch screens of independently implemented adaptations."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from strategy_models import StrategyModel, objective
from sequence_compare import FEATURES, BASE_FEATURES, engineer_features, save_json, save_checkpoint


def rc_codes(df):
    first=df.iloc[::80]
    r=first.R.to_numpy(); c=first.C.to_numpy()
    assert np.isin(r,[5,20,50]).all() and np.isin(c,[10,20,50]).all()
    return np.concatenate([r[:,None]==np.array([5,20,50]),c[:,None]==np.array([10,20,50])],axis=1).astype(np.float32)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=5)
    a=p.parse_args(); torch.set_num_threads(8)
    assert torch.cuda.is_available() and a.epochs>0
    df=pd.read_csv(a.data,usecols=['breath_id',*BASE_FEATURES,'pressure'])
    ids=df.breath_id.to_numpy().reshape(-1,80); n=len(ids)
    assert n==67905 and (ids==ids[:,:1]).all() and len(np.unique(ids[:,0]))==n
    valid_ids=np.random.RandomState(42).permutation(ids[:5000,0])[:500]
    valid=np.isin(ids[:,0],valid_ids); assert valid.sum()==500
    y=df.pressure.to_numpy(dtype=np.float32).reshape(n,80)
    insp=df.u_out.to_numpy().reshape(n,80)==0
    rc=rc_codes(df)
    flat=engineer_features(df)[FEATURES].to_numpy(dtype=np.float32); del df
    tasks=[('mse_control','lstm'),('mae','lstm'),('mae','tcn'),('delta','lstm'),('categorical','lstm'),('cnn','lstm')]
    for strategy,arch in tasks:
        parent=a.parents/arch; folder=a.output/strategy/arch; folder.mkdir(parents=True,exist_ok=True)
        pc=json.loads((parent/'config.json').read_text())
        config=dict(strategy=strategy,model=arch,seeds=pc['seeds'],features=FEATURES,
                    training_breaths=int((~valid).sum()),validation_breaths=500,epochs=a.epochs,
                    learning_rate=.0003,optimizer='fresh Adam for every branch',delta_weight=.2 if strategy=='delta' else 0,
                    parent=str(parent.resolve()),gpu=torch.cuda.get_device_name(0))
        if (folder/'config.json').exists(): assert json.loads((folder/'config.json').read_text())==config
        save_json(folder/'config.json',config); results=[]
        for seed in config['seeds']:
            source=torch.load(parent/f'seed_{seed}_best.pt',map_location='cpu')
            assert np.array_equal(source['validation_breath_ids'],valid_ids)
            x=((flat-source['mean'])/source['scale']).reshape(n,80,-1)
            data=TensorDataset(torch.from_numpy(x[~valid]),torch.from_numpy(y[~valid]),
                               torch.from_numpy(insp[~valid]),torch.from_numpy(rc[~valid]))
            xv=torch.from_numpy(x[valid]); rv=torch.from_numpy(rc[valid]); vm=insp[valid]; truth=y[valid][vm]; del x
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model=StrategyModel(arch,strategy).cuda(); model.base.load_state_dict(source['model'])
            opt=torch.optim.Adam(model.parameters(),lr=.0003)
            generator=torch.Generator(); generator.set_state(source['loader_rng'])
            loader=DataLoader(data,batch_size=51,shuffle=True,generator=generator,pin_memory=True)
            last=folder/f'seed_{seed}_last.pt'; bestpath=folder/f'seed_{seed}_best.pt'
            def predict():
                model.eval()
                with torch.no_grad():
                    return torch.cat([model(z.cuda(),r.cuda()).cpu() for z,r in zip(xv.split(51),rv.split(51))]).numpy()
            def state():
                return dict(model=model.state_dict(),optimizer=opt.state_dict(),loader_rng=generator.get_state(),
                            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=config,
                            seed=seed,mean=source['mean'],scale=source['scale'],validation_breath_ids=valid_ids,
                            history=history,best_mae=best,best_epoch=best_epoch,elapsed_seconds=elapsed)
            if last.exists():
                ck=torch.load(last,map_location='cpu'); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
                generator.set_state(ck['loader_rng']); torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
                history,best,best_epoch,elapsed=(ck[k] for k in ['history','best_mae','best_epoch','elapsed_seconds'])
            else:
                history=[]; elapsed=0.; best_epoch=0
                best=float(np.abs(predict()[vm]-truth).mean())
                assert abs(best-source['best_mae'])<1e-5,'adapter changed initial prediction'
                save_checkpoint(bestpath,state())
            for epoch in range(len(history)+1,a.epochs+1):
                model.train(); torch.cuda.synchronize(); start=time.perf_counter(); total=0.; count=0
                for bx,by,bm,br in loader:
                    bx,by,bm,br=[t.cuda(non_blocking=True) for t in (bx,by,bm,br)]
                    opt.zero_grad(set_to_none=True)
                    if strategy=='delta': pred,aux=model(bx,br,auxiliary=True)
                    else: pred=model(bx,br); aux=None
                    loss=objective(pred,by,bm,strategy,aux)
                    assert torch.isfinite(loss),'nonfinite loss'
                    loss.backward(); opt.step(); total+=loss.item()*len(by); count+=len(by)
                pred=predict(); mae=float(np.abs(pred[vm]-truth).mean()); assert np.isfinite(mae)
                torch.cuda.synchronize(); seconds=time.perf_counter()-start; elapsed+=seconds
                improved=mae<best
                if improved: best,best_epoch=mae,epoch
                row=dict(strategy=strategy,model=arch,seed=seed,epoch=epoch,validation_mae=mae,
                         mean_batch_loss=total/count,epoch_seconds=seconds,elapsed_seconds=elapsed)
                history.append(row); save_checkpoint(last,state())
                if improved: save_checkpoint(bestpath,state())
                save_json(folder/f'seed_{seed}_history.json',history); print(json.dumps(row),flush=True)
            ck=torch.load(bestpath,map_location='cpu'); model.load_state_dict(ck['model']); pred=predict()
            assert abs(float(np.abs(pred[vm]-truth).mean())-best)<1e-5
            np.savez_compressed(folder/f'seed_{seed}_validation.npz',predictions=pred,truth=y[valid],inspiratory=vm,breath_ids=ids[valid,0])
            results.append(dict(seed=seed,parent_mae=source['best_mae'],best_mae=best,best_epoch=best_epoch,
                                elapsed_seconds=elapsed,epochs_completed=len(history)))
            save_json(folder/'results.json',results)
        print('TASK COMPLETE',strategy,arch,flush=True)
    print('STRATEGIES_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    main()
