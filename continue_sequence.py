"""Paired warm-start experiments: lower LR, loss ablation, or expanded data."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sequence_compare import make_model, FEATURES, BASE_FEATURES, engineer_features, save_json, save_checkpoint


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-breaths',type=int,default=20000,help='0 means all')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--loss',choices=['mse','inspiratory_mae'],default='mse')
    p.add_argument('--lr',type=float,default=.0003)
    a=p.parse_args()
    if a.epochs<1 or a.lr<=0 or (a.max_breaths!=0 and a.max_breaths<5000):
        p.error('invalid training budget')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(8)
    start=time.perf_counter()
    df=pd.read_csv(a.data,usecols=['breath_id',*BASE_FEATURES,'pressure'],nrows=a.max_breaths*80 if a.max_breaths else None)
    ids=df.breath_id.to_numpy().reshape(-1,80); n=len(ids)
    assert (ids==ids[:,:1]).all() and len(np.unique(ids[:,0]))==n
    valid_ids=np.random.RandomState(42).permutation(ids[:5000,0])[:500]
    valid=np.isin(ids[:,0],valid_ids); assert valid.sum()==500
    y=df.pressure.to_numpy(dtype=np.float32).reshape(n,80)
    insp=df.u_out.to_numpy().reshape(n,80)==0
    flat=engineer_features(df)[FEATURES].to_numpy(dtype=np.float32)
    del df
    a.output.mkdir(parents=True,exist_ok=True)
    print('PREPARED',n,'breaths',time.perf_counter()-start,'seconds',flush=True)
    for name in ('gru','lstm','tcn'):
        parent=a.parents/name; folder=a.output/name; folder.mkdir(exist_ok=True)
        pc=json.loads((parent/'config.json').read_text())
        config=dict(model=name,seeds=pc['seeds'],features=FEATURES,max_breaths=n,
                    training_breaths=int((~valid).sum()),validation_breaths=500,
                    learning_rate=a.lr,loss=a.loss,epochs=a.epochs,parent=str(parent.resolve()),
                    normalization='frozen parent training-only normalization',gpu=torch.cuda.get_device_name(0))
        if (folder/'config.json').exists():
            assert json.loads((folder/'config.json').read_text())==config,'resume config mismatch'
        save_json(folder/'config.json',config); results=[]
        for seed in config['seeds']:
            parent_ck=torch.load(parent/f'seed_{seed}_best.pt',map_location='cpu')
            assert np.array_equal(parent_ck['validation_breath_ids'],valid_ids)
            mean,scale=parent_ck['mean'],parent_ck['scale']
            x=((flat-mean)/scale).reshape(n,80,-1)
            data=TensorDataset(torch.from_numpy(x[~valid]),torch.from_numpy(y[~valid]),torch.from_numpy(insp[~valid]))
            xv=torch.from_numpy(x[valid]); vm=insp[valid]; truth=y[valid][vm]
            del x
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model=make_model(name,len(FEATURES)).cuda()
            optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
            generator=torch.Generator()
            last=folder/f'seed_{seed}_last.pt'; bestpath=folder/f'seed_{seed}_best.pt'
            ck=torch.load(last,map_location='cpu') if last.exists() else parent_ck
            model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
            for group in optimizer.param_groups: group['lr']=a.lr
            generator.set_state(ck['loader_rng']); torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
            loader=DataLoader(data,batch_size=51,shuffle=True,generator=generator,pin_memory=True)
            if last.exists():
                history,best,best_epoch,stale,elapsed=(ck[k] for k in ['history','best_mae','best_epoch','stale','elapsed_seconds'])
            else:
                history=[]; best=parent_ck['best_mae']; best_epoch=0; stale=0; elapsed=0.
                initial=dict(parent_ck,config=config,history=[],best_epoch=0,stale=0,elapsed_seconds=0.,parent_best_epoch=parent_ck['best_epoch'])
                save_checkpoint(bestpath,initial)
            for epoch in range(len(history)+1,a.epochs+1):
                if stale>=5: break
                torch.cuda.synchronize(); ts=time.perf_counter(); model.train(); total=0.; count=0
                for bx,by,bm in loader:
                    bx,by,bm=bx.cuda(non_blocking=True),by.cuda(non_blocking=True),bm.cuda(non_blocking=True)
                    optimizer.zero_grad(set_to_none=True); pred=model(bx)
                    if a.loss=='mse':
                        loss=(pred-by).square().mean(); weight=by.numel()
                    else:
                        loss=(pred[bm]-by[bm]).abs().mean(); weight=int(bm.sum())
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                    loss.backward(); optimizer.step(); total+=loss.item()*weight; count+=weight
                model.eval()
                with torch.no_grad():
                    pred=torch.cat([model(z.cuda()).cpu() for z in xv.split(51)]).numpy()
                mae=float(np.abs(pred[vm]-truth).mean())
                assert np.isfinite(mae)
                torch.cuda.synchronize(); duration=time.perf_counter()-ts; elapsed+=duration
                improved=mae<best
                if improved: best,best_epoch,stale=mae,epoch,0
                else: stale+=1
                history.append(dict(model=name,seed=seed,epoch=epoch,train_loss=total/count,
                                    validation_mae=mae,epoch_seconds=duration,elapsed_seconds=elapsed))
                state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),loader_rng=generator.get_state(),
                           torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history,
                           best_mae=best,best_epoch=best_epoch,stale=stale,elapsed_seconds=elapsed,
                           mean=mean,scale=scale,validation_breath_ids=valid_ids,config=config,seed=seed,
                           parent_best_epoch=parent_ck['best_epoch'])
                save_checkpoint(last,state)
                if improved: save_checkpoint(bestpath,state)
                save_json(folder/f'seed_{seed}_history.json',history)
                print(json.dumps(history[-1]),flush=True)
            bestck=torch.load(bestpath,map_location='cpu'); model.load_state_dict(bestck['model']); model.eval()
            with torch.no_grad():
                pred=torch.cat([model(z.cuda()).cpu() for z in xv.split(51)]).numpy()
            assert abs(float(np.abs(pred[vm]-truth).mean())-best)<1e-5
            np.savez_compressed(folder/f'seed_{seed}_validation.npz',predictions=pred,truth=y[valid],inspiratory=vm,breath_ids=ids[valid,0])
            results.append(dict(seed=seed,parent_mae=parent_ck['best_mae'],best_mae=best,best_epoch=best_epoch,
                                epochs_completed=len(history),elapsed_seconds=elapsed))
            save_json(folder/'results.json',results)
        print('MODEL COMPLETE',name,json.dumps(results),flush=True)
    print('CONTINUATION COMPLETE',flush=True)


if __name__=='__main__':
    main()
