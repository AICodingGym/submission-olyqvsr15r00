"""Leakage-safe cross-fitting on breaths absent from older warm-start checkpoints."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sequence_compare import FEATURES,BASE_FEATURES,engineer_features,save_json,save_checkpoint
from strategy_models import StrategyModel


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=10)
    a=p.parse_args();torch.set_num_threads(8);assert torch.cuda.is_available()
    df=pd.read_csv(a.data,usecols=['breath_id',*BASE_FEATURES,'pressure'])
    ids=df.breath_id.to_numpy().reshape(-1,80)[:,0];n=len(ids);assert n==67905
    primary_ids=np.random.RandomState(42).permutation(ids[:5000])[:500]
    primary=np.isin(ids,primary_ids)
    rng=np.random.RandomState(20261113)
    eligible=rng.permutation(np.arange(20000,n)); pool=np.sort(eligible[:12000]);inner=np.sort(eligible[12000:13000])
    assert not np.intersect1d(pool,inner).size and not primary[pool].any() and not primary[inner].any()
    fold=np.full(n,-1,dtype=int)
    rc=df.iloc[::80][['R','C']].to_numpy()
    for setting in np.unique(rc[pool],axis=0):
        ix=pool[(rc[pool]==setting).all(1)];rng.shuffle(ix);fold[ix]=np.arange(len(ix))%3
    y=df.pressure.to_numpy(dtype=np.float32).reshape(n,80)
    insp=df.u_out.to_numpy().reshape(n,80)==0
    flat=engineer_features(df)[FEATURES].to_numpy(dtype=np.float32);del df
    a.output.mkdir(parents=True,exist_ok=True)
    config=dict(epochs=a.epochs,pool_breaths=12000,internal_validation_breaths=1000,primary_validation_breaths=500,
                folds=3,split_seed=20261113,learning_rate=.0003,loss='inspiratory MAE',parent=str(a.parents.resolve()))
    if (a.output/'config.json').exists(): assert json.loads((a.output/'config.json').read_text())==config
    save_json(a.output/'config.json',config)
    np.savez_compressed(a.output/'split.npz',ids=ids,pool=pool,inner=inner,fold=fold,primary=np.flatnonzero(primary))
    output=np.full((n,80),np.nan,dtype=np.float32); valpred=[]; results=[]
    for f in range(3):
        held=np.flatnonzero(fold==f)
        train=~primary;train[held]=False;train[inner]=False
        assert not train[held].any() and not train[inner].any() and train[:20000].sum()==19500
        members=[];vmembers=[]
        for arch,strategy in [('lstm','cnn'),('tcn','mae')]:
            pc=json.loads((a.parents/arch/'config.json').read_text())
            assert pc['max_breaths']==20000,'warm start must exclude every OOF/inner breath'
            seed=pc['seeds'][f];source=torch.load(a.parents/arch/f'seed_{seed}_best.pt',map_location='cpu')
            assert source['config']['max_breaths']==20000
            assert np.array_equal(source['validation_breath_ids'],primary_ids)
            mean,scale=source['mean'],source['scale'];x=((flat-mean)/scale).reshape(n,80,-1)
            ds=TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(y[train]),torch.from_numpy(insp[train]))
            xi=torch.from_numpy(x[inner]);xh=torch.from_numpy(x[held]);xv=torch.from_numpy(x[primary]);del x
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=StrategyModel(arch,strategy).cuda();model.base.load_state_dict(source['model'])
            opt=torch.optim.Adam(model.parameters(),lr=.0003);gen=torch.Generator().manual_seed(seed)
            loader=DataLoader(ds,batch_size=51,shuffle=True,generator=gen,pin_memory=True)
            last=a.output/f'fold{f}_{arch}_last.pt';bestpath=a.output/f'fold{f}_{arch}_best.pt'
            def predict(tensor):
                model.eval()
                with torch.no_grad():return torch.cat([model(b.cuda(),None).cpu() for b in tensor.split(128)]).numpy()
            def state():
                return dict(model=model.state_dict(),optimizer=opt.state_dict(),loader_rng=gen.get_state(),torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(),history=history,best_mae=best,best_epoch=best_epoch,elapsed_seconds=elapsed,
                            mean=mean,scale=scale,fold=f,seed=seed,architecture=arch,strategy=strategy,config=config,
                            train_breath_ids=ids[train],held_breath_ids=ids[held],inner_breath_ids=ids[inner])
            if last.exists():
                ck=torch.load(last,map_location='cpu');model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
                gen.set_state(ck['loader_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
                history,best,best_epoch,elapsed=(ck[k] for k in ['history','best_mae','best_epoch','elapsed_seconds'])
            else:
                history=[];elapsed=0.;best_epoch=0;best=float(np.abs(predict(xi)[insp[inner]]-y[inner][insp[inner]]).mean())
                save_checkpoint(bestpath,state())
            for epoch in range(len(history)+1,a.epochs+1):
                model.train();torch.cuda.synchronize();start=time.perf_counter();total=0.;count=0
                for bx,by,bm in loader:
                    bx,by,bm=[b.cuda(non_blocking=True) for b in (bx,by,bm)]
                    opt.zero_grad(set_to_none=True);pred=model(bx,None);loss=(pred[bm]-by[bm]).abs().mean()
                    assert torch.isfinite(loss);loss.backward();opt.step();total+=loss.item()*len(by);count+=len(by)
                pred=predict(xi);mae=float(np.abs(pred[insp[inner]]-y[inner][insp[inner]]).mean());assert np.isfinite(mae)
                torch.cuda.synchronize();seconds=time.perf_counter()-start;elapsed+=seconds
                improved=mae<best
                if improved:best,best_epoch=mae,epoch
                row=dict(fold=f,architecture=arch,epoch=epoch,internal_mae=mae,train_loss=total/count,epoch_seconds=seconds)
                history.append(row);save_checkpoint(last,state())
                if improved:save_checkpoint(bestpath,state())
                save_json(a.output/f'fold{f}_{arch}_history.json',history);print(json.dumps(row),flush=True)
            ck=torch.load(bestpath,map_location='cpu');model.load_state_dict(ck['model'])
            assert abs(float(np.abs(predict(xi)[insp[inner]]-y[inner][insp[inner]]).mean())-best)<1e-5
            hp=predict(xh);vp=predict(xv);members.append(hp);vmembers.append(vp)
            np.savez_compressed(a.output/f'fold{f}_{arch}_predictions.npz',held_ids=ids[held],held_predictions=hp,primary_ids=ids[primary],primary_predictions=vp)
            results.append(dict(fold=f,architecture=arch,seed=seed,internal_mae=best,best_epoch=best_epoch,elapsed_seconds=elapsed,
                                train_breaths=int(train.sum()),oof_breaths=len(held)))
            save_json(a.output/'teacher_results.json',results)
        output[held]=np.mean(members,axis=0);valpred.append(np.mean(vmembers,axis=0))
        print('FOLD COMPLETE',f,flush=True)
    assert np.isfinite(output[pool]).all()
    np.savez_compressed(a.output/'crossfit_predictions.npz',oof_ids=ids[pool],oof_predictions=output[pool],oof_truth=y[pool],oof_mask=insp[pool],
                        oof_fold=fold[pool],primary_ids=ids[primary],primary_predictions=np.mean(valpred,axis=0))
    print('CROSSFIT_COMPLETE',flush=True)


if __name__=='__main__': main()
