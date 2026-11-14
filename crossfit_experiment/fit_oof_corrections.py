"""Fit corrections only to out-of-fold training residuals; evaluate original holdout."""
import argparse,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sequence_compare import FEATURES,engineer_features


def design(local, pred):
    # local contains input features only; pressure labels are never supplied here.
    idx={k:i for i,k in enumerate(FEATURES)}
    u=local[:,:,idx['u_in']]; t=local[:,:,idx['time_step']]; out=local[:,:,idx['u_out']]
    r=local[:,0,idx['R']]; c=local[:,0,idx['C']]; insp=out==0
    count=insp.sum(1)
    pm=(pred*insp).sum(1)/count
    peak=np.where(insp,pred,-np.inf).max(1)
    summary=np.column_stack([r,c,u.mean(1),u.std(1),u.min(1),u.max(1),u[:,0],u[:,-1],
                             count,pm,peak,pred[:,0],pred[:,-1],pred.std(1)])
    breath=np.column_stack([summary,u,t,out,pred]).astype(np.float32)
    prev=np.concatenate([pred[:,:1],pred[:,:-1]],axis=1)
    nex=np.concatenate([pred[:,1:],pred[:,-1:]],axis=1)
    row=np.concatenate([local,np.broadcast_to(summary[:,None,:],(*pred.shape,summary.shape[1])),
                        pred[:,:,None],(pred-prev)[:,:,None],(nex-pred)[:,:,None]],axis=2).astype(np.float32)
    assert np.isfinite(breath).all() and np.isfinite(row).all()
    return breath,row


def diagnostics(pred,truth,mask,local):
    err=pred-truth; ae=np.abs(err); r=local[:,0,FEATURES.index('R')]; c=local[:,0,FEATURES.index('C')]
    score=lambda m:dict(rows=int(m.sum()),mae=float(ae[m].mean()),bias=float(err[m].mean()))
    report={'overall':score(mask),'lung':{},'time':{}}
    for rr,cc in np.unique(np.column_stack([r,c]),axis=0):
        report['lung'][f'{rr:g},{cc:g}']=score(mask&((r==rr)&(c==cc))[:,None])
    step=np.arange(80)[None,:]
    for label,lo,hi in [('0-4',0,5),('5-9',5,10),('10-19',10,20),('20-29',20,30),('30+',30,80)]:
        m=mask&(step>=lo)&(step<hi)
        if m.any():report['time'][label]=score(m)
    return report


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--crossfit',type=Path,required=True)
    p.add_argument('--selected',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.perf_counter();o=np.load(a.crossfit);v=np.load(a.selected)
    assert not np.intersect1d(o['oof_ids'],v['breath_ids']).size
    keep=np.concatenate([o['oof_ids'],v['breath_ids']]);parts=[]
    for chunk in pd.read_csv(a.data,usecols=['breath_id',*['R','C','time_step','u_in','u_out']],chunksize=400000):
        parts.append(chunk.loc[chunk.breath_id.isin(keep)])
    df=pd.concat(parts,ignore_index=True);order=df.breath_id.to_numpy().reshape(-1,80)
    assert (order==order[:,:1]).all();lookup={int(b):i for i,b in enumerate(order[:,0])}
    raw=engineer_features(df)[FEATURES].to_numpy(dtype=np.float32).reshape(-1,80,len(FEATURES));del df
    train=raw[[lookup[int(i)] for i in o['oof_ids']]];valid=raw[[lookup[int(i)] for i in v['breath_ids']]]
    op=o['oof_predictions'];om=o['oof_mask'];ot=o['oof_truth'];vp=(v['cnn_lstm']+v['mae_tcn'])/2
    vm=v['inspiratory'];vt=v['truth']
    assert np.array_equal(train[:,:,FEATURES.index('u_out')]==0,om)
    assert np.array_equal(valid[:,:,FEATURES.index('u_out')]==0,vm)
    xb,xr=design(train,op);vb,vr=design(valid,vp)
    offset_target=np.array([np.median((y-p)[m]) for y,p,m in zip(ot,op,om)])
    offset=HistGradientBoostingRegressor(loss='absolute_error',max_iter=150,max_leaf_nodes=15,
           min_samples_leaf=80,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=20261113)
    offset.fit(xb,offset_target,sample_weight=om.sum(1));print('OFFSET FIT COMPLETE',flush=True)
    row=HistGradientBoostingRegressor(loss='absolute_error',max_iter=150,max_leaf_nodes=15,
        min_samples_leaf=100,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=20261113)
    row.fit(xr[om],(ot-op)[om]);print('ROW FIT COMPLETE',flush=True)
    correction=offset.predict(vb)[:,None];rcorrection=row.predict(vr.reshape(-1,vr.shape[-1])).reshape(vp.shape)
    r50=(valid[:,0,FEATURES.index('R')]==50)[:,None];late=np.arange(80)[None,:]>=10
    candidates={'unchanged':vp,'offset_half':vp+.5*correction,'offset_r50_half':vp+.5*correction*r50,
                'row_half':vp+.5*rcorrection*vm,'row_late_half':vp+.5*rcorrection*vm*late}
    metrics={k:diagnostics(pred,vt,vm,valid) for k,pred in candidates.items()}
    for pred in candidates.values():assert np.isfinite(pred).all()
    best=min(metrics,key=lambda k:metrics[k]['overall']['mae'])
    fold_stats={str(f):diagnostics(op[o['oof_fold']==f],ot[o['oof_fold']==f],om[o['oof_fold']==f],train[o['oof_fold']==f]) for f in range(3)}
    report=dict(oof_diagnostics=diagnostics(op,ot,om,train),fold_diagnostics=fold_stats,
                validation=metrics,best=best,elapsed_seconds=time.perf_counter()-start,
                oof_breaths=len(op),oof_rows=int(om.sum()),offset_target_quantiles=np.quantile(offset_target,[0,.1,.5,.9,1]).tolist(),
                offset_model_validation_prediction_quantiles=np.quantile(correction,[0,.1,.5,.9,1]).tolist(),
                label_dependent_oof_offset_oracle_mae=float(np.abs((ot-op-offset_target[:,None])[om]).mean()))
    a.output.mkdir(parents=True,exist_ok=True)
    joblib.dump({'offset':offset,'row':row,'features':FEATURES,'strength':.5},a.output/'corrections.joblib')
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.output/'validation_predictions.npz',**candidates,truth=vt,inspiratory=vm,breath_ids=v['breath_ids'])
    print(json.dumps({k:val['overall'] for k,val in metrics.items()},indent=2),flush=True)
    print('OOF_EXPERIMENT_COMPLETE',flush=True)


if __name__=='__main__':main()
