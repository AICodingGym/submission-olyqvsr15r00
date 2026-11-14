"""Read-only model diagnostics from saved validation predictions; no fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'error_analysis'; OUT.mkdir(exist_ok=True)
p=np.load(ROOT/'strategy_experiment/results/ensemble_predictions.npz')
ids=p['breath_ids']; mask=p['inspiratory']; truth=p['truth']
pred=(p['cnn_lstm']+p['mae_tcn'])/2
old=(p['parent_lstm']+p['parent_tcn'])/2
parts=[]
for chunk in pd.read_csv(ROOT/'data/ventilator-pressure-prediction/train.csv',chunksize=400000):
 parts.append(chunk.loc[chunk.breath_id.isin(ids)])
df=pd.concat(parts,ignore_index=True)
assert np.array_equal(df.breath_id.to_numpy().reshape(-1,80)[:,0],ids)
assert np.allclose(df.pressure.to_numpy().reshape(-1,80),truth,atol=1e-5,rtol=0)
assert np.array_equal(df.u_out.to_numpy().reshape(-1,80)==0,mask)
df['step']=df.groupby('breath_id',sort=False).cumcount()
df['prediction']=pred.reshape(-1); df['old_prediction']=old.reshape(-1)
df['error']=df.prediction-df.pressure; df['absolute_error']=df.error.abs()
df['old_absolute_error']=(df.old_prediction-df.pressure).abs()
df['model_disagreement']=np.abs(p['cnn_lstm']-p['mae_tcn']).reshape(-1)
df['u_in_change']=df.groupby('breath_id',sort=False).u_in.diff().abs().fillna(0)
v=df.loc[df.u_out.eq(0)].copy(); total=v.absolute_error.sum()
def summary(frame):
 return dict(rows=len(frame),breaths=int(frame.breath_id.nunique()),mae=float(frame.absolute_error.mean()),
             bias=float(frame.error.mean()),old_mae=float(frame.old_absolute_error.mean()),
             row_share=float(len(frame)/len(v)),error_share=float(frame.absolute_error.sum()/total))
def groups(column):
 return [{'group':str(key),**summary(g)} for key,g in v.groupby(column,observed=True,sort=True)]
v['time_block']=pd.cut(v.step,[-1,0,4,9,19,29,39,79],labels=['0','1-4','5-9','10-19','20-29','30-39','40-79'])
v['pressure_band']=pd.cut(v.pressure,[-np.inf,0,5,10,20,30,40,np.inf],labels=['<=0','0-5','5-10','10-20','20-30','30-40','>40'])
v['lung']=v.R.astype(str)+','+v.C.astype(str)
breath=v.groupby('breath_id').agg(mae=('absolute_error','mean'),bias=('error','mean'),absolute_error_sum=('absolute_error','sum'),
                                rows=('step','size'),R=('R','first'),C=('C','first'),peak_pressure=('pressure','max'))
peaks=v.loc[v.groupby('breath_id').pressure.idxmax()]
concentration={}
for percent in [1,5,10,20]:
 worst=breath.nlargest(int(np.ceil(len(breath)*percent/100)),'mae').index
 selected=v.breath_id.isin(worst)
 concentration[str(percent)]={'worst':summary(v.loc[selected]),'remaining':summary(v.loc[~selected])}
threshold=float(v.model_disagreement.quantile(.9))
history=[]
for branch,arch in [('cnn','lstm'),('mae','tcn')]:
 for path in sorted((ROOT/'strategy_experiment/results'/branch/arch).glob('*_history.json')):
  h=json.loads(path.read_text()); best=min(h,key=lambda r:r['validation_mae'])
  history.append({'branch':branch+'_'+arch,'seed':h[0]['seed'],'best_epoch':best['epoch'],
                  'epochs':len(h),'first_mae':h[0]['validation_mae'],'last_mae':h[-1]['validation_mae'],
                  'first_training_loss':h[0]['mean_batch_loss'],'last_training_loss':h[-1]['mean_batch_loss']})
report={'overall':summary(v),'by_time_block':groups('time_block'),'by_timestep':groups('step'),
        'by_pressure':groups('pressure_band'),'by_lung':groups('lung'),'peak_rows':summary(peaks),
        'nonpeak_rows':summary(v.drop(peaks.index)),'concentration_by_breath_mae':concentration,
        'worst_breaths':breath.sort_values('mae',ascending=False).head(15).reset_index().to_dict(orient='records'),
        'disagreement':{'threshold':threshold,'high':summary(v.loc[v.model_disagreement>threshold]),'rest':summary(v.loc[v.model_disagreement<=threshold])},
        'selected_training_histories':history}
(OUT/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Error analysis of the selected ensemble','',
       'Saved predictions only: no training or submission. Same 500 validation breaths.',
       'Signed bias = prediction minus observed pressure. Groups are descriptive, not causal.',
       'Pressure-based groups and peak locations use validation labels for diagnosis only.', '']
for title,key in [('Time within breath','by_time_block'),('Observed pressure','by_pressure'),('Lung setting (R,C)','by_lung')]:
 lines += ['## '+title,'','| Group | Rows | MAE | Signed bias | Share of total absolute error |','|---|---:|---:|---:|---:|']
 for g in report[key]:
  lines.append(f"| {g['group']} | {g['rows']} | {g['mae']:.4f} | {g['bias']:+.4f} | {100*g['error_share']:.1f}% |")
 lines.append('')
lines+=['## Error concentration','','| Worst breaths by MAE | Their error share | Remaining-row MAE |','|---|---:|---:|']
for k,g in concentration.items():
 lines.append(f"| {k}% | {100*g['worst']['error_share']:.1f}% | {g['remaining']['mae']:.4f} |")
lines+=['','Full timestep statistics, worst-breath metadata, peak-row analysis, disagreement',
        'groups and training histories are in diagnostics.json. All results use a repeatedly',
        'tuned holdout; validate any resulting intervention independently.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:value for k,value in report.items() if k not in ['by_timestep','worst_breaths']},indent=2))
print('WORST BREATHS',json.dumps(report['worst_breaths'][:5]))
