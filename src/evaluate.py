"""Strictly training-only CV, no test or sample submission access."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
 os.environ[k]='1'
from pathlib import Path
import json,time,traceback,sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss,confusion_matrix, f1_score
from recipes import SPECS,BLENDS,build
ROOT=Path('/mnt/data'); OUT=ROOT/'vehicle_round3'; TRAIN=ROOT/'vehicle_train_10000(1).csv'
train=pd.read_csv(TRAIN)
cols=[x for x in train.columns if x not in ['vehicle_class','vehicle_id']]
assert train[cols+['vehicle_class']].notna().all().all()
assert train.groupby(cols)['vehicle_class'].nunique().max()==1
uniq=train.drop_duplicates(subset=cols).reset_index(drop=True)
X=uniq[cols];y=uniq.vehicle_class.to_numpy()
assert len(uniq)==7500
folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=2026).split(X,y))
results=[]; cache={}
for i,(name,kind) in enumerate(SPECS):
 start=time.monotonic();p=np.full((len(y),3),np.nan);fscore=[];trscore=[]
 try:
  for j,(it,iv) in enumerate(folds):
   m=build(name).fit(X.iloc[it],y[it]); pred=m.predict(X.iloc[iv]); prob=m.predict_proba(X.iloc[iv]);
   p[iv]=prob;fscore.append(float(accuracy_score(y[iv],pred)));trscore.append(float(accuracy_score(y[it],m.predict(X.iloc[it]))))
  cache[name]=p
  results.append(dict(name=name,kind=kind,cv_mean=np.mean(fscore),cv_std=np.std(fscore,ddof=1),train_mean=np.mean(trscore),gap=np.mean(trscore)-np.mean(fscore),cv_logloss=log_loss(y,p),fold_scores=fscore,seconds=round(time.monotonic()-start,2),error=''))
  print(f'{i+1}/{len(SPECS)} {name:30} {np.mean(fscore):.5f} gap={np.mean(trscore)-np.mean(fscore):.4f} t={time.monotonic()-start:.1f}',flush=True)
 except Exception as e:
  results.append(dict(name=name,kind=kind,cv_mean=np.nan,cv_std=np.nan,train_mean=np.nan,gap=np.nan,cv_logloss=np.nan,fold_scores=[],seconds=round(time.monotonic()-start,2),error=str(e)))
  print('ERROR',name,repr(e),flush=True)
 pd.DataFrame(results).to_csv(OUT/'screen_scores_partial.csv',index=False)
for name,mix in BLENDS.items():
 try:
  pp=sum(w*cache[k] for k,w in mix)
  scores=[accuracy_score(y[iv],np.argmax(pp[iv],axis=1)) for it,iv in folds]
  results.append(dict(name=name,kind='blend',cv_mean=np.mean(scores),cv_std=np.std(scores,ddof=1),train_mean=np.nan,gap=np.nan,cv_logloss=log_loss(y,pp),fold_scores=scores,seconds=0,error=''))
 except Exception as e: print('BLEND_ERROR',name,repr(e),flush=True)
res=pd.DataFrame(results).sort_values('cv_mean',ascending=False)
res.to_csv(OUT/'training_only_screen.csv',index=False)
# Holdout remains untouched; repeated 5-fold as an audit, NOT used to crown a winner.
# Pre-specified 12 representative variants and 2 ensembles; no hidden test access.
audit=['logit_c0p5','logit_c1','logit_c2','logit_c5','logit_domain_c1','logit_polyint_c0p005','logit_spline_c0p3','svm_domain_c4_g0p003','svm_domain_c8_g0p003','svm_raw_c8_g0p003','xgb_d5_reg','xgb_d5_baseline']
rows=[]
for ii,name in enumerate(audit):
 t=time.monotonic()
 for seed in [42,13]:
  fold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
  for j,(it,iv) in enumerate(fold.split(X,y)):
   m=build(name).fit(X.iloc[it],y[it]);va=accuracy_score(y[iv],m.predict(X.iloc[iv]));tr=accuracy_score(y[it],m.predict(X.iloc[it]));
   rows.append(dict(name=name,seed=seed,fold=j+1,cv_accuracy=va,train_accuracy=tr))
 print('AUDIT',ii+1,len(audit),name, 'mean',np.mean([z['cv_accuracy'] for z in rows if z['name']==name]),'time',round(time.monotonic()-t,1),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'audit_partial.csv',index=False)
pd.DataFrame(rows).to_csv(OUT/'repeat_folds.csv',index=False)
agg=pd.DataFrame(rows).groupby('name').agg(cv_mean=('cv_accuracy','mean'),cv_std=('cv_accuracy','std'),train_mean=('train_accuracy','mean')).reset_index()
agg['gap']=agg['train_mean']-agg['cv_mean']
agg.to_csv(OUT/'repeat_audit_summary.csv',index=False)
summary=dict(n_original=len(train),n_unique=len(uniq),n_duplicate_removed=len(train)-len(uniq),classes=uniq.vehicle_class.value_counts().sort_index().to_dict(),columns=cols,screen=int(res.cv_mean.notna().sum()),repeat=int(agg.shape[0]),top5=res.head(5)[['name','cv_mean','gap']].to_dict(orient='records'))
(OUT/'data_summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'evaluation_complete.flag').write_text('TRAINING ONLY COMPLETE')
print('TRAINING_ONLY_EVALUATION_COMPLETE',len(res),'candidates',flush=True)
