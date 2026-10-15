"""Create locked multiple non-neural Kaggle files. No test read until all fits complete."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
 os.environ[k]='1'
from pathlib import Path
import pandas as pd,numpy as np, json, zipfile,time,joblib
from recipes import build,BLENDS,SPECS
ROOT=Path('/mnt/data');OUT=ROOT/'vehicle_round3'; SUBS=OUT/'submissions';SUBS.mkdir(exist_ok=True)
assert (OUT/'evaluation_complete.flag').exists(), 'Must finish train-only model comparison before inferencing.'

# Frozen curated slate of 37 candidates: preserve variety, no single preferred winner.
SINGLE=[
 'logit_c0p3','logit_c0p5','logit_c0p7','logit_c1','logit_c1p5','logit_c2','logit_c3','logit_c5','logit_c10',
 'logit_domain_c0p3','logit_domain_c1','logit_domain_c3',
 'logit_power_c0p3','logit_power_c1',
 'svm_domain_c2_g0p001','svm_domain_c4_g0p001','svm_domain_c8_g0p001',
 'svm_domain_c2_g0p003','svm_domain_c4_g0p003','svm_domain_c8_g0p003',
 'svm_domain_c2_g0p006','svm_domain_c4_g0p006',
 'svm_raw_c4_g0p003','svm_raw_c8_g0p003',
 'svm_poly2_c0p5','svm_poly2_c2',
 'xgb_d4_reg','xgb_d5_reg','xgb_d5_slow','xgb_d5_baseline']
FINAL=SINGLE+list(BLENDS)
assert len(FINAL)==len(set(FINAL))==37
# Freeze entire selected roster BEFORE loading the test data.
manifest={'single_models':SINGLE,'ensembles':BLENDS,'final_submission_names':FINAL,'note':'Selection frozen on training-only evidence, before test inference. No neural networks.'}
(OUT/'frozen_candidate_registry.json').write_text(json.dumps(manifest,indent=2))
training=pd.read_csv(ROOT/'vehicle_train_10000(1).csv')
cols=[c for c in training if c not in ('vehicle_id','vehicle_class')]
unique=training.drop_duplicates(subset=cols).reset_index(drop=True)
X=unique[cols]; y=unique.vehicle_class.to_numpy()
models={}
for idx,name in enumerate(SINGLE,1):
 t=time.perf_counter()
 models[name]=build(name).fit(X,y)
 print('FITTED',idx,len(SINGLE),name,round(time.perf_counter()-t,2),flush=True)
# Now and ONLY now load unseen test data, for inference alone.
test=pd.read_csv(ROOT/'vehicle_test(1).csv')
sample=pd.read_csv(ROOT/'vehicle_sample_submission(1).csv')
assert set(test.columns)==set(cols+['vehicle_id'])
assert list(sample.columns)==['vehicle_id','vehicle_class']
assert len(test)==len(sample) and sample.vehicle_id.equals(test.vehicle_id)
assert all(pd.api.types.is_numeric_dtype(test[c]) for c in cols)
assert test[cols].notna().all().all()
# No descriptive stats/plots/candidate decisions from unseen test features.
predictions={}; probabilities={}; classes=np.array([0,1,2])
for name in SINGLE:
 m=models[name]
 p=np.asarray(m.predict_proba(test[cols]),dtype=float)
 assert p.shape==(len(test),3) and np.isfinite(p).all() and np.allclose(p.sum(axis=1),1.0,atol=1e-5)
 # For SVC use original classifier prediction, consistent with archived reference candidate.
 pred=np.asarray(m.predict(test[cols]),dtype=int)
 assert set(pred).issubset({0,1,2})
 predictions[name]=pred;probabilities[name]=p
for name,terms in BLENDS.items():
 p=sum(weight*probabilities[m] for m,weight in terms)
 predictions[name]=np.argmax(p,axis=1).astype(int)
for idx,name in enumerate(FINAL,1):
 sub=sample[['vehicle_id','vehicle_class']].copy()
 sub['vehicle_class']=predictions[name]
 fn=SUBS/(name+'.csv');sub.to_csv(fn,index=False)
 assert list(pd.read_csv(fn,nrows=0).columns)==['vehicle_id','vehicle_class']
 assert len(sub)==len(sample)
 assert set(sub.vehicle_class.unique()).issubset({0,1,2})
 print('CREATED',idx,len(FINAL),fn.name,flush=True)
scorecard=pd.read_csv(OUT/'training_only_screen.csv')
idxscores=scorecard.set_index('name')
registry=[]
for name in FINAL:
 row=idxscores.loc[name]
 m=build(name) if name in SINGLE else None
 pars=json.dumps(m.get_params(deep=False),default=str) if m else json.dumps(BLENDS[name])
 registry.append(dict(candidate=name,filename=name+'.csv',family=('blend' if name in BLENDS else 'individual'),cv_3fold_accuracy=row.cv_mean,cv_3fold_std=row.cv_std,train_cv_gap=row.gap,cv_logloss=row.cv_logloss,parameters=pars))
reg=pd.DataFrame(registry).sort_values('cv_3fold_accuracy',ascending=False)
reg.to_csv(OUT/'37_candidate_manifest.csv',index=False)
# A convenience ZIP containing all deliverables and reproducible scripts.
zipdest=ROOT/'vehicle_optimized_round3_37_submissions.zip'
with zipfile.ZipFile(zipdest,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(SUBS.glob('*.csv')):z.write(p,'submissions/'+p.name)
 for f in ['37_candidate_manifest.csv','training_only_screen.csv','repeat_audit_summary.csv','repeat_folds.csv','data_summary.json','frozen_candidate_registry.json','recipes.py','evaluate.py','create_submissions.py','README.md']:
  q=OUT/f
  if q.exists(): z.write(q,f)
assert len(list(SUBS.glob('*.csv')))==37
print('BATCH_COMPLETE',str(zipdest),flush=True)
