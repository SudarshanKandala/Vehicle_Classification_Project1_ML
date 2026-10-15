"""Batch inference ONLY after training-only CV and candidate registry are completely frozen.
No test inspection for feature engineering, tuning, selection, or experiment comparisons.
Produces independent submit-ready Kaggle CSVs for every pre-specified model/blend/stack.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import json,time,zipfile,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from vehicle_batch_models import build,INDIVIDUAL,WEIGHTED,ENSEMBLES,model_prob
from vehicle_batch_stacking import build_stack,STACK_NAMES
ROOT=Path('/mnt/data');OUT=ROOT/'vehicle_batch';S=OUT/'submissions';S.mkdir(parents=True,exist_ok=True)
REG=json.loads((OUT/'candidate_registry.json').read_text());CV=json.loads((OUT/'training_only_cv.json').read_text())
assert REG['individual']==INDIVIDUAL,'Candidate list must exactly match frozen train-only registry'
assert REG['ensembles']==list(ENSEMBLES),'Blends must exactly match frozen train-only registry'
assert REG.get('stacking')==STACK_NAMES,'Stacked models must exactly match frozen train-only registry'
assert len(CV)==len(INDIVIDUAL)+len(ENSEMBLES)+len(STACK_NAMES),'All configurations must finish CV before test file can open'

# Train and freeze all candidate models solely with original train labels and features.
orig=pd.read_csv(ROOT/'vehicle_train_10000(1).csv'); cols=[c for c in orig if c not in ('vehicle_id','vehicle_class')]
unique=orig.drop_duplicates(subset=cols).reset_index(drop=True)
assert len(unique)==7500 and len(orig)==10000
models={};train_seconds={}
print('FULL_TRAINING_START',len(INDIVIDUAL),'base',len(STACK_NAMES),'stacks','-- NO TEST OPEN',flush=True)
for i,name in enumerate(INDIVIDUAL,1):
    start=time.time();m=build(name);d=orig if name in WEIGHTED else unique
    m.fit(d[cols],d.vehicle_class.to_numpy())
    models[name]=m;train_seconds[name]=time.time()-start
    print(f'TRAINED {i}/{len(INDIVIDUAL)} {name} {train_seconds[name]:.1f}s',flush=True)
for i,name in enumerate(STACK_NAMES,1):
    start=time.time();m=build_stack(name);m.fit(unique[cols],unique.vehicle_class.to_numpy())
    models[name]=m;train_seconds[name]=time.time()-start
    print(f'STACK_TRAINED {i}/{len(STACK_NAMES)} {name} {train_seconds[name]:.1f}s',flush=True)

# Finalized configurations (no model selection) are frozen at this point.
print('MODEL_REGISTRY_LOCKED_AND_FITTED',len(models),flush=True)
# First-ever opening of test data in this standalone process is solely to predict.
test=pd.read_csv(ROOT/'vehicle_test(1).csv')
sample=pd.read_csv(ROOT/'vehicle_sample_submission(1).csv')
assert list(sample.columns)==['vehicle_id','vehicle_class']
assert set(test.columns)==set(cols+['vehicle_id'])
assert len(test)==len(sample) and test.vehicle_id.equals(sample.vehicle_id)
assert test.vehicle_id.is_unique
assert all(pd.api.types.is_numeric_dtype(test[c]) for c in cols)
xtest=test[cols]

def write_csv(name,pred):
    pred=np.asarray(pred,dtype=int).reshape(-1)
    assert len(pred)==len(sample) and set(np.unique(pred)).issubset({0,1,2})
    sub=sample[['vehicle_id','vehicle_class']].copy();sub['vehicle_class']=pred
    path=S/f'{name}.csv';sub.to_csv(path,index=False)
    assert len(pd.read_csv(path))==len(sample)
    return path
paths=[];model_probs={}
for name in INDIVIDUAL:
    m=models[name];p=model_prob(m,xtest)
    assert p.shape==(len(test),3) and np.isfinite(p).all()
    model_probs[name]=p
    paths.append(write_csv(name,p.argmax(axis=1)))
    print('SUBMISSION',name,flush=True)
for name,members in ENSEMBLES.items():
    p=sum(w*model_probs[k] for k,w in members.items())/sum(members.values())
    paths.append(write_csv(name,p.argmax(axis=1)))
    print('SUBMISSION',name,flush=True)
for name in STACK_NAMES:
    p=models[name].predict_proba(xtest)
    assert p.shape==(len(test),3) and np.isfinite(p).all()
    paths.append(write_csv(name,p.argmax(axis=1)))
    print('SUBMISSION',name,flush=True)
assert len(paths)==len(INDIVIDUAL)+len(ENSEMBLES)+len(STACK_NAMES)==44
result=pd.DataFrame(CV).drop(columns=['fold_scores','members'],errors='ignore').copy()
result['filename']=result['name']+'.csv'
result['training_seconds_final']=result['name'].map(train_seconds)
result.sort_values('cv_mean',ascending=False).to_csv(OUT/'model_scorecard.csv',index=False)
manifest={'metric':'accuracy','test_data_used_only_for_predictions':True,'test_used_for_model_selection':False,'contains_neural_networks':False,'cv':'StratifiedKFold(3,shuffle=True,random_state=2026) after training-only exact feature deduplication','total_candidate_submissions':len(paths),'n_training_rows':len(orig),'n_unique_training_rows':len(unique),'n_submission_rows':len(sample),'submission_files':[p.name for p in paths],'train_versions':{'sklearn':'1.8.0'},'no_automatic_winner':True}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
archive=ROOT/'vehicle_all_44_submissions.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in paths:z.write(p,arcname=f'submissions/{p.name}')
    for file in ['MANIFEST.json','model_scorecard.csv','training_only_model_comparison.csv','training_only_cv.json','training_only_eda.json','candidate_registry.json']:
        z.write(OUT/file,arcname=file)
    for file in ['vehicle_batch_models.py','vehicle_batch_validate.py','vehicle_batch_stacking.py','vehicle_batch_inference.py']:
        z.write(ROOT/file,arcname=f'code/{file}')
print('BATCH_READY',str(archive),'COUNT',len(paths),'TOTAL_MB',round(archive.stat().st_size/1e6,3),flush=True)
