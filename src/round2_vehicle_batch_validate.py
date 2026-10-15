"""Training-only EDA and non-neural models on identical duplicate-safe CV folds. NEVER OPENS TEST."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import warnings,json,time,traceback,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.base import clone
from vehicle_batch_models import *
warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path('/mnt/data'); out=ROOT/'vehicle_batch';out.mkdir(exist_ok=True)
total=pd.read_csv(ROOT/'vehicle_train_10000(1).csv')
f=[c for c in total if c not in ('vehicle_id','vehicle_class')]
assert not total[f+['vehicle_class']].isna().any().any()
conflicts=total.groupby(f,dropna=False)['vehicle_class'].nunique(); assert int(conflicts.max())==1
freq=total.groupby(f,dropna=False).size().rename('_original_frequency')
uni=total.drop_duplicates(subset=f).merge(freq.reset_index(),on=f,validate='one_to_one').reset_index(drop=True)
assert len(uni)==7500 and len(total)==10000
X=uni[f];y=uni.vehicle_class.to_numpy(); folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=2026).split(X,y))
ED=dict(train_rows=int(len(total)),unique_rows=int(len(uni)),duplicate_rows=int(len(total)-len(uni)),missing=int(total.isna().sum().sum()),classes=total.vehicle_class.value_counts().sort_index().to_dict(),unique_classes=uni.vehicle_class.value_counts().sort_index().to_dict(),feature_names=f,feature_count=len(f),constant_features=[c for c in f if total[c].nunique()==1],numeric_features=int(sum(pd.api.types.is_numeric_dtype(total[c]) for c in f)),max_abs_pairwise_corr=float(np.abs(uni[f].corr().to_numpy()-np.eye(len(f))).max()),row_frequency=uni._original_frequency.value_counts().sort_index().to_dict())
(out/'training_only_eda.json').write_text(json.dumps(ED,indent=2));print('EDA',ED,flush=True)
records=[];oofs={};failures={}
for i,n in enumerate(INDIVIDUAL):
    start=time.time()
    oof=np.zeros((len(y),3));scores=[];training=[]
    try:
        for fold,(it,iv) in enumerate(folds):
            subset=train_rows_for_fold(uni,it,f,n in WEIGHTED)
            model=build(n)
            model.fit(subset[f],subset.vehicle_class.to_numpy())
            p=model_prob(model,X.iloc[iv]);assert np.isfinite(p).all() and p.shape==(len(iv),3)
            oof[iv]=p
            scores.append(accuracy_score(y[iv],p.argmax(axis=1)))
            training.append(accuracy_score(subset.vehicle_class,model.predict(subset[f]).reshape(-1)))
        rec=dict(name=n,kind='model',cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),train_mean=float(np.mean(training)),gap=float(np.mean(training)-np.mean(scores)),macro_f1=float(f1_score(y,oof.argmax(axis=1),average='macro')),oof_logloss=float(log_loss(y,np.clip(oof,1e-10,1))),fold_scores=[float(s) for s in scores],seconds=round(time.time()-start,1),feature_set='engineered' if 'engineered' in n else ('select14' if 'select14' in n else ('pca15' if 'pca15' in n else 'raw')),weighted_duplicates=n in WEIGHTED)
        records.append(rec);oofs[n]=oof
        print(f'{i+1}/{len(INDIVIDUAL)} {n}: {rec["cv_mean"]:.5f} +/- {rec["cv_std"]:.4f}, gap {rec["gap"]:.4f}, {rec["seconds"]:.1f}s',flush=True)
    except Exception as e:
        failures[n]=str(e)
        print('FAILED',n,type(e).__name__,str(e),flush=True)
    (out/'training_only_cv.json').write_text(json.dumps(records,indent=2))
for n,members in ENSEMBLES.items():
    if not set(members).issubset(oofs):
        failures[n]='Missing fitted OOF member';continue
    p=sum(w*oofs[k] for k,w in members.items())/sum(members.values())
    scores=[accuracy_score(y[iv],p[iv].argmax(axis=1)) for it,iv in folds]
    rec=dict(name=n,kind='pre_specified_blend',cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),train_mean=None,gap=None,macro_f1=float(f1_score(y,p.argmax(axis=1),average='macro')),oof_logloss=float(log_loss(y,p)),fold_scores=[float(s) for s in scores],seconds=0,feature_set='mix',weighted_duplicates=None,members=members)
    records.append(rec)
    print('BLEND',n,round(rec['cv_mean'],5),flush=True)
(out/'training_only_cv.json').write_text(json.dumps(records,indent=2))
pd.DataFrame(records).drop(columns=['fold_scores','members'],errors='ignore').sort_values('cv_mean',ascending=False).to_csv(out/'training_only_model_comparison.csv',index=False)
(out/'errors.json').write_text(json.dumps(failures,indent=2));(out/'candidate_registry.json').write_text(json.dumps({'individual':list(oofs),'ensembles':[r['name'] for r in records if r['kind']=='pre_specified_blend']},indent=2))
print('CV_COMPLETE',len(oofs),'models',len(records)-len(oofs),'blends','ERRORS',failures,flush=True)
