"""Two leakage-safe cross-fitted non-neural stacking pipelines. No test file used."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from vehicle_batch_models import build
STACK_NAMES=['stack_linear_meta','stack_mixed_meta']
def build_stack(name):
    if name=='stack_linear_meta':
        members=['lda_shrink010','logit_c03','svm_rbf_wide']
    elif name=='stack_mixed_meta':
        members=['lda_shrink010','svm_rbf_wide','xgb_depth3']
    else:raise KeyError(name)
    return StackingClassifier(estimators=[(s,build(s)) for s in members],final_estimator=make_pipeline(StandardScaler(),LogisticRegression(C=.2,max_iter=700)),cv=3,passthrough=False,stack_method='predict_proba',n_jobs=1)

if __name__=='__main__':
    import json,time
    from pathlib import Path
    import pandas as pd,numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score,f1_score,log_loss
    root=Path('/mnt/data');out=root/'vehicle_batch'
    total=pd.read_csv(root/'vehicle_train_10000(1).csv')
    f=[c for c in total if c not in ('vehicle_id','vehicle_class')]
    uni=total.drop_duplicates(subset=f).reset_index(drop=True)
    X=uni[f];y=uni.vehicle_class.to_numpy()
    folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=2026).split(X,y))
    result=[]
    for n in STACK_NAMES:
        start=time.time();oof=np.zeros((len(y),3));cv=[];tr=[]
        try:
            for i,(it,iv) in enumerate(folds,1):
                m=build_stack(n);m.fit(X.iloc[it],y[it]);p=m.predict_proba(X.iloc[iv]);oof[iv]=p
                cv.append(accuracy_score(y[iv],p.argmax(axis=1)));tr.append(accuracy_score(y[it],m.predict(X.iloc[it])))
                print(n,i,round(cv[-1],5),flush=True)
            rec=dict(name=n,kind='crossfitted_stacking',cv_mean=float(np.mean(cv)),cv_std=float(np.std(cv,ddof=1)),train_mean=float(np.mean(tr)),gap=float(np.mean(tr)-np.mean(cv)),macro_f1=float(f1_score(y,oof.argmax(axis=1),average='macro')),oof_logloss=float(log_loss(y,oof)),fold_scores=[float(s) for s in cv],seconds=round(time.time()-start,1),feature_set='raw',weighted_duplicates=False)
            result.append(rec)
        except Exception as e:print('STACK_FAILURE',n,repr(e),flush=True)
    if result:
        base=json.loads((out/'training_only_cv.json').read_text());base+=result
        (out/'training_only_cv.json').write_text(json.dumps(base,indent=2))
        pd.DataFrame(base).drop(columns=['fold_scores','members'],errors='ignore').sort_values('cv_mean',ascending=False).to_csv(out/'training_only_model_comparison.csv',index=False)
        reg=json.loads((out/'candidate_registry.json').read_text());reg['stacking']=[r['name'] for r in result]
        (out/'candidate_registry.json').write_text(json.dumps(reg,indent=2))
        print('STACK_DONE',[(r['name'],r['cv_mean']) for r in result],flush=True)
