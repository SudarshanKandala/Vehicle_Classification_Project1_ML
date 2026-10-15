"""Locked non-neural candidate roster; no test file access permitted."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer, PolynomialFeatures, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator,TransformerMixin

class DomainFeatures(BaseEstimator,TransformerMixin):
    """Matches previous engineered feature candidate, deterministic per row."""
    def fit(self,X,y=None): self.feature_names_in_=list(X.columns);return self
    def transform(self,X):
        a=X.copy();e=1e-8
        a['power_weight']=a.horsepower_hp/(a.vehicle_weight_kg+e)
        a['torque_weight']=a.torque_nm/(a.vehicle_weight_kg+e)
        a['hp_displacement']=a.horsepower_hp/(a.engine_displacement_l+e)
        a['aspect_height_width']=a.vehicle_height_mm/(a.vehicle_width_mm+e)
        a['length_wheelbase']=a.vehicle_length_mm/(a.wheelbase_mm+e)
        a['wheelbase_width']=a.wheelbase_mm/(a.vehicle_width_mm+e)
        a['vehicle_volume']=a.vehicle_length_mm*a.vehicle_width_mm*a.vehicle_height_mm/1e9
        a['cargo_volume_ratio']=a.cargo_capacity_l/(a.vehicle_volume+e)
        a['efficiency_difference']=a.highway_fuel_efficiency_kmpl-a.city_fuel_efficiency_kmpl
        a['efficiency_ratio']=a.highway_fuel_efficiency_kmpl/(a.city_fuel_efficiency_kmpl+e)
        a['acceleration_x_hp']=a.acceleration_0_100_kmh_s*a.horsepower_hp
        a['tank_per_weight']=a.fuel_tank_capacity_l/(a.vehicle_weight_kg+e)
        return a

# All parameter combinations pre-specified by training-only hypotheses.
LOGIT_C=[.15,.3,.5,.7,1.,1.5,2.,3.,5.,10.]
SVM_ENG=[(2,.001),(4,.001),(8,.001),(2,.003),(4,.003),(8,.003),(16,.003),(2,.006),(4,.006),(8,.006),(4,.01)]
SVM_RAW=[(4,.003),(8,.003),(8,.006),(16,.006)]
XGB_PARAMS=[
    ('xgb_d4_reg',dict(n_estimators=260,max_depth=4,learning_rate=.045,min_child_weight=8,subsample=.85,colsample_bytree=.9,reg_lambda=10,reg_alpha=.5)),
    ('xgb_d5_reg',dict(n_estimators=280,max_depth=5,learning_rate=.035,min_child_weight=10,subsample=.8,colsample_bytree=.85,reg_lambda=12,reg_alpha=1.)),
    ('xgb_d5_slow',dict(n_estimators=420,max_depth=5,learning_rate=.023,min_child_weight=8,subsample=.85,colsample_bytree=.85,reg_lambda=12,reg_alpha=.5)),
    ('xgb_d3_reg',dict(n_estimators=340,max_depth=3,learning_rate=.04,min_child_weight=6,subsample=.9,colsample_bytree=.9,reg_lambda=8,reg_alpha=.3)),
    ('xgb_d5_baseline',dict(n_estimators=290,max_depth=5,learning_rate=.042,min_child_weight=6,subsample=.8,colsample_bytree=.8,reg_lambda=8,reg_alpha=.4)),
]

SPECS=[]
for c in LOGIT_C: SPECS.append((f'logit_c{c:g}'.replace('.','p'), 'logistic'))
for c in [.3,1.,3.,10.]:SPECS.append((f'logit_domain_c{c:g}'.replace('.','p'),'logistic_domain'))
for c in [.001,.005,.02,.08]:SPECS.append((f'logit_polyint_c{c:g}'.replace('.','p'),'logistic_poly'))
for c in [.03,.3,1.]:SPECS.append((f'logit_spline_c{c:g}'.replace('.','p'),'logistic_spline'))
for c in [.3,1.]:SPECS.append((f'logit_quantile_c{c:g}'.replace('.','p'),'logistic_quantile'))
for c in [.3,1.]:SPECS.append((f'logit_power_c{c:g}'.replace('.','p'),'logistic_power'))
for c,g in SVM_ENG:SPECS.append((f'svm_domain_c{c:g}_g{g:g}'.replace('.','p'),'svm_domain'))
for c,g in SVM_RAW:SPECS.append((f'svm_raw_c{c:g}_g{g:g}'.replace('.','p'),'svm_raw'))
for c in [.5,2.,8.]: SPECS.append((f'svm_poly2_c{c:g}'.replace('.','p'),'svm_poly'))
for name,_ in XGB_PARAMS: SPECS.append((name,'xgboost'))
# Historically strong candidates preserved in repeated audit
SPECS += [('lda_auto_reference','lda'),('logit_c1_reference','reference'),('svm_domain_reference','reference')]

# Fixed ensembles—not tuned using public leaderboard scores. Individual components are CV-fitted within outer folds.
BLENDS={
  'blend_logit_svmdomain_70_30': [('logit_c1','0.7'),('svm_domain_c4_g0p003','0.3')],
  'blend_logit_svmdomain_50_50': [('logit_c1','0.5'),('svm_domain_c4_g0p003','0.5')],
  'blend_logit_svmdomain_30_70': [('logit_c1','0.3'),('svm_domain_c4_g0p003','0.7')],
  'blend_logit_xgb_75_25': [('logit_c1','0.75'),('xgb_d5_baseline','0.25')],
  'blend_logit_xgb_50_50': [('logit_c1','0.5'),('xgb_d5_baseline','0.5')],
  'blend_logit_c3_svm_60_40': [('logit_c3','0.6'),('svm_domain_c8_g0p003','0.4')],
  'blend_logit_svmdomain_xgb': [('logit_c1','0.5'),('svm_domain_c4_g0p003','0.3'),('xgb_d5_baseline','0.2')],
}
BLENDS={k:[(name,float(w)) for name,w in v] for k,v in BLENDS.items()}

def build(name):
    if name=='lda_auto_reference':return make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))
    if name=='logit_c1_reference':name='logit_c1'
    if name=='svm_domain_reference':name='svm_domain_c4_g0p003'
    if name.startswith('logit_c'):
        c=float(name[7:].replace('p','.'))
        return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1600,solver='lbfgs'))
    if name.startswith('logit_domain_c'):
        c=float(name.split('_c')[-1].replace('p','.'))
        return make_pipeline(DomainFeatures(),StandardScaler(),LogisticRegression(C=c,max_iter=1600))
    if name.startswith('logit_polyint_c'):
        c=float(name.split('_c')[-1].replace('p','.'))
        return make_pipeline(StandardScaler(),PolynomialFeatures(degree=2,interaction_only=True,include_bias=False),LogisticRegression(C=c,max_iter=1700))
    if name.startswith('logit_spline_c'):
        c=float(name.split('_c')[-1].replace('p','.'))
        return make_pipeline(StandardScaler(),SplineTransformer(n_knots=4,degree=2,include_bias=False),LogisticRegression(C=c,max_iter=1700))
    if name.startswith('logit_quantile_c'):
        c=float(name.split('_c')[-1].replace('p','.'))
        return make_pipeline(QuantileTransformer(n_quantiles=500,output_distribution='normal',random_state=42),LogisticRegression(C=c,max_iter=1700))
    if name.startswith('logit_power_c'):
        c=float(name.split('_c')[-1].replace('p','.'))
        return make_pipeline(PowerTransformer(method='yeo-johnson',standardize=True),LogisticRegression(C=c,max_iter=1700))
    if name.startswith('svm_domain'):
        c=float(name.split('_c')[1].split('_g')[0].replace('p','.'))
        g=float(name.split('_g')[1].replace('p','.'))
        return make_pipeline(DomainFeatures(),StandardScaler(),SVC(kernel='rbf',C=c,gamma=g,probability=True,random_state=42))
    if name.startswith('svm_raw'):
        c=float(name.split('_c')[1].split('_g')[0].replace('p','.'))
        g=float(name.split('_g')[1].replace('p','.'))
        return make_pipeline(StandardScaler(),SVC(kernel='rbf',C=c,gamma=g,probability=True,random_state=42))
    if name.startswith('svm_poly2_c'):
        c=float(name.split('_c')[-1].replace('p','.'))
        return make_pipeline(StandardScaler(),SVC(kernel='poly',degree=2,coef0=2,C=c,gamma=.025,probability=True,random_state=42))
    for nam,pars in XGB_PARAMS:
        if name==nam:
            return XGBClassifier(**pars,tree_method='hist',eval_metric='mlogloss',n_jobs=1,random_state=42)
    raise ValueError('unknown recipe '+name)
