"""Non-neural model factory for vehicle Kaggle batch. NO test data is accessed here."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier,
                              AdaBoostClassifier,GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

TARGET='vehicle_class'; IDENT='vehicle_id'
SEED=42
class DomainFeatures(BaseEstimator,TransformerMixin):
    """Deterministic row-wise physical ratios; learns no statistics from any data."""
    def fit(self, X, y=None):
        self.feature_names_in_=list(X.columns)
        return self
    def transform(self,X):
        a=X.copy()
        eps=1.e-8
        a['power_weight']=a.horsepower_hp/(a.vehicle_weight_kg+eps)
        a['torque_weight']=a.torque_nm/(a.vehicle_weight_kg+eps)
        a['hp_displacement']=a.horsepower_hp/(a.engine_displacement_l+eps)
        a['aspect_height_width']=a.vehicle_height_mm/(a.vehicle_width_mm+eps)
        a['length_wheelbase']=a.vehicle_length_mm/(a.wheelbase_mm+eps)
        a['wheelbase_width']=a.wheelbase_mm/(a.vehicle_width_mm+eps)
        a['vehicle_volume']=a.vehicle_length_mm*a.vehicle_width_mm*a.vehicle_height_mm/1e9
        a['cargo_volume_ratio']=a.cargo_capacity_l/(a.vehicle_volume+eps)
        a['efficiency_difference']=a.highway_fuel_efficiency_kmpl-a.city_fuel_efficiency_kmpl
        a['efficiency_ratio']=a.highway_fuel_efficiency_kmpl/(a.city_fuel_efficiency_kmpl+eps)
        a['acceleration_x_hp']=a.acceleration_0_100_kmh_s*a.horsepower_hp
        a['tank_per_weight']=a.fuel_tank_capacity_l/(a.vehicle_weight_kg+eps)
        return a

# Each config fixed independently of hidden Kaggle feature/labels. All offered to user, no model selected.
INDIVIDUAL = [
 'lda_auto','lda_shrink005','lda_shrink010','lda_shrink025',
 'logit_c003','logit_c03','logit_c1', 'ridge_c1',
 'svm_linear','svm_rbf_wide','svm_rbf_mid','svm_poly2',
 'qda_reg02','naive_bayes','knn_31',
 'rf_depth7','rf_depth11','extratrees_12','histboost_depth4',
 'adaboost_stumps','gradboost_depth2',
 'xgb_depth3','xgb_depth5','lgb_depth4','lgb_depth6',
 'catboost_depth4','catboost_depth6',
 'lda_engineered','logit_engineered','svm_engineered',
 'logit_select14','logit_pca15',
 'lda_dupweighted','logit_dupweighted','cat_dupweighted'
]
WEIGHTED={'lda_dupweighted','logit_dupweighted','cat_dupweighted'}
ENSEMBLES={
 'blend_linear_3':{'lda_shrink010':1/3,'logit_c03':1/3,'svm_rbf_wide':1/3},
 'blend_lda_svm':{'lda_auto':.5,'svm_rbf_mid':.5},
 'blend_boost_3':{'xgb_depth3':1/3,'lgb_depth4':1/3,'catboost_depth4':1/3},
 'blend_tree_4':{'rf_depth7':.25,'extratrees_12':.25,'xgb_depth3':.25,'catboost_depth4':.25},
 'blend_crossfamily_5':{'lda_shrink010':.35,'svm_rbf_mid':.25,'catboost_depth4':.15,'xgb_depth3':.15,'lgb_depth4':.10},
 'blend_lda_logit':{'lda_auto':.5,'logit_c03':.5},
 'blend_linear_weighted':{'lda_dupweighted':.5,'logit_dupweighted':.5},
}

def build(name):
    if name.startswith('lda_'):
        if name=='lda_auto': return make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))
        if name in ('lda_shrink005','lda_shrink010','lda_shrink025'):
            shrink={'lda_shrink005':.05,'lda_shrink010':.10,'lda_shrink025':.25}[name]
            return make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage=shrink))
        if name=='lda_engineered':return make_pipeline(DomainFeatures(),StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage=.10))
        if name=='lda_dupweighted':return make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage=.10))
    if name.startswith('logit_'):
        if name in ('logit_c003','logit_c03','logit_c1'):
            cc={'logit_c003':.03,'logit_c03':.3,'logit_c1':1.0}[name]
            return make_pipeline(StandardScaler(), LogisticRegression(C=cc,max_iter=1500))
        if name=='logit_engineered':return make_pipeline(DomainFeatures(),StandardScaler(),LogisticRegression(C=.1,max_iter=1500))
        if name=='logit_select14':return make_pipeline(StandardScaler(),SelectKBest(f_classif,k=14),LogisticRegression(C=.1,max_iter=1500))
        if name=='logit_pca15':return make_pipeline(StandardScaler(),PCA(n_components=15),LogisticRegression(C=.1,max_iter=1500))
        if name=='logit_dupweighted':return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500))
    if name=='ridge_c1':return make_pipeline(StandardScaler(),RidgeClassifier(alpha=10.0))
    if name=='svm_linear':return make_pipeline(StandardScaler(),SVC(kernel='linear',C=.025,probability=True,random_state=SEED))
    if name=='svm_rbf_wide':return make_pipeline(StandardScaler(),SVC(kernel='rbf',C=16,gamma=.001,probability=True,random_state=SEED))
    if name=='svm_rbf_mid':return make_pipeline(StandardScaler(),SVC(kernel='rbf',C=4,gamma=.005,probability=True,random_state=SEED))
    if name=='svm_poly2':return make_pipeline(StandardScaler(),SVC(kernel='poly',degree=2,coef0=2,C=.5,gamma=.025,probability=True,random_state=SEED))
    if name=='svm_engineered':return make_pipeline(DomainFeatures(),StandardScaler(),SVC(kernel='rbf',C=4,gamma=.003,probability=True,random_state=SEED))
    if name=='qda_reg02':return make_pipeline(StandardScaler(),QuadraticDiscriminantAnalysis(reg_param=.2))
    if name=='naive_bayes':return make_pipeline(StandardScaler(),GaussianNB(var_smoothing=.005))
    if name=='knn_31':return make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=31,weights='distance',n_jobs=2))
    if name=='rf_depth7':return RandomForestClassifier(n_estimators=280,max_depth=7,min_samples_leaf=5,max_features=.8,n_jobs=2,random_state=SEED)
    if name=='rf_depth11':return RandomForestClassifier(n_estimators=320,max_depth=11,min_samples_leaf=3,max_features=.8,n_jobs=2,random_state=SEED)
    if name=='extratrees_12':return ExtraTreesClassifier(n_estimators=320,max_depth=12,min_samples_leaf=3,max_features=.8,n_jobs=2,random_state=SEED)
    if name=='histboost_depth4':return HistGradientBoostingClassifier(max_iter=160,max_depth=4,max_leaf_nodes=13,learning_rate=.045,min_samples_leaf=35,l2_regularization=4,random_state=SEED)
    if name=='adaboost_stumps':return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2,min_samples_leaf=30,random_state=SEED),n_estimators=120,learning_rate=.06,random_state=SEED)
    if name=='gradboost_depth2':return GradientBoostingClassifier(n_estimators=140,max_depth=2,min_samples_leaf=25,learning_rate=.045,subsample=.85,random_state=SEED)
    if name=='xgb_depth3':return XGBClassifier(n_estimators=280,max_depth=3,learning_rate=.055,min_child_weight=5,subsample=.85,colsample_bytree=.85,reg_lambda=5,reg_alpha=.2,tree_method='hist',eval_metric='mlogloss',n_jobs=2,random_state=SEED)
    if name=='xgb_depth5':return XGBClassifier(n_estimators=290,max_depth=5,learning_rate=.042,min_child_weight=6,subsample=.8,colsample_bytree=.8,reg_lambda=8,reg_alpha=.4,tree_method='hist',eval_metric='mlogloss',n_jobs=2,random_state=SEED)
    if name=='lgb_depth4':return LGBMClassifier(n_estimators=270,learning_rate=.042,max_depth=4,num_leaves=13,min_child_samples=35,colsample_bytree=.85,subsample=.85,subsample_freq=1,reg_lambda=5,reg_alpha=.3,verbosity=-1,n_jobs=2,random_state=SEED)
    if name=='lgb_depth6':return LGBMClassifier(n_estimators=290,learning_rate=.035,max_depth=6,num_leaves=21,min_child_samples=40,colsample_bytree=.8,subsample=.8,subsample_freq=1,reg_lambda=6,reg_alpha=.5,verbosity=-1,n_jobs=2,random_state=SEED)
    if name=='catboost_depth4':return CatBoostClassifier(iterations=270,depth=4,learning_rate=.055,l2_leaf_reg=6,random_strength=.7,loss_function='MultiClass',verbose=False,thread_count=2,random_seed=SEED)
    if name=='catboost_depth6':return CatBoostClassifier(iterations=290,depth=6,learning_rate=.045,l2_leaf_reg=8,random_strength=1.0,loss_function='MultiClass',verbose=False,thread_count=2,random_seed=SEED)
    if name=='cat_dupweighted':return CatBoostClassifier(iterations=270,depth=4,learning_rate=.055,l2_leaf_reg=6,random_strength=.7,loss_function='MultiClass',verbose=False,thread_count=2,random_seed=SEED)
    raise ValueError(name)


def model_prob(model,X):
    if hasattr(model,'predict_proba'):
        p=np.asarray(model.predict_proba(X),dtype=float)
        assert p.ndim==2 and p.shape[1]==3
        return p
    # Ridge classifier has no probabilities; return predictions as one-hot for isolated diagnostic only
    z=np.asarray(model.predict(X),dtype=int)
    return np.eye(3)[z]


def train_rows_for_fold(train_data, train_ix, feature_cols, weighted):
    """training folds consist of unique entries; weighted variants repeat only training-fold originals."""
    d=train_data.iloc[train_ix]
    if weighted:
        return d.loc[d.index.repeat(d['_original_frequency'].to_numpy())]
    return d
