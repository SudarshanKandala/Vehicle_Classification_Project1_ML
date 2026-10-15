"""Build train-only diagnostic report for batch deliverable. Does not read test data."""
import json,zipfile
from pathlib import Path
import numpy as np,pandas as pd
root=Path('/mnt/data');o=root/'vehicle_batch'
a=pd.read_csv(root/'vehicle_train_10000(1).csv');cols=[c for c in a if c not in ('vehicle_id','vehicle_class')]
u=a.drop_duplicates(subset=cols);r=pd.read_csv(o/'model_scorecard.csv')
co=u[cols].corr().abs();pairs=co.where(np.triu(np.ones(co.shape,dtype=bool),1)).stack().sort_values(ascending=False).head(8)
g=u.groupby('vehicle_class')[cols].mean();diff=((g.max()-g.min())/u[cols].std()).sort_values(ascending=False).head(8)
lines=[
'# Vehicle Type Classification — 44 non-neural submission candidates',
'',
'## Competition constraints',
'- Objective: multiclass **classification accuracy**; classes 0, 1, 2.',
'- Required CSV columns: `vehicle_id,vehicle_class`.',
'- User-reported most recent public score: **0.896** for the earlier LDA submission, *not* a score for any submission in this new batch.',
'- User-requested rule: **do not designate one winning model**; all 44 candidates have independent files for user selection.',
'- **No neural networks**, and no test-feature inspection or learning; test was used only for final prediction generation after all candidate configurations were frozen.',
'',
'## Step 1 — Training-only EDA and leakage audit',
'- Original train: 10,000 rows × 22 columns (20 numeric features, 1 ID, 1 target).',
'- Unique feature rows: 7,500; remaining 2,500 are exact copies with matching targets.',
'- No missing values; no constant training features.',
'- Class counts in unique training rows: '+str(u.vehicle_class.value_counts().sort_index().to_dict())+'.',
'- All duplicates removed before train/validation splitting. Group leakage is thereby blocked even if copies are scattered in input ordering.',
'- `vehicle_id` excluded from predictors, since it is an identifier rather than a defensible source of vehicle information.',
'- Duplicate-weighted candidate configurations repeat ONLY the training portion of the unique-fold data during CV; validation copies are never present in the fitted fold.',
'- Numeric features are highly collinear, supporting regularized linear and discriminant models as well as a diverse tree baseline.',
'',
'**Largest absolute feature correlations (unique training data only):**',
]
for (c1,c2),v in pairs.items():lines.append(f'- `{c1}` / `{c2}`: {v:.3f}')
lines +=['','**Largest standardized class-mean differences (exploratory only):**']
for c,v in diff.items():lines.append(f'- `{c}`: {v:.2f} pooled training-standard-deviation units')
lines +=[
'',
'## Steps 2–5 — Validation, baseline models, regularization and family comparison',
'- Primary screening: 3-fold `StratifiedKFold(shuffle=True, random_state=2026)` over **7,500 unique rows**, identical folds for all candidates.',
'- Evaluated 35 individually fitted estimators, 7 fixed probability blends and 2 leakage-safe cross-fitted stacking estimators.',
'- All feature normalization, dimensionality reduction, feature selection and meta-model training occur inside the relevant CV training folds.',
'- Different regularization strengths, RBF/poly kernels, boosting trees/depth, forest depth/leaf count, and categorical-free numerical pipelines are represented; see `code/vehicle_batch_models.py` for exact hyperparameters.',
'- Earlier separate training-only research also screened 121 successfully fitted configurations and evaluated repeated 5-fold CV finalists. It is archived as historical context, not recomputed in this batch.',
'- All 44 candidates are offered **without automatic winner selection**, regardless of CV rank.',
'',
'## Steps 6–8 — Overfitting and feature / robustness checks',
'- Tree models often show larger train–CV gaps than LDA/logistic regression on the same folds; see `model_scorecard.csv`.',
'- Represented feature variations: deterministic vehicle ratio/interaction features, fold-fitted SelectKBest and PCA, plus unique-row vs duplicate-weighted learning.',
'- Represented ensembles: linear mixtures, boosted-tree mixtures, cross-family mixtures, and nested cross-fitted stacking. No pseudo-labeling or test-derived statistics.',
'- Seed stability experiments for earlier finalists are described in archived previous reports. Full repeated-seed validation for each of the 44 submitted configs was **not performed**; 3-fold results are screening-level evidence only.',
'- No fresh untouched independent holdout score is claimed for these 44 candidates. The previous holdout was used in earlier work, so it should not be repeatedly mined for selection.',
'',
'## Step 9 — Model selection deliberately delegated to user',
'- Per request, **no single candidate is designated as winner** and no models are removed based on test scores.',
'- The public leaderboard is a noisy estimate; a 90.30% visible fifth-place score is a user-provided historical snapshot, not a score earned by this batch.',
'- 3-fold CV accuracies remain approximately 84.73–89.31%. Therefore top-five performance cannot be asserted.',
'',
'## Step 10 — Inference and submissions',
'- First all 35 base models and 2 stack models were fitted on their locked training-only datasets (unique rows unless explicitly duplicate-weighted).',
'- Only after finishing all model training, test was loaded solely for batch prediction and output-schema checks.',
'- 44 CSVs were validated as 2,500 rows with exactly `vehicle_id,vehicle_class` columns and labels from {0,1,2}. No test-label access and no performance calculations against test.',
'- `submissions/` directory holds each model CSV; `model_scorecard.csv` gives corresponding training-only CV scores and names.',
'- All scripts under `code/` are reproducible; individual configurations are defined by `build` in `vehicle_batch_models.py`.',
'',
'## Full local validation scorecard (not Kaggle test scores)',
'',
'| Candidate filename | Family | 3-fold CV accuracy | CV SD | Train–CV gap (pp) |',
'|---|---|---:|---:|---:|',
]
for row in r.itertuples():
    gap='—' if pd.isna(row.gap) else f'{row.gap*100:.2f}'
    lines.append(f'| `{row.filename}` | {row.kind} | {row.cv_mean*100:.2f}% | {row.cv_std*100:.2f} pp | {gap} |')
lines +=['','## How to submit', '1. Extract `vehicle_all_44_submissions.zip`.','2. Upload one `.csv` at a time from its `submissions/` folder using the Kaggle competition submission page.','3. Track public scores separately. They are not model-training metrics and cannot be used to infer hidden labels.','','**Notes:** Prior versions and user-reported leaderboard scores are not used to revise the fixed model slate. No Kaggle submissions were made automatically.']
(o/'README.md').write_text('\n'.join(lines)+'\n')
z=root/'vehicle_all_44_submissions.zip'
with zipfile.ZipFile(z,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zz:
    zz.write(o/'README.md',arcname='README.md')
    zz.write(root/'vehicle_batch_build_report.py',arcname='code/vehicle_batch_build_report.py')
    for fn in ['vehicle_optimization_summary.md','vehicle_optimize_phase1.json','vehicle_repeated_candidates.json','vehicle_model_report.md']:
        if (root/fn).exists():zz.write(root/fn,arcname='historical/'+fn)
print('VERIFIED_REPORT',str(o/'README.md'),len(r),'configs',round(z.stat().st_size/1024),'KiB')
