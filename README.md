# Vehicle Type Classification — Kaggle College Project

**Best reported public Kaggle leaderboard accuracy:** 0.90700 (90.70%), **reported rank:** #1 on 15 October 2026. The private leaderboard score is not yet known.

This repository preserves the scripts and training-only validation experiments for a three-class synthetic vehicle dataset with 20 numeric predictors. The winning predictor is a **50:50 probability blend** of standardized Logistic Regression and a depth-5 XGBoost classifier. Neural networks were not used.

## Requirements

Python 3.10+ recommended. Install `pip install -r requirements.txt`.

## Data

Place the supplied Kaggle train, test and sample submission CSVs under `data/` (see `data/README.md`). The dataset is not redistributed here.

## Reproduce the winning submission

```bash
python src/generate_winning_submission.py --data-dir data --output blend_logit_xgb_50_50.csv
```

The script trains both locked model recipes on deduplicated labeled training data, then reads the test CSV only for inference. It writes `vehicle_id,vehicle_class`.

## Exact winning recipe

- Logistic Regression: `StandardScaler`, `C=1`, solver `lbfgs`, `max_iter=1600`.
- XGBoost: 290 trees, maximum depth 5, learning rate 0.042, min child weight 6, row/column subsampling 0.8, L1 alpha 0.4, L2 lambda 8, seed 42.
- `p = 0.5*p_logistic + 0.5*p_xgboost`; class prediction via argmax.
- Source of record: `src/recipes.py` and `src/create_submissions.py` from Round 3.

## Research and validation

The initial batch compared 35 individual classifiers, 7 blends, and 2 cross-fitted stackers; public scores for all 44 were supplied by the student/team and preserved in `experiments/public_leaderboard_44_reported.csv`. A second batch screened 58 candidates and generated 37 Round 3 submissions; the 50:50 blend's user-reported public score was 0.90700.

Training-only preprocessing and validation were used throughout; duplicates were not allowed to cross training/validation folds. The original models often showed training–validation gaps, so local CV and public leaderboard results are reported separately. The public score covers ~40% of hidden test rows, and first place may change on the private leaderboard.

## Files

- `src/recipes.py`: original Round 3 candidate factory and registered blends.
- `src/evaluate.py`: training-only split, CV, robustness evaluation. **Note:** archived script uses absolute `/mnt/data` paths; adjust `ROOT` to run in another environment.
- `src/create_submissions.py`: archived Round 3 generator of 37 CSVs; likewise adjust `ROOT`.
- `src/generate_winning_submission.py`: portable single-command champion reconstruction.
- `src/round2_*`: original Round 2 training, validation and candidate functions, archived for methodology transparency.
- `experiments/`: scorecards, Round 3 manifest, OOF/repeated-CV audit, public-score log.

## Link for the college report

Replace the placeholder in the report with the **actual GitHub repository URL** after uploading this package. Do not claim private leaderboard performance before it is released.
