"""Reproduce the public-leaderboard-winning 50/50 Logistic Regression + XGBoost ensemble.

Model recipe copied from the archived Round 3 `recipes.py` / `create_submissions.py`.
No neural networks. No test analysis or test-driven model selection.

Usage:
    python src/generate_winning_submission.py --data-dir data \
        --output blend_logit_xgb_50_50.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from recipes import build

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=Path('data'))
    ap.add_argument('--output', type=Path, default=Path('blend_logit_xgb_50_50.csv'))
    a = ap.parse_args()
    # Lock both architectures and ensemble weights before reading test data.
    names = ('logit_c1', 'xgb_d5_baseline')
    weights = (0.5, 0.5)
    train = pd.read_csv(a.data_dir / 'vehicle_train_10000(1).csv')
    columns = [c for c in train if c not in ('vehicle_id', 'vehicle_class')]
    assert len(columns) == 20
    assert train.groupby(columns)['vehicle_class'].nunique().max() == 1
    unique = train.drop_duplicates(subset=columns).reset_index(drop=True)
    X, y = unique[columns], unique['vehicle_class']
    models = {name: build(name).fit(X, y) for name in names}
    # Inference only; absolutely no training/tuning/test-distribution inspection below.
    test = pd.read_csv(a.data_dir / 'vehicle_test(1).csv')
    sample = pd.read_csv(a.data_dir / 'vehicle_sample_submission(1).csv')
    assert list(sample.columns) == ['vehicle_id', 'vehicle_class']
    assert list(test.columns) == ['vehicle_id', *columns]
    assert len(test) == len(sample)
    assert test['vehicle_id'].equals(sample['vehicle_id'])
    probs = [models[name].predict_proba(test[columns]) for name in names]
    assert all(p.shape == (len(test), 3) for p in probs)
    combined = sum(w * p for w, p in zip(weights, probs))
    assert np.isfinite(combined).all()
    result = pd.DataFrame({'vehicle_id': sample['vehicle_id'],
                           'vehicle_class': np.argmax(combined, axis=1)})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False)
    print(f'Saved {len(result)} predictions: {a.output}')

if __name__ == '__main__':
    main()
