"""
src/fink_extras.py

Referee-requested diagnostics for the Fink binary classifiers:

1. Ranking metrics (ROC-AUC, average precision): temperature scaling hitting
   its bound shows *scalar monotone rescaling* fails, but only ranking metrics
   can distinguish "poorly calibrated but still ranks SNIa above non-SNIa"
   from "no discriminative signal at all". Monotone recalibration preserves
   ranking, so AUC ~ 0.5 means NO monotone post-hoc method can help.

2. Isotonic regression (5-fold CV): a much more flexible monotone calibrator
   than temperature scaling. If isotonic also fails to produce calibrated
   scores, the "cannot rescale away" claim is supported beyond scalar T.

3. Early-phase proxy subsets via `ndethist` (number of prior detections at
   the retrieved alert): Fink documents rf_snia_vs_nonia as targeting
   *rising, early-phase* SNIa candidates. We check whether RF performs
   better in its documented regime. ndethist at the retrieved (most recent)
   alert is an imperfect phase proxy — stated as such in the paper.

Reads:  data/raw/fink_predictions.csv
Writes: results/tables/fink_extras.json
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from calibration import compute_binned_ce, bootstrap_binned_ce

ROOT = os.path.join(os.path.dirname(__file__), '..')
FINK = os.path.join(ROOT, 'data', 'raw', 'fink_predictions.csv')
OUT = os.path.join(ROOT, 'results', 'tables', 'fink_extras.json')

SEED = 42


def brier(scores, y):
    return float(np.mean((scores - y) ** 2))


def isotonic_cv_ece(scores, y, n_splits=5):
    """Fit isotonic regression on train folds, evaluate ECE on held-out folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.empty_like(scores, dtype=float)
    for tr, te in skf.split(scores.reshape(-1, 1), y):
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(scores[tr], y[tr])
        oof[te] = iso.predict(scores[te])
    return compute_binned_ce(oof, y), brier(oof, y)


def analyse(name, scores, y):
    res = {
        'n': int(len(y)),
        'snia_fraction': round(float(y.mean()), 4),
        'ece': round(compute_binned_ce(scores, y), 4),
        'brier': round(brier(scores, y), 4),
        'roc_auc': round(float(roc_auc_score(y, scores)), 4),
        'pr_auc': round(float(average_precision_score(y, scores)), 4),
    }
    iso_ece, iso_brier = isotonic_cv_ece(scores, y)
    res['isotonic_cv'] = {'ece': round(iso_ece, 4), 'brier': round(iso_brier, 4)}
    return res


def main():
    df = pd.read_csv(FINK)
    y = (df['alerce_class'] == 'SNIa').astype(float).values

    results = {}
    print(f"{'Subset':<22} {'clf':<5} {'N':>6} {'ECE':>7} {'iso-ECE':>8} "
          f"{'ROC-AUC':>8} {'PR-AUC':>7} {'Brier':>7}")
    print("-" * 76)

    for clf, col in [('rf', 'rf_snia_vs_nonia'), ('snn', 'snn_snia_vs_nonia')]:
        results[clf] = {}
        subsets = {
            'full': np.ones(len(df), dtype=bool),
            'ndethist_le5': df['ndethist'].values <= 5,
            'ndethist_le3': df['ndethist'].values <= 3,
        }
        for sub_name, mask in subsets.items():
            r = analyse(f"{clf}/{sub_name}", df[col].values[mask], y[mask])
            results[clf][sub_name] = r
            print(f"{sub_name:<22} {clf:<5} {r['n']:>6} {r['ece']:>7.4f} "
                  f"{r['isotonic_cv']['ece']:>8.4f} {r['roc_auc']:>8.4f} "
                  f"{r['pr_auc']:>7.4f} {r['brier']:>7.4f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT}")

    rf_auc = results['rf']['full']['roc_auc']
    snn_auc = results['snn']['full']['roc_auc']
    print(f"\nInterpretation guide:")
    print(f"  ROC-AUC 0.5 = no ranking signal; monotone recalibration preserves AUC,")
    print(f"  so low AUC means NO monotone post-hoc method can recover calibration.")
    print(f"  RF AUC = {rf_auc}, SNN AUC = {snn_auc}")


if __name__ == '__main__':
    main()
