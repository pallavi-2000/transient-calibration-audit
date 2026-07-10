"""
src/alerce_perclass_definitions.py

Per-class calibration under three explicit definitions.

The paper's original "per-class ECE" conditions on the TRUE class: for each
class k it computes top-label ECE over the subset {i : y_i = k}. A referee
correctly noted this is not standard classwise calibration — it measures a
class-conditional confidence deficit (how much confidence the classifier
assigns to objects that ARE class k), not P(y=k | p_k = p) = p.

This script computes all three quantities so the paper can state its
definition precisely and show whether the headline pattern (best-classified
classes are worst-calibrated) survives the standard definitions:

  A. true-class-conditioned top-label ECE   (the paper's original table)
  B. predicted-class-stratified top-label ECE
       restrict to {i : argmax_j p_ij = k}, bin top-1 confidence vs correct
  C. one-vs-rest classwise ECE
       ALL objects, bin p_k vs indicator(y = k)   [standard classwise def.]

Reads:  data/processed/alerce_dataset.csv  (TDE excluded — taxonomy gap)
Writes: results/tables/perclass_definitions.json
"""

import json
import os

import numpy as np
import pandas as pd

from calibration import compute_binned_ce, bootstrap_binned_ce

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATASET = os.path.join(ROOT, 'data', 'processed', 'alerce_dataset.csv')
OUT = os.path.join(ROOT, 'results', 'tables', 'perclass_definitions.json')

CLASSES = ['SNIa', 'SNII', 'SNIbc', 'SLSN']


def main():
    df = pd.read_csv(DATASET)
    df = df[df['alerce_class'] != 'TDE'].reset_index(drop=True)
    n = len(df)
    print(f"Objects (TDE excluded): {n}")

    conf = df['confidence'].values
    correct = df['correct'].values.astype(float)
    true_cls = df['alerce_class'].values
    pred_cls = df['predicted_class'].values

    results = {}
    print(f"\n{'Class':<7} {'defn':<28} {'N':>6} {'ECE':>8} {'95% CI':>20}")
    print("-" * 74)

    for k in CLASSES:
        results[k] = {}

        # A. true-class-conditioned (paper's original definition)
        mask = true_cls == k
        ece = compute_binned_ce(conf[mask], correct[mask])
        _, lo, hi = bootstrap_binned_ce(conf[mask], correct[mask])
        results[k]['true_class_conditioned'] = {
            'n': int(mask.sum()), 'ece': round(ece, 4), 'ci_lo': lo, 'ci_hi': hi}
        print(f"{k:<7} {'A: true-class-conditioned':<28} {mask.sum():>6} {ece:>8.4f} "
              f"{'[' + str(lo) + ', ' + str(hi) + ']':>20}")

        # B. predicted-class-stratified top-label ECE
        mask = pred_cls == k
        if mask.sum() >= 20:
            ece = compute_binned_ce(conf[mask], correct[mask])
            _, lo, hi = bootstrap_binned_ce(conf[mask], correct[mask])
            results[k]['predicted_class_stratified'] = {
                'n': int(mask.sum()), 'ece': round(ece, 4), 'ci_lo': lo, 'ci_hi': hi}
            print(f"{k:<7} {'B: predicted-class strat.':<28} {mask.sum():>6} {ece:>8.4f} "
                  f"{'[' + str(lo) + ', ' + str(hi) + ']':>20}")
        else:
            results[k]['predicted_class_stratified'] = {'n': int(mask.sum()),
                                                        'note': 'n < 20, not reported'}
            print(f"{k:<7} {'B: predicted-class strat.':<28} {mask.sum():>6}    (n<20)")

        # C. one-vs-rest classwise ECE over ALL objects
        pk = df[k].values  # raw 15-class probability for class k
        yk = (true_cls == k).astype(float)
        ece = compute_binned_ce(pk, yk)
        _, lo, hi = bootstrap_binned_ce(pk, yk)
        results[k]['one_vs_rest'] = {
            'n': int(n), 'ece': round(ece, 4), 'ci_lo': lo, 'ci_hi': hi}
        print(f"{k:<7} {'C: one-vs-rest (all obj.)':<28} {n:>6} {ece:>8.4f} "
              f"{'[' + str(lo) + ', ' + str(hi) + ']':>20}")

        # accuracy for context
        mask_t = true_cls == k
        results[k]['accuracy'] = round(float((pred_cls[mask_t] == k).mean()), 4)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT}")

    print("\nDoes the accuracy-vs-calibration inversion survive each definition?")
    for defn, key in [('A', 'true_class_conditioned'),
                      ('B', 'predicted_class_stratified'),
                      ('C', 'one_vs_rest')]:
        vals = [(k, results[k]['accuracy'], results[k][key].get('ece'))
                for k in CLASSES if results[k][key].get('ece') is not None]
        print(f"  {defn}: " + "  ".join(f"{k}(acc {a:.2f}, ECE {e:.3f})" for k, a, e in vals))


if __name__ == '__main__':
    main()
