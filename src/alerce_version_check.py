"""
src/alerce_version_check.py

Version-stratified calibration check for ALeRCE.

The original 1,606-object collection (collect_alerce.py) discarded the
`classifier_version` field returned alongside each probability vector,
so the dataset was analysed as if it came from one stationary model.
In fact ALeRCE's production lc_classifier is versioned; a spot check
during this audit found at least two distinct classifier_version
strings already present among objects in this very dataset
(hierarchical_rf_1.1.0 and lc_classifier_1.1.13).

collect_alerce_versions.py re-queries the live API to recover
classifier_version for every object (see data/raw/alerce_versions.csv).
This script merges that back onto the analysis dataset and checks
whether the headline ECE=0.259 / T=0.359 result is stable across
versions, or whether it is an artefact of pooling calibration curves
from genuinely different models.

ECE here is the same quantity as the paper's headline: top-1 ECE over
the full 15-class output (confidence = max over all 15 classes,
correct = top-1 class matches spectroscopic label). An earlier draft
of this script used the 4-class renormalised probabilities, which
made the pooled number (0.250) inconsistent with the 15-class
headline (0.259) — a referee caught this.

Caveat on version assignment: the tags come from a later re-query.
If the broker recomputed an object's classification between our
original probability retrieval and the re-query, the tag may refer
to a newer output than the probabilities we analyse. We cannot rule
this out from the API alone; the check should be read as
best-available, not perfect, version attribution.

Writes: results/tables/alerce_version_stratified_ece.json
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from calibration import compute_ece, bootstrap_ece

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATASET_PATH = os.path.join(ROOT, 'data', 'processed', 'alerce_dataset.csv')
VERSIONS_PATH = os.path.join(ROOT, 'data', 'raw', 'alerce_versions.csv')
PREDICTIONS_PATH = os.path.join(ROOT, 'data', 'raw', 'alerce_predictions.csv')
OUT_PATH = os.path.join(ROOT, 'results', 'tables', 'alerce_version_stratified_ece.json')

# full 15-class output space, in CSV column order
ALL_CLASS_COLS = ['SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar',
                  'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP',
                  'Periodic-Other']
MIN_N = 100  # minimum objects for a version-specific ECE/T to be reported


def fit_temperature(y_true_idx, y_proba):
    def objective(T):
        scaled = softmax(np.log(np.clip(y_proba, 1e-10, 1)) / T, axis=1)
        ece, _ = compute_ece(y_true_idx, scaled)
        return ece
    res = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
    return res.x


def main():
    if not os.path.exists(VERSIONS_PATH):
        print(f"ERROR: {VERSIONS_PATH} not found.")
        print("Run collect_alerce_versions.py first (re-queries the live ALeRCE API).")
        return

    versions = pd.read_csv(VERSIONS_PATH)
    dataset = pd.read_csv(DATASET_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    print(f"Version tags collected: {len(versions)} / {len(predictions)} objects "
          f"({len(versions) / len(predictions) * 100:.1f}%)")

    merged = dataset.merge(versions, on='oid', how='inner')
    merged = merged[merged['alerce_class'] != 'TDE'].reset_index(drop=True)

    print(f"Merged (TDE excluded): {len(merged)} objects")
    print("\nVersion distribution:")
    print(merged['classifier_version'].value_counts().to_string())

    # 15-class basis: same quantity as the paper's headline ECE
    class_to_idx = {c: i for i, c in enumerate(ALL_CLASS_COLS)}
    y_true_all = merged['alerce_class'].map(class_to_idx).values
    y_proba_all = merged[ALL_CLASS_COLS].values
    y_proba_all = y_proba_all / y_proba_all.sum(axis=1, keepdims=True)

    results = {'basis': '15-class top-1 ECE (matches headline)'}

    # ── pooled (all versions merged) ──────────────────────────────────────
    pooled_ece, pooled_lo, pooled_hi = bootstrap_ece(y_true_all, y_proba_all)
    pooled_T = fit_temperature(y_true_all, y_proba_all)
    results['pooled'] = {
        'n': int(len(merged)), 'ece': pooled_ece, 'ci_lo': pooled_lo, 'ci_hi': pooled_hi,
        'T': round(float(pooled_T), 4),
    }
    print(f"\nPooled (this merged subset): ECE={pooled_ece} [{pooled_lo}, {pooled_hi}], T={pooled_T:.4f}")

    # ── per version ──────────────────────────────────────────────────────
    results['by_version'] = {}
    print(f"\n{'Version':<24} {'N':>6} {'ECE':>8} {'95% CI':>18} {'T':>8}")
    print("-" * 68)
    for version, group in merged.groupby('classifier_version'):
        n = len(group)
        if n < MIN_N:
            print(f"{version:<24} {n:>6}   (below n={MIN_N} threshold, skipped)")
            results['by_version'][version] = {'n': int(n), 'note': f'n < {MIN_N}, ECE not reported'}
            continue

        y_true = group['alerce_class'].map(class_to_idx).values
        y_proba = group[ALL_CLASS_COLS].values
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        ece, lo, hi = bootstrap_ece(y_true, y_proba)
        T = fit_temperature(y_true, y_proba)

        print(f"{version:<24} {n:>6} {ece:>8.4f} {'[' + str(lo) + ', ' + str(hi) + ']':>18} {T:>8.4f}")
        results['by_version'][version] = {
            'n': int(n), 'ece': ece, 'ci_lo': lo, 'ci_hi': hi, 'T': round(float(T), 4),
        }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == '__main__':
    main()
