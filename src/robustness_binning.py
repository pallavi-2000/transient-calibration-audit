"""
src/robustness_binning.py

Robustness check: does the choice of binning scheme (equal-width, the
paper's default, vs equal-mass/adaptive) change the calibration story?

Standard ECE bins by equal-width confidence intervals. For a classifier
whose scores are heavily concentrated near one end — Fink's
rf_snia_vs_nonia has mean score 0.011, so ~90% of objects fall in the
first of 10 equal-width bins — the other bins are estimated from very
few points, and the metric's exact value can be sensitive to bin count
and placement (Nixon et al. 2019; Roelofs et al. 2022). This script
recomputes every headline ECE using equal-mass (adaptive) bins instead,
so every bin gets ~n/10 samples regardless of score skew, and reports
both side by side with bootstrap CIs.

Reads already-collected data (no network calls):
    data/processed/alerce_dataset.csv
    data/raw/fink_predictions.csv

Writes:
    results/tables/robustness_binning.json
"""

import json
import os

import numpy as np
import pandas as pd

from calibration import compute_binned_ce, bootstrap_binned_ce

ROOT = os.path.join(os.path.dirname(__file__), '..')
ALERCE_PATH = os.path.join(ROOT, 'data', 'processed', 'alerce_dataset.csv')
FINK_PATH = os.path.join(ROOT, 'data', 'raw', 'fink_predictions.csv')
OUT_PATH = os.path.join(ROOT, 'results', 'tables', 'robustness_binning.json')


def summarize(values, targets, label):
    ew_point = compute_binned_ce(values, targets, n_bins=10, adaptive=False)
    ew_mean, ew_lo, ew_hi = bootstrap_binned_ce(values, targets, n_bins=10, adaptive=False)
    am_point = compute_binned_ce(values, targets, n_bins=10, adaptive=True)
    am_mean, am_lo, am_hi = bootstrap_binned_ce(values, targets, n_bins=10, adaptive=True)

    print(f"\n{label}  (n={len(values)})")
    print(f"  Equal-width : {ew_point:.4f}  (95% CI [{ew_lo}, {ew_hi}])")
    print(f"  Equal-mass  : {am_point:.4f}  (95% CI [{am_lo}, {am_hi}])")

    return {
        "n": int(len(values)),
        "equal_width": {"point": round(ew_point, 4), "ci_lo": ew_lo, "ci_hi": ew_hi},
        "equal_mass":  {"point": round(am_point, 4), "ci_lo": am_lo, "ci_hi": am_hi},
    }


def main():
    results = {}

    # ── ALeRCE: top-1 confidence vs correctness, TDE excluded (taxonomy gap) ──
    alerce = pd.read_csv(ALERCE_PATH)
    alerce = alerce[alerce['alerce_class'] != 'TDE']
    results['alerce_aggregate'] = summarize(
        alerce['confidence'].values, alerce['correct'].values.astype(float),
        "ALeRCE aggregate (top-1 confidence vs. correctness)"
    )

    # ── Fink: raw score vs true-SNIa indicator (matches the paper's binary ECE) ──
    fink = pd.read_csv(FINK_PATH)
    is_snia = (fink['alerce_class'] == 'SNIa').astype(float).values

    results['fink_rf'] = summarize(
        fink['rf_snia_vs_nonia'].values, is_snia,
        "Fink RF (rf_snia_vs_nonia) — the most skewed score distribution in the study"
    )
    results['fink_snn'] = summarize(
        fink['snn_snia_vs_nonia'].values, is_snia,
        "Fink SNN (snn_snia_vs_nonia)"
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")

    print("\n" + "=" * 60)
    print("SUMMARY: does adaptive binning change the qualitative story?")
    print("=" * 60)
    for key, r in results.items():
        ew, am = r['equal_width']['point'], r['equal_mass']['point']
        rel_change = (am - ew) / ew * 100 if ew else float('nan')
        print(f"  {key:16s}  equal-width={ew:.4f}  equal-mass={am:.4f}  "
              f"({rel_change:+.1f}% relative change)")


if __name__ == '__main__':
    main()
