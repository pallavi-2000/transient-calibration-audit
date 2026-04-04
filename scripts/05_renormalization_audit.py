"""
Step 2: Renormalization Audit
==============================

Addresses reviewer concerns #1 and #2:
  1. The hierarchical compounding problem
  2. Renormalization artifacts

ALeRCE's lc_classifier outputs 15-class probabilities.
We renormalized to 4 transient classes. This script quantifies
how much that renormalization distorts the probabilities and
whether ECE changes on the raw 15-class vector.

Run from project root:
    python3 scripts/05_renormalization_audit.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import compute_ece, bootstrap_ece, fit_temperature_cv


def main():
    print("=" * 60)
    print("RENORMALIZATION AUDIT")
    print("Reviewer concerns #1 (hierarchy) and #2 (renormalization)")
    print("=" * 60)

    # Load data
    alerce = pd.read_csv("data/raw/alerce_classifications.csv")
    sample = pd.read_csv("data/ground_truth/bts_sample.csv")

    merged = alerce.merge(
        sample[["ZTFID", "alerce_class"]],
        left_on="oid", right_on="ZTFID"
    )
    # Exclude TDEs (ALeRCE has no TDE class)
    merged = merged[merged["alerce_class"] != "TDE"].copy()

    print(f"\nTotal objects: {len(merged)}")

    # ---- Define class groups ----
    transient_cols = ["SNIa", "SNIbc", "SNII", "SLSN"]
    nontransient_cols = [
        "QSO", "AGN", "Blazar", "CV/Nova", "YSO",
        "LPV", "E", "DSCT", "RRL", "CEP", "Periodic-Other"
    ]
    all_15_cols = transient_cols + nontransient_cols

    # Convert to numeric
    for col in all_15_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # ---- ANALYSIS 1: Non-transient probability mass ----
    print(f"\n{'='*50}")
    print("ANALYSIS 1: Non-transient probability mass")
    print(f"{'='*50}")

    available_nontrans = [c for c in nontransient_cols if c in merged.columns]
    merged["p_transient"] = merged[transient_cols].sum(axis=1)
    merged["p_nontransient"] = merged[available_nontrans].sum(axis=1)

    print(f"\nProbability mass distribution (all {len(merged)} objects):")
    print(f"  Mean P(transient):       {merged['p_transient'].mean():.3f}")
    print(f"  Mean P(non-transient):   {merged['p_nontransient'].mean():.3f}")
    print(f"  Median P(transient):     {merged['p_transient'].median():.3f}")
    print(f"  Min P(transient):        {merged['p_transient'].min():.3f}")
    print(f"  Max P(non-transient):    {merged['p_nontransient'].max():.3f}")

    # Quantify the artifact
    thresholds = [0.10, 0.20, 0.30, 0.50]
    print(f"\nObjects with significant non-transient probability:")
    for t in thresholds:
        n = (merged["p_nontransient"] > t).sum()
        pct = n / len(merged) * 100
        print(f"  P(non-transient) > {t:.0%}: {n:4d} ({pct:.1f}%)")

    # ---- ANALYSIS 2: Renormalization inflation ----
    print(f"\n{'='*50}")
    print("ANALYSIS 2: Renormalization inflation factor")
    print(f"{'='*50}")

    merged["inflation_factor"] = 1.0 / merged["p_transient"]

    print(f"\nInflation factor = 1 / P(transient):")
    print(f"  Mean:   {merged['inflation_factor'].mean():.3f}x")
    print(f"  Median: {merged['inflation_factor'].median():.3f}x")
    print(f"  Max:    {merged['inflation_factor'].max():.3f}x")
    print(f"  Objects with >2x inflation: "
          f"{(merged['inflation_factor'] > 2).sum()}")
    print(f"  Objects with >3x inflation: "
          f"{(merged['inflation_factor'] > 3).sum()}")

    # Show worst cases
    worst = merged.nlargest(5, "inflation_factor")
    print(f"\nTop 5 most inflated objects:")
    for _, row in worst.iterrows():
        trans_probs = row[transient_cols].astype(float)
        top_class = trans_probs.idxmax()
        raw_p = float(trans_probs[top_class])
        renorm_p = raw_p / float(row["p_transient"])
        print(f"  {row['oid']}: P({top_class})={raw_p:.3f} -> "
              f"{renorm_p:.3f} ({float(row['inflation_factor']):.1f}x inflation), "
              f"P(non-trans)={float(row['p_nontransient']):.3f}")

    # ---- ANALYSIS 3: ECE on raw 15-class vs renormalized 4-class ----
    print(f"\n{'='*50}")
    print("ANALYSIS 3: ECE comparison — raw 15-class vs renormalized 4-class")
    print(f"{'='*50}")

    class_to_int_4 = {c: i for i, c in enumerate(transient_cols)}
    labels_4 = merged["alerce_class"].map(class_to_int_4).values

    # Renormalized 4-class (what we've been using)
    proba_4 = merged[transient_cols].astype(float).values
    proba_4 = proba_4 / proba_4.sum(axis=1, keepdims=True)

    ece_4, _ = compute_ece(labels_4, proba_4)
    boot_4 = bootstrap_ece(labels_4, proba_4)

    print(f"\nRenormalized 4-class ECE: {boot_4['ece']:.3f} "
          f"[{boot_4['ci_lower']:.3f}, {boot_4['ci_upper']:.3f}]")

    # Raw 15-class (full vector, map labels to 15-class indices)
    # For 15-class ECE, we need labels in the 15-class space
    all_15_available = [c for c in all_15_cols if c in merged.columns]
    class_to_int_15 = {c: i for i, c in enumerate(all_15_available)}

    # Map alerce_class to 15-class index
    labels_15 = merged["alerce_class"].map(class_to_int_15).values
    proba_15 = merged[all_15_available].astype(float).values

    # Check probabilities sum to ~1
    row_sums = proba_15.sum(axis=1)
    print(f"\nRaw 15-class probability sums: "
          f"mean={row_sums.mean():.4f}, min={row_sums.min():.4f}, "
          f"max={row_sums.max():.4f}")

    # Normalize if needed (should be close to 1 already)
    proba_15 = proba_15 / proba_15.sum(axis=1, keepdims=True)

    ece_15, _ = compute_ece(labels_15, proba_15)
    boot_15 = bootstrap_ece(labels_15, proba_15)

    print(f"Raw 15-class ECE:          {boot_15['ece']:.3f} "
          f"[{boot_15['ci_lower']:.3f}, {boot_15['ci_upper']:.3f}]")

    print(f"\nDifference: {abs(ece_4 - ece_15):.3f}")
    if ece_15 > ece_4:
        print("  Raw 15-class has HIGHER ECE (worse)")
        print("  -> Renormalization actually HELPS by focusing on relevant classes")
    elif ece_15 < ece_4:
        print("  Raw 15-class has LOWER ECE (better)")
        print("  -> Renormalization INFLATES miscalibration — this is the artifact")
    else:
        print("  No significant difference")

    # ---- ANALYSIS 4: Temperature scaling on raw 15-class ----
    print(f"\n{'='*50}")
    print("ANALYSIS 4: Temperature scaling — raw 15-class")
    print(f"{'='*50}")

    ts_15 = fit_temperature_cv(labels_15, proba_15, n_folds=5)
    print(f"  T = {ts_15['T_mean']:.4f}")
    print(f"  ECE before: {ts_15['ece_before']:.4f}")
    print(f"  ECE after:  {ts_15['ece_after']:.4f}")
    print(f"  Recommended: {ts_15['recommended']}")

    # Compare with 4-class temp scaling
    ts_4 = fit_temperature_cv(labels_4, proba_4, n_folds=5)
    print(f"\nComparison:")
    print(f"  {'Metric':<25} {'4-class (renorm)':<20} {'15-class (raw)':<20}")
    print(f"  {'ECE baseline':<25} {boot_4['ece']:<20.3f} {boot_15['ece']:<20.3f}")
    print(f"  {'T':<25} {ts_4['T_mean']:<20.4f} {ts_15['T_mean']:<20.4f}")
    print(f"  {'ECE after T':<25} {ts_4['ece_after']:<20.4f} {ts_15['ece_after']:<20.4f}")

    # ---- ANALYSIS 5: Per-class impact ----
    print(f"\n{'='*50}")
    print("ANALYSIS 5: Per-class renormalization impact")
    print(f"{'='*50}")

    for cls in transient_cols:
        idx = class_to_int_4[cls]
        mask = labels_4 == idx

        raw_conf = merged.loc[mask, cls].astype(float).values
        renorm_conf = proba_4[mask, idx]

        print(f"\n  {cls} (n={mask.sum()}):")
        print(f"    Raw P({cls}):        mean={raw_conf.mean():.3f}, "
              f"max={raw_conf.max():.3f}")
        print(f"    Renormalized P({cls}): mean={renorm_conf.mean():.3f}, "
              f"max={renorm_conf.max():.3f}")
        print(f"    Inflation:           {renorm_conf.mean()/raw_conf.mean():.2f}x")

    # ---- Save results ----
    results = {
        "n_objects": int(len(merged)),
        "p_transient_mean": float(merged["p_transient"].mean()),
        "p_nontransient_mean": float(merged["p_nontransient"].mean()),
        "inflation_factor_mean": float(merged["inflation_factor"].mean()),
        "inflation_factor_max": float(merged["inflation_factor"].max()),
        "n_gt_10pct_nontransient": int((merged["p_nontransient"] > 0.10).sum()),
        "n_gt_50pct_nontransient": int((merged["p_nontransient"] > 0.50).sum()),
        "ece_4class_renormalized": boot_4["ece"],
        "ece_15class_raw": boot_15["ece"],
        "ts_4class": {
            "T": ts_4["T_mean"],
            "ece_after": ts_4["ece_after"],
        },
        "ts_15class": {
            "T": ts_15["T_mean"],
            "ece_after": ts_15["ece_after"],
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/renormalization_audit.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Mean non-transient probability: {merged['p_nontransient'].mean():.1%}")
    print(f"Objects with >50% non-transient: {(merged['p_nontransient'] > 0.50).sum()}")
    print(f"Mean inflation factor: {merged['inflation_factor'].mean():.2f}x")
    print(f"ECE (4-class renormalized): {boot_4['ece']:.3f}")
    print(f"ECE (15-class raw):         {boot_15['ece']:.3f}")
    print(f"\nResults saved to: results/renormalization_audit.json")


if __name__ == "__main__":
    main()
