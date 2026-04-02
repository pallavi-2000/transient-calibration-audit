"""
ALeRCE Calibration Analysis
=============================

Complete analysis of ALeRCE's Balanced Random Forest light curve
classifier (lc_classifier v1.1.13, 15-class).

Key findings:
  - Systematic underconfidence: accuracy ~0.75, mean conf ~0.45
  - ECE ~0.27 (severe miscalibration)
  - Temperature scaling (T~0.36) reduces ECE to ~0.08 (71% improvement)
  - Underconfidence inverts the Guo et al. 2017 overconfidence norm

Usage:
    python scripts/01_alerce_analysis.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import (
    compute_ece, compute_classwise_ece, brier_score,
    fit_temperature_cv, apply_temperature, bootstrap_ece,
    auto_calibrate
)
from src.plotting import (
    reliability_diagram, reliability_diagram_comparison,
    per_class_reliability_grid
)


def main():
    print("=" * 60)
    print("ALeRCE CALIBRATION ANALYSIS")
    print("Classifier: Balanced Random Forest (lc_classifier v1.1.13)")
    print("=" * 60)

    # Create output directories
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ---- Load and prepare data ----
    alerce = pd.read_csv("data/raw/alerce_classifications.csv")
    sample = pd.read_csv("data/ground_truth/bts_sample.csv")

    merged = alerce.merge(
        sample[["ZTFID", "alerce_class"]],
        left_on="oid", right_on="ZTFID"
    )

    # Exclude TDEs (ALeRCE has no TDE class)
    merged = merged[merged["alerce_class"] != "TDE"].copy()

    # Multi-class probabilities (4 transient classes, renormalized)
    trans_classes = ["SNIa", "SNIbc", "SNII", "SLSN"]
    class_to_int = {c: i for i, c in enumerate(trans_classes)}

    proba = merged[trans_classes].apply(pd.to_numeric, errors="coerce").values
    proba = proba / proba.sum(axis=1, keepdims=True)
    labels = merged["alerce_class"].map(class_to_int).values

    print(f"\nSample size: {len(labels)} objects (TDEs excluded)")
    print(f"Class distribution:")
    for cls, idx in class_to_int.items():
        print(f"  {cls}: {(labels == idx).sum()}")

    # ---- Overall statistics ----
    confidences = np.max(proba, axis=1)
    predictions = np.argmax(proba, axis=1)
    accuracy = np.mean(predictions == labels)

    print(f"\nOverall accuracy: {accuracy:.3f}")
    print(f"Mean confidence: {confidences.mean():.3f}")
    print(f"Accuracy - Confidence gap: {accuracy - confidences.mean():+.3f}")
    print(f"Direction: {'underconfident' if accuracy > confidences.mean() else 'overconfident'}")

    # ---- ECE with bootstrap CI ----
    print(f"\n--- ECE (equal-mass binning, 15 bins) ---")
    boot = bootstrap_ece(labels, proba, n_bins=15)
    print(f"ECE: {boot['ece']:.3f} [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")

    # Also compute equal-width for comparison
    ece_ew, bins_ew = compute_ece(labels, proba, n_bins=15, strategy="equal_width")
    print(f"ECE (equal-width): {ece_ew:.3f}")

    # Brier score
    brier = brier_score(labels, proba)
    print(f"Brier score: {brier['brier_score']:.3f}")

    # ---- Figure 1: Aggregate reliability diagram ----
    ece_em, bins_em = compute_ece(labels, proba, n_bins=15, strategy="equal_mass")
    reliability_diagram(
        bins_em,
        title="ALeRCE Light Curve Classifier",
        save_path="figures/fig1_alerce_reliability.pdf",
        color="#0072B2"
    )
    # Also save PNG for quick viewing
    reliability_diagram(
        bins_em,
        title="ALeRCE Light Curve Classifier",
        save_path="figures/fig1_alerce_reliability.png",
        color="#0072B2"
    )

    # ---- Per-class ECE ----
    print(f"\n--- Per-class ECE ---")
    per_class = compute_classwise_ece(labels, proba, n_bins=15,
                                      class_names=trans_classes)

    for cls, stats in per_class.items():
        print(f"  {cls:6s}: ECE={stats['ece']:.3f}  "
              f"acc={stats['accuracy']:.3f}  "
              f"conf={stats['mean_confidence']:.3f}  "
              f"gap={stats['gap']:+.3f}")

    # ---- Figure 2: Per-class reliability diagrams ----
    # Compute per-class bin data for plotting
    class_bin_data = {}
    for k, cls_name in enumerate(trans_classes):
        is_k = (labels == k).astype(float)
        probs_k = proba[:, k]

        quantiles = np.linspace(0, 100, 16)
        bin_edges = np.percentile(probs_k, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0 + 1e-8
        bin_edges = np.unique(bin_edges)

        bins = []
        for i in range(len(bin_edges) - 1):
            in_bin = (probs_k >= bin_edges[i]) & (probs_k < bin_edges[i + 1])
            count = in_bin.sum()
            if count > 0:
                bins.append({
                    "confidence": float(probs_k[in_bin].mean()),
                    "accuracy": float(is_k[in_bin].mean()),
                    "count": int(count),
                    "gap": float(is_k[in_bin].mean() - probs_k[in_bin].mean()),
                })
        class_bin_data[cls_name] = bins

    per_class_reliability_grid(
        per_class, class_bin_data,
        title="ALeRCE Per-Class Calibration",
        save_path="figures/fig2_alerce_perclass.pdf"
    )
    per_class_reliability_grid(
        per_class, class_bin_data,
        title="ALeRCE Per-Class Calibration",
        save_path="figures/fig2_alerce_perclass.png"
    )

    # ---- Temperature scaling (5-fold CV) ----
    print(f"\n--- Temperature Scaling (5-fold CV) ---")
    ts = fit_temperature_cv(labels, proba, n_folds=5)
    print(f"T = {ts['T_mean']:.3f} +/- {ts['T_std']:.3f}")
    print(f"ECE before: {ts['ece_before']:.3f}")
    print(f"ECE after:  {ts['ece_after']:.3f}")
    print(f"Improvement: {ts['improvement']:.3f} ({ts['improvement']/ts['ece_before']*100:.0f}%)")
    print(f"Recommended: {ts['recommended']}")

    # ---- Figure 3: Before/After temperature scaling ----
    T_global = ts["T_mean"]
    proba_calibrated = apply_temperature(proba, T_global)
    _, bins_after = compute_ece(labels, proba_calibrated, n_bins=15, strategy="equal_mass")

    reliability_diagram_comparison(
        bins_em, bins_after,
        T_value=T_global,
        title="ALeRCE",
        save_path="figures/fig3_alerce_tempscaling.pdf"
    )
    reliability_diagram_comparison(
        bins_em, bins_after,
        T_value=T_global,
        title="ALeRCE",
        save_path="figures/fig3_alerce_tempscaling.png"
    )

    # ---- Save results as JSON ----
    results = {
        "classifier": "ALeRCE lc_classifier v1.1.13",
        "architecture": "Balanced Random Forest",
        "n_objects": int(len(labels)),
        "n_classes": len(trans_classes),
        "class_names": trans_classes,
        "class_counts": {c: int((labels == i).sum()) for c, i in class_to_int.items()},
        "accuracy": float(accuracy),
        "mean_confidence": float(confidences.mean()),
        "direction": "underconfident",
        "ece": boot["ece"],
        "ece_ci": [boot["ci_lower"], boot["ci_upper"]],
        "ece_equal_width": float(ece_ew),
        "brier_score": brier["brier_score"],
        "per_class": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                          for kk, vv in v.items()}
                      for k, v in per_class.items()},
        "temperature_scaling": {
            "T": ts["T_mean"],
            "T_std": ts["T_std"],
            "ece_before": ts["ece_before"],
            "ece_after": ts["ece_after"],
            "improvement_pct": round(ts["improvement"] / ts["ece_before"] * 100, 1),
            "recommended": ts["recommended"],
        },
    }

    with open("results/alerce_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: results/alerce_results.json")

    print(f"\nFigures saved:")
    print(f"  figures/fig1_alerce_reliability.pdf")
    print(f"  figures/fig2_alerce_perclass.pdf")
    print(f"  figures/fig3_alerce_tempscaling.pdf")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
