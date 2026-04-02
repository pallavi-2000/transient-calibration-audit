"""
Fink Calibration Analysis
===========================

Analysis of Fink's two binary classifiers:
  - Random Forest (rf_snia_vs_nonia): structurally broken, ECE ~0.41
  - SuperNNova RNN (snn_snia_vs_nonia): miscalibrated, ECE ~0.20

Key findings:
  - RF is 94% zeros — degenerate, not suitable for probability calibration
  - SNN has discriminative power but temperature scaling hits optimizer bound
  - Both classifiers ask a binary question: "Is this SN Ia?"

Usage:
    python scripts/02_fink_analysis.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import (
    compute_ece, bootstrap_ece, brier_score,
    fit_temperature_cv, apply_temperature, auto_calibrate
)
from src.plotting import reliability_diagram, reliability_diagram_comparison


def analyze_binary_classifier(scores, true_labels, name, color):
    """Analyze a single binary classifier."""
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")

    n_total = len(scores)
    n_positive = true_labels.sum()
    n_zero = (scores == 0).sum()

    print(f"Objects: {n_total}")
    print(f"SN Ia (positive): {n_positive} ({n_positive/n_total*100:.1f}%)")
    print(f"Scores at exactly 0.0: {n_zero} ({n_zero/n_total*100:.0f}%)")
    print(f"Mean score: {scores.mean():.3f}")

    # ECE
    boot = bootstrap_ece(true_labels, scores, n_bins=15)
    print(f"ECE: {boot['ece']:.3f} [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")

    ece_ew, bins_ew = compute_ece(true_labels, scores, n_bins=15, strategy="equal_width")
    print(f"ECE (equal-width): {ece_ew:.3f}")

    brier = brier_score(true_labels, scores)
    print(f"Brier: {brier['brier_score']:.3f}")

    # Auto-calibrate
    result = auto_calibrate(true_labels, scores)
    print(f"Recommendation: {result['recommendation']}")
    if "reason" in result:
        print(f"Reason: {result.get('reason', result.get('summary', ''))}")
    if "summary" in result:
        print(f"Summary: {result['summary']}")

    # Reliability diagram
    ece_em, bins_em = compute_ece(true_labels, scores, n_bins=15, strategy="equal_mass")

    reliability_diagram(
        bins_em,
        title=f"Fink {name}",
        save_path=f"figures/fig_fink_{name.lower().replace(' ', '_')}.pdf",
        color=color,
    )
    reliability_diagram(
        bins_em,
        title=f"Fink {name}",
        save_path=f"figures/fig_fink_{name.lower().replace(' ', '_')}.png",
        color=color,
    )

    return {
        "name": name,
        "n_objects": int(n_total),
        "n_positive": int(n_positive),
        "prevalence": float(n_positive / n_total),
        "n_zeros": int(n_zero),
        "zero_fraction": float(n_zero / n_total),
        "mean_score": float(scores.mean()),
        "ece": boot["ece"],
        "ece_ci": [boot["ci_lower"], boot["ci_upper"]],
        "ece_equal_width": float(ece_ew),
        "brier_score": brier["brier_score"],
        "recommendation": result["recommendation"],
        "degenerate": result.get("degenerate", False),
    }


def main():
    print("=" * 60)
    print("FINK CALIBRATION ANALYSIS")
    print("Classifiers: RF (rf_snia_vs_nonia), SNN (snn_snia_vs_nonia)")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load data
    fink = pd.read_csv("data/raw/fink_classifications.csv")
    sample = pd.read_csv("data/ground_truth/bts_sample.csv")

    merged = fink.merge(
        sample[["ZTFID", "alerce_class"]],
        left_on="oid", right_on="ZTFID"
    )

    # Binary ground truth: SN Ia vs everything else
    is_snia = (merged["alerce_class"] == "SNIa").astype(int).values

    print(f"\nTotal objects: {len(merged)}")
    print(f"SN Ia: {is_snia.sum()} ({is_snia.mean()*100:.1f}%)")
    print(f"Not SN Ia: {len(merged) - is_snia.sum()}")

    # Analyze RF
    rf_scores = pd.to_numeric(merged["rf_snia_vs_nonia"], errors="coerce").values
    valid_rf = ~np.isnan(rf_scores)
    rf_result = analyze_binary_classifier(
        rf_scores[valid_rf], is_snia[valid_rf],
        "Random Forest", "#D55E00"
    )

    # Analyze SNN
    snn_scores = pd.to_numeric(merged["snn_snia_vs_nonia"], errors="coerce").values
    valid_snn = ~np.isnan(snn_scores)
    snn_result = analyze_binary_classifier(
        snn_scores[valid_snn], is_snia[valid_snn],
        "SuperNNova", "#0072B2"
    )

    # ---- Precision-Recall analysis for RF (since probabilities are useless) ----
    print(f"\n{'='*50}")
    print("RF Precision-Recall Analysis (since RF is degenerate)")
    print(f"{'='*50}")

    rf_clean = rf_scores[valid_rf]
    is_snia_rf = is_snia[valid_rf]

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"{'Threshold':>10} {'Predicted+':>12} {'True+':>8} {'Precision':>10} {'Recall':>8}")
    for t in thresholds:
        predicted_pos = rf_clean >= t
        n_predicted = predicted_pos.sum()
        if n_predicted > 0:
            true_pos = (predicted_pos & (is_snia_rf == 1)).sum()
            precision = true_pos / n_predicted
            recall = true_pos / is_snia_rf.sum()
            print(f"{t:>10.1f} {n_predicted:>12} {true_pos:>8} {precision:>10.3f} {recall:>8.3f}")
        else:
            print(f"{t:>10.1f} {0:>12} {'N/A':>8} {'N/A':>10} {'N/A':>8}")

    # Save results
    results = {
        "classifier_type": "binary (SN Ia vs not SN Ia)",
        "random_forest": rf_result,
        "supernnova": snn_result,
        "notes": {
            "rf_degenerate": "94% of RF scores are exactly 0.0. Not suitable for probability calibration.",
            "snn_bound_hit": "Temperature scaling optimizer hits upper bound (T=10), suggesting structural issues.",
            "fink_api_change": "API URL changed Jan 2025: fink-portal.org -> api.fink-portal.org",
        }
    }

    with open("results/fink_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: results/fink_results.json")
    print(f"\nFigures saved:")
    print(f"  figures/fig_fink_random_forest.pdf")
    print(f"  figures/fig_fink_supernnova.pdf")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
