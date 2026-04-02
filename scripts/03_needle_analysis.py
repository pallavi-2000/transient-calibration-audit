"""
NEEDLE Calibration Analysis
==============================

Analysis of NEEDLE's 3-class hybrid CNN+DNN classifier
(Sheng et al. 2024, MNRAS 531, 2474).

Key findings:
  - Aggregate ECE = 0.050 (appears well-calibrated)
  - But per-class analysis reveals class-asymmetric miscalibration:
    - SLSN-I: overconfident (conf ~95%, acc ~80%)
    - TDE: underconfident (conf ~86%, acc ~100%)
  - Global temperature scaling WORSENS ECE (0.05 -> 0.13)
  - Root cause: inverse-frequency class weighting (~80:1 ratio)
  - This is itself a publishable finding about structural miscalibration

Usage:
    python scripts/03_needle_analysis.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import (
    compute_ece, compute_classwise_ece, brier_score,
    fit_temperature_cv, apply_temperature, bootstrap_ece,
    fit_per_class_temperature, apply_per_class_temperature,
    auto_calibrate
)
from src.plotting import (
    reliability_diagram, reliability_diagram_comparison,
    per_class_reliability_grid
)


def main():
    print("=" * 60)
    print("NEEDLE CALIBRATION ANALYSIS")
    print("Classifier: Hybrid CNN+DNN (Sheng et al. 2024)")
    print("Classes: SN, SLSN-I, TDE")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load predictions
    data = np.load("data/processed/needle_predictions.npz", allow_pickle=True)
    probs = data["probs"]
    labels = data["labels"]
    ztf_ids = data["ztf_ids"]
    model_ids = data["model_ids"]
    class_names = list(data["class_names"])

    print(f"\nTotal predictions: {len(probs)} (across {len(np.unique(model_ids))} models)")
    print(f"Unique objects: {len(set(ztf_ids))}")
    print(f"Classes: {class_names}")

    for k, name in enumerate(class_names):
        n = (labels == k).sum()
        print(f"  {name}: {n} predictions")

    # ---- Overall statistics ----
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == labels)

    print(f"\nOverall accuracy: {accuracy:.3f}")
    print(f"Mean confidence: {confidences.mean():.3f}")
    print(f"Accuracy - Confidence gap: {accuracy - confidences.mean():+.3f}")

    # ---- Aggregate ECE ----
    print(f"\n--- Aggregate ECE ---")
    boot = bootstrap_ece(labels, probs, n_bins=15)
    print(f"ECE: {boot['ece']:.3f} [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")

    brier = brier_score(labels, probs)
    print(f"Brier score: {brier['brier_score']:.3f}")

    # ---- Figure: Aggregate reliability diagram ----
    ece_em, bins_em = compute_ece(labels, probs, n_bins=15, strategy="equal_mass")
    reliability_diagram(
        bins_em,
        title="NEEDLE Classifier (Aggregate)",
        save_path="figures/fig_needle_reliability.pdf",
        color="#009E73"
    )
    reliability_diagram(
        bins_em,
        title="NEEDLE Classifier (Aggregate)",
        save_path="figures/fig_needle_reliability.png",
        color="#009E73"
    )

    # ---- Per-class ECE — THIS IS THE KEY FINDING ----
    print(f"\n--- Per-class ECE (reveals asymmetry) ---")
    per_class = compute_classwise_ece(labels, probs, n_bins=15,
                                      class_names=class_names)

    for cls, stats in per_class.items():
        direction = "overconfident" if stats["gap"] < 0 else "underconfident"
        print(f"  {cls:7s}: ECE={stats['ece']:.3f}  "
              f"acc={stats['accuracy']:.3f}  "
              f"conf={stats['mean_confidence']:.3f}  "
              f"gap={stats['gap']:+.3f} ({direction})")

    # ---- Figure: Per-class reliability diagrams ----
    class_bin_data = {}
    for k, cls_name in enumerate(class_names):
        is_k = (labels == k).astype(float)
        probs_k = probs[:, k]

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
        title="NEEDLE Per-Class Calibration",
        save_path="figures/fig_needle_perclass.pdf"
    )
    per_class_reliability_grid(
        per_class, class_bin_data,
        title="NEEDLE Per-Class Calibration",
        save_path="figures/fig_needle_perclass.png"
    )

    # ---- Global temperature scaling (should FAIL) ----
    print(f"\n--- Global Temperature Scaling ---")
    ts = fit_temperature_cv(labels, probs, n_folds=5)
    print(f"T = {ts['T_mean']:.3f} +/- {ts['T_std']:.3f}")
    print(f"ECE before: {ts['ece_before']:.3f}")
    print(f"ECE after:  {ts['ece_after']:.3f}")
    print(f"Recommended: {ts['recommended']} — {ts['reason']}")

    # ---- Per-class temperature scaling ----
    print(f"\n--- Per-class Temperature Scaling ---")
    class_Ts = fit_per_class_temperature(labels, probs, class_names=class_names)

    print("Per-class temperatures:")
    for cls, T in class_Ts.items():
        direction = "soften (overconfident)" if T > 1 else "sharpen (underconfident)"
        print(f"  {cls:7s}: T={T:.3f} — {direction}")

    probs_perclass = apply_per_class_temperature(probs, class_Ts, class_names)
    ece_perclass, _ = compute_ece(labels, probs_perclass, n_bins=15)
    print(f"ECE after per-class T: {ece_perclass:.3f}")

    # Compare all three
    print(f"\n--- Comparison ---")
    print(f"{'Method':<25} {'ECE':<10}")
    print(f"{'Raw':25s} {boot['ece']:.3f}")
    print(f"{'Global T={:.3f}'.format(ts['T_mean']):25s} {ts['ece_after']:.3f}")
    print(f"{'Per-class T':25s} {ece_perclass:.3f}")

    # ---- Silent failures ----
    print(f"\n--- Silent Failures (confident AND wrong) ---")
    for threshold in [0.9, 0.8, 0.7]:
        confident = confidences >= threshold
        wrong = predictions != labels
        silent = confident & wrong
        print(f"  Threshold >= {threshold:.1f}: "
              f"{silent.sum()} silent failures / {confident.sum()} confident predictions")

        if silent.sum() > 0:
            for idx in np.where(silent)[0]:
                pred_cls = class_names[predictions[idx]]
                true_cls = class_names[labels[idx]]
                conf = confidences[idx]
                print(f"    {ztf_ids[idx]}: predicted {pred_cls} ({conf:.3f}), actual {true_cls}")

    # ---- Save results ----
    results = {
        "classifier": "NEEDLE (Sheng et al. 2024)",
        "architecture": "Hybrid CNN + DNN",
        "n_predictions": int(len(probs)),
        "n_unique_objects": int(len(set(ztf_ids))),
        "n_models": int(len(np.unique(model_ids))),
        "class_names": class_names,
        "class_counts": {class_names[k]: int((labels == k).sum()) for k in range(len(class_names))},
        "accuracy": float(accuracy),
        "mean_confidence": float(confidences.mean()),
        "ece": boot["ece"],
        "ece_ci": [boot["ci_lower"], boot["ci_upper"]],
        "brier_score": brier["brier_score"],
        "per_class": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                          for kk, vv in v.items()}
                      for k, v in per_class.items()},
        "global_temperature": {
            "T": ts["T_mean"],
            "ece_before": ts["ece_before"],
            "ece_after": ts["ece_after"],
            "recommended": ts["recommended"],
            "reason": ts["reason"],
        },
        "per_class_temperature": {
            "class_Ts": {k: round(v, 3) for k, v in class_Ts.items()},
            "ece_after": float(ece_perclass),
        },
        "key_finding": (
            "Aggregate ECE (0.050) masks class-asymmetric miscalibration. "
            "SLSN-I is overconfident (driven by ~123x inverse-frequency weight), "
            "TDE is underconfident. Global temperature scaling worsens ECE "
            "because it cannot simultaneously soften and sharpen different classes."
        ),
    }

    with open("results/needle_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: results/needle_results.json")
    print(f"\nFigures saved:")
    print(f"  figures/fig_needle_reliability.pdf")
    print(f"  figures/fig_needle_perclass.pdf")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
