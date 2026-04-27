"""
ALeRCE Dual-Task Analysis (Revision Step 1.2)
==============================================

Reviewer concern: "Did you artificially restrict ALeRCE to 4 classes?
What is the native output?"

This script validates that restricting ALeRCE's 15-class native output to
the 4 transient classes (SNIa, SNIbc, SNII, SLSN) does not inflate/deflate
ECE relative to our paper baseline of 0.271.

What it does:
  1. Load raw ALeRCE 15-class predictions + BTS ground truth
  2. Compute ECE on native 15-class output (all 15 probability columns)
  3. Restrict to 4 transient columns and renormalize
  4. Recompute ECE on 4-class restricted (should ≈ 0.271)
  5. Compare: restriction does not introduce calibration bias

Actual CSV columns (lc_classifier v1.1.13):
  SNIa, SNIbc, SNII, SLSN, QSO, AGN, Blazar, CV/Nova,
  YSO, LPV, E, DSCT, RRL, CEP, Periodic-Other

Usage:
  python scripts/09_alerce_dual_task.py

Outputs:
  results/alerce_dual_task_results.json
  figures/fig_alerce_15class_vs_4class.pdf
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import compute_ece, bootstrap_ece
from src.plotting import reliability_diagram


# Actual ALeRCE lc_classifier v1.1.13 column order
ALERCE_15_COLS = [
    "SNIa", "SNIbc", "SNII", "SLSN",
    "QSO", "AGN", "Blazar", "CV/Nova",
    "YSO", "LPV", "E", "DSCT", "RRL", "CEP", "Periodic-Other",
]

# 4 transient classes used in our paper
TRANSIENT_4_COLS = ["SNIa", "SNIbc", "SNII", "SLSN"]

# Ground truth class → 15-class column index
ALERCE_CLASS_TO_15 = {cname: i for i, cname in enumerate(ALERCE_15_COLS)}

# Ground truth class → 4-class index
ALERCE_CLASS_TO_4 = {cname: i for i, cname in enumerate(TRANSIENT_4_COLS)}


def load_data(
    pred_path="data/raw/alerce_classifications.csv",
    truth_path="data/ground_truth/bts_sample.csv",
):
    """Load ALeRCE predictions and merge with BTS ground truth."""
    for p in [pred_path, "/Users/pallavisati/Desktop/transient-calibration-audit/" + pred_path]:
        if os.path.exists(p):
            pred_path = p
            break
    else:
        raise FileNotFoundError(f"ALeRCE CSV not found: {pred_path}")

    for p in [truth_path, "/Users/pallavisati/Desktop/transient-calibration-audit/" + truth_path]:
        if os.path.exists(p):
            truth_path = p
            break
    else:
        raise FileNotFoundError(f"BTS sample not found: {truth_path}")

    print(f"Loading predictions: {pred_path}")
    print(f"Loading ground truth: {truth_path}")

    preds = pd.read_csv(pred_path)
    truth = pd.read_csv(truth_path)

    merged = preds.merge(
        truth[["ZTFID", "alerce_class"]],
        left_on="oid", right_on="ZTFID",
        how="inner",
    )

    # Exclude TDE (ALeRCE has no TDE class)
    merged = merged[merged["alerce_class"] != "TDE"].copy()
    merged = merged[merged["alerce_class"].isin(TRANSIENT_4_COLS)].copy()

    N = len(merged)
    print(f"\nMerged: {N} objects (TDEs excluded)")
    print(f"Class distribution:")
    for cls in TRANSIENT_4_COLS:
        n = (merged["alerce_class"] == cls).sum()
        print(f"  {cls:8s}: {n:4d} ({100*n/N:.1f}%)")

    # Extract 15-class probabilities
    probs_15 = merged[ALERCE_15_COLS].values.astype(np.float64)

    # Map true classes to 15-class and 4-class indices
    labels_15 = np.array([ALERCE_CLASS_TO_15[c] for c in merged["alerce_class"]])
    labels_4  = np.array([ALERCE_CLASS_TO_4[c]  for c in merged["alerce_class"]])

    ztf_ids = merged["oid"].values.astype(str)

    return {
        "probs_15": probs_15,
        "labels_15": labels_15,
        "labels_4": labels_4,
        "ztf_ids": ztf_ids,
        "n": N,
    }


def make_4class_probs(probs_15):
    """Restrict to 4 transient columns and renormalize."""
    # Columns 0-3 are SNIa, SNIbc, SNII, SLSN
    probs_4 = probs_15[:, :4].copy()
    row_sums = probs_4.sum(axis=1, keepdims=True)
    probs_4 = probs_4 / row_sums
    return probs_4


def compute_calibration(probs, labels, class_names, label=""):
    """Compute aggregate ECE, per-class ECE, Brier, silent failures."""
    pred_class = probs.argmax(axis=1)
    pred_confidence = probs.max(axis=1)
    correct = (pred_class == labels).astype(int)

    # Use 2D (labels, probs) path — required for multi-class classifiers where
    # max confidence can be < 0.5 (e.g. ALeRCE BRF with 4 classes).
    # The 1D (correct, pred_conf) path inverts confidence for objects with
    # max_conf < 0.5, producing severely wrong ECE.
    boot = bootstrap_ece(labels, probs, n_bins=15)
    ece_agg = boot["ece"]
    ece_ci = [boot["ci_lower"], boot["ci_upper"]]

    N, K = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1
    brier = ((probs - one_hot) ** 2).sum(axis=1).mean()

    per_class = {}
    for k, cname in enumerate(class_names):
        class_prob = probs[:, k]
        class_true = (labels == k).astype(int)
        ece_k, _ = compute_ece(class_true, class_prob, n_bins=15, strategy="equal_mass")
        mask_pred_k = (pred_class == k)
        if mask_pred_k.sum() > 0:
            acc = (labels[mask_pred_k] == k).mean()
            conf = probs[mask_pred_k, k].mean()
        else:
            acc = conf = None
        per_class[cname] = {
            "ece": float(ece_k),
            "accuracy_when_predicted": float(acc) if acc is not None else None,
            "mean_confidence_when_predicted": float(conf) if conf is not None else None,
            "gap": float(acc - conf) if acc is not None else None,
            "n_true": int((labels == k).sum()),
            "n_predicted": int(mask_pred_k.sum()),
        }

    silent_mask = (pred_confidence >= 0.90) & (correct == 0)
    n_silent = int(silent_mask.sum())

    print(f"\n--- {label} ---")
    print(f"N: {N} | K classes: {K} | Accuracy: {correct.mean():.3f}")
    print(f"ECE: {ece_agg:.3f} [{ece_ci[0]:.3f}, {ece_ci[1]:.3f}]")
    print(f"Brier: {brier:.3f}")
    print(f"Silent failures (≥90% conf, wrong): {n_silent} ({100*n_silent/N:.1f}%)")
    print(f"Per-class:")
    for cname, m in per_class.items():
        if m["gap"] is not None:
            direction = "underconfident" if m["gap"] > 0 else "overconfident"
            print(f"  {cname:14s}: n_true={m['n_true']:4d}, n_pred={m['n_predicted']:4d}, "
                  f"ECE={m['ece']:.3f}, acc={m['accuracy_when_predicted']:.3f}, "
                  f"conf={m['mean_confidence_when_predicted']:.3f}, "
                  f"gap={m['gap']:+.3f} ({direction})")
        else:
            print(f"  {cname:14s}: n_true={m['n_true']:4d}, n_pred=0 (never predicted)")

    return {
        "label": label,
        "n_objects": N,
        "n_classes": K,
        "overall_accuracy": float(correct.mean()),
        "ece_aggregate": float(ece_agg),
        "ece_ci": [float(x) for x in ece_ci],
        "brier_score": float(brier),
        "n_silent_failures": n_silent,
        "per_class": per_class,
    }


def make_comparison_figure(metrics_15, metrics_4, save_path):
    """Side-by-side: 15-class vs 4-class ECE comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Aggregate ECE
    ax = axes[0]
    labels = ["15-class\n(native)", "4-class\n(restricted)"]
    eces = [metrics_15["ece_aggregate"], metrics_4["ece_aggregate"]]
    ci_lows = [metrics_15["ece_ci"][0], metrics_4["ece_ci"][0]]
    ci_highs = [metrics_15["ece_ci"][1], metrics_4["ece_ci"][1]]
    errors = [
        [e - l for e, l in zip(eces, ci_lows)],
        [h - e for e, h in zip(eces, ci_highs)],
    ]

    bars = ax.bar(labels, eces, yerr=errors, capsize=8,
                  color=["#888888", "#0072B2"], edgecolor="black")
    for bar, ece in zip(bars, eces):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{ece:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(0.271, color="red", linestyle="--", alpha=0.7, label="Paper baseline (0.271)")
    ax.axhline(0.05, color="green", linestyle=":", alpha=0.6, label="Well-calibrated (< 0.05)")
    ax.axhline(0.10, color="orange", linestyle="--", alpha=0.5, label="Acceptable (< 0.10)")
    ax.set_ylabel("Aggregate ECE")
    ax.set_title("ALeRCE: 15-Class vs 4-Class Restriction")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(eces) * 1.3, 0.40))
    ax.grid(alpha=0.3, axis="y")

    # Per-class ECE for the 4-class restricted analysis
    ax = axes[1]
    classes = list(metrics_4["per_class"].keys())
    x = np.arange(len(classes))
    ece_vals = [metrics_4["per_class"][c]["ece"] for c in classes]
    acc_vals = [metrics_4["per_class"][c]["accuracy_when_predicted"] or 0 for c in classes]
    conf_vals = [metrics_4["per_class"][c]["mean_confidence_when_predicted"] or 0 for c in classes]

    width = 0.25
    ax.bar(x - width, ece_vals, width, label="Per-class ECE", color="#0072B2", edgecolor="black")
    ax.bar(x,         acc_vals,  width, label="Accuracy (pred)", color="#009E73", edgecolor="black")
    ax.bar(x + width, conf_vals, width, label="Mean Confidence (pred)", color="#D55E00", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Value")
    ax.set_title("ALeRCE 4-Class: Per-Class Breakdown")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nComparison figure saved: {save_path}")


def main():
    print("=" * 80)
    print("ALERCE DUAL-TASK ANALYSIS")
    print("Reviewer concern: Is 4-class restriction fair?")
    print("=" * 80)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    data = load_data()

    print("\n" + "=" * 80)
    print("1. 15-CLASS NATIVE (full ALeRCE output)")
    print("=" * 80)
    metrics_15 = compute_calibration(
        data["probs_15"],
        data["labels_15"],
        ALERCE_15_COLS,
        label="15-class (native)",
    )

    print("\n" + "=" * 80)
    print("2. 4-CLASS RESTRICTED (SNIa, SNIbc, SNII, SLSN — renormalized)")
    print("=" * 80)
    probs_4 = make_4class_probs(data["probs_15"])
    metrics_4 = compute_calibration(
        probs_4,
        data["labels_4"],
        TRANSIENT_4_COLS,
        label="4-class (restricted)",
    )

    print("\n" + "=" * 80)
    print("3. COMPARISON: TASK COMPLEXITY")
    print("=" * 80)
    diff = metrics_4["ece_aggregate"] - metrics_15["ece_aggregate"]
    print(f"\n15-class ECE: {metrics_15['ece_aggregate']:.3f} [{metrics_15['ece_ci'][0]:.3f}, {metrics_15['ece_ci'][1]:.3f}]")
    print(f"4-class ECE:  {metrics_4['ece_aggregate']:.3f} [{metrics_4['ece_ci'][0]:.3f}, {metrics_4['ece_ci'][1]:.3f}]")
    print(f"Difference:   {diff:+.3f} (4-class − 15-class)")
    print(f"Paper baseline for 4-class: 0.271")

    passes = abs(metrics_4["ece_aggregate"] - 0.271) < 0.05
    if passes:
        print(f"\n✓ Validation PASSED: 4-class ECE ({metrics_4['ece_aggregate']:.3f}) ≈ baseline (0.271)")
        print(f"  Class restriction does NOT introduce significant calibration bias.")
    else:
        print(f"\n⚠ WARNING: 4-class ECE ({metrics_4['ece_aggregate']:.3f}) diverges from baseline (0.271)")
        print(f"  Investigate class-mapping or data mismatch.")

    # Reliability diagrams — use 2D path to handle max_conf < 0.5 correctly
    _, bins_4 = compute_ece(data["labels_4"], probs_4, n_bins=15, strategy="equal_mass")
    reliability_diagram(
        bins_4,
        title=f"ALeRCE 4-Class Restricted (N={data['n']})",
        save_path="figures/fig_alerce_4class_reliability.pdf",
        color="#0072B2",
    )

    _, bins_15 = compute_ece(data["labels_15"], data["probs_15"], n_bins=15, strategy="equal_mass")
    reliability_diagram(
        bins_15,
        title=f"ALeRCE 15-Class Native (N={data['n']})",
        save_path="figures/fig_alerce_15class_reliability.pdf",
        color="#888888",
    )

    make_comparison_figure(metrics_15, metrics_4, "figures/fig_alerce_15class_vs_4class.pdf")

    results = {
        "metadata": {
            "description": "ALeRCE dual-task analysis for paper revision Step 1.2",
            "question": "Does restricting 15-class to 4-class introduce calibration bias?",
            "methodology": (
                "15-class: native ALeRCE output on all 15 class columns. "
                "4-class: select SNIa/SNIbc/SNII/SLSN columns and renormalize."
            ),
        },
        "native_15class": metrics_15,
        "restricted_4class": metrics_4,
        "validation": {
            "baseline_4class_ece": 0.271,
            "observed_4class_ece": metrics_4["ece_aggregate"],
            "absolute_difference": abs(metrics_4["ece_aggregate"] - 0.271),
            "passes_validation": passes,
        },
    }

    with open("results/alerce_dual_task_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY FOR PAPER")
    print("=" * 80)
    print(f"\nALeRCE 15-class (native, harder problem):")
    print(f"  ECE: {metrics_15['ece_aggregate']:.3f} [{metrics_15['ece_ci'][0]:.3f}, {metrics_15['ece_ci'][1]:.3f}]")
    print(f"  Accuracy: {metrics_15['overall_accuracy']:.3f}")
    print(f"\nALeRCE 4-class (restricted, paper baseline):")
    print(f"  ECE: {metrics_4['ece_aggregate']:.3f} [{metrics_4['ece_ci'][0]:.3f}, {metrics_4['ece_ci'][1]:.3f}]")
    print(f"  Accuracy: {metrics_4['overall_accuracy']:.3f}")
    print(f"  Baseline match: {'✓ YES' if passes else '✗ NO'}")
    print(f"\nResults: results/alerce_dual_task_results.json")
    print(f"Figures: figures/")
    print("=" * 80)


if __name__ == "__main__":
    main()
