"""
Bin Sensitivity Analysis (Revision Step 1.5)
=============================================

Reviewer concern: "Why 15 bins? ECE is sensitive to bin count."

This script computes ECE across bin counts [5, 10, 15, 20] for all four
classifiers, quantifying how much the choice of 15 bins matters.

Usage:
  python scripts/12_bin_sensitivity_analysis.py

Outputs:
  results/bin_sensitivity_results.json
  figures/fig_bin_sensitivity_analysis.pdf
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import compute_ece

BIN_COUNTS = [5, 10, 15, 20]

# ── Data loaders ─────────────────────────────────────────────────────────────

def load_alerce():
    pred = pd.read_csv("data/raw/alerce_classifications.csv")
    truth = pd.read_csv("data/ground_truth/bts_sample.csv")
    merged = pred.merge(truth[["ZTFID","alerce_class"]], left_on="oid", right_on="ZTFID")
    merged = merged[merged["alerce_class"] != "TDE"]
    cols = ["SNIa","SNIbc","SNII","SLSN"]
    probs = merged[cols].values.astype(float)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = np.array([cols.index(c) for c in merged["alerce_class"]])
    return probs, labels, "ALeRCE (4-class)", f"N={len(labels)}"


def load_fink_rf():
    pred = pd.read_csv("data/raw/fink_classifications.csv")
    truth = pd.read_csv("data/ground_truth/bts_sample.csv")
    merged = pred.merge(truth[["ZTFID","alerce_class"]], left_on="oid", right_on="ZTFID")
    scores = merged["rf_snia_vs_nonia"].values.astype(float)
    labels = (merged["alerce_class"] == "SNIa").astype(int).values
    mask = scores > 0
    return scores[mask], labels[mask], "Fink RF (binary, non-zero)", f"N={mask.sum()}"


def load_fink_snn():
    pred = pd.read_csv("data/raw/fink_classifications.csv")
    truth = pd.read_csv("data/ground_truth/bts_sample.csv")
    merged = pred.merge(truth[["ZTFID","alerce_class"]], left_on="oid", right_on="ZTFID")
    scores = merged["snn_snia_vs_nonia"].values.astype(float)
    labels = (merged["alerce_class"] == "SNIa").astype(int).values
    mask = scores > 0
    return scores[mask], labels[mask], "Fink SNN (binary, non-zero)", f"N={mask.sum()}"


def load_needle():
    path = "data/processed/needle_predictions.npz"
    data = np.load(path, allow_pickle=True)
    probs  = data["probs"].astype(np.float64)
    labels = data["labels"].astype(int)
    ztf_ids = data["ztf_ids"].astype(str)
    class_names = data["class_names"].astype(str)

    # Deduplicate to object level
    unique_ids = np.unique(ztf_ids)
    K = probs.shape[1]
    dedup_probs  = np.zeros((len(unique_ids), K))
    dedup_labels = np.zeros(len(unique_ids), dtype=int)
    for i, uid in enumerate(unique_ids):
        mask = ztf_ids == uid
        dedup_probs[i]  = probs[mask].mean(axis=0)
        dedup_labels[i] = labels[mask][0]
    dedup_probs /= dedup_probs.sum(axis=1, keepdims=True)
    return dedup_probs, dedup_labels, "NEEDLE (3-class, object-level)", f"N={len(dedup_labels)}"


# ── Core analysis ─────────────────────────────────────────────────────────────

def bin_sensitivity(data, labels, name, size_label, is_binary=False):
    ece_values = []
    for n_bins in BIN_COUNTS:
        if is_binary:
            # 1D path: correct for binary classifiers
            ece, _ = compute_ece(labels, data, n_bins=n_bins, strategy="equal_mass")
        else:
            # 2D path: required for multi-class (max_conf can be < 0.5)
            ece, _ = compute_ece(labels, data, n_bins=n_bins, strategy="equal_mass")
        ece_values.append(float(ece))

    ece_arr = np.array(ece_values)
    mean_ece = float(ece_arr.mean())
    std_ece  = float(ece_arr.std())
    rng      = float(ece_arr.max() - ece_arr.min())
    rel_pct  = 100 * rng / mean_ece if mean_ece > 0 else 0.0

    if rel_pct < 2:
        stability = "Excellent (< 2%)"
    elif rel_pct < 5:
        stability = "Good (2–5%)"
    elif rel_pct < 10:
        stability = "Acceptable (5–10%)"
    else:
        stability = "Poor (> 10%)"

    print(f"\n{name} [{size_label}]:")
    print(f"  Bin counts : {BIN_COUNTS}")
    print(f"  ECE values : {[f'{v:.4f}' for v in ece_values]}")
    print(f"  Mean: {mean_ece:.4f}  Std: {std_ece:.4f}")
    print(f"  Range: {rng:.4f} ({ece_arr.min():.4f} – {ece_arr.max():.4f}), {rel_pct:.1f}% relative")
    print(f"  Stability: {stability}")

    return {
        "name": name,
        "n": size_label,
        "bin_counts": BIN_COUNTS,
        "ece_values": ece_values,
        "mean": mean_ece,
        "std": std_ece,
        "range": rng,
        "relative_change_pct": rel_pct,
        "stability": stability,
    }


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(results, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    for ax, res, col in zip(axes, results, colors):
        eces = res["ece_values"]
        ax.plot(BIN_COUNTS, eces, "o-", color=col, linewidth=2, markersize=7)
        ax.axvline(15, color="gray", linestyle="--", alpha=0.6, label="15 bins (paper)")

        # Shade ±5% band around 15-bin value
        ece_15 = eces[BIN_COUNTS.index(15)]
        ax.axhspan(ece_15 * 0.95, ece_15 * 1.05, alpha=0.10, color=col, label="±5% band")

        for x, y in zip(BIN_COUNTS, eces):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)

        ax.set_xlabel("Number of bins")
        ax.set_ylabel("ECE")
        ax.set_title(f"{res['name']}\n[{res['n']}, {res['relative_change_pct']:.1f}% range]",
                     fontsize=9)
        ax.set_xticks(BIN_COUNTS)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        y_lo = min(eces) * 0.85
        y_hi = max(eces) * 1.15
        ax.set_ylim(y_lo, y_hi)

    plt.suptitle("ECE Bin-Count Sensitivity (equal-mass binning)", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nFigure saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("STEP 1.5: BIN SENSITIVITY ANALYSIS")
    print("How sensitive is ECE to bin count (5, 10, 15, 20)?")
    print("=" * 80)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("\nLoading classifiers...")

    alerce_probs, alerce_labels, alerce_name, alerce_n = load_alerce()
    print(f"  ✓ ALeRCE (4-class)")

    rf_scores, rf_labels, rf_name, rf_n = load_fink_rf()
    print(f"  ✓ Fink RF (binary, non-zero)")

    snn_scores, snn_labels, snn_name, snn_n = load_fink_snn()
    print(f"  ✓ Fink SNN (binary)")

    needle_probs, needle_labels, needle_name, needle_n = load_needle()
    print(f"  ✓ NEEDLE (object-level)")

    print("\n" + "=" * 80)
    print("BIN SENSITIVITY ANALYSIS")
    print("=" * 80)

    results = [
        bin_sensitivity(alerce_probs,  alerce_labels,  alerce_name,  alerce_n,  is_binary=False),
        bin_sensitivity(rf_scores,     rf_labels,      rf_name,      rf_n,      is_binary=True),
        bin_sensitivity(snn_scores,    snn_labels,     snn_name,     snn_n,     is_binary=True),
        bin_sensitivity(needle_probs,  needle_labels,  needle_name,  needle_n,  is_binary=False),
    ]

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    all_pass = all(r["relative_change_pct"] < 10 for r in results)
    for r in results:
        flag = "✓" if r["relative_change_pct"] < 10 else "⚠"
        print(f"  {flag} {r['name']}: {r['relative_change_pct']:.1f}% — {r['stability']}")

    print(f"\n{'PASSES' if all_pass else 'CONDITIONAL'}: "
          f"{'All classifiers stable (<10%)' if all_pass else 'One or more classifiers show >10% sensitivity'}")

    make_figure(results, "figures/fig_bin_sensitivity_analysis.pdf")

    output = {
        "metadata": {
            "description": "ECE bin-count sensitivity analysis (Step 1.5)",
            "bin_counts_tested": BIN_COUNTS,
            "paper_bin_count": 15,
            "stability_thresholds": {
                "excellent": "< 2%",
                "good": "2–5%",
                "acceptable": "5–10%",
                "poor": "> 10%",
            },
        },
        "results": results,
        "overall_passes": all_pass,
    }

    with open("results/bin_sensitivity_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults: results/bin_sensitivity_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
