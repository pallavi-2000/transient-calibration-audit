"""
ALeRCE Operational Gain — Proper Cross-Validation (Revision Step 1.3)
======================================================================

Reviewer concern: "The 21× operational gain may reflect data leakage —
temperature T was fit and evaluated on the same 1114 objects."

This script validates the paper's claim using stratified 5-fold CV:
  - T is fit on the calibration set (4 folds)
  - Operational gain is measured on the held-out fold (1 fold)
  - Repeat 5 times, report mean ± std and pooled (all-folds-combined) results

Paper claim:
  Raw:    20 / 1114 objects pass p > 0.80 (1.8%)
  Scaled: 425 / 1114 objects pass p > 0.80 (38.1%) at 91% precision
  Gain:   21×

Expected CV result:
  Pooled gain close to 21× (no leakage expected because T is a single scalar
  that changes very little across folds; see T mean ± std in output)

Usage:
  python scripts/10_alerce_operational_gain_cv.py

Outputs:
  results/alerce_operational_gain_cv_results.json
  figures/fig_alerce_operational_gain_cv.pdf
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import (
    compute_ece, bootstrap_ece,
    fit_temperature, apply_temperature,
    _stratified_kfold,
)

TRANSIENT_4_COLS = ["SNIa", "SNIbc", "SNII", "SLSN"]
THRESHOLD = 0.80
PAPER_GAIN = 21.0
PAPER_PRECISION = 0.911
PAPER_BEFORE = 20
PAPER_AFTER = 425
N_FOLDS = 5


def load_data(
    pred_path="data/raw/alerce_classifications.csv",
    truth_path="data/ground_truth/bts_sample.csv",
):
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

    preds = pd.read_csv(pred_path)
    truth = pd.read_csv(truth_path)

    merged = preds.merge(
        truth[["ZTFID", "alerce_class"]],
        left_on="oid", right_on="ZTFID",
        how="inner",
    )
    merged = merged[merged["alerce_class"] != "TDE"].copy()
    merged = merged[merged["alerce_class"].isin(TRANSIENT_4_COLS)].copy()

    probs = merged[TRANSIENT_4_COLS].values.astype(np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)

    class_to_idx = {c: i for i, c in enumerate(TRANSIENT_4_COLS)}
    labels = np.array([class_to_idx[c] for c in merged["alerce_class"]])

    N = len(labels)
    print(f"Loaded {N} objects, {len(TRANSIENT_4_COLS)} classes")
    print(f"Class distribution:")
    for i, cls in enumerate(TRANSIENT_4_COLS):
        n = (labels == i).sum()
        print(f"  {cls:8s}: {n:4d} ({100*n/N:.1f}%)")

    return probs, labels


def operational_metrics(raw_probs, scaled_probs, labels, threshold=THRESHOLD):
    """Count high-confidence objects and compute precision before/after scaling."""
    raw_conf = raw_probs.max(axis=1)
    sca_conf = scaled_probs.max(axis=1)

    raw_pred = raw_probs.argmax(axis=1)
    sca_pred = scaled_probs.argmax(axis=1)

    raw_mask = raw_conf >= threshold
    sca_mask = sca_conf >= threshold

    n_before = int(raw_mask.sum())
    n_after  = int(sca_mask.sum())
    gain = n_after / max(n_before, 1)

    raw_correct = (raw_pred == labels)
    sca_correct = (sca_pred == labels)

    prec_before = float(raw_correct[raw_mask].mean()) if n_before > 0 else float("nan")
    prec_after  = float(sca_correct[sca_mask].mean()) if n_after  > 0 else float("nan")

    return {
        "n_before": n_before,
        "n_after":  n_after,
        "gain":     float(gain),
        "precision_before": prec_before,
        "precision_after":  prec_after,
        "n_total": len(labels),
    }


def run_cv(probs, labels):
    folds = _stratified_kfold(labels, n_folds=N_FOLDS, random_state=42)

    fold_results = []
    all_raw_probs_test   = np.zeros_like(probs)
    all_scaled_probs_test = np.zeros_like(probs)

    print(f"\n{'='*80}")
    print(f"STRATIFIED {N_FOLDS}-FOLD CROSS-VALIDATION")
    print(f"{'='*80}")

    for fold_i, (cal_idx, test_idx) in enumerate(folds):
        probs_cal  = probs[cal_idx];  labels_cal  = labels[cal_idx]
        probs_test = probs[test_idx]; labels_test = labels[test_idx]

        # Fit T on calibration set
        T = fit_temperature(labels_cal, probs_cal)

        # Apply T to test set
        scaled_test = apply_temperature(probs_test, T)

        # ECE before and after on test set (2D path)
        ece_before, _ = compute_ece(labels_test, probs_test)
        ece_after,  _ = compute_ece(labels_test, scaled_test)

        # Operational metrics on test set
        ops = operational_metrics(probs_test, scaled_test, labels_test)

        fold_results.append({
            "fold": fold_i + 1,
            "n_train": len(cal_idx),
            "n_test":  len(test_idx),
            "T": float(T),
            "ece_before": float(ece_before),
            "ece_after":  float(ece_after),
            **ops,
        })

        # Accumulate for pooled analysis
        all_raw_probs_test[test_idx]    = probs_test
        all_scaled_probs_test[test_idx] = scaled_test

        print(f"\nFold {fold_i+1}/{N_FOLDS}:")
        print(f"  Train: {len(cal_idx):4d}  |  Test: {len(test_idx)}")
        print(f"  T = {T:.4f}")
        print(f"  ECE: {ece_before:.3f} → {ece_after:.3f}  "
              f"(improvement: {ece_before - ece_after:.3f})")
        print(f"  p>{THRESHOLD}: {ops['n_before']} → {ops['n_after']}  "
              f"(gain: {ops['gain']:.1f}×)")
        if not np.isnan(ops['precision_after']):
            print(f"  Precision after scaling: {ops['precision_after']:.3f}")
        else:
            print(f"  Precision after scaling: N/A (no objects above threshold)")

    # Pooled analysis (all held-out test sets, each object appears exactly once)
    pooled = operational_metrics(all_raw_probs_test, all_scaled_probs_test, labels)
    ece_before_pooled, _ = compute_ece(labels, all_raw_probs_test)
    ece_after_pooled,  _ = compute_ece(labels, all_scaled_probs_test)

    return fold_results, pooled, ece_before_pooled, ece_after_pooled


def print_summary(fold_results, pooled, ece_before_pooled, ece_after_pooled):
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATED RESULTS")
    print(f"{'='*80}")

    Ts    = [r["T"] for r in fold_results]
    gains = [r["gain"] for r in fold_results]
    precs = [r["precision_after"] for r in fold_results if not np.isnan(r["precision_after"])]
    eces_b = [r["ece_before"] for r in fold_results]
    eces_a = [r["ece_after"]  for r in fold_results]

    print(f"\nTemperature T:")
    print(f"  Mean: {np.mean(Ts):.4f} ± {np.std(Ts):.4f}")
    print(f"  Range: [{min(Ts):.4f}, {max(Ts):.4f}]")
    print(f"  Paper: T = 0.357")

    print(f"\nOperational Gain (p > {THRESHOLD}), per-fold:")
    print(f"  Mean: {np.mean(gains):.1f}× ± {np.std(gains):.1f}×")
    print(f"  Range: [{min(gains):.1f}×, {max(gains):.1f}×]")

    print(f"\nOperational Gain (p > {THRESHOLD}), POOLED (all held-out test sets):")
    print(f"  Before:    {pooled['n_before']:4d} / {pooled['n_total']} objects ({100*pooled['n_before']/pooled['n_total']:.1f}%)")
    print(f"  After:     {pooled['n_after']:4d} / {pooled['n_total']} objects ({100*pooled['n_after']/pooled['n_total']:.1f}%)")
    print(f"  Gain:      {pooled['gain']:.1f}×")
    print(f"  Paper:     {PAPER_BEFORE} → {PAPER_AFTER} ({PAPER_GAIN:.0f}×)")

    print(f"\nPrecision at p > {THRESHOLD}, per-fold:")
    print(f"  Mean: {np.mean(precs):.3f} ± {np.std(precs):.3f}")
    print(f"  Range: [{min(precs):.3f}, {max(precs):.3f}]")

    print(f"\nPrecision at p > {THRESHOLD}, POOLED:")
    print(f"  After scaling: {pooled['precision_after']:.3f}")
    print(f"  Paper: {PAPER_PRECISION:.3f}")

    print(f"\nECE Improvement, per-fold:")
    print(f"  Before: {np.mean(eces_b):.3f} ± {np.std(eces_b):.3f}")
    print(f"  After:  {np.mean(eces_a):.3f} ± {np.std(eces_a):.3f}")
    print(f"  Improvement: {np.mean(eces_b)-np.mean(eces_a):.3f}")

    print(f"\nECE Improvement, POOLED:")
    print(f"  Before: {ece_before_pooled:.3f}  (paper: 0.271)")
    print(f"  After:  {ece_after_pooled:.3f}  (paper: 0.097)")

    # Validation verdict
    print(f"\n{'='*80}")
    print(f"VALIDATION VERDICT")
    print(f"{'='*80}")
    pooled_gain = pooled["gain"]
    pooled_prec = pooled["precision_after"]

    if pooled_gain >= 15 and pooled_prec >= 0.85:
        verdict = "PASSES"
        detail = f"Pooled gain {pooled_gain:.1f}× (paper: {PAPER_GAIN:.0f}×), precision {pooled_prec:.3f} (paper: {PAPER_PRECISION:.3f})"
    elif pooled_gain >= 10:
        verdict = "CONDITIONAL"
        detail = f"Gain {pooled_gain:.1f}× lower than paper claim; still substantial"
    else:
        verdict = "FAILS"
        detail = f"Gain {pooled_gain:.1f}× collapses significantly from paper claim"

    print(f"\n{verdict}: {detail}")


def make_figure(fold_results, pooled, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    folds = [r["fold"] for r in fold_results]
    Ts    = [r["T"] for r in fold_results]
    gains = [r["gain"] for r in fold_results]
    precs = [r["precision_after"] for r in fold_results]
    eces_b = [r["ece_before"] for r in fold_results]
    eces_a = [r["ece_after"]  for r in fold_results]

    # Panel 1: Temperature per fold
    ax = axes[0]
    ax.bar(folds, Ts, color="#0072B2", edgecolor="black")
    ax.axhline(0.357, color="red", linestyle="--", label="Paper T=0.357")
    ax.axhline(np.mean(Ts), color="gray", linestyle=":", label=f"CV mean={np.mean(Ts):.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Temperature T")
    ax.set_title("Temperature per Fold")
    ax.legend(fontsize=8)
    ax.set_xticks(folds)
    ax.set_ylim(0, max(max(Ts) * 1.2, 0.5))
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: Operational gain per fold + pooled
    ax = axes[1]
    x = folds + [6]
    heights = gains + [pooled["gain"]]
    colors = ["#0072B2"] * 5 + ["#D55E00"]
    bars = ax.bar(x, heights, color=colors, edgecolor="black")
    ax.axhline(PAPER_GAIN, color="red", linestyle="--", label=f"Paper {PAPER_GAIN:.0f}×")
    ax.axhline(10, color="orange", linestyle=":", alpha=0.7, label="Min acceptable (10×)")
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"{h:.1f}×", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in folds] + ["Pooled"])
    ax.set_ylabel(f"Gain factor (p > {THRESHOLD})")
    ax.set_title("Operational Gain (Held-Out)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Panel 3: ECE before/after per fold
    ax = axes[2]
    xf = np.arange(len(folds))
    width = 0.35
    ax.bar(xf - width/2, eces_b, width, label="Before scaling", color="#888888", edgecolor="black")
    ax.bar(xf + width/2, eces_a, width, label="After scaling",  color="#0072B2",  edgecolor="black")
    ax.axhline(0.271, color="gray",   linestyle="--", alpha=0.5, label="Paper before (0.271)")
    ax.axhline(0.097, color="#0072B2", linestyle="--", alpha=0.5, label="Paper after (0.097)")
    ax.set_xticks(xf)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_ylabel("ECE")
    ax.set_title("ECE Before/After Temperature Scaling")
    ax.legend(fontsize=7)
    ax.set_ylim(0, max(max(eces_b) * 1.2, 0.35))
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nFigure saved: {save_path}")


def main():
    print("=" * 80)
    print("STEP 1.3: HELD-OUT OPERATIONAL GAIN (PROPER CV)")
    print("Reviewer concern: Is 21× gain real or data leakage?")
    print("=" * 80)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    probs, labels = load_data()

    # Full-dataset baseline (no CV — reproduces paper's claim)
    print(f"\n{'='*80}")
    print("FULL-DATASET BASELINE (no CV — what the paper computed)")
    print("=" * 80)
    T_full = fit_temperature(labels, probs)
    scaled_full = apply_temperature(probs, T_full)
    ops_full = operational_metrics(probs, scaled_full, labels)
    ece_full_before, _ = compute_ece(labels, probs)
    ece_full_after,  _ = compute_ece(labels, scaled_full)
    print(f"T = {T_full:.4f}")
    print(f"ECE: {ece_full_before:.3f} → {ece_full_after:.3f}")
    print(f"p>{THRESHOLD}: {ops_full['n_before']} → {ops_full['n_after']} ({ops_full['gain']:.1f}×)")
    print(f"Precision after scaling: {ops_full['precision_after']:.3f}")

    # Cross-validated results
    fold_results, pooled, ece_before_pooled, ece_after_pooled = run_cv(probs, labels)

    print_summary(fold_results, pooled, ece_before_pooled, ece_after_pooled)

    make_figure(fold_results, pooled, "figures/fig_alerce_operational_gain_cv.pdf")

    Ts    = [r["T"] for r in fold_results]
    gains = [r["gain"] for r in fold_results]
    precs = [r["precision_after"] for r in fold_results if not np.isnan(r["precision_after"])]

    results = {
        "metadata": {
            "description": "ALeRCE operational gain with proper held-out CV (Step 1.3)",
            "threshold": THRESHOLD,
            "n_folds": N_FOLDS,
            "paper_claim_gain": PAPER_GAIN,
            "paper_claim_precision": PAPER_PRECISION,
        },
        "full_dataset_baseline": {
            "T": T_full,
            "ece_before": ece_full_before,
            "ece_after":  ece_full_after,
            **ops_full,
        },
        "per_fold": fold_results,
        "cv_summary": {
            "T_mean": float(np.mean(Ts)),
            "T_std":  float(np.std(Ts)),
            "gain_mean_per_fold": float(np.mean(gains)),
            "gain_std_per_fold":  float(np.std(gains)),
            "precision_mean_per_fold": float(np.mean(precs)),
            "precision_std_per_fold":  float(np.std(precs)),
        },
        "pooled_held_out": {
            "ece_before": float(ece_before_pooled),
            "ece_after":  float(ece_after_pooled),
            **pooled,
        },
        "validation": {
            "passes": pooled["gain"] >= 15 and pooled["precision_after"] >= 0.85,
            "gain_vs_paper": pooled["gain"] / PAPER_GAIN,
            "precision_vs_paper": pooled["precision_after"] / PAPER_PRECISION,
        },
    }

    with open("results/alerce_operational_gain_cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults: results/alerce_operational_gain_cv_results.json")
    print(f"Figures: figures/fig_alerce_operational_gain_cv.pdf")
    print("=" * 80)


if __name__ == "__main__":
    main()
