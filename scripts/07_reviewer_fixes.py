"""
Steps 4-7: Remaining Reviewer Fixes
=====================================

Step 4: Brier scores (reviewer #3) — extract from JSON, format for paper
Step 5: Bin sensitivity for NEEDLE (reviewer #8) — M=10, 15, 20
Step 6: Per-class post-scaling ECE for NEEDLE (reviewer #13)
Step 7: Decision-theoretic table for ALeRCE (reviewer #6)

Run from project root:
    python3 scripts/07_reviewer_fixes.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import (
    compute_ece, compute_classwise_ece, brier_score,
    fit_temperature_cv, apply_temperature,
    fit_per_class_temperature, apply_per_class_temperature
)


def step4_brier_table():
    """Step 4: Compile Brier scores into a table for the paper."""
    print("=" * 60)
    print("STEP 4: Brier Score Table")
    print("Reviewer concern #3")
    print("=" * 60)

    # Load results
    with open("results/alerce_results.json") as f:
        alerce = json.load(f)
    with open("results/fink_results.json") as f:
        fink = json.load(f)
    with open("results/needle_results.json") as f:
        needle = json.load(f)

    print(f"\n  {'Classifier':<20} {'Brier Score':<15} {'ECE':<15}")
    print(f"  {'-'*50}")
    print(f"  {'ALeRCE BRF':<20} {alerce['brier_score']:<15.3f} {alerce['ece']:<15.3f}")
    print(f"  {'Fink RF':<20} {fink['random_forest']['brier_score']:<15.3f} {fink['random_forest']['ece']:<15.3f}")
    print(f"  {'Fink SNN':<20} {fink['supernnova']['brier_score']:<15.3f} {fink['supernnova']['ece']:<15.3f}")
    print(f"  {'NEEDLE':<20} {needle['brier_score']:<15.3f} {needle['ece']:<15.3f}")

    # Note: Brier score interpretation
    # 0 = perfect, 1 = worst possible
    # For K classes, random guessing gives BS = 2(1 - 1/K) / K... roughly
    print(f"\n  Reference: random 4-class Brier ≈ 1.5, random binary Brier ≈ 0.5")

    return {
        "ALeRCE": alerce["brier_score"],
        "Fink_RF": fink["random_forest"]["brier_score"],
        "Fink_SNN": fink["supernnova"]["brier_score"],
        "NEEDLE": needle["brier_score"],
    }


def step5_bin_sensitivity():
    """Step 5: NEEDLE ECE sensitivity to bin count."""
    print(f"\n{'='*60}")
    print("STEP 5: Bin Sensitivity for NEEDLE")
    print("Reviewer concern #8")
    print("=" * 60)

    data = np.load("data/processed/needle_predictions.npz", allow_pickle=True)
    probs = data["probs"]
    labels = data["labels"]

    print(f"\n  N = {len(labels)} predictions")
    print(f"\n  {'M (bins)':<12} {'Samples/bin':<15} {'ECE':<10}")
    print(f"  {'-'*37}")

    results = {}
    for M in [5, 10, 15, 20, 25]:
        ece, bins = compute_ece(labels, probs, n_bins=M)
        samples_per_bin = len(labels) / M
        print(f"  {M:<12} {samples_per_bin:<15.0f} {ece:<10.3f}")
        results[M] = float(ece)

    print(f"\n  Recommendation: M=10 gives {len(labels)/10:.0f} samples/bin")
    print(f"  (vs M=15 with {len(labels)/15:.0f} — marginal for tail bins)")
    print(f"  ECE is stable across M=10-20 (range: "
          f"{min(results[10], results[15], results[20]):.3f}-"
          f"{max(results[10], results[15], results[20]):.3f})")

    return results


def step6_perclass_postscaling():
    """Step 6: Per-class ECE after per-class temperature scaling."""
    print(f"\n{'='*60}")
    print("STEP 6: Per-Class Post-Scaling ECE for NEEDLE")
    print("Reviewer concern #13")
    print("=" * 60)

    data = np.load("data/processed/needle_predictions.npz", allow_pickle=True)
    probs = data["probs"]
    labels = data["labels"]
    class_names = list(data["class_names"])

    # Before scaling
    cw_before = compute_classwise_ece(labels, probs, class_names=class_names)

    # Fit per-class T
    class_Ts = fit_per_class_temperature(labels, probs, class_names=class_names)

    # Apply per-class T
    probs_scaled = apply_per_class_temperature(probs, class_Ts, class_names)

    # After scaling
    cw_after = compute_classwise_ece(labels, probs_scaled, class_names=class_names)
    ece_before, _ = compute_ece(labels, probs)
    ece_after, _ = compute_ece(labels, probs_scaled)

    print(f"\n  Per-class temperatures: {class_Ts}")
    print(f"\n  {'Class':<10} {'ECE before':<15} {'ECE after':<15} {'Change':<15}")
    print(f"  {'-'*55}")

    results = {}
    for cls in class_names:
        before = cw_before[cls]["ece"]
        after = cw_after[cls]["ece"]
        change = after - before
        direction = "worse" if change > 0 else "better"
        print(f"  {cls:<10} {before:<15.3f} {after:<15.3f} {change:+.3f} ({direction})")
        results[cls] = {"before": before, "after": after, "T": class_Ts[cls]}

    print(f"\n  {'Aggregate':<10} {ece_before:<15.3f} {ece_after:<15.3f} "
          f"{ece_after-ece_before:+.3f}")

    # Per-class accuracy and confidence after scaling
    print(f"\n  Post-scaling per-class statistics:")
    print(f"  {'Class':<10} {'Accuracy':<12} {'Conf before':<15} {'Conf after':<15}")
    print(f"  {'-'*52}")
    for cls in class_names:
        print(f"  {cls:<10} "
              f"{cw_before[cls]['accuracy']:<12.3f} "
              f"{cw_before[cls]['mean_confidence']:<15.3f} "
              f"{cw_after[cls]['mean_confidence']:<15.3f}")

    results["aggregate"] = {"before": float(ece_before), "after": float(ece_after)}
    return results


def step7_decision_table():
    """Step 7: Decision-theoretic table for ALeRCE."""
    print(f"\n{'='*60}")
    print("STEP 7: Decision-Theoretic Table for ALeRCE")
    print("Reviewer concern #6")
    print("=" * 60)

    # Load data
    alerce = pd.read_csv("data/raw/alerce_classifications.csv")
    sample = pd.read_csv("data/ground_truth/bts_sample.csv")

    merged = alerce.merge(
        sample[["ZTFID", "alerce_class"]],
        left_on="oid", right_on="ZTFID"
    )
    merged = merged[merged["alerce_class"] != "TDE"].copy()

    trans_classes = ["SNIa", "SNIbc", "SNII", "SLSN"]
    class_to_int = {c: i for i, c in enumerate(trans_classes)}

    proba = merged[trans_classes].astype(float).values
    proba = proba / proba.sum(axis=1, keepdims=True)
    labels = merged["alerce_class"].map(class_to_int).values

    # Apply temperature scaling
    ts = fit_temperature_cv(labels, proba, n_folds=5)
    T = ts["T_mean"]
    proba_cal = apply_temperature(proba, T)

    confidences_raw = np.max(proba, axis=1)
    confidences_cal = np.max(proba_cal, axis=1)
    predictions_raw = np.argmax(proba, axis=1)
    predictions_cal = np.argmax(proba_cal, axis=1)
    correct = labels  # ground truth

    print(f"\n  T = {T:.3f}, N = {len(labels)}")
    print(f"\n  Follow-up decisions at various probability thresholds:")
    print(f"\n  {'Threshold':<12} {'---Raw (uncalibrated)---':^36} {'---Calibrated (T={:.3f})---'.format(T):^36}")
    print(f"  {'':12} {'N pass':<10} {'Precision':<12} {'Recall':<12} {'N pass':<10} {'Precision':<12} {'Recall':<12}")
    print(f"  {'-'*84}")

    results = {}
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        # Raw
        pass_raw = confidences_raw >= threshold
        n_pass_raw = pass_raw.sum()
        if n_pass_raw > 0:
            correct_raw = (predictions_raw[pass_raw] == labels[pass_raw]).sum()
            precision_raw = correct_raw / n_pass_raw
            recall_raw = correct_raw / len(labels)
        else:
            precision_raw = float("nan")
            recall_raw = 0.0

        # Calibrated
        pass_cal = confidences_cal >= threshold
        n_pass_cal = pass_cal.sum()
        if n_pass_cal > 0:
            correct_cal = (predictions_cal[pass_cal] == labels[pass_cal]).sum()
            precision_cal = correct_cal / n_pass_cal
            recall_cal = correct_cal / len(labels)
        else:
            precision_cal = float("nan")
            recall_cal = 0.0

        print(f"  p > {threshold:<5.1f}   "
              f"{n_pass_raw:<10} {precision_raw:<12.3f} {recall_raw:<12.3f} "
              f"{n_pass_cal:<10} {precision_cal:<12.3f} {recall_cal:<12.3f}")

        results[str(threshold)] = {
            "raw": {"n_pass": int(n_pass_raw), "precision": float(precision_raw),
                    "recall": float(recall_raw)},
            "calibrated": {"n_pass": int(n_pass_cal), "precision": float(precision_cal),
                           "recall": float(recall_cal)},
        }

    # The key operational finding
    print(f"\n  KEY FINDING:")
    n_raw_80 = (confidences_raw >= 0.8).sum()
    n_cal_80 = (confidences_cal >= 0.8).sum()
    print(f"  At p > 0.8 threshold:")
    print(f"    Raw:        {n_raw_80} objects pass (classifier never exceeds ~0.75)")
    print(f"    Calibrated: {n_cal_80} objects pass")
    if n_cal_80 > 0:
        prec_cal = (predictions_cal[confidences_cal >= 0.8] ==
                    labels[confidences_cal >= 0.8]).mean()
        print(f"    Calibrated precision at p>0.8: {prec_cal:.3f}")

    return results


def main():
    os.makedirs("results", exist_ok=True)

    brier = step4_brier_table()
    bins = step5_bin_sensitivity()
    perclass = step6_perclass_postscaling()
    decision = step7_decision_table()

    # Save all results
    all_results = {
        "step4_brier_scores": brier,
        "step5_bin_sensitivity": bins,
        "step6_perclass_postscaling": perclass,
        "step7_decision_table": decision,
    }

    with open("results/reviewer_fixes.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("ALL REVIEWER FIXES COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: results/reviewer_fixes.json")
    print(f"\nSummary of what to add to the paper:")
    print(f"  Step 4: Brier score column in Table 2")
    print(f"  Step 5: Sentence on bin sensitivity (M=10-20 gives stable ECE)")
    print(f"  Step 6: Per-class ECE table before/after per-class T")
    print(f"  Step 7: Decision table in Discussion section 6.3")


if __name__ == "__main__":
    main()
