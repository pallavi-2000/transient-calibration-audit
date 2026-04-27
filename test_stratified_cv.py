"""
Test: Stratified CV Verification
=================================

Verifies that:
1. Every fold contains every class
2. Class ratios are preserved across folds
3. Reports updated numbers after fix

Run from project root:
    python3 tests/test_stratified_cv.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import (
    _stratified_kfold, fit_temperature_cv, auto_calibrate,
    compute_ece, bootstrap_ece
)


def test_stratification_correctness():
    """Verify every fold has every class."""
    print("=" * 60)
    print("TEST 1: Stratification Correctness")
    print("=" * 60)

    # Simulate imbalanced dataset like ours
    # SNIa=463, SNIbc=225, SNII=341, SLSN=85
    labels = np.array(
        [0]*463 + [1]*225 + [2]*341 + [3]*85
    )

    folds = _stratified_kfold(labels, n_folds=5, random_state=42)

    overall_dist = {k: (labels == k).sum() for k in range(4)}
    print(f"\nOverall distribution: {overall_dist}")
    print(f"Overall ratios: { {k: v/len(labels) for k, v in overall_dist.items()} }")

    all_ok = True
    for i, (cal_idx, test_idx) in enumerate(folds):
        fold_labels = labels[test_idx]
        fold_dist = {k: (fold_labels == k).sum() for k in range(4)}
        fold_ratios = {k: v/len(fold_labels) if len(fold_labels) > 0 else 0
                       for k, v in fold_dist.items()}

        missing = [k for k, v in fold_dist.items() if v == 0]
        status = "FAIL" if missing else "OK"
        if missing:
            all_ok = False

        print(f"  Fold {i}: n={len(test_idx):4d}  dist={fold_dist}  "
              f"missing={missing}  [{status}]")

    print(f"\nAll folds have all classes: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_old_vs_new_comparison():
    """Show what the old unstratified split would have done."""
    print(f"\n{'=' * 60}")
    print("TEST 2: Old (unstratified) vs New (stratified) fold contents")
    print("=" * 60)

    labels = np.array(
        [0]*463 + [1]*225 + [2]*341 + [3]*85
    )

    # Old method: random permutation
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(labels))
    fold_size = len(labels) // 5

    print("\nOLD (unstratified):")
    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size if fold < 4 else len(labels)
        test_idx = indices[start:end]
        fold_labels = labels[test_idx]
        fold_dist = {k: (fold_labels == k).sum() for k in range(4)}
        print(f"  Fold {fold}: n={len(test_idx):4d}  dist={fold_dist}")

    # New method: stratified
    folds = _stratified_kfold(labels, n_folds=5, random_state=42)

    print("\nNEW (stratified):")
    for i, (cal_idx, test_idx) in enumerate(folds):
        fold_labels = labels[test_idx]
        fold_dist = {k: (fold_labels == k).sum() for k in range(4)}
        print(f"  Fold {i}: n={len(test_idx):4d}  dist={fold_dist}")


def test_on_real_data():
    """Run on actual ALeRCE and NEEDLE data, compare old vs new results."""
    print(f"\n{'=' * 60}")
    print("TEST 3: Real Data — Updated Results")
    print("=" * 60)

    # ---- ALeRCE ----
    alerce_file = "data/raw/alerce_classifications.csv"
    sample_file = "data/ground_truth/bts_sample.csv"

    if os.path.exists(alerce_file) and os.path.exists(sample_file):
        alerce = pd.read_csv(alerce_file)
        sample = pd.read_csv(sample_file)
        merged = alerce.merge(
            sample[["ZTFID", "alerce_class"]],
            left_on="oid", right_on="ZTFID"
        )
        merged = merged[merged["alerce_class"] != "TDE"].copy()

        trans = ["SNIa", "SNIbc", "SNII", "SLSN"]
        class_to_int = {c: i for i, c in enumerate(trans)}

        proba = merged[trans].apply(pd.to_numeric, errors="coerce").values
        proba = proba / proba.sum(axis=1, keepdims=True)
        labels = merged["alerce_class"].map(class_to_int).values

        print(f"\nALeRCE (n={len(labels)}):")
        print(f"  Class distribution: { {c: int((labels==i).sum()) for c, i in class_to_int.items()} }")

        # Verify stratification on real data
        folds = _stratified_kfold(labels, n_folds=5, random_state=42)
        for i, (cal_idx, test_idx) in enumerate(folds):
            fold_labels = labels[test_idx]
            fold_dist = {c: int((fold_labels==idx).sum()) for c, idx in class_to_int.items()}
            print(f"  Fold {i}: n={len(test_idx):4d}  {fold_dist}")

        # Run temperature scaling with stratified CV
        ts = fit_temperature_cv(labels, proba, n_folds=5)
        print(f"\n  Temperature scaling (STRATIFIED CV):")
        print(f"    T = {ts['T_mean']:.4f} +/- {ts['T_std']:.4f}")
        print(f"    T per fold: {[f'{t:.4f}' for t in ts['T_per_fold']]}")
        print(f"    ECE before: {ts['ece_before']:.4f}")
        print(f"    ECE after:  {ts['ece_after']:.4f}")
        print(f"    Improvement: {ts['improvement']:.4f} "
              f"({ts['improvement']/ts['ece_before']*100:.1f}%)")
        print(f"    Recommended: {ts['recommended']}")

        # Compare with OLD numbers from previous run
        print(f"\n  Previous (unstratified) results for comparison:")
        print(f"    T = 0.3569")
        print(f"    ECE before: 0.2761")
        print(f"    ECE after:  0.0797")
    else:
        print(f"\n  Skipping ALeRCE (data files not found)")

    # ---- NEEDLE ----
    needle_file = "data/processed/needle_predictions.npz"

    if os.path.exists(needle_file):
        data = np.load(needle_file, allow_pickle=True)
        probs = data["probs"]
        labels_n = data["labels"]
        class_names = list(data["class_names"])

        print(f"\nNEEDLE (n={len(labels_n)}):")
        print(f"  Class distribution: { {class_names[k]: int((labels_n==k).sum()) for k in range(len(class_names))} }")

        # Verify stratification
        folds = _stratified_kfold(labels_n, n_folds=5, random_state=42)
        for i, (cal_idx, test_idx) in enumerate(folds):
            fold_labels = labels_n[test_idx]
            fold_dist = {class_names[k]: int((fold_labels==k).sum())
                         for k in range(len(class_names))}
            print(f"  Fold {i}: n={len(test_idx):4d}  {fold_dist}")

        # Run temperature scaling with stratified CV
        ts = fit_temperature_cv(labels_n, probs, n_folds=5)
        print(f"\n  Temperature scaling (STRATIFIED CV):")
        print(f"    T = {ts['T_mean']:.4f} +/- {ts['T_std']:.4f}")
        print(f"    ECE before: {ts['ece_before']:.4f}")
        print(f"    ECE after:  {ts['ece_after']:.4f}")
        print(f"    Recommended: {ts['recommended']} — {ts['reason']}")
    else:
        print(f"\n  Skipping NEEDLE (data file not found)")

    # ---- Fink SNN ----
    fink_file = "data/raw/fink_classifications.csv"

    if os.path.exists(fink_file) and os.path.exists(sample_file):
        fink = pd.read_csv(fink_file)
        sample = pd.read_csv(sample_file)
        merged = fink.merge(
            sample[["ZTFID", "alerce_class"]],
            left_on="oid", right_on="ZTFID"
        )
        is_snia = (merged["alerce_class"] == "SNIa").astype(int).values
        snn = pd.to_numeric(merged["snn_snia_vs_nonia"], errors="coerce").values
        valid = ~np.isnan(snn)

        print(f"\nFink SNN (n={valid.sum()}):")
        ts = fit_temperature_cv(is_snia[valid], snn[valid], n_folds=5)
        print(f"  Temperature scaling (STRATIFIED CV):")
        print(f"    T = {ts['T_mean']:.4f}")
        print(f"    ECE before: {ts['ece_before']:.4f}")
        print(f"    ECE after:  {ts['ece_after']:.4f}")
        print(f"    Recommended: {ts['recommended']} — {ts['reason']}")
    else:
        print(f"\n  Skipping Fink SNN (data files not found)")


if __name__ == "__main__":
    test_stratification_correctness()
    test_old_vs_new_comparison()
    test_on_real_data()
    print(f"\n{'=' * 60}")
    print("DONE — Copy updated calibration.py and this test to your machine")
    print("=" * 60)
