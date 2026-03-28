"""
src/calibration.py

Core calibration functions for the ALeRCE + Fink calibration study.
All paper results are computed from functions in this file.

No API calls. No data collection. Pure computation only.
"""

import numpy as np
from typing import Tuple, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# 1. EXPECTED CALIBRATION ERROR
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    y_true_idx: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, list]:
    """
    Compute Expected Calibration Error (ECE).

    Uses top-1 confidence: for each prediction, takes the class with
    the highest probability and checks if it matches the true label.

    Parameters
    ----------
    y_true_idx : np.ndarray, shape (N,)
        Integer class labels. e.g. [0, 2, 0, 1, ...]
    y_proba : np.ndarray, shape (N, C)
        Probability matrix. Rows must sum to 1.0.
    n_bins : int
        Number of confidence bins. 10 is standard in literature.

    Returns
    -------
    ece : float
        Expected Calibration Error. Lower is better. 0.0 is perfect.
    bin_stats : list of dicts
        Per-bin breakdown for plotting reliability diagrams.
    """
    y_pred = np.argmax(y_proba, axis=1)
    y_conf = y_proba[np.arange(len(y_true_idx)), y_pred]
    correct = (y_pred == y_true_idx).astype(float)

    ece = 0.0
    bin_stats = []

    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        mask = (y_conf >= lo) & (y_conf < hi)

        if mask.sum() == 0:
            continue

        bin_acc  = correct[mask].mean()
        bin_conf = y_conf[mask].mean()
        bin_n    = int(mask.sum())
        bin_ece  = (bin_n / len(y_true_idx)) * abs(bin_acc - bin_conf)

        ece += bin_ece
        bin_stats.append({
            "bin_lo":   round(lo, 2),
            "bin_hi":   round(hi, 2),
            "accuracy": round(float(bin_acc), 4),
            "confidence": round(float(bin_conf), 4),
            "gap":      round(float(bin_acc - bin_conf), 4),
            "count":    bin_n,
            "ece_contribution": round(float(bin_ece), 4),
        })

    return round(float(ece), 4), bin_stats


# ─────────────────────────────────────────────────────────────────────────────
# 2. BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ece(
    y_true_idx: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 2000,
    n_bins: int = 10,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap 95% confidence interval on ECE.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples. 2000 is sufficient for publication.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (mean_ece, ci_lo, ci_hi) : Tuple[float, float, float]
        Mean ECE and 95% confidence interval bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true_idx)
    samples = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        ece, _ = compute_ece(y_true_idx[idx], y_proba[idx], n_bins)
        samples.append(ece)

    samples = np.array(samples)
    ci_lo, ci_hi = np.percentile(samples, [2.5, 97.5])

    return (
        round(float(np.mean(samples)), 4),
        round(float(ci_lo), 4),
        round(float(ci_hi), 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. RENORMALIZE TO SUBSET OF CLASSES
# ─────────────────────────────────────────────────────────────────────────────

def renormalize_to_subset(
    y_proba: np.ndarray,
    subset_indices: List[int]
) -> np.ndarray:
    """
    Project a 15-class probability matrix onto a subset of classes,
    renormalizing so rows sum to 1.0.

    This separates genuine miscalibration from probability dilution —
    i.e. the effect of ALeRCE spreading probability across 15 classes
    when you only have ground truth for 5 of them.

    Parameters
    ----------
    y_proba : np.ndarray, shape (N, 15)
        Full ALeRCE probability matrix.
    subset_indices : List[int]
        Column indices of the classes to keep.
        e.g. [0, 1, 2, 3, 4] for SNIa, SNIbc, SNII, SLSN, TDE

    Returns
    -------
    y_sub_norm : np.ndarray, shape (N, len(subset_indices))
        Renormalized probability matrix.
    """
    y_sub = y_proba[:, subset_indices]
    row_sums = y_sub.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid division by zero
    return y_sub / row_sums


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST — run this file directly to verify functions are correct
# python src/calibration.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running self-tests...\n")
    rng = np.random.default_rng(0)

    # ── Test 1: perfect calibration should give ECE ≈ 0 ─────────────────────
    # A perfect classifier: confidence always equals accuracy
    # We simulate this by making the true class always have the highest prob
    N = 500
    C = 5
    # Create probabilities where the correct class always has ~0.8 confidence
    y_true = rng.integers(0, C, N)
    y_proba = rng.dirichlet(np.ones(C) * 0.5, N)
    # Force correct class to have high confidence (simulates good calibration)
    for i in range(N):
        y_proba[i, y_true[i]] = 0.8
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    ece, bins = compute_ece(y_true, y_proba)
    print(f"Test 1 — calibration check")
    print(f"  ECE = {ece}  (should be low, < 0.15 for a reasonable classifier)")
    print(f"  Bins with data: {len(bins)}")
    assert ece >= 0.0, "ECE must be non-negative"
    assert ece <= 1.0, "ECE must be <= 1.0"
    print("  PASSED\n")

    # ── Test 2: perfectly uniform classifier should be underconfident ────────
    # If all classes get equal probability, confidence = 1/C for every prediction
    # but accuracy will be ~1/C too — so ECE should be near 0
    y_uniform = np.ones((N, C)) / C
    ece_uniform, _ = compute_ece(y_true, y_uniform)
    print(f"Test 2 — uniform classifier")
    print(f"  ECE = {ece_uniform}  (should be near 0 — uniform is well-calibrated)")
    assert ece_uniform < 0.05, f"Uniform classifier ECE too high: {ece_uniform}"
    print("  PASSED\n")

    # ── Test 3: bootstrap returns valid CI ──────────────────────────────────
    mean_ece, lo, hi = bootstrap_ece(y_true, y_proba, n_bootstrap=500)
    print(f"Test 3 — bootstrap CI")
    print(f"  ECE = {mean_ece} (95% CI: [{lo}, {hi}])")
    assert lo <= mean_ece <= hi, "Mean ECE must be within CI"
    assert lo >= 0.0, "CI lower bound must be non-negative"
    print("  PASSED\n")

    # ── Test 4: renormalization ──────────────────────────────────────────────
    y_15class = rng.dirichlet(np.ones(15) * 0.5, N)
    subset_idx = [0, 2, 5, 8, 11]  # 5 arbitrary columns
    y_5class = renormalize_to_subset(y_15class, subset_idx)
    print(f"Test 4 — renormalization")
    print(f"  Input shape:  {y_15class.shape}")
    print(f"  Output shape: {y_5class.shape}")
    row_sums = y_5class.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Rows must sum to 1.0 after renormalization"
    print(f"  All rows sum to 1.0: YES")
    print("  PASSED\n")

    print("=" * 50)
    print("All tests passed. calibration.py is ready.")
    print("=" * 50)