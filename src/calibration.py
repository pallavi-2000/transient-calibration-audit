"""
Calibration Metrics and Methods
================================

Core module for computing calibration metrics (ECE, reliability
diagrams) and applying post-hoc calibration (temperature scaling).

Handles three cases:
  - Multi-class (ALeRCE 15-class, NEEDLE 3-class)
  - Binary (Fink SN Ia vs not)

References:
  - Guo et al. 2017: On Calibration of Modern Neural Networks
  - Naeini et al. 2015: Obtaining Well Calibrated Probabilities
  - Nixon et al. 2019: Measuring Calibration in Deep Learning
"""

import numpy as np


def compute_ece(y_true, y_proba, n_bins=10, strategy="equal_width"):
    """
    Compute Expected Calibration Error.

    Multi-class (ALeRCE, NEEDLE):
        y_true = array of class labels [0, 1, 2, ...]
        y_proba = (N, K) array of probabilities

    Binary (Fink):
        y_true = array of 0/1 labels
        y_proba = (N,) array of scores

    Parameters
    ----------
    y_true : np.ndarray
        True class labels. Shape (N,)
    y_proba : np.ndarray
        Predicted probabilities. (N, K) multi-class or (N,) binary.
    n_bins : int
        Number of confidence bins.
    strategy : str
        "equal_width" or "equal_mass"

    Returns
    -------
    ece : float
    bin_data : list of dict
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    if y_proba.ndim == 1:
        confidences = np.maximum(y_proba, 1 - y_proba)
        predictions = (y_proba >= 0.5).astype(int)
    else:
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)

    correct = (predictions == y_true).astype(float)

    if strategy == "equal_width":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "equal_mass":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(confidences, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0 + 1e-8
        bin_edges = np.unique(bin_edges)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ece = 0.0
    bin_data = []
    n_total = len(y_true)

    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == len(bin_edges) - 2:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        count = in_bin.sum()

        if count > 0:
            bin_acc = correct[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            bin_ece = (count / n_total) * abs(bin_acc - bin_conf)
            ece += bin_ece

            bin_data.append({
                "bin_lower": lower,
                "bin_upper": upper,
                "count": int(count),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "gap": float(bin_acc - bin_conf),
                "ece_contribution": float(bin_ece),
            })

    return float(ece), bin_data
