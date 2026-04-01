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


# ============================================================
# ECE COMPUTATION
# ============================================================

def compute_ece(y_true, y_proba, n_bins=10, strategy="equal_width"):
    """
    Compute Expected Calibration Error.

    Multi-class: y_proba is (N, K), y_true is class labels.
    Binary: y_proba is (N,), y_true is 0/1.
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


# ============================================================
# PER-CLASS ECE
# ============================================================

def compute_per_class_ece(y_true, y_proba, n_bins=10, class_names=None):
    """
    Compute ECE separately for each class.

    This catches class-asymmetric miscalibration like NEEDLE's:
    SLSN-I overconfident, TDE underconfident.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    n_classes = y_proba.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    results = {}

    for k in range(n_classes):
        is_class_k = (y_true == k).astype(float)
        probs_k = y_proba[:, k]

        mask = (probs_k > 0.01) | (is_class_k == 1)

        if mask.sum() < 5:
            results[class_names[k]] = {
                "ece": float("nan"),
                "accuracy": float("nan"),
                "mean_confidence": float("nan"),
                "gap": float("nan"),
                "count": int(mask.sum()),
            }
            continue

        probs_masked = probs_k[mask]
        true_masked = is_class_k[mask]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece_k = 0.0
        n_total = len(probs_masked)

        for i in range(n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i == n_bins - 1:
                in_bin = (probs_masked >= lower) & (probs_masked <= upper)
            else:
                in_bin = (probs_masked >= lower) & (probs_masked < upper)

            count = in_bin.sum()
            if count > 0:
                bin_acc = true_masked[in_bin].mean()
                bin_conf = probs_masked[in_bin].mean()
                ece_k += (count / n_total) * abs(bin_acc - bin_conf)

        predicted_k = np.argmax(y_proba, axis=1) == k
        if predicted_k.sum() > 0:
            acc_k = is_class_k[predicted_k].mean()
            conf_k = y_proba[predicted_k, k].mean()
        else:
            acc_k = float("nan")
            conf_k = float("nan")

        results[class_names[k]] = {
            "ece": float(ece_k),
            "accuracy": float(acc_k),
            "mean_confidence": float(conf_k),
            "gap": float(acc_k - conf_k) if not np.isnan(acc_k) else float("nan"),
            "count": int(mask.sum()),
        }

    return results


# ============================================================
# TEMPERATURE SCALING — INTERNALS
# ============================================================

def _proba_to_logits(proba):
    proba = np.clip(proba, 1e-10, 1 - 1e-10)
    return np.log(proba)


def _logits_to_proba(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _binary_logit(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))


def _binary_sigmoid(logit):
    return 1 / (1 + np.exp(-logit))


def _nll_multiclass(T, logits, labels):
    scaled = logits / T
    proba = _logits_to_proba(scaled)
    log_probs = np.log(np.clip(proba[np.arange(len(labels)), labels], 1e-10, 1.0))
    return -np.mean(log_probs)


def _nll_binary(T, logits, labels):
    scaled = logits / T
    proba = _binary_sigmoid(scaled)
    log_probs = labels * np.log(np.clip(proba, 1e-10, 1.0)) + \
                (1 - labels) * np.log(np.clip(1 - proba, 1e-10, 1.0))
    return -np.mean(log_probs)


# ============================================================
# TEMPERATURE SCALING — PUBLIC API
# ============================================================

def fit_temperature(y_true, y_proba, T_range=(0.01, 10.0)):
    """
    Fit a single temperature T by minimizing NLL.

    T < 1 sharpens (fixes underconfidence).
    T > 1 softens (fixes overconfidence).
    """
    from scipy.optimize import minimize_scalar

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    if y_proba.ndim == 1:
        logits = _binary_logit(y_proba)
        result = minimize_scalar(
            _nll_binary, bounds=T_range, args=(logits, y_true), method="bounded"
        )
    else:
        logits = _proba_to_logits(y_proba)
        result = minimize_scalar(
            _nll_multiclass, bounds=T_range, args=(logits, y_true), method="bounded"
        )

    return float(result.x)


def apply_temperature(y_proba, T):
    """Apply temperature scaling to probabilities."""
    y_proba = np.asarray(y_proba, dtype=np.float64)

    if y_proba.ndim == 1:
        logits = _binary_logit(y_proba)
        return _binary_sigmoid(logits / T)
    else:
        logits = _proba_to_logits(y_proba)
        scaled = logits / T
        return _logits_to_proba(scaled)


def fit_temperature_cv(y_true, y_proba, n_folds=5, random_state=42):
    """
    Fit temperature scaling with cross-validation.

    Correct methodology: fit T on calibration folds,
    evaluate on held-out fold.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    n = len(y_true)
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)

    fold_size = n // n_folds
    fold_Ts = []
    fold_eces_before = []
    fold_eces_after = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        cal_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        T = fit_temperature(y_true[cal_idx], y_proba[cal_idx])

        ece_before, _ = compute_ece(y_true[test_idx], y_proba[test_idx])
        calibrated = apply_temperature(y_proba[test_idx], T)
        ece_after, _ = compute_ece(y_true[test_idx], calibrated)

        fold_Ts.append(T)
        fold_eces_before.append(ece_before)
        fold_eces_after.append(ece_after)

    ece_before = float(np.mean(fold_eces_before))
    ece_after = float(np.mean(fold_eces_after))
    improvement = ece_before - ece_after

    # Decide whether scaling is recommended
    scaling_worsens = improvement < 0
    already_good = ece_before < 0.1
    scaling_recommended = (not scaling_worsens) and (not already_good)

    if scaling_worsens:
        reason = "Temperature scaling worsens ECE — class-asymmetric miscalibration"
    elif already_good:
        reason = "ECE already below 0.1 — scaling unnecessary"
    else:
        reason = "Scaling recommended"

    return {
        "T_mean": float(np.mean(fold_Ts)),
        "T_std": float(np.std(fold_Ts)),
        "T_per_fold": [float(t) for t in fold_Ts],
        "ece_before_mean": ece_before,
        "ece_after_mean": ece_after,
        "improvement": improvement,
        "scaling_recommended": scaling_recommended,
        "reason": reason,
    }


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_ece(y_true, y_proba, n_bootstrap=1000, n_bins=10,
                  confidence=0.95, random_state=42):
    """Compute ECE with bootstrap confidence intervals."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    rng = np.random.RandomState(random_state)
    n = len(y_true)

    ece_point, _ = compute_ece(y_true, y_proba, n_bins=n_bins)

    bootstrap_eces = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        ece_boot, _ = compute_ece(y_true[idx], y_proba[idx], n_bins=n_bins)
        bootstrap_eces.append(ece_boot)

    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(bootstrap_eces, 100 * alpha)
    ci_upper = np.percentile(bootstrap_eces, 100 * (1 - alpha))

    return {
        "ece": float(ece_point),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }
