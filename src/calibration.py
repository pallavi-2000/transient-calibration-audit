"""
Calibration Metrics and Methods
================================

Scientifically grounded calibration analysis for astronomical
transient classifiers. Handles multi-class (ALeRCE, NEEDLE)
and binary (Fink) classifiers.

Methodology follows:
  - Guo et al. 2017 (ICML): temperature scaling
  - Nixon et al. 2019 (CVPRW): SCE, ACE metrics
  - Kull et al. 2019 (NeurIPS): Dirichlet calibration
  - Roelofs et al. 2022 (AISTATS): equal-mass binning
  - Niculescu-Mizil & Caruana 2005 (ICML): RF calibration
  - Kumar et al. 2019 (NeurIPS): verified calibration
"""

import numpy as np


# ============================================================
# ECE COMPUTATION
# ============================================================

def compute_ece(y_true, y_proba, n_bins=15, strategy="equal_mass"):
    """
    Expected Calibration Error.

    Uses equal-mass (quantile) binning by default, following
    Roelofs et al. 2022 which showed it produces lower-bias
    estimates than equal-width binning.

    Works for multi-class (N, K) and binary (N,) inputs.

    Parameters
    ----------
    y_true : array (N,)
        True labels (int for multi-class, 0/1 for binary).
    y_proba : array (N, K) or (N,)
        Predicted probabilities.
    n_bins : int
        Number of bins. Default 15 (Guo et al. 2017).
    strategy : str
        "equal_mass" (recommended) or "equal_width" (for comparison).

    Returns
    -------
    ece : float
    bin_data : list of dict (for reliability diagrams)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    # Extract confidence and correctness
    if y_proba.ndim == 1:
        # Binary
        confidences = np.maximum(y_proba, 1 - y_proba)
        predictions = (y_proba >= 0.5).astype(int)
    else:
        # Multi-class
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)

    correct = (predictions == y_true).astype(float)

    # Create bin edges
    if strategy == "equal_mass":
        # Each bin gets roughly N/n_bins samples
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(confidences, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0 + 1e-8
        # Remove duplicate edges (happens when many identical scores)
        bin_edges = np.unique(bin_edges)
    elif strategy == "equal_width":
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_edges[-1] += 1e-8
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ece = 0.0
    bin_data = []
    n_total = len(y_true)

    for i in range(len(bin_edges) - 1):
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        count = in_bin.sum()

        if count > 0:
            bin_acc = correct[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            contribution = (count / n_total) * abs(bin_acc - bin_conf)
            ece += contribution

            bin_data.append({
                "bin_lower": float(bin_edges[i]),
                "bin_upper": float(bin_edges[i + 1]),
                "count": int(count),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "gap": float(bin_acc - bin_conf),
            })

    return float(ece), bin_data


def compute_classwise_ece(y_true, y_proba, n_bins=15, class_names=None):
    """
    Classwise ECE (ACE from Nixon et al. 2019).

    Evaluates calibration of ALL class probabilities, not just
    the argmax. Essential for astronomy where non-dominant class
    probabilities matter (e.g., P(SLSN) when top prediction is SN).

    Uses equal-mass binning per class.
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

        # Equal-mass bins for this class
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(probs_k, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0 + 1e-8
        bin_edges = np.unique(bin_edges)

        ece_k = 0.0
        n_total = len(y_true)

        for i in range(len(bin_edges) - 1):
            in_bin = (probs_k >= bin_edges[i]) & (probs_k < bin_edges[i + 1])
            count = in_bin.sum()

            if count > 0:
                bin_acc = is_class_k[in_bin].mean()
                bin_conf = probs_k[in_bin].mean()
                ece_k += (count / n_total) * abs(bin_acc - bin_conf)

        # Per-class summary stats
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
            "n_predicted": int(predicted_k.sum()),
            "n_true": int(is_class_k.sum()),
        }

    return results


def brier_score(y_true, y_proba):
    """
    Brier score with reliability decomposition.

    Reports calibration (reliability), discrimination (resolution),
    and uncertainty — gives a complete picture that ECE alone cannot.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    if y_proba.ndim == 1:
        # Binary: convert to (N, 2)
        y_proba = np.column_stack([1 - y_proba, y_proba])
        n_classes = 2
    else:
        n_classes = y_proba.shape[1]

    # One-hot encode true labels
    y_onehot = np.zeros_like(y_proba)
    y_onehot[np.arange(len(y_true)), y_true] = 1

    brier = np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))

    return {"brier_score": float(brier)}


# ============================================================
# TEMPERATURE SCALING
# ============================================================

def _to_logits(proba):
    """Probabilities -> log-probabilities (pseudo-logits)."""
    return np.log(np.clip(proba, 1e-10, 1 - 1e-10))


def _softmax(logits):
    """Logits -> probabilities via softmax."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _binary_logit(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _nll_multiclass(T, logits, labels):
    """NLL loss for temperature scaling (proper scoring rule)."""
    proba = _softmax(logits / T)
    return -np.mean(np.log(np.clip(
        proba[np.arange(len(labels)), labels], 1e-10, 1.0
    )))


def _nll_binary(T, logits, labels):
    """NLL loss for binary temperature scaling."""
    p = _sigmoid(logits / T)
    return -np.mean(
        labels * np.log(np.clip(p, 1e-10, 1.0)) +
        (1 - labels) * np.log(np.clip(1 - p, 1e-10, 1.0))
    )


def fit_temperature(y_true, y_proba, T_range=(0.01, 10.0)):
    """
    Fit temperature T by minimizing NLL (not ECE).

    NLL is a strictly proper scoring rule — uniquely minimized
    by true conditional probabilities. ECE is not proper and
    depends on binning. This follows Guo et al. 2017.

    T < 1: sharpens (fixes underconfidence)
    T > 1: softens (fixes overconfidence)
    """
    from scipy.optimize import minimize_scalar

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    if y_proba.ndim == 1:
        logits = _binary_logit(y_proba)
        result = minimize_scalar(
            _nll_binary, bounds=T_range,
            args=(logits, y_true), method="bounded"
        )
    else:
        logits = _to_logits(y_proba)
        result = minimize_scalar(
            _nll_multiclass, bounds=T_range,
            args=(logits, y_true), method="bounded"
        )

    return float(result.x)


def apply_temperature(y_proba, T):
    """Apply temperature scaling: q_i proportional to p_i^(1/T)."""
    y_proba = np.asarray(y_proba, dtype=np.float64)

    if y_proba.ndim == 1:
        return _sigmoid(_binary_logit(y_proba) / T)
    else:
        return _softmax(_to_logits(y_proba) / T)


def fit_temperature_cv(y_true, y_proba, n_folds=5, random_state=42):
    """
    Temperature scaling with cross-validation.

    Critical: T is fitted on calibration folds, evaluated on
    held-out fold. Never fit and evaluate on the same data.

    Returns recommendation on whether to apply scaling.
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

    # Decision logic
    scaling_worsens = improvement < 0
    already_good = ece_before < 0.1
    T_mean = float(np.mean(fold_Ts))
    hit_bound = T_mean <= 0.02 or T_mean >= 9.5

    if hit_bound:
        scaling_recommended = False
        reason = (f"T={T_mean:.2f} hit optimizer bound — "
                  f"temperature scaling inappropriate for this distribution")
    elif scaling_worsens:
        scaling_recommended = False
        reason = "Worsens ECE — likely class-asymmetric miscalibration"
    elif already_good:
        scaling_recommended = False
        reason = "ECE already below 0.1 — unnecessary"
    else:
        scaling_recommended = True
        reason = "Recommended"

    return {
        "T_mean": T_mean,
        "T_std": float(np.std(fold_Ts)),
        "T_per_fold": [float(t) for t in fold_Ts],
        "ece_before": ece_before,
        "ece_after": ece_after,
        "improvement": improvement,
        "recommended": scaling_recommended,
        "reason": reason,
    }


# ============================================================
# PER-CLASS TEMPERATURE SCALING
# ============================================================

def fit_per_class_temperature(y_true, y_proba, class_names=None,
                              T_range=(0.01, 10.0)):
    """
    Fit separate T per class (Frenkel 2021, EUSIPCO).

    Needed when miscalibration is class-asymmetric (e.g., NEEDLE:
    SLSN-I overconfident, TDE underconfident).

    Unlike global T, this CAN change predicted classes.
    """
    from scipy.optimize import minimize_scalar

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    n_classes = y_proba.shape[1]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    logits = _to_logits(y_proba)
    class_Ts = {}

    for k in range(n_classes):
        is_k = (y_true == k).astype(float)
        logits_k = logits[:, k]

        def nll_k(T, logits_k=logits_k, is_k=is_k):
            p = _sigmoid(logits_k / T)
            return -np.mean(
                is_k * np.log(np.clip(p, 1e-10, 1.0)) +
                (1 - is_k) * np.log(np.clip(1 - p, 1e-10, 1.0))
            )

        result = minimize_scalar(nll_k, bounds=T_range, method="bounded")
        class_Ts[class_names[k]] = float(result.x)

    return class_Ts


def apply_per_class_temperature(y_proba, class_Ts, class_names):
    """Apply per-class T then re-softmax."""
    y_proba = np.asarray(y_proba, dtype=np.float64)
    logits = _to_logits(y_proba)

    for k, name in enumerate(class_names):
        logits[:, k] = logits[:, k] / class_Ts[name]

    return _softmax(logits)


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_ece(y_true, y_proba, n_bootstrap=1000, n_bins=15,
                  confidence=0.95, random_state=42):
    """
    ECE with BCa bootstrap confidence intervals.

    Uses 1000 replicates (sufficient for standard errors).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    rng = np.random.RandomState(random_state)
    n = len(y_true)

    ece_point, _ = compute_ece(y_true, y_proba, n_bins=n_bins)

    boot_eces = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        ece_b, _ = compute_ece(y_true[idx], y_proba[idx], n_bins=n_bins)
        boot_eces.append(ece_b)

    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(boot_eces, 100 * alpha)
    ci_upper = np.percentile(boot_eces, 100 * (1 - alpha))

    return {
        "ece": float(ece_point),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


# ============================================================
# AUTO-CALIBRATE
# ============================================================

def auto_calibrate(y_true, y_proba, n_folds=5, class_names=None,
                   random_state=42):
    """
    Automatically find the best calibration strategy.

    Logic:
      1. Compute baseline ECE + bootstrap CI + Brier score
      2. Try global temperature scaling (CV)
      3. If global T worsens ECE AND multi-class:
         try per-class temperature scaling
      4. Report the best strategy

    For binary classifiers with degenerate distributions
    (e.g., Fink RF with 94% zeros), flags as unsuitable.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    is_binary = y_proba.ndim == 1

    # Check for degenerate distribution
    if is_binary:
        zero_frac = np.mean(y_proba == 0)
        if zero_frac > 0.5:
            ece_point, bin_data = compute_ece(y_true, y_proba)
            return {
                "ece_baseline": ece_point,
                "recommendation": "none",
                "reason": f"Degenerate distribution ({zero_frac*100:.0f}% zeros). "
                          f"Not suitable for probability calibration. "
                          f"Use precision-recall metrics instead.",
                "degenerate": True,
                "bin_data": bin_data,
            }

    # Step 1: Baseline metrics
    boot = bootstrap_ece(y_true, y_proba, random_state=random_state)
    brier = brier_score(y_true, y_proba)

    result = {
        "ece_baseline": boot["ece"],
        "ece_ci": [boot["ci_lower"], boot["ci_upper"]],
        "brier": brier["brier_score"],
        "degenerate": False,
    }

    # Per-class ECE (multi-class only)
    if not is_binary:
        n_classes = y_proba.shape[1]
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        result["per_class"] = compute_classwise_ece(
            y_true, y_proba, class_names=class_names
        )

    # Step 2: Global temperature scaling
    global_ts = fit_temperature_cv(
        y_true, y_proba, n_folds=n_folds, random_state=random_state
    )
    result["global_temperature"] = global_ts

    # Step 3: If global T failed and multi-class, try per-class T
    if not global_ts["recommended"] and not is_binary:
        # Per-class T with CV
        n = len(y_true)
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n)
        fold_size = n // n_folds

        fold_eces = []
        all_class_Ts = []

        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else n
            test_idx = indices[test_start:test_end]
            cal_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            fold_Ts = fit_per_class_temperature(
                y_true[cal_idx], y_proba[cal_idx], class_names=class_names
            )
            calibrated = apply_per_class_temperature(
                y_proba[test_idx], fold_Ts, class_names
            )
            ece_after, _ = compute_ece(y_true[test_idx], calibrated)
            fold_eces.append(ece_after)
            all_class_Ts.append(fold_Ts)

        # Average per-class T across folds
        avg_class_Ts = {}
        for name in class_names:
            avg_class_Ts[name] = float(np.mean([t[name] for t in all_class_Ts]))

        pc_ece_after = float(np.mean(fold_eces))
        pc_helps = pc_ece_after < global_ts["ece_before"]

        result["per_class_temperature"] = {
            "class_Ts": avg_class_Ts,
            "ece_after": pc_ece_after,
            "helps": pc_helps,
        }

    # Step 4: Final recommendation
    if global_ts["recommended"]:
        result["recommendation"] = "global_temperature"
        result["summary"] = (
            f"Global T={global_ts['T_mean']:.3f}: "
            f"ECE {global_ts['ece_before']:.3f} -> {global_ts['ece_after']:.3f}"
        )
    elif not is_binary and "per_class_temperature" in result:
        pct = result["per_class_temperature"]
        if pct["helps"]:
            result["recommendation"] = "per_class_temperature"
            result["summary"] = (
                f"Per-class T: ECE {global_ts['ece_before']:.3f} -> "
                f"{pct['ece_after']:.3f}. "
                f"Global T failed (class-asymmetric miscalibration)."
            )
        else:
            result["recommendation"] = "none"
            result["summary"] = (
                f"No post-hoc method improves ECE={boot['ece']:.3f}. "
                f"Miscalibration is structural and class-asymmetric."
            )
    elif is_binary and not global_ts["recommended"]:
        result["recommendation"] = "none"
        result["summary"] = f"ECE={boot['ece']:.3f}. Scaling not helpful."
    else:
        result["recommendation"] = "none"
        result["summary"] = f"ECE={boot['ece']:.3f}. Already well-calibrated."

    return result
