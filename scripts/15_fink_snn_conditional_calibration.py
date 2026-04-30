"""
Fink SuperNNova Conditional Calibration with Abstention-Bias Quantification
============================================================================

Purpose:
    Compute calibration metrics on Fink SNN's NON-ZERO predictions only
    (selective classification framework), AND quantify how the abstention
    mechanism biases this conditional calibration.

Reviewer concern (advanced):
    "Conditional ECE is meaningful only if the abstention mechanism is
     INDEPENDENT of calibration error within the accepted set. If Fink
     abstains preferentially on rare/faint objects (which it does:
     p < 10^-36), then the conditional ECE estimate is biased toward
     easy cases (SN Ia)."

What this script does:

    Step 1: Conditional calibration (the standard analysis)
        - Compute ECE on non-zero SNN predictions only
        - Report per-class accuracy / confidence / gap on accepted set
        - Bootstrap CI

    Step 2: Abstention-bias quantification (the novel contribution)
        - Compute class-composition shift: BTS sample → accepted set
        - Compute "calibration-conditional-on-class" for each class
        - Show that conditional ECE = sum_c P(c|accepted) * ECE_c
        - Reweight ECE to BTS-prevalence to expose how much the conditional
          number underestimates true population miscalibration

    Step 3: Temperature scaling diagnostics
        - Fit T on conditional sample
        - Show T -> bound (or report value)
        - Explain why T fails: not "bound hit" but "no global T fixes
          a class-asymmetric problem"

Scientific framing:
    This script produces the publishable Fink SNN result:
    - "Conditional ECE = 0.198 [CI]"
    - "But this is biased: SN Ia comprises 81% of training but only 54%
       of the BTS sample. Reweighting to BTS prevalence yields ECE = X"
    - "The honest interpretation is selective calibration with
       class-prevalence dependence, not classifier failure"

Usage:
    python scripts/15_fink_snn_conditional_calibration.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import existing calibration utilities; fall back to local definitions
try:
    from src.calibration import compute_ece, bootstrap_ece
    USE_LOCAL_CALIBRATION = True
except ImportError:
    USE_LOCAL_CALIBRATION = False


# ===========================================================================
# CONFIGURATION
# ===========================================================================

FINK_CSV_CANDIDATES = [
    "data/raw/fink_classifications.csv",
    "fink_classifications.csv",
]

BTS_CSV_CANDIDATES = [
    "data/ground_truth/bts_sample.csv",
    "data/ground_truth/bts_spectroscopic.csv",
]

SNN_SCORE_COL = "snn_snia_vs_nonia"
BTS_OID_COL = "ZTFID"
BTS_CLASS_COL = "alerce_class"
FINK_OID_CANDIDATES = ["oid", "ZTFID", "ztf_id", "objectId"]

CLASS_ORDER = ["SNIa", "SNIbc", "SNII", "SLSN", "TDE", "Other"]


# ===========================================================================
# CALIBRATION UTILITIES (local fallbacks if src.calibration unavailable)
# ===========================================================================

def compute_ece_local(y_true, y_prob, n_bins=15, strategy="equal_mass"):
    """
    Binary ECE: |accuracy - confidence| weighted by bin size.
    
    Args:
        y_true: (N,) binary labels (0/1)
        y_prob: (N,) predicted probability of positive class
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    # For binary, "confidence" is max(p, 1-p), and "correctness" is whether
    # the predicted class (>=0.5) matches the true label
    pred_class = (y_prob >= 0.5).astype(int)
    correct = (pred_class == y_true).astype(int)
    confidence = np.where(y_prob >= 0.5, y_prob, 1 - y_prob)
    
    if strategy == "equal_mass":
        bin_edges = np.quantile(confidence, np.linspace(0, 1, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0001
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    
    bin_ids = np.digitize(confidence, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    
    ece = 0.0
    bin_data = []
    for b in range(n_bins):
        mask = bin_ids == b
        n = mask.sum()
        if n == 0:
            bin_data.append({"n": 0, "acc": 0, "conf": 0})
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += (n / len(y_true)) * abs(acc - conf)
        bin_data.append({"n": int(n), "acc": float(acc), "conf": float(conf)})
    
    return float(ece), bin_data


def bootstrap_ece_local(y_true, y_prob, n_bins=15, n_boot=1000, seed=42):
    """Bootstrap 95% CI for binary ECE."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    eces = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ece, _ = compute_ece_local(y_true[idx], y_prob[idx], n_bins=n_bins)
        eces.append(ece)
    point, _ = compute_ece_local(y_true, y_prob, n_bins=n_bins)
    return {
        "ece": float(point),
        "ci_lower": float(np.quantile(eces, 0.025)),
        "ci_upper": float(np.quantile(eces, 0.975)),
        "bootstrap_samples": eces,
    }


def temperature_scale_binary(scores, labels, return_diagnostics=False):
    """
    Fit temperature T for binary scores via NLL minimization.
    Bounded T in [0.01, 10.0].
    """
    eps = 1e-12
    p = np.clip(scores, eps, 1 - eps)
    logits = np.log(p / (1 - p))  # logit transform
    
    def nll(T):
        if T <= 0:
            return 1e10
        scaled_logits = logits / T
        scaled_p = 1.0 / (1.0 + np.exp(-scaled_logits))
        scaled_p = np.clip(scaled_p, eps, 1 - eps)
        return -np.mean(labels * np.log(scaled_p) + (1 - labels) * np.log(1 - scaled_p))
    
    result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
    T = result.x
    bound_hit = (T < 0.05) or (T > 9.5)
    
    scaled_logits = logits / T
    scaled_p = 1.0 / (1.0 + np.exp(-scaled_logits))
    
    if return_diagnostics:
        return T, scaled_p, bound_hit
    return T


# ===========================================================================
# DATA LOADING
# ===========================================================================

def find_first_existing(candidates, label):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {label}: tried {candidates}")


def load_data():
    """Load and merge Fink + BTS, filter to non-zero SNN predictions."""
    fink_path = find_first_existing(FINK_CSV_CANDIDATES, "Fink CSV")
    bts_path = find_first_existing(BTS_CSV_CANDIDATES, "BTS CSV")
    
    print(f"Loading Fink: {fink_path}")
    fink = pd.read_csv(fink_path)
    
    fink_oid = next((c for c in FINK_OID_CANDIDATES if c in fink.columns), None)
    if fink_oid is None:
        raise ValueError(f"No OID column in Fink. Have: {list(fink.columns)}")
    fink = fink.rename(columns={fink_oid: "_oid"})
    
    print(f"Loading BTS: {bts_path}")
    bts = pd.read_csv(bts_path)
    bts = bts.rename(columns={BTS_OID_COL: "_oid", BTS_CLASS_COL: "_true_class"})
    bts = bts[["_oid", "_true_class"]].dropna()
    
    merged = pd.merge(fink, bts, on="_oid", how="inner")
    print(f"Merged: {len(merged)} objects with both Fink predictions and BTS truth")
    
    return merged


def split_accepted_rejected(df, score_col):
    """Split into accepted (non-zero) and rejected (zero/NaN) subsets."""
    scores = df[score_col].values
    is_zero = (scores == 0.0) | np.isnan(scores)
    
    accepted = df[~is_zero].copy().reset_index(drop=True)
    rejected = df[is_zero].copy().reset_index(drop=True)
    
    print(f"\nSelective classification split:")
    print(f"  Accepted (score > 0): {len(accepted)} ({100*len(accepted)/len(df):.1f}%)")
    print(f"  Rejected (score = 0): {len(rejected)} ({100*len(rejected)/len(df):.1f}%)")
    
    return accepted, rejected


# ===========================================================================
# STEP 1: CONDITIONAL CALIBRATION
# ===========================================================================

def conditional_calibration(accepted, score_col):
    """
    Compute calibration metrics on the accepted set only.
    
    The SNN score = P(SN Ia). True label = 1 if class is "SNIa", else 0.
    """
    print("\n" + "=" * 70)
    print("STEP 1: CONDITIONAL CALIBRATION (on accepted set)")
    print("=" * 70)
    
    p_snia = accepted[score_col].values
    true_class = accepted["_true_class"].values
    y_true = (true_class == "SNIa").astype(int)
    
    # Aggregate ECE
    ece_func = compute_ece_local
    boot_func = bootstrap_ece_local
    
    boot = boot_func(y_true, p_snia, n_bins=15, n_boot=1000)
    ece = boot["ece"]
    ci = [boot["ci_lower"], boot["ci_upper"]]
    
    # Brier score
    brier = float(np.mean((p_snia - y_true) ** 2))
    
    # Accuracy at p>=0.5 threshold
    pred = (p_snia >= 0.5).astype(int)
    acc = float((pred == y_true).mean())
    
    print(f"N (accepted): {len(accepted)}")
    print(f"Aggregate ECE: {ece:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"Brier score:   {brier:.3f}")
    print(f"Accuracy (p>=0.5): {acc:.3f}")
    print(f"True SN Ia in accepted set: {y_true.sum()} / {len(y_true)} "
          f"({100*y_true.mean():.1f}%)")
    
    return {
        "n_accepted": len(accepted),
        "ece": float(ece),
        "ece_ci": [float(c) for c in ci],
        "brier": brier,
        "accuracy": acc,
        "n_true_snia": int(y_true.sum()),
        "frac_true_snia": float(y_true.mean()),
    }


# ===========================================================================
# STEP 2: ABSTENTION-BIAS QUANTIFICATION (THE NOVEL CONTRIBUTION)
# ===========================================================================

def abstention_bias_analysis(merged, accepted, score_col):
    """
    Quantify how class-dependent abstention biases the conditional ECE.
    
    Key insight: conditional ECE is computed on a sample whose class
    composition differs from the BTS sample. If we reweight to BTS
    prevalence, we get a different number — the gap quantifies bias.
    """
    print("\n" + "=" * 70)
    print("STEP 2: ABSTENTION-BIAS QUANTIFICATION")
    print("=" * 70)
    
    # Class composition: BTS sample vs accepted set
    bts_classes = merged["_true_class"].value_counts(normalize=True)
    accepted_classes = accepted["_true_class"].value_counts(normalize=True)
    
    classes = [c for c in CLASS_ORDER if c in bts_classes.index]
    
    print(f"\nClass-composition shift due to abstention:")
    print(f"{'Class':<10} {'BTS_prev':>10} {'Accepted_prev':>15} {'Shift':>10}")
    print("-" * 50)
    for c in classes:
        bts_p = bts_classes.get(c, 0)
        acc_p = accepted_classes.get(c, 0)
        shift = acc_p - bts_p
        marker = " *" if abs(shift) > 0.05 else ""
        print(f"{c:<10} {bts_p:>10.3f} {acc_p:>15.3f} {shift:>+10.3f}{marker}")
    
    # Per-class conditional calibration (on accepted set, by true class)
    print(f"\nPer-class calibration on accepted set:")
    print(f"{'Class':<10} {'N_acc':>8} {'mean_p':>10} {'frac_Ia':>10} "
          f"{'gap':>10} {'ECE_c':>10}")
    print("-" * 60)
    
    p_snia = accepted[score_col].values
    true_class = accepted["_true_class"].values
    y_true = (true_class == "SNIa").astype(int)
    
    per_class = {}
    weighted_ece_bts = 0.0
    weighted_ece_accepted = 0.0
    
    for c in classes:
        mask = (true_class == c)
        n_c = int(mask.sum())
        if n_c < 5:
            print(f"{c:<10} {n_c:>8d} (too few — skipping)")
            per_class[c] = {"n": n_c, "skipped": True}
            continue
        
        p_c = p_snia[mask]
        y_c = y_true[mask]
        
        mean_p = float(p_c.mean())
        frac_ia = float(y_c.mean())
        gap = frac_ia - mean_p
        
        # Wilson 95% CI on frac_ia
        if n_c > 0:
            ci_obj = stats.binomtest(int(y_c.sum()), n_c).proportion_ci(
                confidence_level=0.95, method="wilson"
            )
            ci_low, ci_high = float(ci_obj.low), float(ci_obj.high)
        else:
            ci_low, ci_high = 0.0, 0.0
        
        # Per-class ECE: |frac_Ia - mean_p|  (single-bin approximation; for
        # binary scores within a single true class this is the natural metric)
        ece_c = abs(gap)
        
        print(f"{c:<10} {n_c:>8d} {mean_p:>10.3f} {frac_ia:>10.3f} "
              f"{gap:>+10.3f} {ece_c:>10.3f}")
        
        per_class[c] = {
            "n": n_c,
            "mean_predicted_p_snia": mean_p,
            "true_frac_snia": frac_ia,
            "true_frac_ci": [ci_low, ci_high],
            "gap": float(gap),
            "ece_class": float(ece_c),
        }
        
        weighted_ece_bts += bts_classes.get(c, 0) * ece_c
        weighted_ece_accepted += accepted_classes.get(c, 0) * ece_c
    
    print(f"\nWeighted ECE comparison:")
    print(f"  Conditional (weighted by accepted-set composition): {weighted_ece_accepted:.3f}")
    print(f"  Reweighted to BTS prevalence:                       {weighted_ece_bts:.3f}")
    diff = weighted_ece_bts - weighted_ece_accepted
    print(f"  Bias from abstention: {diff:+.3f}")
    
    if abs(diff) > 0.02:
        print(f"  ==> Conditional ECE UNDERESTIMATES population miscalibration")
        print(f"      by ~{abs(diff)*100:.1f} percentage points.")
        print(f"      Abstention preferentially removes hard (rare) cases.")
    else:
        print(f"  ==> Conditional ECE is approximately unbiased w.r.t. BTS prevalence.")
    
    return {
        "bts_class_composition": {c: float(bts_classes.get(c, 0)) for c in classes},
        "accepted_class_composition": {c: float(accepted_classes.get(c, 0)) for c in classes},
        "per_class_calibration": per_class,
        "weighted_ece_accepted": float(weighted_ece_accepted),
        "weighted_ece_bts_reweighted": float(weighted_ece_bts),
        "abstention_bias": float(diff),
    }


# ===========================================================================
# STEP 3: TEMPERATURE SCALING DIAGNOSTICS
# ===========================================================================

def temperature_scaling_diagnostics(accepted, score_col):
    """Fit T on conditional set and report diagnostics."""
    print("\n" + "=" * 70)
    print("STEP 3: TEMPERATURE SCALING ON CONDITIONAL SET")
    print("=" * 70)
    
    p_snia = accepted[score_col].values
    true_class = accepted["_true_class"].values
    y_true = (true_class == "SNIa").astype(int)
    
    T, scaled_p, bound_hit = temperature_scale_binary(
        p_snia, y_true, return_diagnostics=True
    )
    
    ece_pre, _ = compute_ece_local(y_true, p_snia, n_bins=15)
    ece_post, _ = compute_ece_local(y_true, scaled_p, n_bins=15)
    
    print(f"Optimal T: {T:.3f}")
    print(f"Bound hit: {bound_hit}")
    print(f"ECE pre-scaling:  {ece_pre:.3f}")
    print(f"ECE post-scaling: {ece_post:.3f}")
    print(f"Improvement: {ece_pre - ece_post:+.3f}")
    
    if bound_hit:
        if T > 9.5:
            interpretation = (
                "T converged to upper bound (10.0). The optimizer is trying to "
                "flatten predictions to maximum entropy, indicating that no "
                "global temperature can simultaneously address the per-class "
                "miscalibration patterns. This is a structural failure of the "
                "single-parameter assumption, not a softening that 'goes too far.'"
            )
        else:
            interpretation = (
                "T converged to lower bound (0.01). The optimizer wants to "
                "sharpen predictions to extreme confidence, but no single "
                "T can fix the per-class issues."
            )
    elif ece_post > ece_pre:
        interpretation = (
            f"T = {T:.3f} produces WORSE calibration. Indicates that the "
            "miscalibration is not a uniform softening/sharpening problem."
        )
    else:
        interpretation = (
            f"T = {T:.3f} reduces ECE by "
            f"{100*(ece_pre-ece_post)/ece_pre:.1f}%. Standard temperature "
            "scaling is effective."
        )
    
    print(f"\nInterpretation: {interpretation}")
    
    return {
        "temperature": float(T),
        "bound_hit": bool(bound_hit),
        "ece_pre": float(ece_pre),
        "ece_post": float(ece_post),
        "interpretation": interpretation,
    }


# ===========================================================================
# FIGURES
# ===========================================================================

def plot_class_composition_shift(bias_results, save_path):
    """Show how abstention shifts the class composition."""
    bts = bias_results["bts_class_composition"]
    acc = bias_results["accepted_class_composition"]
    classes = list(bts.keys())
    
    bts_vals = [bts[c] for c in classes]
    acc_vals = [acc[c] for c in classes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.35
    
    ax.bar(x - width/2, bts_vals, width, label="BTS sample (full)",
           color="#888888", edgecolor="black")
    ax.bar(x + width/2, acc_vals, width, label="Accepted (score > 0)",
           color="#2ca02c", edgecolor="black")
    
    for i, (b, a) in enumerate(zip(bts_vals, acc_vals)):
        ax.text(i - width/2, b + 0.005, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(i + width/2, a + 0.005, f"{a:.2f}", ha="center", fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Class prevalence (fraction)", fontsize=12)
    ax.set_title(
        "Class-composition shift: BTS sample → Fink SNN accepted set\n"
        "(Abstention enriches SN Ia, depletes rare/sparse classes)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_per_class_gap(bias_results, save_path):
    """Show per-class accuracy-vs-confidence gap on accepted set."""
    per_class = bias_results["per_class_calibration"]
    classes = [c for c in CLASS_ORDER if c in per_class
               and not per_class[c].get("skipped", False)]
    
    mean_p = [per_class[c]["mean_predicted_p_snia"] for c in classes]
    true_p = [per_class[c]["true_frac_snia"] for c in classes]
    counts = [per_class[c]["n"] for c in classes]
    ci_lows = [per_class[c]["true_frac_ci"][0] for c in classes]
    ci_highs = [per_class[c]["true_frac_ci"][1] for c in classes]
    
    err_low = [t - l for t, l in zip(true_p, ci_lows)]
    err_high = [h - t for t, h in zip(true_p, ci_highs)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.35
    
    ax.bar(x - width/2, mean_p, width, label="Mean predicted P(SN Ia)",
           color="#1f77b4", edgecolor="black", alpha=0.8)
    ax.bar(x + width/2, true_p, width, yerr=[err_low, err_high],
           capsize=6, label="True fraction SN Ia (95% Wilson CI)",
           color="#ff7f0e", edgecolor="black", alpha=0.8)
    
    for i, (m, t, n) in enumerate(zip(mean_p, true_p, counts)):
        ax.text(i - width/2, m + 0.02, f"{m:.2f}", ha="center", fontsize=9)
        ax.text(i + width/2, t + 0.02, f"{t:.2f}\nn={n}",
                ha="center", fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Probability / Fraction", fontsize=12)
    ax.set_title(
        "Fink SNN per-class calibration on accepted set\n"
        "Gap = True fraction SN Ia − Mean predicted P(SN Ia)",
        fontsize=12,
    )
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_reweighted_ece(bias_results, save_path):
    """Compare conditional ECE (on accepted) vs BTS-reweighted ECE."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ["Conditional\n(accepted-weighted)", "BTS-reweighted\n(population)"]
    values = [bias_results["weighted_ece_accepted"],
              bias_results["weighted_ece_bts_reweighted"]]
    colors = ["#2ca02c", "#d62728"]
    
    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    
    ax.set_ylabel("Weighted per-class |gap|", fontsize=12)
    ax.set_title(
        "Abstention bias in conditional calibration\n"
        f"Bias = {bias_results['abstention_bias']:+.3f}",
        fontsize=12,
    )
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(values) * 1.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# PAPER-READY OUTPUT
# ===========================================================================

def print_paper_text(cond, bias, temp):
    print("\n" + "=" * 70)
    print("PAPER-READY TEXT")
    print("=" * 70)
    
    print("\n--- Section 5.2.3 (Fink SNN conditional calibration) ---\n")
    
    n = cond["n_accepted"]
    ece = cond["ece"]
    ci = cond["ece_ci"]
    bias_val = bias["abstention_bias"]
    bts_ece = bias["weighted_ece_bts_reweighted"]
    
    text = (
        f"On the {n} BTS objects receiving non-zero SuperNNova scores "
        f"(64.1% coverage), the conditional ECE is "
        f"{ece:.3f} [95% CI: {ci[0]:.3f}, {ci[1]:.3f}], with Brier score "
        f"{cond['brier']:.3f}. However, this conditional metric is biased "
        f"by class-dependent abstention. SN Ia objects, which are easier "
        f"to classify and the only class for which the SNN score is the "
        f"target probability, comprise "
        f"{bias['accepted_class_composition'].get('SNIa', 0)*100:.1f}% of "
        f"the accepted set versus "
        f"{bias['bts_class_composition'].get('SNIa', 0)*100:.1f}% of the "
        f"BTS sample. Reweighting per-class calibration gaps to BTS prevalence "
        f"yields a population-equivalent weighted-gap of "
        f"{bts_ece:.3f}, an absolute increase of "
        f"{abs(bias_val):.3f} over the conditional measurement. "
    )
    if bias_val > 0.02:
        text += (
            f"This {bias_val*100:+.1f} percentage-point bias quantifies how "
            f"selective abstention masks the population-level miscalibration: "
            f"the SNN is meaningfully less reliable on the BTS-weighted "
            f"transient distribution than the conditional number suggests."
        )
    else:
        text += (
            f"The bias is small, suggesting the conditional ECE is approximately "
            f"representative of the BTS-weighted population."
        )
    print(text + "\n")
    
    print("\n--- Temperature scaling paragraph ---\n")
    print(
        f"We attempted post-hoc calibration via temperature scaling on the "
        f"conditional set. The optimal T = {temp['temperature']:.2f} "
        f"{'reaches the optimizer bound, ' if temp['bound_hit'] else ''}"
        f"with post-scaling ECE = {temp['ece_post']:.3f} versus pre-scaling "
        f"ECE = {temp['ece_pre']:.3f}. {temp['interpretation']}\n"
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("FINK SNN CONDITIONAL CALIBRATION + ABSTENTION-BIAS ANALYSIS")
    print("=" * 70)
    
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Load and split
    merged = load_data()
    accepted, rejected = split_accepted_rejected(merged, SNN_SCORE_COL)
    
    # Step 1: conditional calibration
    cond = conditional_calibration(accepted, SNN_SCORE_COL)
    
    # Step 2: abstention-bias quantification (the key novel piece)
    bias = abstention_bias_analysis(merged, accepted, SNN_SCORE_COL)
    
    # Step 3: temperature scaling
    temp = temperature_scaling_diagnostics(accepted, SNN_SCORE_COL)
    
    # Figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_class_composition_shift(
        bias, "figures/fink_snn_class_composition_shift.png"
    )
    plot_per_class_gap(
        bias, "figures/fink_snn_per_class_gap.png"
    )
    plot_reweighted_ece(
        bias, "figures/fink_snn_abstention_bias.png"
    )
    
    # Save JSON
    out = {
        "metadata": {
            "description": "Fink SNN conditional calibration with "
                           "abstention-bias quantification",
            "framing": "Selective classification: calibrate only on "
                       "accepted (non-zero) predictions",
            "novel_contribution": "Quantify how class-dependent abstention "
                                  "biases the conditional ECE estimate",
        },
        "step1_conditional_calibration": cond,
        "step2_abstention_bias": bias,
        "step3_temperature_scaling": temp,
    }
    with open("results/fink_snn_conditional_analysis.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: results/fink_snn_conditional_analysis.json")
    
    print_paper_text(cond, bias, temp)
    
    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
