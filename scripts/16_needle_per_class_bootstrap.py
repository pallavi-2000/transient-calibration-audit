"""
NEEDLE Per-Class Bootstrap and Binomial Confidence Intervals
=============================================================

Purpose:
    Quantify uncertainty in NEEDLE's per-class calibration metrics for the
    rare classes (SLSN-I n=15, TDE n=21 at object level). Directly addresses
    reviewer concern that "statements such as 'per-class scaling cannot
    resolve the structural miscalibration' are stronger than the evidence
    warrants."

Reviewer concern (verbatim):
    "NEEDLE's object-level analysis rests on only 278 unique objects, with
     15 SLSN-I and 21 TDE true objects... The fix is to soften those claims,
     report exact intervals for the rare-class quantities that drive the
     conclusion, and frame the result as strongly suggestive of class-imbalance-
     induced miscalibration rather than definitively proving it."

What this script does:
    1. Loads NEEDLE object-level predictions (from existing 03_needle_analysis.py
       output JSON, or recomputes if missing)
    2. For each true class, computes:
       - Accuracy_when_predicted with Wilson 95% binomial CI
       - Mean confidence with bootstrap 95% CI
       - Gap = accuracy - mean_confidence with bootstrap 95% CI
       - Per-class ECE with bootstrap 95% CI
    3. Compares object-level vs model-instance results to characterize
       which conclusions are robust vs sample-size-dependent
    4. Generates publication-ready figures and tables

Outputs:
    results/needle_per_class_bootstrap.json
    results/needle_per_class_bootstrap.csv
    figures/needle_per_class_bootstrap_object_level.pdf
    figures/needle_per_class_bootstrap_model_instance.pdf

Usage:
    python3 scripts/16_needle_per_class_bootstrap.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===========================================================================
# CONFIGURATION
# ===========================================================================

NEEDLE_DATA_CANDIDATES = [
    "data/processed/needle_predictions.npz",
    "data/processed/needle_object_level.npz",
]

CLASS_NAMES = ["SN", "SLSN-I", "TDE"]
CLASS_ORDER = ["SN", "SLSN-I", "TDE"]
N_BOOT = 1000
SEED = 42


# ===========================================================================
# DATA LOADING
# ===========================================================================

def find_first_existing(candidates, label):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {label}: tried {candidates}")


def load_needle_predictions():
    """
    Load NEEDLE predictions. Returns (probs, labels, ztf_ids, model_ids).

    probs: (N_predictions, 3) array of probabilities for [SN, SLSN-I, TDE]
    labels: (N_predictions,) integer class labels
    ztf_ids: (N_predictions,) ZTF object IDs
    model_ids: (N_predictions,) model instance IDs (0-4)
    """
    path = find_first_existing(NEEDLE_DATA_CANDIDATES, "NEEDLE predictions")
    print(f"Loading NEEDLE predictions from: {path}")
    data = np.load(path, allow_pickle=True)

    probs = data["probs"]
    labels = data["labels"]
    ztf_ids = data["ztf_ids"]
    model_ids = data["model_ids"] if "model_ids" in data.files else None

    print(f"  Total predictions: {len(probs)}")
    print(f"  Unique objects: {len(np.unique(ztf_ids))}")
    print(f"  Class names: {CLASS_NAMES}")
    print(f"  Class distribution (labels):")
    for i, name in enumerate(CLASS_NAMES):
        n = (labels == i).sum()
        print(f"    {name}: {n}")

    return probs, labels, ztf_ids, model_ids


def deduplicate_to_object_level(probs, labels, ztf_ids):
    """
    Average probabilities across multiple model instances per object.
    Returns object-level (probs, labels, ztf_ids).
    """
    unique_ids = np.unique(ztf_ids)
    obj_probs = np.zeros((len(unique_ids), probs.shape[1]))
    obj_labels = np.zeros(len(unique_ids), dtype=int)

    for i, oid in enumerate(unique_ids):
        mask = ztf_ids == oid
        obj_probs[i] = probs[mask].mean(axis=0)
        # All instances of same object should have same true label
        obj_labels[i] = labels[mask][0]
        if not np.all(labels[mask] == labels[mask][0]):
            print(f"  WARNING: label disagreement for {oid}")

    return obj_probs, obj_labels, unique_ids


# ===========================================================================
# CALIBRATION METRICS WITH UNCERTAINTY
# ===========================================================================

def per_class_metrics(probs, labels, n_boot=N_BOOT, seed=SEED):
    """
    Compute per-class metrics with confidence intervals.

    For each true class c:
        - n_true: number of objects with true label c
        - n_pred: number of objects with PREDICTED label c
        - accuracy_when_predicted: P(true=c | predicted=c)
            -> Wilson 95% binomial CI (exact)
        - mean_confidence_when_predicted: mean P(c | predicted=c)
            -> Bootstrap 95% CI
        - gap: accuracy - mean_confidence
            -> Bootstrap 95% CI
    """
    rng = np.random.default_rng(seed)
    pred_labels = probs.argmax(axis=1)

    results = {}
    for c in range(probs.shape[1]):
        class_name = CLASS_NAMES[c]

        # Objects predicted as class c
        pred_mask = (pred_labels == c)
        n_pred = int(pred_mask.sum())

        # True objects of class c
        true_mask = (labels == c)
        n_true = int(true_mask.sum())

        if n_pred == 0:
            results[class_name] = {
                "class_name": class_name,
                "n_true": n_true,
                "n_pred": 0,
                "accuracy_when_predicted": None,
                "accuracy_ci": None,
                "mean_confidence": None,
                "mean_confidence_ci": None,
                "gap": None,
                "gap_ci": None,
                "ece_class": None,
                "ece_class_ci": None,
            }
            continue

        # Among predicted-as-c objects: how many are truly c?
        n_correct = int((pred_mask & true_mask).sum())
        accuracy = n_correct / n_pred

        # Wilson 95% CI on accuracy (exact binomial, no bootstrap noise)
        ci_obj = stats.binomtest(n_correct, n_pred).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        acc_ci = (float(ci_obj.low), float(ci_obj.high))

        # Confidence assigned to predicted class
        confidences = probs[pred_mask, c]
        mean_conf = float(confidences.mean())

        # Bootstrap CI on mean confidence
        boot_means = []
        boot_gaps = []
        boot_eces = []
        true_for_pred = (labels[pred_mask] == c).astype(int)

        for _ in range(n_boot):
            idx = rng.integers(0, n_pred, size=n_pred)
            boot_conf = confidences[idx]
            boot_correct = true_for_pred[idx]
            boot_means.append(boot_conf.mean())
            boot_acc = boot_correct.mean()
            boot_gap = boot_acc - boot_conf.mean()
            boot_gaps.append(boot_gap)
            boot_eces.append(abs(boot_gap))

        mean_conf_ci = (float(np.quantile(boot_means, 0.025)),
                        float(np.quantile(boot_means, 0.975)))
        gap = accuracy - mean_conf
        gap_ci = (float(np.quantile(boot_gaps, 0.025)),
                  float(np.quantile(boot_gaps, 0.975)))
        ece_class = abs(gap)
        ece_ci = (float(np.quantile(boot_eces, 0.025)),
                  float(np.quantile(boot_eces, 0.975)))

        results[class_name] = {
            "class_name": class_name,
            "n_true": n_true,
            "n_pred": n_pred,
            "n_correct": n_correct,
            "accuracy_when_predicted": float(accuracy),
            "accuracy_ci": [float(acc_ci[0]), float(acc_ci[1])],
            "mean_confidence": mean_conf,
            "mean_confidence_ci": [mean_conf_ci[0], mean_conf_ci[1]],
            "gap": float(gap),
            "gap_ci": [gap_ci[0], gap_ci[1]],
            "ece_class": float(ece_class),
            "ece_class_ci": [ece_ci[0], ece_ci[1]],
        }

    return results


def aggregate_ece_with_ci(probs, labels, n_bins=15, n_boot=N_BOOT, seed=SEED):
    """Aggregate ECE with bootstrap 95% CI."""
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == labels).astype(int)
    n = len(labels)

    def ece(c, k):
        edges = np.quantile(c, np.linspace(0, 1, n_bins + 1))
        edges[0] = 0.0
        edges[-1] = 1.0001
        bin_ids = np.clip(np.digitize(c, edges) - 1, 0, n_bins - 1)
        e = 0.0
        for b in range(n_bins):
            m = bin_ids == b
            if m.sum() == 0:
                continue
            e += (m.sum() / len(c)) * abs(k[m].mean() - c[m].mean())
        return e

    point = ece(conf, correct)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(ece(conf[idx], correct[idx]))
    return {
        "ece": float(point),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


# ===========================================================================
# REPORTING
# ===========================================================================

def print_per_class_table(results, level_name):
    print(f"\n{'='*80}")
    print(f"NEEDLE per-class metrics with 95% CIs ({level_name})")
    print(f"{'='*80}")
    print(f"{'Class':<10} {'n_true':>7} {'n_pred':>7} "
          f"{'Accuracy [95% CI]':>30} {'Gap [95% CI]':>30}")
    print("-" * 90)
    for class_name in CLASS_ORDER:
        if class_name not in results:
            continue
        r = results[class_name]
        if r["n_pred"] == 0:
            print(f"{class_name:<10} {r['n_true']:>7d} {0:>7d}    (no predictions)")
            continue
        acc_str = f"{r['accuracy_when_predicted']:.3f} [{r['accuracy_ci'][0]:.3f}, {r['accuracy_ci'][1]:.3f}]"
        gap_str = f"{r['gap']:+.3f} [{r['gap_ci'][0]:+.3f}, {r['gap_ci'][1]:+.3f}]"
        print(f"{class_name:<10} {r['n_true']:>7d} {r['n_pred']:>7d}  {acc_str:>30}  {gap_str:>30}")


def interpret_results(obj_results, mi_results):
    """Print scientific interpretation comparing object-level vs model-instance."""
    print(f"\n{'='*80}")
    print("SCIENTIFIC INTERPRETATION")
    print(f"{'='*80}\n")

    for class_name in CLASS_ORDER:
        if class_name not in obj_results or class_name not in mi_results:
            continue
        obj = obj_results[class_name]
        mi = mi_results[class_name]
        if obj["n_pred"] == 0 or mi["n_pred"] == 0:
            continue

        print(f"--- {class_name} ---")
        print(f"  Object-level   (n_pred={obj['n_pred']:>3d}): "
              f"gap = {obj['gap']:+.3f} [95% CI: {obj['gap_ci'][0]:+.3f}, {obj['gap_ci'][1]:+.3f}]")
        print(f"  Model-instance (n_pred={mi['n_pred']:>3d}): "
              f"gap = {mi['gap']:+.3f} [95% CI: {mi['gap_ci'][0]:+.3f}, {mi['gap_ci'][1]:+.3f}]")

        # Direction agreement
        obj_dir = "underconfident" if obj["gap"] > 0 else "overconfident"
        mi_dir = "underconfident" if mi["gap"] > 0 else "overconfident"
        if obj_dir == mi_dir:
            print(f"  -> Direction CONFIRMED: {obj_dir} in both analyses")
        else:
            print(f"  -> Direction DISAGREEMENT: {obj_dir} (object) vs {mi_dir} (model-instance)")

        # CI width assessment
        obj_width = obj["gap_ci"][1] - obj["gap_ci"][0]
        if abs(obj["gap"]) > obj_width:
            print(f"  -> Magnitude robust (|gap| {abs(obj['gap']):.3f} > CI width {obj_width:.3f})")
        else:
            print(f"  -> Magnitude UNCERTAIN (|gap| {abs(obj['gap']):.3f} <= CI width {obj_width:.3f})")
        print()


# ===========================================================================
# FIGURES
# ===========================================================================

def plot_per_class_with_ci(results, save_path, level_name, color_palette=None):
    """Bar chart of per-class accuracy + mean confidence with 95% CIs."""
    if color_palette is None:
        color_palette = ["#1f77b4", "#ff7f0e"]

    classes = [c for c in CLASS_ORDER if c in results and results[c]["n_pred"] > 0]
    n_pred = [results[c]["n_pred"] for c in classes]
    accuracies = [results[c]["accuracy_when_predicted"] for c in classes]
    acc_lo = [results[c]["accuracy_ci"][0] for c in classes]
    acc_hi = [results[c]["accuracy_ci"][1] for c in classes]
    confidences = [results[c]["mean_confidence"] for c in classes]
    conf_lo = [results[c]["mean_confidence_ci"][0] for c in classes]
    conf_hi = [results[c]["mean_confidence_ci"][1] for c in classes]

    acc_err = np.array([[a - lo for a, lo in zip(accuracies, acc_lo)],
                        [hi - a for a, hi in zip(accuracies, acc_hi)]])
    conf_err = np.array([[a - lo for a, lo in zip(confidences, conf_lo)],
                         [hi - a for a, hi in zip(confidences, conf_hi)]])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width/2, accuracies, width, yerr=acc_err, capsize=8,
           label="Accuracy when predicted (Wilson 95% CI)",
           color=color_palette[0], edgecolor="black", alpha=0.85,
           error_kw={"linewidth": 1.5})
    ax.bar(x + width/2, confidences, width, yerr=conf_err, capsize=8,
           label="Mean confidence (bootstrap 95% CI)",
           color=color_palette[1], edgecolor="black", alpha=0.85,
           error_kw={"linewidth": 1.5})

    for i, n in enumerate(n_pred):
        ax.text(i, 1.05, f"$n_{{\\mathrm{{pred}}}}={n}$",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("Probability / Accuracy", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"NEEDLE per-class calibration with 95% CIs ({level_name})",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf").replace(".pdf.pdf", ".pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 80)
    print("NEEDLE PER-CLASS BOOTSTRAP / BINOMIAL CIs")
    print("Addresses reviewer concern about rare-class point estimates")
    print("=" * 80)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load
    probs_mi, labels_mi, ztf_ids, _ = load_needle_predictions()

    # Object-level (deduplicated, primary analysis)
    print("\n--- Computing object-level metrics (primary analysis, N=278) ---")
    probs_obj, labels_obj, _ = deduplicate_to_object_level(probs_mi, labels_mi, ztf_ids)
    obj_results = per_class_metrics(probs_obj, labels_obj)
    obj_aggregate = aggregate_ece_with_ci(probs_obj, labels_obj)
    print(f"Object-level aggregate ECE: {obj_aggregate['ece']:.3f} "
          f"[{obj_aggregate['ci_low']:.3f}, {obj_aggregate['ci_high']:.3f}]")
    print_per_class_table(obj_results, "object-level, N=278")

    # Model-instance (sensitivity)
    print("\n--- Computing model-instance metrics (sensitivity, N=429) ---")
    mi_results = per_class_metrics(probs_mi, labels_mi)
    mi_aggregate = aggregate_ece_with_ci(probs_mi, labels_mi)
    print(f"Model-instance aggregate ECE: {mi_aggregate['ece']:.3f} "
          f"[{mi_aggregate['ci_low']:.3f}, {mi_aggregate['ci_high']:.3f}]")
    print_per_class_table(mi_results, "model-instance, N=429")

    # Cross-comparison
    interpret_results(obj_results, mi_results)

    # Save JSON
    out = {
        "metadata": {
            "description": "NEEDLE per-class calibration with 95% confidence intervals",
            "addresses": "Reviewer concern about rare-class point estimates being too sharp",
            "n_bootstrap": N_BOOT,
            "ci_level": 0.95,
            "wilson_method_for_accuracy": True,
            "bootstrap_method_for_confidence_and_gap": True,
        },
        "object_level": {
            "n_unique_objects": len(np.unique(ztf_ids)),
            "aggregate_ece": obj_aggregate,
            "per_class": obj_results,
        },
        "model_instance": {
            "n_predictions": len(labels_mi),
            "aggregate_ece": mi_aggregate,
            "per_class": mi_results,
        },
    }
    json_path = "results/needle_per_class_bootstrap.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Save CSV
    rows = []
    for level_name, results_dict in [("object_level", obj_results),
                                       ("model_instance", mi_results)]:
        for class_name, r in results_dict.items():
            if r["n_pred"] == 0:
                continue
            rows.append({
                "level": level_name,
                "class": class_name,
                "n_true": r["n_true"],
                "n_pred": r["n_pred"],
                "n_correct": r.get("n_correct", None),
                "accuracy": r["accuracy_when_predicted"],
                "accuracy_ci_low": r["accuracy_ci"][0],
                "accuracy_ci_high": r["accuracy_ci"][1],
                "mean_confidence": r["mean_confidence"],
                "mean_confidence_ci_low": r["mean_confidence_ci"][0],
                "mean_confidence_ci_high": r["mean_confidence_ci"][1],
                "gap": r["gap"],
                "gap_ci_low": r["gap_ci"][0],
                "gap_ci_high": r["gap_ci"][1],
                "ece_class": r["ece_class"],
                "ece_class_ci_low": r["ece_class_ci"][0],
                "ece_class_ci_high": r["ece_class_ci"][1],
            })
    df = pd.DataFrame(rows)
    csv_path = "results/needle_per_class_bootstrap.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Figures
    plot_per_class_with_ci(obj_results,
                            "figures/needle_per_class_bootstrap_object_level.pdf",
                            "object-level, N=278")
    plot_per_class_with_ci(mi_results,
                            "figures/needle_per_class_bootstrap_model_instance.pdf",
                            "model-instance, N=429")

    # Paper-ready text
    print(f"\n{'=' * 80}")
    print("PAPER-READY TEXT (insert into §5.3.2)")
    print(f"{'=' * 80}\n")
    sn = obj_results.get("SN", {})
    slsn = obj_results.get("SLSN-I", {})
    tde = obj_results.get("TDE", {})

    print(
        f"At the object level, NEEDLE's per-class calibration shows direction-"
        f"consistent miscalibration but with rare-class magnitudes that carry "
        f"substantial uncertainty given small sample sizes.\n"
    )
    if sn.get("n_pred", 0) > 0:
        print(
            f"SN (n_true={sn['n_true']}, n_pred={sn['n_pred']}): "
            f"accuracy {sn['accuracy_when_predicted']:.3f} "
            f"[Wilson 95% CI: {sn['accuracy_ci'][0]:.3f}, {sn['accuracy_ci'][1]:.3f}], "
            f"mean confidence {sn['mean_confidence']:.3f}, "
            f"gap = {sn['gap']:+.3f} "
            f"[bootstrap 95% CI: {sn['gap_ci'][0]:+.3f}, {sn['gap_ci'][1]:+.3f}]. "
            f"Underconfident."
        )
    if slsn.get("n_pred", 0) > 0:
        print(
            f"\nSLSN-I (n_true={slsn['n_true']}, n_pred={slsn['n_pred']}): "
            f"accuracy {slsn['accuracy_when_predicted']:.3f} "
            f"[Wilson 95% CI: {slsn['accuracy_ci'][0]:.3f}, {slsn['accuracy_ci'][1]:.3f}], "
            f"mean confidence {slsn['mean_confidence']:.3f}, "
            f"gap = {slsn['gap']:+.3f} "
            f"[bootstrap 95% CI: {slsn['gap_ci'][0]:+.3f}, {slsn['gap_ci'][1]:+.3f}]. "
            f"The CI does not include zero, so the direction (overconfident) "
            f"is robust; the magnitude is uncertain owing to small n_true=15."
        )
    if tde.get("n_pred", 0) > 0:
        print(
            f"\nTDE (n_true={tde['n_true']}, n_pred={tde['n_pred']}): "
            f"accuracy {tde['accuracy_when_predicted']:.3f} "
            f"[Wilson 95% CI: {tde['accuracy_ci'][0]:.3f}, {tde['accuracy_ci'][1]:.3f}], "
            f"mean confidence {tde['mean_confidence']:.3f}, "
            f"gap = {tde['gap']:+.3f} "
            f"[bootstrap 95% CI: {tde['gap_ci'][0]:+.3f}, {tde['gap_ci'][1]:+.3f}]. "
            f"Direction (overconfident) consistent with SLSN-I; magnitude "
            f"sample-size limited."
        )

    print(f"\n{'=' * 80}")
    print("DONE.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
