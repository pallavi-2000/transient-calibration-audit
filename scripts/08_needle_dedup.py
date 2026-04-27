"""
NEEDLE Deduplication Analysis (Revision Step 1.1)
===================================================

Addresses reviewer concern: "NEEDLE duplication problem"
    The current main table uses 429 predictions across only 278 unique objects
    (5 models x their respective held-out test sets). Reviewer recommends
    promoting object-level (deduplicated) results to main, with model-instance
    as sensitivity analysis.

This script:
  1. Loads NEEDLE predictions from data/processed/needle_predictions.npz
  2. Deduplicates to object-level via mean-of-available-predictions
  3. Computes ECE, per-class ECE, silent failures on DEDUPLICATED set
  4. Compares to original (model-instance) results
  5. Generates new reliability diagrams labeled "object-level"
  6. Outputs JSON with canonical numbers for paper

Data format expected (from .npz):
    probs:       shape (N, K)  float32  - class probabilities
    labels:      shape (N,)    int64    - true class indices
    model_ids:   shape (N,)    int32    - which of 5 models
    ztf_ids:     shape (N,)    str      - ZTF object identifiers
    class_names: shape (K,)    str      - class name strings

Usage:
    python scripts/08_needle_dedup.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import compute_ece, bootstrap_ece
from src.plotting import reliability_diagram


def load_needle_predictions(path="data/processed/needle_predictions.npz"):
    """Load NEEDLE predictions from .npz file."""
    if not os.path.exists(path):
        for alt in [
            "/Users/pallavisati/Desktop/transient-calibration-audit/data/processed/needle_predictions.npz",
            "data/needle_predictions.npz",
            "results/needle_predictions.npz",
        ]:
            if os.path.exists(alt):
                path = alt
                break
        else:
            raise FileNotFoundError("Could not find NEEDLE predictions .npz file.")

    print(f"Loading from: {path}")
    data = np.load(path, allow_pickle=True)

    result = {
        "probs": data["probs"].astype(np.float64),
        "labels": data["labels"].astype(int),
        "model_ids": data["model_ids"].astype(int),
        "ztf_ids": data["ztf_ids"].astype(str),
        "class_names": data["class_names"].astype(str),
    }

    N, K = result["probs"].shape
    print(f"  N predictions: {N}")
    print(f"  K classes: {K} ({', '.join(result['class_names'])})")
    print(f"  Unique objects: {len(np.unique(result['ztf_ids']))}")
    print(f"  Models: {sorted(np.unique(result['model_ids']))}")

    print(f"  True-class distribution:")
    for k, cname in enumerate(result["class_names"]):
        n_k = (result["labels"] == k).sum()
        print(f"    {cname}: {n_k} ({100*n_k/N:.1f}%)")

    return result


def deduplicate_by_object(data, method="mean"):
    """Collapse model-instance predictions to object-level via mean + renorm."""
    ztf_ids = data["ztf_ids"]
    probs = data["probs"]
    labels = data["labels"]
    K = probs.shape[1]

    unique_ids = np.unique(ztf_ids)
    N_unique = len(unique_ids)

    print(f"\nDeduplication: {len(ztf_ids)} predictions -> {N_unique} unique objects")

    dedup_probs = np.zeros((N_unique, K))
    dedup_labels = np.zeros(N_unique, dtype=int)
    dedup_n_models = np.zeros(N_unique, dtype=int)

    for i, ztfid in enumerate(unique_ids):
        mask = (ztf_ids == ztfid)
        dedup_probs[i] = probs[mask].mean(axis=0)
        dedup_labels[i] = labels[mask][0]
        dedup_n_models[i] = mask.sum()

        if len(np.unique(labels[mask])) > 1:
            print(f"  WARNING: {ztfid} has inconsistent labels across models")

    # Renormalize
    row_sums = dedup_probs.sum(axis=1, keepdims=True)
    dedup_probs = dedup_probs / row_sums

    n_with_1 = (dedup_n_models == 1).sum()
    n_with_2plus = (dedup_n_models >= 2).sum()
    print(f"  Objects with 1 model prediction:   {n_with_1}")
    print(f"  Objects with 2+ model predictions: {n_with_2plus}")
    print(f"  Mean models per object: {dedup_n_models.mean():.2f}")

    return {
        "probs": dedup_probs,
        "labels": dedup_labels,
        "ztf_ids": unique_ids,
        "class_names": data["class_names"],
        "n_models_per_object": dedup_n_models,
    }


def compute_metrics(data, label=""):
    """Compute aggregate ECE, per-class ECE, Brier, silent failures."""
    probs = data["probs"]
    labels = data["labels"]
    class_names = data["class_names"]
    ztf_ids = data["ztf_ids"]

    pred_class = probs.argmax(axis=1)
    pred_confidence = probs.max(axis=1)
    correct = (pred_class == labels).astype(int)

    # Use 2D (labels, probs) path for both point estimate and bootstrap CI.
    # The 1D (correct, pred_confidence) path inverts predictions when max_conf < 0.5
    # (possible after probability averaging in dedup), producing inconsistent results
    # where the CI bounds don't bracket the point estimate.
    boot = bootstrap_ece(labels, probs, n_bins=15)
    ece_agg = boot["ece"]
    ece_ci = [boot["ci_lower"], boot["ci_upper"]]

    N, K = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1
    brier = ((probs - one_hot) ** 2).sum(axis=1).mean()

    per_class = {}
    for k, cname in enumerate(class_names):
        class_prob = probs[:, k]
        class_true = (labels == k).astype(int)
        ece_k, _ = compute_ece(class_true, class_prob, n_bins=15, strategy="equal_mass")

        mask_pred_k = (pred_class == k)
        if mask_pred_k.sum() > 0:
            acc_when_pred = (labels[mask_pred_k] == k).mean()
            conf_when_pred = probs[mask_pred_k, k].mean()
        else:
            acc_when_pred = None
            conf_when_pred = None

        per_class[cname] = {
            "ece": float(ece_k),
            "accuracy_when_predicted": float(acc_when_pred) if acc_when_pred is not None else None,
            "mean_confidence_when_predicted": float(conf_when_pred) if conf_when_pred is not None else None,
            "gap": float(acc_when_pred - conf_when_pred) if acc_when_pred is not None else None,
            "n_true": int((labels == k).sum()),
            "n_predicted": int(mask_pred_k.sum()),
        }

    # Silent failures
    silent_mask = (pred_confidence >= 0.90) & (correct == 0)
    n_silent = int(silent_mask.sum())
    silent_objects = []
    if n_silent > 0:
        for i in np.where(silent_mask)[0]:
            silent_objects.append({
                "ztf_id": str(ztf_ids[i]),
                "predicted_class": str(class_names[pred_class[i]]),
                "true_class": str(class_names[labels[i]]),
                "confidence": float(pred_confidence[i]),
            })

    print(f"\n--- Metrics: {label} ---")
    print(f"N objects: {N}")
    print(f"Overall accuracy: {correct.mean():.3f}")
    print(f"Aggregate ECE: {ece_agg:.3f} [{ece_ci[0]:.3f}, {ece_ci[1]:.3f}]")
    print(f"Brier score: {brier:.3f}")
    print(f"Silent failures (>=90% conf, wrong): {n_silent}")
    print(f"Per-class:")
    for cname, m in per_class.items():
        if m["gap"] is not None:
            direction = "underconfident" if m["gap"] > 0 else "overconfident"
            print(f"  {cname}: n_true={m['n_true']}, n_pred={m['n_predicted']}, "
                  f"ECE={m['ece']:.3f}, acc={m['accuracy_when_predicted']:.3f}, "
                  f"conf={m['mean_confidence_when_predicted']:.3f}, "
                  f"gap={m['gap']:+.3f} ({direction})")
        else:
            print(f"  {cname}: n_true={m['n_true']}, n_pred=0 (never predicted)")

    return {
        "label": label,
        "n_objects": int(N),
        "overall_accuracy": float(correct.mean()),
        "ece_aggregate": float(ece_agg),
        "ece_ci": [float(x) for x in ece_ci],
        "brier_score": float(brier),
        "n_silent_failures": n_silent,
        "silent_objects": silent_objects,
        "per_class": per_class,
    }


def compute_inter_model_agreement(data):
    """For objects with 2+ model predictions, compute class-agreement rate."""
    probs = data["probs"]
    ztf_ids = data["ztf_ids"]
    pred_class = probs.argmax(axis=1)

    df = pd.DataFrame({"ztf_id": ztf_ids, "pred_class": pred_class})
    obj_counts = df["ztf_id"].value_counts()
    multi_obj_ids = obj_counts[obj_counts > 1].index

    agreements = []
    for oid in multi_obj_ids:
        preds = df.loc[df["ztf_id"] == oid, "pred_class"].values
        agreements.append(len(np.unique(preds)) == 1)

    if agreements:
        rate = sum(agreements) / len(agreements) * 100
    else:
        rate = None

    print(f"\n--- Inter-model Agreement ---")
    print(f"Objects with predictions from 2+ models: {len(agreements)}")
    if rate is not None:
        print(f"Agreement rate: {rate:.1f}%")

    return {
        "n_multi_model_objects": len(agreements),
        "agreement_rate_percent": rate,
    }


def make_comparison_figure(full_metrics, dedup_metrics, save_path):
    """Side-by-side bar chart: model-instance vs object-level ECE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    labels = ["Model-instance\n(sensitivity)", "Object-level\n(primary)"]
    eces = [full_metrics["ece_aggregate"], dedup_metrics["ece_aggregate"]]
    ci_lows = [full_metrics["ece_ci"][0], dedup_metrics["ece_ci"][0]]
    ci_highs = [full_metrics["ece_ci"][1], dedup_metrics["ece_ci"][1]]
    errors = [
        [max(0.0, e - l) for e, l in zip(eces, ci_lows)],
        [max(0.0, h - e) for e, h in zip(eces, ci_highs)],
    ]

    bars = ax.bar(labels, eces, yerr=errors, capsize=8,
                   color=["#888888", "#0072B2"], edgecolor="black")
    for bar, ece in zip(bars, eces):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{ece:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.axhline(0.05, color="green", linestyle=":", alpha=0.6,
               label="Well-calibrated (< 0.05)")
    ax.axhline(0.10, color="orange", linestyle="--", alpha=0.6,
               label="Acceptable (< 0.10)")
    ax.set_ylabel("Aggregate ECE")
    ax.set_title("NEEDLE Aggregate Calibration")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(max(eces) + 0.03, 0.12))
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    classes = list(full_metrics["per_class"].keys())
    x = np.arange(len(classes))
    width = 0.35

    full_ece = [full_metrics["per_class"][c]["ece"] for c in classes]
    dedup_ece = [dedup_metrics["per_class"][c]["ece"] for c in classes]

    ax.bar(x - width / 2, full_ece, width, label="Model-instance",
           color="#888888", edgecolor="black")
    ax.bar(x + width / 2, dedup_ece, width, label="Object-level",
           color="#0072B2", edgecolor="black")

    for i, (f, d) in enumerate(zip(full_ece, dedup_ece)):
        ax.text(i - width / 2, f + 0.005, f"{f:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, d + 0.005, f"{d:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Per-class ECE")
    ax.set_title("NEEDLE Per-Class Calibration")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Comparison figure saved: {save_path}")


def main():
    print("=" * 80)
    print("NEEDLE DEDUPLICATION ANALYSIS")
    print("Reviewer concern: object-level vs model-instance evaluation")
    print("=" * 80)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    data = load_needle_predictions()

    print("\n" + "=" * 80)
    print("1. FULL SET (model-instance, N=429)")
    print("=" * 80)
    full_metrics = compute_metrics(data, label="Full (model-instance)")

    print("\n" + "=" * 80)
    print("2. DEDUPLICATION (object-level)")
    print("=" * 80)
    dedup = deduplicate_by_object(data, method="mean")
    dedup_metrics = compute_metrics(dedup, label="Deduplicated (object-level)")

    print("\n" + "=" * 80)
    print("3. INTER-MODEL CLASS AGREEMENT")
    print("=" * 80)
    agreement = compute_inter_model_agreement(data)

    print("\n" + "=" * 80)
    print("4. GENERATING FIGURES")
    print("=" * 80)
    make_comparison_figure(
        full_metrics, dedup_metrics,
        "figures/fig_needle_comparison_dedup_vs_full.pdf"
    )

    probs = dedup["probs"]
    labels_arr = dedup["labels"]

    _, bins = compute_ece(labels_arr, probs, n_bins=15, strategy="equal_mass")
    reliability_diagram(
        bins,
        title=f"NEEDLE Object-Level (N={len(labels_arr)})",
        save_path="figures/fig_needle_dedup_reliability.pdf",
        color="#0072B2",
    )
    reliability_diagram(
        bins,
        title=f"NEEDLE Object-Level (N={len(labels_arr)})",
        save_path="figures/fig_needle_dedup_reliability.png",
        color="#0072B2",
    )
    print("Object-level reliability diagram: figures/fig_needle_dedup_reliability.pdf")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: NUMBERS FOR PAPER")
    print("=" * 80)

    print(f"\n[PROMOTE TO PRIMARY in Table 2 and abstract]")
    print(f"NEEDLE object-level (N={dedup_metrics['n_objects']}):")
    print(f"  Aggregate ECE: {dedup_metrics['ece_aggregate']:.3f} "
          f"[{dedup_metrics['ece_ci'][0]:.3f}, {dedup_metrics['ece_ci'][1]:.3f}]")
    print(f"  Brier score:   {dedup_metrics['brier_score']:.3f}")
    print(f"  Overall accuracy: {dedup_metrics['overall_accuracy']:.3f}")
    print(f"  Silent failures: {dedup_metrics['n_silent_failures']}")

    print(f"\n[REPORT AS SENSITIVITY / ROBUSTNESS]")
    print(f"NEEDLE model-instance (N={full_metrics['n_objects']}):")
    print(f"  Aggregate ECE: {full_metrics['ece_aggregate']:.3f} "
          f"[{full_metrics['ece_ci'][0]:.3f}, {full_metrics['ece_ci'][1]:.3f}]")
    print(f"  Silent failures: {full_metrics['n_silent_failures']}")
    if agreement["agreement_rate_percent"] is not None:
        print(f"  Inter-model class agreement: "
              f"{agreement['agreement_rate_percent']:.1f}% "
              f"(over {agreement['n_multi_model_objects']} objects "
              f"with predictions from 2+ models)")

    print(f"\n[PER-CLASS, object-level]")
    for cname, m in dedup_metrics["per_class"].items():
        if m["gap"] is not None:
            direction = "underconfident" if m["gap"] > 0 else "overconfident"
            print(f"  {cname}: n={m['n_true']} true / {m['n_predicted']} pred, "
                  f"ECE={m['ece']:.3f}, "
                  f"acc={m['accuracy_when_predicted']:.3f}, "
                  f"conf={m['mean_confidence_when_predicted']:.3f}, "
                  f"gap={m['gap']:+.3f} ({direction})")

    if dedup_metrics["n_silent_failures"] > 0:
        print(f"\n[SILENT FAILURES at object level]")
        for sf in dedup_metrics["silent_objects"]:
            print(f"  {sf['ztf_id']}: predicted {sf['predicted_class']} "
                  f"(conf={sf['confidence']:.3f}), true {sf['true_class']}")

    results = {
        "metadata": {
            "description": "NEEDLE dedup analysis for paper revision Step 1.1",
            "primary": "object-level (dedup)",
            "sensitivity": "model-instance (full)",
        },
        "primary_object_level": dedup_metrics,
        "sensitivity_model_instance": full_metrics,
        "inter_model_agreement": agreement,
    }

    with open("results/needle_dedup_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved: results/needle_dedup_results.json")
    print(f"Figures saved in: figures/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()