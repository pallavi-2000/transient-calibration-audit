"""
Step 3: NEEDLE Duplicate Handling
==================================

Addresses reviewer concern #5:
  "429 predictions across 278 unique objects — 151 predictions
   are duplicates. How are these handled?"

This script:
  1. Computes per-model ECE separately (are models consistent?)
  2. Reports inter-model variance on shared objects
  3. Compares ECE: all predictions vs deduplicated (averaged)
  4. Identifies objects with high inter-model disagreement

Run from project root:
    python3 scripts/06_needle_duplicates.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import compute_ece, bootstrap_ece, compute_classwise_ece


def main():
    print("=" * 60)
    print("NEEDLE DUPLICATE HANDLING AUDIT")
    print("Reviewer concern #5")
    print("=" * 60)

    # Load data
    data = np.load("data/processed/needle_predictions.npz", allow_pickle=True)
    probs = data["probs"]
    labels = data["labels"]
    ztf_ids = data["ztf_ids"]
    model_ids = data["model_ids"]
    class_names = list(data["class_names"])

    n_total = len(probs)
    n_unique = len(set(ztf_ids))
    n_models = len(np.unique(model_ids))
    n_duplicates = n_total - n_unique

    print(f"\nTotal predictions: {n_total}")
    print(f"Unique objects:   {n_unique}")
    print(f"Duplicate predictions: {n_duplicates}")
    print(f"Models: {n_models}")
    print(f"Classes: {class_names}")

    # ---- ANALYSIS 1: Per-model ECE ----
    print(f"\n{'='*50}")
    print("ANALYSIS 1: Per-model ECE")
    print(f"{'='*50}")

    model_results = {}
    for m in range(n_models):
        mask = model_ids == m
        m_probs = probs[mask]
        m_labels = labels[mask]
        m_ids = ztf_ids[mask]

        ece, _ = compute_ece(m_labels, m_probs)
        acc = np.mean(np.argmax(m_probs, axis=1) == m_labels)
        conf = np.mean(np.max(m_probs, axis=1))

        model_results[m] = {
            "n": int(mask.sum()),
            "ece": float(ece),
            "accuracy": float(acc),
            "mean_confidence": float(conf),
        }

        print(f"\n  Model {m}: n={mask.sum()}")
        print(f"    ECE:        {ece:.3f}")
        print(f"    Accuracy:   {acc:.3f}")
        print(f"    Confidence: {conf:.3f}")

        # Per-class breakdown
        for k, cls in enumerate(class_names):
            cls_mask = m_labels == k
            if cls_mask.sum() > 0:
                cls_acc = np.mean(np.argmax(m_probs[cls_mask], axis=1) == m_labels[cls_mask])
                cls_conf = np.mean(m_probs[cls_mask, k])
                print(f"    {cls:7s}: n={cls_mask.sum():3d}, "
                      f"acc={cls_acc:.3f}, conf={cls_conf:.3f}")

    # Summary statistics
    eces = [r["ece"] for r in model_results.values()]
    accs = [r["accuracy"] for r in model_results.values()]
    print(f"\n  ECE across models: mean={np.mean(eces):.3f}, "
          f"std={np.std(eces):.3f}, range=[{min(eces):.3f}, {max(eces):.3f}]")
    print(f"  Accuracy across models: mean={np.mean(accs):.3f}, "
          f"std={np.std(accs):.3f}")

    # ---- ANALYSIS 2: Inter-model agreement on shared objects ----
    print(f"\n{'='*50}")
    print("ANALYSIS 2: Inter-model agreement on shared objects")
    print(f"{'='*50}")

    # Find objects that appear in multiple models
    from collections import Counter
    id_counts = Counter(ztf_ids)
    multi_model_ids = {oid for oid, count in id_counts.items() if count > 1}
    single_model_ids = {oid for oid, count in id_counts.items() if count == 1}

    print(f"\n  Objects in 1 model:  {len(single_model_ids)}")
    print(f"  Objects in 2+ models: {len(multi_model_ids)}")

    # Distribution of appearances
    count_dist = Counter(id_counts.values())
    for n_appearances, n_objects in sorted(count_dist.items()):
        print(f"    Appearing in {n_appearances} model(s): {n_objects} objects")

    # For multi-model objects, compute prediction variance
    print(f"\n  Prediction consistency for multi-model objects:")
    disagreements = []
    high_variance_objects = []

    for oid in multi_model_ids:
        mask = ztf_ids == oid
        obj_probs = probs[mask]
        obj_labels = labels[mask]
        obj_models = model_ids[mask]

        # All labels should be the same (same object)
        assert len(set(obj_labels)) == 1, f"{oid}: inconsistent labels!"

        true_label = obj_labels[0]
        true_class = class_names[true_label]

        # Predicted classes across models
        pred_classes = np.argmax(obj_probs, axis=1)
        all_agree = len(set(pred_classes)) == 1

        # Probability variance
        prob_std = obj_probs.std(axis=0)
        max_prob_std = prob_std.max()

        if not all_agree:
            pred_names = [class_names[p] for p in pred_classes]
            disagreements.append({
                "oid": oid,
                "true_class": true_class,
                "predictions": pred_names,
                "max_prob_std": float(max_prob_std),
            })

        if max_prob_std > 0.1:
            high_variance_objects.append({
                "oid": oid,
                "true_class": true_class,
                "max_prob_std": float(max_prob_std),
                "probs_per_model": obj_probs.tolist(),
            })

    n_multi = len(multi_model_ids)
    n_disagree = len(disagreements)
    pct_agree = (n_multi - n_disagree) / n_multi * 100

    print(f"    Models agree on class: {n_multi - n_disagree}/{n_multi} "
          f"({pct_agree:.1f}%)")
    print(f"    Models disagree: {n_disagree}/{n_multi} "
          f"({100-pct_agree:.1f}%)")

    if disagreements:
        print(f"\n  Objects where models disagree:")
        for d in disagreements[:10]:  # Show up to 10
            print(f"    {d['oid']} (true: {d['true_class']}): "
                  f"predictions={d['predictions']}, "
                  f"max_std={d['max_prob_std']:.3f}")

    print(f"\n  Objects with high probability variance (std > 0.1): "
          f"{len(high_variance_objects)}")

    # ---- ANALYSIS 3: Deduplicated ECE (averaged predictions) ----
    print(f"\n{'='*50}")
    print("ANALYSIS 3: ECE — all predictions vs deduplicated (averaged)")
    print(f"{'='*50}")

    # Method: average probabilities across models for each object
    unique_ids_list = sorted(set(ztf_ids))
    avg_probs = []
    avg_labels = []

    for oid in unique_ids_list:
        mask = ztf_ids == oid
        obj_probs = probs[mask]
        obj_labels = labels[mask]

        avg_probs.append(obj_probs.mean(axis=0))
        avg_labels.append(obj_labels[0])

    avg_probs = np.array(avg_probs)
    avg_labels = np.array(avg_labels)

    # ECE on all predictions (treating each as independent)
    boot_all = bootstrap_ece(labels, probs)
    print(f"\n  All predictions (N={n_total}, treating each as independent):")
    print(f"    ECE: {boot_all['ece']:.3f} "
          f"[{boot_all['ci_lower']:.3f}, {boot_all['ci_upper']:.3f}]")

    # ECE on deduplicated (averaged) predictions
    boot_avg = bootstrap_ece(avg_labels, avg_probs)
    print(f"\n  Deduplicated — model-averaged (N={len(avg_labels)}):")
    print(f"    ECE: {boot_avg['ece']:.3f} "
          f"[{boot_avg['ci_lower']:.3f}, {boot_avg['ci_upper']:.3f}]")

    # ECE difference
    print(f"\n  Difference: {abs(boot_all['ece'] - boot_avg['ece']):.3f}")

    # Per-class comparison
    print(f"\n  Per-class ECE comparison:")
    cw_all = compute_classwise_ece(labels, probs, class_names=class_names)
    cw_avg = compute_classwise_ece(avg_labels, avg_probs, class_names=class_names)

    print(f"    {'Class':<10} {'All (N={})'.format(n_total):<20} "
          f"{'Dedup (N={})'.format(len(avg_labels)):<20}")
    for cls in class_names:
        print(f"    {cls:<10} ECE={cw_all[cls]['ece']:.3f}  "
              f"acc={cw_all[cls]['accuracy']:.3f}     "
              f"ECE={cw_avg[cls]['ece']:.3f}  "
              f"acc={cw_avg[cls]['accuracy']:.3f}")

    # Accuracy comparison
    acc_all = np.mean(np.argmax(probs, axis=1) == labels)
    acc_avg = np.mean(np.argmax(avg_probs, axis=1) == avg_labels)
    print(f"\n  Accuracy: all={acc_all:.3f}, dedup={acc_avg:.3f}")

    # ---- Save results ----
    results = {
        "n_total_predictions": int(n_total),
        "n_unique_objects": int(n_unique),
        "n_duplicates": int(n_duplicates),
        "n_models": int(n_models),
        "per_model_ece": {f"model_{m}": model_results[m] for m in range(n_models)},
        "ece_std_across_models": float(np.std(eces)),
        "inter_model_agreement": {
            "n_multi_model_objects": int(n_multi),
            "n_disagree": int(n_disagree),
            "agreement_pct": float(pct_agree),
        },
        "ece_all_predictions": boot_all["ece"],
        "ece_deduplicated": boot_avg["ece"],
        "n_high_variance_objects": len(high_variance_objects),
        "disagreement_details": disagreements[:10],
    }

    os.makedirs("results", exist_ok=True)
    with open("results/needle_duplicates.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Summary ----
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Per-model ECE std: {np.std(eces):.3f} "
          f"(models are {'consistent' if np.std(eces) < 0.03 else 'variable'})")
    print(f"Inter-model agreement: {pct_agree:.1f}%")
    print(f"ECE (all {n_total} predictions): {boot_all['ece']:.3f}")
    print(f"ECE (deduplicated {n_unique} objects): {boot_avg['ece']:.3f}")
    print(f"Difference: {abs(boot_all['ece'] - boot_avg['ece']):.3f}")
    print(f"\nResults saved to: results/needle_duplicates.json")


if __name__ == "__main__":
    main()
