"""
Step 4: Temperature scaling for NEEDLE.

Applies post-hoc temperature scaling to correct NEEDLE's calibration.
Uses proper held-out methodology (5-fold CV on the test predictions).

Temperature scaling: p_calibrated = softmax(logits / T)
- T < 1: sharpens probabilities (fixes underconfidence)
- T > 1: softens probabilities (fixes overconfidence)
- T = 1: no change

Methodology:
- Split the pooled test predictions into 5 folds
- For each fold: fit T on 4 folds, evaluate on the held-out fold
- Report mean T, ECE before/after, and improvement
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PRED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'needle_predictions.npz')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['SN', 'SLSN-I', 'TDE']

# ALeRCE comparison values
ALERCE_T = 0.36
ALERCE_ECE_BEFORE = 0.259
ALERCE_ECE_AFTER = 0.058  # approximate


def probs_to_logits(probs, eps=1e-10):
    """Convert softmax probabilities back to logits."""
    probs = np.clip(probs, eps, 1 - eps)
    return np.log(probs)


def apply_temperature(logits, T):
    """Apply temperature scaling: softmax(logits / T)."""
    scaled = logits / T
    return softmax(scaled, axis=1)


def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error."""
    pred_classes = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (pred_classes == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            avg_conf = np.mean(confidences[in_bin])
            avg_acc = np.mean(accuracies[in_bin])
            ece += np.mean(in_bin) * abs(avg_acc - avg_conf)
    
    return ece


def compute_nll(probs, labels):
    """Compute negative log-likelihood (the objective for fitting T)."""
    eps = 1e-10
    probs = np.clip(probs, eps, 1 - eps)
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels.astype(int)]))
    return nll


def compute_brier(probs, labels):
    """Compute Brier score."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels.astype(int)] = 1
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def fit_temperature(logits, labels):
    """Find optimal temperature by minimizing NLL."""
    def objective(T):
        scaled_probs = apply_temperature(logits, T)
        return compute_nll(scaled_probs, labels)
    
    result = minimize_scalar(objective, bounds=(0.01, 10.0), method='bounded')
    return result.x


def run_cross_validation(probs, labels, groups, n_folds=5):
    """
    5-fold cross-validation for temperature scaling.

    This avoids the methodological issue of fitting T on the same data
    you evaluate on (which was identified as a critical fix in your
    ALeRCE paper revision).

    Folds are grouped by ZTF object ID (GroupKFold), not by row. NEEDLE-TH
    is a 5-model ensemble and the pooled 429 predictions cover only 278
    unique objects — 43 objects appear 2-5x (once per model that held
    them out). A plain row-wise shuffle-split can put one appearance of
    an object in the training fold and another appearance of the SAME
    object in the "held-out" test fold, leaking its true label (and a
    correlated confidence pattern) across the split. GroupKFold guarantees
    every appearance of a given object lands in the same fold.
    """
    logits = probs_to_logits(probs)
    n = len(probs)

    gkf = GroupKFold(n_splits=n_folds)

    fold_T_values = []
    all_test_true = []
    all_test_proba_before = []
    all_test_proba_after = []

    print(f"\nRunning {n_folds}-fold group-aware (by ZTF ID) cross-validation...")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(np.arange(n), groups=groups)):
        # Fit T on training folds
        T = fit_temperature(logits[train_idx], labels[train_idx])
        fold_T_values.append(T)
        
        # Evaluate on held-out fold
        test_probs_before = probs[test_idx]
        test_probs_after = apply_temperature(logits[test_idx], T)
        test_labels = labels[test_idx]
        
        ece_before = compute_ece(test_probs_before, test_labels)
        ece_after = compute_ece(test_probs_after, test_labels)
        brier_before = compute_brier(test_probs_before, test_labels)
        brier_after = compute_brier(test_probs_after, test_labels)
        
        print(f"  Fold {fold+1}: T={T:.4f}  ECE {ece_before:.4f}\u2192{ece_after:.4f}  "
              f"Brier {brier_before:.4f}\u2192{brier_after:.4f}")
        
        all_test_true.append(test_labels)
        all_test_proba_before.append(test_probs_before)
        all_test_proba_after.append(test_probs_after)
    
    return fold_T_values, all_test_true, all_test_proba_before, all_test_proba_after


def plot_before_after(probs_before, probs_after, labels, save_path=None):
    """Plot reliability diagrams before and after temperature scaling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, probs, title in [(ax1, probs_before, 'Before temperature scaling'),
                              (ax2, probs_after, 'After temperature scaling')]:
        pred_classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        accuracies = (pred_classes == labels).astype(float)
        
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        confs, accs = [], []
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                avg_conf = np.mean(confidences[in_bin])
                avg_acc = np.mean(accuracies[in_bin])
                ece += np.mean(in_bin) * abs(avg_acc - avg_conf)
                confs.append(avg_conf)
                accs.append(avg_acc)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Perfect')
        colors = ['#5DCAA5' if a >= c else '#E24B4A' for c, a in zip(confs, accs)]
        ax.bar(confs, accs, width=0.08, alpha=0.7, color=colors, edgecolor='white')
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{title}\nECE = {ece:.4f}', fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def main():
    print("NEEDLE Calibration Study — Temperature Scaling")
    print("=" * 60)
    
    # Load predictions
    if not os.path.exists(PRED_PATH):
        print(f"ERROR: {PRED_PATH} not found. Run steps 1-2 first.")
        return
    
    data = np.load(PRED_PATH, allow_pickle=True)
    probs = data['probs']
    labels = data['labels']
    ztf_ids = data['ztf_ids']

    n_unique = len(set(ztf_ids.tolist()))
    print(f"Loaded {len(probs)} predictions ({n_unique} unique objects)")
    if n_unique < len(probs):
        print(f"  NOTE: {len(probs) - n_unique} repeat object-appearances across "
              f"the 5 ensemble models — using GroupKFold(groups=ztf_id) so no "
              f"object crosses the train/test fold boundary.")

    # === Cross-validated temperature scaling (grouped by object ID) ===
    fold_T, all_true, all_before, all_after = run_cross_validation(probs, labels, groups=ztf_ids)
    
    # Combine all folds
    combined_true = np.concatenate(all_true)
    combined_before = np.concatenate(all_before)
    combined_after = np.concatenate(all_after)
    
    # Final metrics
    ece_before = compute_ece(combined_before, combined_true)
    ece_after = compute_ece(combined_after, combined_true)
    brier_before = compute_brier(combined_before, combined_true)
    brier_after = compute_brier(combined_after, combined_true)
    nll_before = compute_nll(combined_before, combined_true)
    nll_after = compute_nll(combined_after, combined_true)
    
    T_mean = np.mean(fold_T)
    T_std = np.std(fold_T)
    ece_reduction = (1 - ece_after / ece_before) * 100 if ece_before > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"COMBINED TEST SET RESULTS ({len(fold_T)}-fold CV)")
    print(f"{'='*60}")
    print(f"\n{'Metric':<16} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"{'-'*48}")
    print(f"{'ECE':<16} {ece_before:>10.4f} {ece_after:>10.4f} {ece_after-ece_before:>+10.4f}")
    print(f"{'Brier':<16} {brier_before:>10.4f} {brier_after:>10.4f} {brier_after-brier_before:>+10.4f}")
    print(f"{'NLL':<16} {nll_before:>10.4f} {nll_after:>10.4f} {nll_after-nll_before:>+10.4f}")
    
    print(f"\nTemperature:")
    print(f"  Mean: {T_mean:.4f}")
    print(f"  Std:  {T_std:.4f}")
    print(f"  Range: [{min(fold_T):.4f}, {max(fold_T):.4f}]")
    print(f"\nECE reduction: {ece_reduction:.1f}%")
    
    if T_mean < 1:
        print(f"\nT < 1 means NEEDLE is UNDERCONFIDENT")
        print(f"(probabilities need sharpening)")
    else:
        print(f"\nT > 1 means NEEDLE is OVERCONFIDENT")
        print(f"(probabilities need softening)")
    
    # === Comparison with ALeRCE ===
    print(f"\n{'='*60}")
    print(f"COMPARISON WITH ALeRCE")
    print(f"{'='*60}")
    print(f"{'':20} {'NEEDLE':>12} {'ALeRCE':>12}")
    print(f"{'-'*46}")
    print(f"{'T (mean)':20} {T_mean:>12.4f} {ALERCE_T:>12.4f}")
    print(f"{'ECE before':20} {ece_before:>12.4f} {ALERCE_ECE_BEFORE:>12.4f}")
    print(f"{'ECE after':20} {ece_after:>12.4f} {ALERCE_ECE_AFTER:>12.4f}")
    print(f"{'ECE reduction':20} {ece_reduction:>11.1f}% {'~78':>11}%")
    
    direction = "underconfident" if T_mean < 1 else "overconfident"
    alerce_dir = "underconfident"
    if direction == alerce_dir:
        print(f"\nBoth classifiers show the SAME pattern: {direction}")
    else:
        print(f"\nDifferent patterns: NEEDLE is {direction}, ALeRCE is {alerce_dir}")
    
    # === Plot ===
    plot_before_after(
        combined_before, combined_after, combined_true,
        save_path=os.path.join(OUTPUT_DIR, 'needle_temperature_scaling.png')
    )
    
    # === Save results ===
    results_path = os.path.join(OUTPUT_DIR, 'needle_temperature_scaling_results.txt')
    with open(results_path, 'w') as f:
        f.write("NEEDLE Temperature Scaling Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Temperature (mean +/- std): {T_mean:.4f} +/- {T_std:.4f}\n")
        f.write(f"Temperature range: [{min(fold_T):.4f}, {max(fold_T):.4f}]\n")
        f.write(f"ECE before: {ece_before:.4f}\n")
        f.write(f"ECE after: {ece_after:.4f}\n")
        f.write(f"ECE reduction: {ece_reduction:.1f}%\n")
        f.write(f"Brier before: {brier_before:.4f}\n")
        f.write(f"Brier after: {brier_after:.4f}\n")
        f.write(f"\nDirection: {'underconfident' if T_mean < 1 else 'overconfident'}\n")
        f.write(f"\nFold T values: {[round(t, 4) for t in fold_T]}\n")
    
    print(f"\nResults saved to: {results_path}")
    print(f"\nDone! You now have calibration results for NEEDLE.")
    print(f"These are ready to share with Xinyue and include in your AJ paper.")


if __name__ == '__main__':
    main()
