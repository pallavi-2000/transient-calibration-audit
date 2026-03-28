"""
Step 3: Calibration analysis of NEEDLE predictions.

Computes:
- Expected Calibration Error (ECE)
- Per-class ECE (SN, SLSN, TDE)
- Reliability diagrams (aggregate + per-class)
- Silent failure analysis
- Comparison with ALeRCE calibration results

Uses the predictions extracted in Step 2.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PRED_PATH = os.path.join(OUTPUT_DIR, 'needle_predictions.npz')

# Your ALeRCE results for comparison
ALERCE_ECE = 0.259
ALERCE_ACCURACY = 0.750
ALERCE_MEAN_CONF = 0.451
ALERCE_T = 0.36

CLASS_NAMES = ['SN', 'SLSN-I', 'TDE']


def compute_ece(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error.
    
    ECE = sum_b (|B_b|/N) * |accuracy(B_b) - confidence(B_b)|
    
    For multiclass: use the predicted class probability as confidence.
    """
    pred_classes = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (pred_classes == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_conf = np.mean(confidences[in_bin])
            avg_acc = np.mean(accuracies[in_bin])
            ece += prop_in_bin * abs(avg_acc - avg_conf)
            bin_data.append({
                'lower': bin_lower,
                'upper': bin_upper,
                'confidence': avg_conf,
                'accuracy': avg_acc,
                'count': np.sum(in_bin),
                'gap': avg_acc - avg_conf  # positive = underconfident
            })
    
    return ece, bin_data


def compute_per_class_ece(probs, labels, n_bins=10):
    """Compute ECE for each class separately."""
    results = {}
    
    for class_id, class_name in enumerate(CLASS_NAMES):
        # For this class: use the probability assigned to this class
        class_probs = probs[:, class_id]
        class_true = (labels == class_id).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(class_probs)
        
        for i in range(n_bins):
            in_bin = (class_probs > bin_boundaries[i]) & (class_probs <= bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                avg_conf = np.mean(class_probs[in_bin])
                avg_acc = np.mean(class_true[in_bin])
                ece += (np.sum(in_bin) / n) * abs(avg_acc - avg_conf)
        
        results[class_name] = ece
    
    return results


def plot_reliability_diagram(probs, labels, title='NEEDLE calibration', save_path=None):
    """
    Plot reliability diagram with gap bars.
    Style: clean, publication-ready.
    """
    ece, bin_data = compute_ece(probs, labels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left panel: Reliability diagram ---
    confs = [b['confidence'] for b in bin_data]
    accs = [b['accuracy'] for b in bin_data]
    counts = [b['count'] for b in bin_data]
    gaps = [b['gap'] for b in bin_data]
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Perfect calibration')
    
    # Bar chart of accuracy
    bar_width = 0.08
    colors = ['#5DCAA5' if g >= 0 else '#E24B4A' for g in gaps]
    ax1.bar(confs, accs, width=bar_width, alpha=0.7, color=colors, edgecolor='white', linewidth=0.5)
    
    # Gap indicators
    for conf, acc, gap in zip(confs, accs, gaps):
        if gap >= 0:  # underconfident
            ax1.plot([conf, conf], [conf, acc], color='#1D9E75', linewidth=2, alpha=0.6)
        else:  # overconfident
            ax1.plot([conf, conf], [acc, conf], color='#E24B4A', linewidth=2, alpha=0.6)
    
    ax1.set_xlabel('Mean predicted confidence', fontsize=12)
    ax1.set_ylabel('Fraction of correct predictions', fontsize=12)
    ax1.set_title(f'{title}\nECE = {ece:.4f}', fontsize=13)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.2)
    
    # --- Right panel: Sample count per bin ---
    ax2.bar(confs, counts, width=bar_width, alpha=0.6, color='#378ADD', edgecolor='white')
    ax2.set_xlabel('Mean predicted confidence', fontsize=12)
    ax2.set_ylabel('Number of predictions', fontsize=12)
    ax2.set_title('Predictions per confidence bin', fontsize=13)
    ax2.grid(alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()
    return ece, bin_data


def plot_per_class_reliability(probs, labels, save_path=None):
    """Plot reliability diagrams for each class separately."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#888780', '#7F77DD', '#D85A30']
    
    for class_id, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        ax = axes[class_id]
        
        class_probs = probs[:, class_id]
        class_true = (labels == class_id).astype(float)
        
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        confs, accs, counts = [], [], []
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (class_probs > bin_boundaries[i]) & (class_probs <= bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                avg_conf = np.mean(class_probs[in_bin])
                avg_acc = np.mean(class_true[in_bin])
                ece += (np.sum(in_bin) / len(class_probs)) * abs(avg_acc - avg_conf)
                confs.append(avg_conf)
                accs.append(avg_acc)
                counts.append(np.sum(in_bin))
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
        ax.bar(confs, accs, width=0.08, alpha=0.7, color=color, edgecolor='white')
        
        ax.set_xlabel('Predicted P(' + class_name + ')', fontsize=11)
        ax.set_ylabel('Fraction actually ' + class_name, fontsize=11)
        ax.set_title(f'{class_name} (ECE = {ece:.4f})', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    
    plt.suptitle('NEEDLE per-class calibration', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def silent_failure_analysis(probs, labels, threshold=0.75):
    """
    Analyze silent failures: cases where the classifier is confident AND wrong.
    These are the most dangerous failure mode for follow-up decisions.
    """
    pred_classes = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    correct = pred_classes == labels
    
    high_conf = confidences >= threshold
    silent_failures = high_conf & ~correct
    
    results = {
        'threshold': threshold,
        'total': len(probs),
        'high_confidence': np.sum(high_conf),
        'silent_failures': np.sum(silent_failures),
        'silent_failure_rate': np.mean(silent_failures) if np.sum(high_conf) > 0 else 0,
        'high_conf_accuracy': np.mean(correct[high_conf]) if np.sum(high_conf) > 0 else 0,
    }
    
    # Per-class silent failures
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_mask = labels == class_id
        class_sf = silent_failures & class_mask
        results[f'{class_name}_silent_failures'] = np.sum(class_sf)
    
    return results


def compare_with_alerce(needle_ece, needle_acc, needle_conf):
    """Print side-by-side comparison with ALeRCE results."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: NEEDLE vs ALeRCE")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'NEEDLE':>12} {'ALeRCE':>12}")
    print(f"{'-'*50}")
    print(f"{'ECE':<25} {needle_ece:>12.4f} {ALERCE_ECE:>12.4f}")
    print(f"{'Accuracy':<25} {needle_acc:>12.3f} {ALERCE_ACCURACY:>12.3f}")
    print(f"{'Mean confidence':<25} {needle_conf:>12.3f} {ALERCE_MEAN_CONF:>12.3f}")
    print(f"{'Acc - Conf (gap)':<25} {needle_acc-needle_conf:>+12.3f} {ALERCE_ACCURACY-ALERCE_MEAN_CONF:>+12.3f}")
    
    if needle_acc > needle_conf:
        print(f"\nNEEDLE is UNDERCONFIDENT (like ALeRCE)")
    else:
        print(f"\nNEEDLE is OVERCONFIDENT (unlike ALeRCE)")


def main():
    print("NEEDLE Calibration Study — Calibration Analysis")
    print("=" * 60)
    
    # Load predictions
    if not os.path.exists(PRED_PATH):
        print(f"ERROR: Predictions file not found at {PRED_PATH}")
        print("Run 02_extract_predictions.py first.")
        return
    
    data = np.load(PRED_PATH, allow_pickle=True)
    probs = data['probs']
    labels = data['labels']
    model_ids = data['model_ids']
    
    print(f"Loaded {len(probs)} predictions from {len(np.unique(model_ids))} models")
    
    pred_classes = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    overall_accuracy = np.mean(pred_classes == labels)
    mean_confidence = np.mean(confidences)
    
    # === Aggregate ECE ===
    print(f"\n{'='*60}")
    print(f"AGGREGATE CALIBRATION")
    print(f"{'='*60}")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    print(f"Mean confidence: {mean_confidence:.3f}")
    print(f"Confidence range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
    
    ece, bin_data = plot_reliability_diagram(
        probs, labels,
        title='NEEDLE-TH aggregate calibration',
        save_path=os.path.join(OUTPUT_DIR, 'needle_reliability_diagram.png')
    )
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    
    print(f"\nCalibration bins:")
    print(f"{'Confidence':>12} {'Accuracy':>10} {'Gap':>8} {'Count':>8}")
    print(f"{'-'*42}")
    for b in bin_data:
        direction = 'UNDER' if b['gap'] > 0.05 else ('OVER' if b['gap'] < -0.05 else 'ok')
        print(f"{b['confidence']:>12.3f} {b['accuracy']:>10.3f} {b['gap']:>+8.3f} {b['count']:>8d}  {direction}")
    
    # === Per-class ECE ===
    print(f"\n{'='*60}")
    print(f"PER-CLASS CALIBRATION")
    print(f"{'='*60}")
    
    per_class = compute_per_class_ece(probs, labels)
    for class_name, class_ece in per_class.items():
        class_mask = labels == CLASS_NAMES.index(class_name)
        class_acc = np.mean(pred_classes[class_mask] == labels[class_mask])
        class_conf = np.mean(confidences[class_mask])
        print(f"  {class_name:>6}: ECE={class_ece:.4f}  acc={class_acc:.3f}  conf={class_conf:.3f}  gap={class_acc-class_conf:+.3f}")
    
    plot_per_class_reliability(
        probs, labels,
        save_path=os.path.join(OUTPUT_DIR, 'needle_per_class_reliability.png')
    )
    
    # === Silent failure analysis ===
    print(f"\n{'='*60}")
    print(f"SILENT FAILURE ANALYSIS")
    print(f"{'='*60}")
    
    for threshold in [0.5, 0.75, 0.9]:
        sf = silent_failure_analysis(probs, labels, threshold)
        print(f"\n  Threshold >= {threshold}:")
        print(f"    High-confidence predictions: {sf['high_confidence']}/{sf['total']}")
        print(f"    Silent failures: {sf['silent_failures']}")
        print(f"    High-conf accuracy: {sf['high_conf_accuracy']:.3f}")
        for cn in CLASS_NAMES:
            print(f"    {cn} silent failures: {sf[f'{cn}_silent_failures']}")
    
    # === Compare with ALeRCE ===
    compare_with_alerce(ece, overall_accuracy, mean_confidence)
    
    # === Save summary ===
    summary_path = os.path.join(OUTPUT_DIR, 'needle_calibration_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("NEEDLE Calibration Study Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total predictions: {len(probs)}\n")
        f.write(f"Models used: {len(np.unique(model_ids))}\n")
        f.write(f"Overall accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Mean confidence: {mean_confidence:.4f}\n")
        f.write(f"ECE: {ece:.4f}\n\n")
        f.write("Per-class ECE:\n")
        for cn, ce in per_class.items():
            f.write(f"  {cn}: {ce:.4f}\n")
        f.write(f"\nComparison:\n")
        f.write(f"  NEEDLE ECE: {ece:.4f}\n")
        f.write(f"  ALeRCE ECE: {ALERCE_ECE:.4f}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"\nNext step: Run 04_temperature_scaling.py")


if __name__ == '__main__':
    main()
