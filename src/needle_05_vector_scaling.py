"""
Step 5: Per-class (vector) temperature scaling for NEEDLE.

Global temperature scaling failed because NEEDLE's miscalibration is
class-asymmetric: SLSN-I is overconfident (conf 95%, acc 80%) while
TDE is underconfident (conf 86%, acc 100%). A single scalar T cannot
fix both simultaneously.

Vector scaling applies a separate temperature T_k to each class logit
before the final softmax, then jointly optimises all T_k by minimising
NLL. This preserves valid probability distributions (sum = 1) while
allowing class-specific corrections.

    p = softmax( [logit_0/T_0, logit_1/T_1, logit_2/T_2] )

Methodology: same 5-fold CV as script 04 for fair comparison.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, log_softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PRED_PATH  = os.path.join(OUTPUT_DIR, 'needle_predictions.npz')

CLASS_NAMES = ['SN', 'SLSN-I', 'TDE']

# Reference values from previous scripts
GLOBAL_T      = 1.5511
ECE_RAW       = 0.0504
ECE_GLOBAL_T  = 0.1300
ALERCE_T      = 0.36
ALERCE_ECE    = 0.259


# ── calibration primitives ────────────────────────────────────────────────────

def probs_to_logits(probs, eps=1e-10):
    return np.log(np.clip(probs, eps, 1 - eps))


def apply_vector_scaling(logits, T_vec):
    """Divide each class logit by its own temperature, then softmax."""
    return softmax(logits / np.array(T_vec), axis=1)


def nll(probs, labels):
    eps = 1e-10
    return -np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels.astype(int)], eps, 1)))


def ece(probs, labels, n_bins=10):
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    acc  = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            e += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return e


def brier(probs, labels):
    oh = np.zeros_like(probs)
    oh[np.arange(len(labels)), labels.astype(int)] = 1
    return np.mean(np.sum((probs - oh) ** 2, axis=1))


def per_class_stats(probs, labels):
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    rows = []
    for k, name in enumerate(CLASS_NAMES):
        mask = labels == k
        if mask.any():
            # one-vs-rest ECE for this class
            cp = probs[:, k]
            ct = (labels == k).astype(float)
            bins = np.linspace(0, 1, 11)
            e = 0.0
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (cp > lo) & (cp <= hi)
                if m.any():
                    e += (m.sum() / len(cp)) * abs(ct[m].mean() - cp[m].mean())
            rows.append({
                'class': name,
                'n': mask.sum(),
                'acc':  (pred[mask] == labels[mask]).mean(),
                'conf': conf[mask].mean(),
                'gap':  (pred[mask] == labels[mask]).mean() - conf[mask].mean(),
                'ece':  e,
            })
    return rows


# ── fitting ───────────────────────────────────────────────────────────────────

def fit_vector_T(logits, labels, init=None):
    """
    Minimise NLL over T_0, T_1, T_2 with T_k ∈ (0.01, 20).
    Returns array [T_SN, T_SLSNI, T_TDE].
    """
    if init is None:
        init = [1.0, 1.0, 1.0]

    def objective(log_T):          # optimise in log-space → always positive
        T = np.exp(log_T)
        p = apply_vector_scaling(logits, T)
        return nll(p, labels)

    res = minimize(
        objective,
        x0=np.log(init),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12},
    )
    return np.exp(res.x)           # back to real temperatures


# ── cross-validation ──────────────────────────────────────────────────────────

def run_cv(probs, labels, n_folds=5):
    logits = probs_to_logits(probs)
    n = len(probs)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)

    fold_size = n // n_folds
    fold_T, all_true, all_before, all_after = [], [], [], []

    print(f"\nRunning {n_folds}-fold CV for vector scaling …")
    print(f"  {'Fold':>4}  {'T_SN':>7}  {'T_SLSNI':>8}  {'T_TDE':>7}  "
          f"{'ECE_bef':>8}  {'ECE_aft':>8}  {'Brier_bef':>10}  {'Brier_aft':>10}")
    print("  " + "─" * 76)

    for fold in range(n_folds):
        start = fold * fold_size
        end   = start + fold_size if fold < n_folds - 1 else n
        te_idx = idx[start:end]
        tr_idx = np.concatenate([idx[:start], idx[end:]])

        T_vec = fit_vector_T(logits[tr_idx], labels[tr_idx])
        fold_T.append(T_vec)

        pb = probs[te_idx]
        pa = apply_vector_scaling(logits[te_idx], T_vec)
        tl = labels[te_idx]

        print(f"  {fold+1:>4}  {T_vec[0]:>7.4f}  {T_vec[1]:>8.4f}  {T_vec[2]:>7.4f}  "
              f"{ece(pb,tl):>8.4f}  {ece(pa,tl):>8.4f}  "
              f"{brier(pb,tl):>10.4f}  {brier(pa,tl):>10.4f}")

        all_true.append(tl)
        all_before.append(pb)
        all_after.append(pa)

    return np.array(fold_T), all_true, all_before, all_after


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(probs_raw, probs_global, probs_vec, labels, save_path=None):
    """
    Three-panel reliability diagram: raw / global-T / vector-T.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    panels = [
        (probs_raw,    f'Raw  (ECE={ece(probs_raw,labels):.4f})',    '#888'),
        (probs_global, f'Global T={GLOBAL_T:.2f}  (ECE={ece(probs_global,labels):.4f})', '#378ADD'),
        (probs_vec,    f'Vector T  (ECE={ece(probs_vec,labels):.4f})',  '#2CA86E'),
    ]

    for ax, (p, title, col) in zip(axes, panels):
        pred = np.argmax(p, axis=1)
        conf = np.max(p, axis=1)
        acc  = (pred == labels).astype(float)
        bins = np.linspace(0, 1, 11)
        confs, accs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                confs.append(conf[m].mean())
                accs.append(acc[m].mean())
        colors = ['#5DCAA5' if a >= c else '#E24B4A' for c, a in zip(confs, accs)]
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5, label='Perfect')
        ax.bar(confs, accs, width=0.08, alpha=0.75, color=colors, edgecolor='white')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Accuracy', fontsize=12)
    plt.suptitle('NEEDLE-TH calibration: raw vs global T vs per-class (vector) T',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_per_class_comparison(probs_raw, probs_vec, labels, save_path=None):
    """
    2×3 grid: per-class reliability diagrams before and after vector scaling.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ['#888780', '#7F77DD', '#D85A30']

    for col, (k, name, col_c) in enumerate(zip(range(3), CLASS_NAMES, colors)):
        for row, (p, tag) in enumerate([(probs_raw, 'Raw'), (probs_vec, 'Vector T')]):
            ax = axes[row][col]
            cp = p[:, k]
            ct = (labels == k).astype(float)
            bins = np.linspace(0, 1, 11)
            confs, accs, e = [], [], 0.0
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (cp > lo) & (cp <= hi)
                if m.any():
                    c_, a_ = cp[m].mean(), ct[m].mean()
                    e += (m.sum() / len(cp)) * abs(a_ - c_)
                    confs.append(c_); accs.append(a_)
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
            ax.bar(confs, accs, width=0.08, alpha=0.75, color=col_c, edgecolor='white')
            ax.set_title(f'{name} — {tag}  (ECE={e:.4f})', fontsize=11)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel(f'P({name})', fontsize=10)
            ax.set_ylabel('Fraction actually ' + name, fontsize=10)
            ax.grid(alpha=0.2)

    plt.suptitle('Per-class calibration: raw vs vector scaling', fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("NEEDLE Calibration Study — Vector (Per-Class) Temperature Scaling")
    print("=" * 65)

    data   = np.load(PRED_PATH, allow_pickle=True)
    probs  = data['probs']
    labels = data['labels']
    print(f"Loaded {len(probs)} predictions")

    # ── cross-validated vector scaling ────────────────────────────────────────
    fold_T, all_true, all_before, all_after = run_cv(probs, labels)

    combined_true   = np.concatenate(all_true)
    combined_before = np.concatenate(all_before)
    combined_after  = np.concatenate(all_after)

    T_mean = fold_T.mean(axis=0)
    T_std  = fold_T.std(axis=0)

    ece_raw = ece(combined_before, combined_true)
    ece_vec = ece(combined_after,  combined_true)
    ece_red = (1 - ece_vec / ece_raw) * 100 if ece_raw > 0 else 0

    brier_raw = brier(combined_before, combined_true)
    brier_vec = brier(combined_after,  combined_true)
    nll_raw   = nll(combined_before, combined_true)
    nll_vec   = nll(combined_after,  combined_true)

    # ── summary tables ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("COMBINED CV RESULTS")
    print(f"{'='*65}")
    print(f"\n  Per-class temperatures (mean ± std across folds):")
    for k, name in enumerate(CLASS_NAMES):
        print(f"    T_{name:<7} = {T_mean[k]:.4f} ± {T_std[k]:.4f}   "
              f"({'overconf → soften' if T_mean[k]>1 else 'underconf → sharpen'})")

    print(f"\n  {'Metric':<16} {'Raw':>10} {'Global T':>10} {'Vector T':>10}")
    print(f"  {'─'*48}")
    print(f"  {'ECE':<16} {ece_raw:>10.4f} {ECE_GLOBAL_T:>10.4f} {ece_vec:>10.4f}")
    print(f"  {'Brier':<16} {brier_raw:>10.4f} {'—':>10} {brier_vec:>10.4f}")
    print(f"  {'NLL':<16} {nll_raw:>10.4f} {'—':>10} {nll_vec:>10.4f}")
    print(f"  {'ECE reduction':<16} {'—':>10} {'-158%':>10} {ece_red:>+9.1f}%")

    print(f"\n{'='*65}")
    print("PER-CLASS STATS  (raw → after vector T)")
    print(f"{'='*65}")
    print(f"  {'Class':<8} {'N':>5}  {'Acc':>6}  "
          f"{'Conf_raw':>9}  {'Conf_vec':>9}  {'ECE_raw':>8}  {'ECE_vec':>8}  {'Gap_raw':>8}  {'Gap_vec':>8}")
    print("  " + "─" * 75)

    stats_raw = per_class_stats(combined_before, combined_true)
    stats_vec = per_class_stats(combined_after,  combined_true)

    # Rebuild per-class ECE for vector-scaled probs
    for r, v in zip(stats_raw, stats_vec):
        print(f"  {r['class']:<8} {r['n']:>5}  {r['acc']:>6.3f}  "
              f"{r['conf']:>9.3f}  {v['conf']:>9.3f}  "
              f"{r['ece']:>8.4f}  {v['ece']:>8.4f}  "
              f"{r['gap']:>+8.3f}  {v['gap']:>+8.3f}")

    print(f"\n{'='*65}")
    print("COMPARISON WITH ALeRCE")
    print(f"{'='*65}")
    print(f"  {'':22} {'NEEDLE (vec T)':>15}  {'ALeRCE':>8}")
    print(f"  {'─'*48}")
    print(f"  {'ECE (raw)':22} {ece_raw:>15.4f}  {ALERCE_ECE:>8.4f}")
    print(f"  {'ECE (after calib)':22} {ece_vec:>15.4f}  {'~0.058':>8}")
    print(f"  {'ECE reduction':22} {ece_red:>14.1f}%  {'~78%':>8}")
    print(f"  {'T_SN':22} {T_mean[0]:>15.4f}  {'—':>8}")
    print(f"  {'T_SLSN-I':22} {T_mean[1]:>15.4f}  {'—':>8}")
    print(f"  {'T_TDE':22} {T_mean[2]:>15.4f}  {'0.36':>8}")

    # ── global-T probs for comparison plot ────────────────────────────────────
    from scipy.special import softmax as sp_softmax
    logits_all   = probs_to_logits(probs)
    probs_global = sp_softmax(logits_all / GLOBAL_T, axis=1)

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_comparison(
        combined_before, probs_global[np.concatenate([
            np.where(np.isin(np.arange(len(probs)), fold_idx))[0]
            for fold_idx in [np.arange(len(probs))]   # full set
        ])], combined_after, combined_true,
        save_path=os.path.join(OUTPUT_DIR, 'needle_vector_scaling_comparison.png'),
    )

    plot_per_class_comparison(
        combined_before, combined_after, combined_true,
        save_path=os.path.join(OUTPUT_DIR, 'needle_vector_scaling_per_class.png'),
    )

    # ── save text results ─────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, 'needle_vector_scaling_results.txt')
    with open(out_path, 'w') as f:
        f.write("NEEDLE Vector (Per-Class) Temperature Scaling Results\n")
        f.write("=" * 55 + "\n\n")
        f.write("Per-class temperatures (mean ± std across 5 folds):\n")
        for k, name in enumerate(CLASS_NAMES):
            f.write(f"  T_{name} = {T_mean[k]:.4f} ± {T_std[k]:.4f}\n")
        f.write(f"\nAll fold T values:\n")
        for i, row in enumerate(fold_T):
            f.write(f"  Fold {i+1}: {[round(t,4) for t in row]}\n")
        f.write(f"\nAggregate metrics:\n")
        f.write(f"  ECE  raw     : {ece_raw:.4f}\n")
        f.write(f"  ECE  global T: {ECE_GLOBAL_T:.4f}\n")
        f.write(f"  ECE  vector T: {ece_vec:.4f}\n")
        f.write(f"  ECE reduction: {ece_red:.1f}%\n")
        f.write(f"  Brier raw    : {brier_raw:.4f}\n")
        f.write(f"  Brier vec T  : {brier_vec:.4f}\n")
        f.write(f"  NLL raw      : {nll_raw:.4f}\n")
        f.write(f"  NLL vec T    : {nll_vec:.4f}\n")
    print(f"\nResults saved → {out_path}")
    print("Plots saved to outputs/")
    print("\nAnalysis complete.")


if __name__ == '__main__':
    main()
