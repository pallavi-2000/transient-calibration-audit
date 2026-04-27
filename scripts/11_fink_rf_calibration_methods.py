"""
Step 1.4: Platt Scaling vs. Isotonic Regression for Fink RF
============================================================

Reviewer concern: "You used temperature scaling on Fink RF. Why not Platt 
scaling or isotonic regression? Compare all three methods."

Goal: Validate that temperature scaling is the right choice for Fink RF,
or identify if another method is better.

What it does:
  1. Load Fink RF predictions (non-zero subset, N=76)
  2. Fit three calibration methods:
     a) Temperature scaling: scale confidence by T, renormalize
     b) Platt scaling: fit logistic regression P(correct | score)
     c) Isotonic regression: learn monotonic mapping score → P(correct)
  3. Compute ECE for each method on held-out data (5-fold CV)
  4. Compare: which reduces ECE most?
  5. Validate that temperature scaling is competitive or better

Expected outcome:
  - All three methods should reduce ECE somewhat
  - Temperature scaling: likely best for RF (simple, works well)
  - Isotonic: may overfit on tiny sample (N=76)
  - Platt: intermediate
  - If isotonic dramatically better: use it; otherwise stick with T-scaling

Usage:
  python scripts/11_fink_rf_calibration_methods.py

Outputs:
  results/fink_rf_calibration_comparison.json
  figures/fig_fink_rf_calibration_methods.pdf
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import compute_ece, bootstrap_ece


def load_fink_rf(
    path="data/raw/fink_classifications.csv",
    truth_path="data/ground_truth/bts_sample.csv",
):
    """
    Load Fink RF predictions (binary: SN Ia vs non-SN Ia).

    Merges predictions with BTS ground truth.
    Returns non-zero predictions only (~76 objects after filtering).
    """
    for p in [path, "/Users/pallavisati/Desktop/transient-calibration-audit/" + path]:
        if os.path.exists(p):
            path = p
            break
    else:
        raise FileNotFoundError(f"Could not find Fink CSV at {path}")

    for p in [truth_path, "/Users/pallavisati/Desktop/transient-calibration-audit/" + truth_path]:
        if os.path.exists(p):
            truth_path = p
            break
    else:
        raise FileNotFoundError(f"Could not find BTS sample at {truth_path}")

    print(f"Loading Fink RF from: {path}")
    df = pd.read_csv(path)
    truth = pd.read_csv(truth_path)

    if "rf_snia_vs_nonia" not in df.columns:
        raise ValueError(f"Expected 'rf_snia_vs_nonia' column. Found: {df.columns.tolist()}")

    # Merge with BTS ground truth
    df = df.merge(truth[["ZTFID", "alerce_class"]], left_on="oid", right_on="ZTFID", how="inner")

    # Binary label: SN Ia (positive class) vs everything else
    is_sn = (df["alerce_class"] == "SNIa").values.astype(bool)

    # Get RF scores
    rf_scores = df["rf_snia_vs_nonia"].values.astype(np.float64)

    # Filter to non-zero (Fink's minimum-epoch gate produces many zeros)
    nonzero_mask = rf_scores > 0
    rf_scores_nz = rf_scores[nonzero_mask]
    is_sn_nz = is_sn[nonzero_mask]
    ztf_ids_nz = df["oid"].values[nonzero_mask]

    N_total = len(rf_scores)
    N_nonzero = len(rf_scores_nz)
    N_zero = N_total - N_nonzero

    print(f"Total Fink RF predictions: {N_total}")
    print(f"  Non-zero (analyzed): {N_nonzero} ({100*N_nonzero/N_total:.1f}%)")
    print(f"  Zeros (filtered out): {N_zero} ({100*N_zero/N_total:.1f}%)")
    print(f"\nNon-zero sample composition:")
    print(f"  SN: {is_sn_nz.sum()} ({100*is_sn_nz.sum()/N_nonzero:.1f}%)")
    print(f"  Non-SN: {(~is_sn_nz).sum()} ({100*(~is_sn_nz).sum()/N_nonzero:.1f}%)")

    return {
        "scores": rf_scores_nz,
        "true_class": is_sn_nz.astype(int),  # 1=SN, 0=non-SN
        "ztf_ids": ztf_ids_nz,
        "n_total": N_total,
        "n_nonzero": N_nonzero,
        "n_zero": N_zero,
    }


def temperature_scaling_fit(scores_train, true_train):
    """Fit temperature scaling: T that minimizes NLL on training set."""
    def nll(T):
        if T <= 0:
            return 1e10
        # Scale scores
        scaled = scores_train ** (1 / T)
        scaled = scaled / (scaled + 1 - scaled)  # Ensure [0,1] for binary
        # Negative log likelihood
        eps = 1e-15
        scaled = np.clip(scaled, eps, 1 - eps)
        nll_val = -np.mean(true_train * np.log(scaled) + (1 - true_train) * np.log(1 - scaled))
        return nll_val
    
    result = minimize(nll, x0=1.0, bounds=[(0.01, 10)], method="L-BFGS-B")
    return result.x[0]


def apply_temperature(scores, T):
    """Apply temperature scaling to binary scores."""
    scaled = scores ** (1 / T)
    # Normalize to [0, 1]
    scaled = scaled / (scaled + 1 - scaled)
    return np.clip(scaled, 0, 1)


def platt_scaling_fit(scores_train, true_train):
    """Fit Platt scaling: logistic regression on scores."""
    from sklearn.linear_model import LogisticRegression
    
    X = scores_train.reshape(-1, 1)
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X, true_train)
    return clf


def apply_platt(scores, clf):
    """Apply Platt scaling."""
    X = scores.reshape(-1, 1)
    return clf.predict_proba(X)[:, 1]


def isotonic_regression_fit(scores_train, true_train):
    """Fit isotonic regression."""
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(scores_train, true_train)
    return iso_reg


def apply_isotonic(scores, iso_reg):
    """Apply isotonic regression."""
    return iso_reg.predict(scores)


def compute_metrics_binary(probs_test, true_test, method_name=""):
    """Compute ECE and Brier for binary classification."""
    # Use 1D path: compute_ece(labels, probs_1d) where probs_1d is P(SN Ia).
    # The 1D path uses confidence = max(p, 1-p) and threshold at 0.5 — correct
    # for binary classifiers. Do NOT reshape to (-1,1) which triggers the 2D
    # path and sets all predictions to class-0, corrupting correctness.
    ece, _ = compute_ece(true_test, probs_test, n_bins=10, strategy="equal_mass")

    brier = float(np.mean((probs_test - true_test) ** 2))

    correct = (probs_test >= 0.5).astype(int)
    is_correct = (correct == true_test).astype(int)

    return {
        "ece": float(ece),
        "brier": brier,
        "accuracy": float(is_correct.mean()),
    }


def main():
    print("=" * 80)
    print("STEP 1.4: FINK RF CALIBRATION METHOD COMPARISON")
    print("Temperature scaling vs. Platt vs. Isotonic regression")
    print("=" * 80)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load data
    data = load_fink_rf()
    scores = data["scores"]
    true_class = data["true_class"]

    print(f"\n" + "=" * 80)
    print("5-FOLD CROSS-VALIDATION: METHOD COMPARISON")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_by_method = {
        "raw": [],
        "temperature": [],
        "platt": [],
        "isotonic": [],
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(scores, true_class)):
        print(f"\nFold {fold + 1}/5:")

        scores_train = scores[train_idx]
        true_train = true_class[train_idx]
        scores_test = scores[test_idx]
        true_test = true_class[test_idx]

        # 1. Raw (no calibration)
        raw_metrics = compute_metrics_binary(scores_test, true_test, "raw")
        results_by_method["raw"].append(raw_metrics)

        # 2. Temperature scaling
        T = temperature_scaling_fit(scores_train, true_train)
        scores_test_T = apply_temperature(scores_test, T)
        T_metrics = compute_metrics_binary(scores_test_T, true_test, "temperature")
        results_by_method["temperature"].append(T_metrics)
        T_metrics["temperature"] = float(T)

        # 3. Platt scaling
        platt_clf = platt_scaling_fit(scores_train, true_train)
        scores_test_platt = apply_platt(scores_test, platt_clf)
        platt_metrics = compute_metrics_binary(scores_test_platt, true_test, "platt")
        results_by_method["platt"].append(platt_metrics)

        # 4. Isotonic regression
        iso_reg = isotonic_regression_fit(scores_train, true_train)
        scores_test_iso = apply_isotonic(scores_test, iso_reg)
        iso_metrics = compute_metrics_binary(scores_test_iso, true_test, "isotonic")
        results_by_method["isotonic"].append(iso_metrics)

        # Print fold results
        print(f"  Raw:        ECE={raw_metrics['ece']:.3f}, Brier={raw_metrics['brier']:.3f}")
        print(f"  Temperature (T={T:.3f}): ECE={T_metrics['ece']:.3f}, Brier={T_metrics['brier']:.3f}")
        print(f"  Platt:      ECE={platt_metrics['ece']:.3f}, Brier={platt_metrics['brier']:.3f}")
        print(f"  Isotonic:   ECE={iso_metrics['ece']:.3f}, Brier={iso_metrics['brier']:.3f}")

    # Aggregate across folds
    print(f"\n" + "=" * 80)
    print("CROSS-VALIDATED RESULTS (mean ± std across 5 folds)")
    print("=" * 80)

    for method in ["raw", "temperature", "platt", "isotonic"]:
        eces = [r["ece"] for r in results_by_method[method]]
        briers = [r["brier"] for r in results_by_method[method]]

        print(f"\n{method.upper()}:")
        print(f"  ECE:  {np.mean(eces):.3f} ± {np.std(eces):.3f}")
        print(f"  Brier: {np.mean(briers):.3f} ± {np.std(briers):.3f}")

    # Determine winner
    temp_ece = np.mean([r["ece"] for r in results_by_method["temperature"]])
    platt_ece = np.mean([r["ece"] for r in results_by_method["platt"]])
    iso_ece = np.mean([r["ece"] for r in results_by_method["isotonic"]])

    winner = min(
        [("temperature", temp_ece), ("platt", platt_ece), ("isotonic", iso_ece)],
        key=lambda x: x[1],
    )[0]

    print(f"\n" + "=" * 80)
    print(f"RECOMMENDATION: {winner.upper()} is best")
    print("=" * 80)

    if winner == "temperature":
        print(f"Temperature scaling achieves lowest ECE ({temp_ece:.3f}).")
        print(f"This is the simplest method and works best for RF scores.")
        print(f"✓ Paper choice of temperature scaling is validated.")
    elif winner == "isotonic":
        print(f"Isotonic regression achieves lowest ECE ({iso_ece:.3f}).")
        print(f"⚠ Consider using isotonic instead of temperature scaling.")
    else:
        print(f"Platt scaling achieves lowest ECE ({platt_ece:.3f}).")
        print(f"⚠ Consider using Platt instead of temperature scaling.")

    # Save results
    summary = {
        "metadata": {
            "description": "Fink RF calibration method comparison",
            "question": "Temperature scaling vs. Platt vs. Isotonic: which is best?",
            "sample_size": data["n_nonzero"],
        },
        "results_by_method": {},
        "aggregate": {},
    }

    for method in ["raw", "temperature", "platt", "isotonic"]:
        eces = [r["ece"] for r in results_by_method[method]]
        briers = [r["brier"] for r in results_by_method[method]]
        accs = [r["accuracy"] for r in results_by_method[method]]

        summary["results_by_method"][method] = {
            "ece_mean": float(np.mean(eces)),
            "ece_std": float(np.std(eces)),
            "brier_mean": float(np.mean(briers)),
            "brier_std": float(np.std(briers)),
            "accuracy_mean": float(np.mean(accs)),
        }
        summary["aggregate"][method] = results_by_method[method]

    summary["recommendation"] = winner
    summary["validation"] = {
        "paper_method": "temperature_scaling",
        "cv_winner": winner,
        "temperature_ece": float(temp_ece),
        "platt_ece": float(platt_ece),
        "isotonic_ece": float(iso_ece),
        "validated": winner == "temperature",
    }

    with open("results/fink_rf_calibration_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = ["raw", "temperature", "platt", "isotonic"]
    eces_by_method = [
        [r["ece"] for r in results_by_method[m]] for m in methods
    ]
    briers_by_method = [
        [r["brier"] for r in results_by_method[m]] for m in methods
    ]

    # --- ECE comparison ---
    ax = axes[0]
    means = [np.mean(e) for e in eces_by_method]
    stds = [np.std(e) for e in eces_by_method]
    colors = ["#888888", "#0072B2", "#ff7f0e", "#2ca02c"]
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, edgecolor="black", alpha=0.7)
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("ECE Comparison Across Calibration Methods (5-fold CV)")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(means) * 1.3)

    # Highlight winner
    winner_idx = np.argmin(means)
    bars[winner_idx].set_edgecolor("red")
    bars[winner_idx].set_linewidth(3)

    # --- Brier score comparison ---
    ax = axes[1]
    means_brier = [np.mean(b) for b in briers_by_method]
    stds_brier = [np.std(b) for b in briers_by_method]
    
    bars = ax.bar(x, means_brier, yerr=stds_brier, capsize=8, color=colors, edgecolor="black", alpha=0.7)
    
    for bar, mean in zip(bars, means_brier):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylabel("Brier Score")
    ax.set_title("Brier Score Comparison (5-fold CV)")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(means_brier) * 1.3)

    # Highlight winner
    winner_idx_brier = np.argmin(means_brier)
    bars[winner_idx_brier].set_edgecolor("red")
    bars[winner_idx_brier].set_linewidth(3)

    plt.tight_layout()
    plt.savefig("figures/fig_fink_rf_calibration_methods.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig_fink_rf_calibration_methods.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"JSON: results/fink_rf_calibration_comparison.json")
    print(f"Figure: figures/fig_fink_rf_calibration_methods.pdf")
    print("=" * 80)


if __name__ == "__main__":
    main()
