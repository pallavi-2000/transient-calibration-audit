"""
Fink Calibration Analysis - Conditional & Unconditional
=========================================================

Analysis of Fink's two binary classifiers with proper handling of
prediction eligibility (selective classification):

  - Random Forest (rf_snia_vs_nonia): implements eligibility gate
  - SuperNNova RNN (snn_snia_vs_nonia): possible abstention mechanism

Key concepts:
  - UNCONDITIONAL: ECE computed on all API outputs (including score=0)
    → Measures "user experience" if zeros treated as probabilities
  
  - CONDITIONAL: ECE computed only on score > 0 (π=1 subset)
    → Measures actual classifier calibration when predictions are made

Reference: Geifman & El-Yaniv 2017 (selective classification)

Usage:
    python scripts/02_fink_analysis_conditional.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import (
    compute_ece, bootstrap_ece, brier_score,
    fit_temperature_cv, apply_temperature, auto_calibrate
)
from src.plotting import reliability_diagram, reliability_diagram_comparison


def diagnose_zero_origin(scores, name="Classifier"):
    """
    Diagnose whether score=0 represents abstention or genuine low probability.
    
    Abstention signature:
      - Large fraction of exact zeros
      - Gap between 0 and minimum non-zero score
      - No scores in (0, epsilon) range
    
    Genuine low-score signature:
      - Continuous distribution near zero
      - Small gap between 0 and next score
    """
    n_total = len(scores)
    n_zero = (scores == 0.0).sum()
    zero_frac = n_zero / n_total
    
    scores_nonzero = scores[scores > 0]
    if len(scores_nonzero) > 0:
        min_nonzero = scores_nonzero.min()
        scores_near_zero = ((scores > 0) & (scores < 0.01)).sum()
    else:
        min_nonzero = np.nan
        scores_near_zero = 0
    
    print(f"\n{'='*60}")
    print(f"ZERO-ORIGIN DIAGNOSTIC: {name}")
    print(f"{'='*60}")
    print(f"Total objects: {n_total}")
    print(f"Score = 0.0: {n_zero} ({zero_frac*100:.1f}%)")
    print(f"Score > 0: {len(scores_nonzero)} ({(1-zero_frac)*100:.1f}%)")
    
    if len(scores_nonzero) > 0:
        print(f"Min non-zero score: {min_nonzero:.6f}")
        print(f"Scores in (0, 0.01): {scores_near_zero}")
        print(f"Gap between 0 and min: {min_nonzero:.6f}")
        
        # Diagnosis
        if zero_frac > 0.5 and min_nonzero > 0.001:
            diagnosis = "ABSTENTION"
            explanation = (
                f"High zero fraction ({zero_frac*100:.0f}%) with large gap "
                f"({min_nonzero:.6f}) suggests eligibility gate. "
                "Score=0 means 'no prediction made', not P(SN Ia)=0."
            )
        elif zero_frac > 0.2 and min_nonzero > 0.0001:
            diagnosis = "LIKELY_ABSTENTION"
            explanation = (
                f"Moderate zero fraction with gap suggests possible eligibility "
                f"threshold. Verify against Fink documentation."
            )
        else:
            diagnosis = "GENUINE_LOW_SCORES"
            explanation = (
                f"Continuous distribution near zero suggests these are actual "
                f"probability estimates, not abstentions."
            )
    else:
        diagnosis = "ALL_ZEROS"
        explanation = "All scores are exactly zero. Classifier did not run on any objects."
    
    print(f"\nDiagnosis: {diagnosis}")
    print(f"Explanation: {explanation}")
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.set_title(f'{name} Score Distribution')
    ax.axvline(0, color='red', linestyle='--', label='Zero')
    if len(scores_nonzero) > 0:
        ax.axvline(min_nonzero, color='orange', linestyle='--', 
                   label=f'Min non-zero: {min_nonzero:.4f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    save_name = name.lower().replace(' ', '_')
    fig.savefig(f"figures/diagnostic_{save_name}_histogram.pdf", bbox_inches='tight')
    fig.savefig(f"figures/diagnostic_{save_name}_histogram.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Histogram saved: figures/diagnostic_{save_name}_histogram.pdf")
    
    return {
        "n_total": int(n_total),
        "n_zero": int(n_zero),
        "zero_fraction": float(zero_frac),
        "n_nonzero": int(len(scores_nonzero)),
        "min_nonzero": float(min_nonzero) if not np.isnan(min_nonzero) else None,
        "n_near_zero": int(scores_near_zero),
        "diagnosis": diagnosis,
        "explanation": explanation,
    }


def analyze_conditional_unconditional(scores, true_labels, name, color):
    """
    Analyze classifier with both unconditional (raw API) and conditional (π=1) ECE.
    
    Unconditional: treats all API outputs as probabilities (including score=0)
    Conditional: restricts to cases where classifier actually made a prediction (score>0)
    """
    print(f"\n{'='*70}")
    print(f"{name.upper()}: CONDITIONAL & UNCONDITIONAL ANALYSIS")
    print(f"{'='*70}")
    
    n_total = len(scores)
    n_positive = true_labels.sum()
    
    # --- UNCONDITIONAL (raw API output) ---
    print(f"\n--- UNCONDITIONAL (Raw API) ---")
    print(f"Total objects: {n_total}")
    print(f"Ground truth SN Ia: {n_positive} ({n_positive/n_total*100:.1f}%)")
    
    boot_raw = bootstrap_ece(true_labels, scores, n_bins=15)
    print(f"ECE (unconditional): {boot_raw['ece']:.3f} [{boot_raw['ci_lower']:.3f}, {boot_raw['ci_upper']:.3f}]")
    
    brier_raw = brier_score(true_labels, scores)
    print(f"Brier score: {brier_raw['brier_score']:.3f}")
    
    ece_raw, bins_raw = compute_ece(true_labels, scores, n_bins=15, strategy="equal_mass")
    
    # --- CONDITIONAL (π=1 subset: score > 0) ---
    mask_predicted = scores > 0
    n_predicted = mask_predicted.sum()
    n_abstained = n_total - n_predicted
    
    print(f"\n--- CONDITIONAL (π=1: Predictions Only) ---")
    print(f"Objects with predictions (score > 0): {n_predicted} ({n_predicted/n_total*100:.1f}%)")
    print(f"Objects abstained (score = 0): {n_abstained} ({n_abstained/n_total*100:.1f}%)")
    
    if n_predicted == 0:
        print("ERROR: No predictions made (all scores are zero). Cannot compute conditional ECE.")
        return None
    
    scores_cond = scores[mask_predicted]
    labels_cond = true_labels[mask_predicted]
    
    boot_cond = bootstrap_ece(labels_cond, scores_cond, n_bins=15)
    print(f"ECE (conditional): {boot_cond['ece']:.3f} [{boot_cond['ci_lower']:.3f}, {boot_cond['ci_upper']:.3f}]")
    
    brier_cond = brier_score(labels_cond, scores_cond)
    print(f"Brier score (conditional): {brier_cond['brier_score']:.3f}")
    
    # Accuracy on conditional subset
    accuracy_cond = ((scores_cond > 0.5) == labels_cond).mean()
    print(f"Accuracy (conditional, threshold=0.5): {accuracy_cond:.3f}")
    
    ece_cond, bins_cond = compute_ece(labels_cond, scores_cond, n_bins=15, strategy="equal_mass")
    
    # --- Temperature scaling on conditional subset ---
    print(f"\n--- Temperature Scaling (Conditional Subset) ---")
    result_cond = auto_calibrate(labels_cond, scores_cond)
    print(f"Recommendation: {result_cond['recommendation']}")
    if "reason" in result_cond:
        print(f"Reason: {result_cond.get('reason', '')}")
    if "summary" in result_cond:
        print(f"Summary: {result_cond['summary']}")
    
    # Extract temperature if fitted
    T_cond = None
    ece_post_cond = None
    if result_cond['recommendation'] in ['temperature_scaling', 'apply_with_caution']:
        if 'temperature' in result_cond:
            T_cond = result_cond['temperature']
            print(f"Fitted temperature: {T_cond:.3f}")
            if 'ece_after' in result_cond:
                ece_post_cond = result_cond['ece_after']
                print(f"ECE after temperature scaling: {ece_post_cond:.3f}")
    
    # --- Generate figures ---
    print(f"\n--- Generating Figures ---")
    
    # Unconditional reliability diagram
    reliability_diagram(
        bins_raw,
        title=f"Fink {name} (Raw API)",
        save_path=f"figures/fig_fink_{name.lower().replace(' ', '_')}_raw.pdf",
        color=color,
    )
    reliability_diagram(
        bins_raw,
        title=f"Fink {name} (Raw API)",
        save_path=f"figures/fig_fink_{name.lower().replace(' ', '_')}_raw.png",
        color=color,
    )
    print(f"  ✓ Raw API diagram: figures/fig_fink_{name.lower().replace(' ', '_')}_raw.pdf")
    
    # Conditional reliability diagram
    reliability_diagram(
        bins_cond,
        title=f"Fink {name} (Conditional: score > 0)",
        save_path=f"figures/fig_fink_{name.lower().replace(' ', '_')}_conditional.pdf",
        color=color,
    )
    reliability_diagram(
        bins_cond,
        title=f"Fink {name} (Conditional: score > 0)",
        save_path=f"figures/fig_fink_{name.lower().replace(' ', '_')}_conditional.png",
        color=color,
    )
    print(f"  ✓ Conditional diagram: figures/fig_fink_{name.lower().replace(' ', '_')}_conditional.pdf")
    
    # --- Return comprehensive results ---
    return {
        "name": name,
        "unconditional": {
            "n_objects": int(n_total),
            "n_positive": int(n_positive),
            "prevalence": float(n_positive / n_total),
            "ece": float(boot_raw["ece"]),
            "ece_ci": [float(boot_raw["ci_lower"]), float(boot_raw["ci_upper"])],
            "brier_score": float(brier_raw["brier_score"]),
        },
        "conditional": {
            "n_predicted": int(n_predicted),
            "n_abstained": int(n_abstained),
            "prediction_coverage": float(n_predicted / n_total),
            "ece": float(boot_cond["ece"]),
            "ece_ci": [float(boot_cond["ci_lower"]), float(boot_cond["ci_upper"])],
            "brier_score": float(brier_cond["brier_score"]),
            "accuracy": float(accuracy_cond),
            "temperature": float(T_cond) if T_cond is not None else None,
            "ece_post_temperature": float(ece_post_cond) if ece_post_cond is not None else None,
            "calibration_recommendation": result_cond['recommendation'],
        },
    }


def main():
    print("=" * 80)
    print("FINK CALIBRATION ANALYSIS: CONDITIONAL & UNCONDITIONAL")
    print("=" * 80)
    print("\nThis script computes:")
    print("  1. UNCONDITIONAL ECE: raw API output (treats score=0 as probability)")
    print("  2. CONDITIONAL ECE: score > 0 subset (actual classifier calibration)")
    print("  3. Zero-origin diagnostics (abstention vs genuine low scores)")
    print("\nClassifiers: RF (rf_snia_vs_nonia), SNN (snn_snia_vs_nonia)")
    print("=" * 80)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load data
    print("\nLoading data...")
    fink = pd.read_csv("data/raw/fink_classifications.csv")
    sample = pd.read_csv("data/ground_truth/bts_sample.csv")

    merged = fink.merge(
        sample[["ZTFID", "alerce_class"]],
        left_on="oid", right_on="ZTFID"
    )

    # Binary ground truth: SN Ia vs everything else
    is_snia = (merged["alerce_class"] == "SNIa").astype(int).values

    print(f"Total objects in merged dataset: {len(merged)}")
    print(f"Ground truth: SN Ia = {is_snia.sum()}, Not SN Ia = {len(merged) - is_snia.sum()}")

    # ========================================
    # RANDOM FOREST
    # ========================================
    print("\n" + "="*80)
    print("RANDOM FOREST ANALYSIS")
    print("="*80)
    
    rf_scores = pd.to_numeric(merged["rf_snia_vs_nonia"], errors="coerce").values
    valid_rf = ~np.isnan(rf_scores)
    rf_scores_clean = rf_scores[valid_rf]
    is_snia_rf = is_snia[valid_rf]
    
    # Zero-origin diagnostic
    rf_diagnostic = diagnose_zero_origin(rf_scores_clean, "Random Forest")
    
    # Conditional/unconditional analysis
    rf_result = analyze_conditional_unconditional(
        rf_scores_clean, is_snia_rf,
        "Random Forest", "#D55E00"
    )
    
    # ========================================
    # SUPERNNOVA
    # ========================================
    print("\n" + "="*80)
    print("SUPERNNOVA ANALYSIS")
    print("="*80)
    
    snn_scores = pd.to_numeric(merged["snn_snia_vs_nonia"], errors="coerce").values
    valid_snn = ~np.isnan(snn_scores)
    snn_scores_clean = snn_scores[valid_snn]
    is_snia_snn = is_snia[valid_snn]
    
    # Zero-origin diagnostic
    snn_diagnostic = diagnose_zero_origin(snn_scores_clean, "SuperNNova")
    
    # Conditional/unconditional analysis
    snn_result = analyze_conditional_unconditional(
        snn_scores_clean, is_snia_snn,
        "SuperNNova", "#0072B2"
    )
    
    # ========================================
    # SUMMARY TABLE
    # ========================================
    print("\n" + "="*80)
    print("SUMMARY TABLE FOR PAPER")
    print("="*80)
    
    print("\n--- Random Forest ---")
    print(f"Raw API ECE: {rf_result['unconditional']['ece']:.3f} "
          f"[{rf_result['unconditional']['ece_ci'][0]:.3f}, "
          f"{rf_result['unconditional']['ece_ci'][1]:.3f}]")
    print(f"Prediction coverage: {rf_result['conditional']['n_predicted']}/{rf_result['unconditional']['n_objects']} "
          f"({rf_result['conditional']['prediction_coverage']*100:.1f}%)")
    print(f"Conditional ECE: {rf_result['conditional']['ece']:.3f} "
          f"[{rf_result['conditional']['ece_ci'][0]:.3f}, "
          f"{rf_result['conditional']['ece_ci'][1]:.3f}]")
    if rf_result['conditional']['temperature'] is not None:
        print(f"Temperature (conditional): {rf_result['conditional']['temperature']:.3f}")
        print(f"ECE post-T (conditional): {rf_result['conditional']['ece_post_temperature']:.3f}")
    
    print("\n--- SuperNNova ---")
    print(f"Raw API ECE: {snn_result['unconditional']['ece']:.3f} "
          f"[{snn_result['unconditional']['ece_ci'][0]:.3f}, "
          f"{snn_result['unconditional']['ece_ci'][1]:.3f}]")
    print(f"Prediction coverage: {snn_result['conditional']['n_predicted']}/{snn_result['unconditional']['n_objects']} "
          f"({snn_result['conditional']['prediction_coverage']*100:.1f}%)")
    print(f"Conditional ECE: {snn_result['conditional']['ece']:.3f} "
          f"[{snn_result['conditional']['ece_ci'][0]:.3f}, "
          f"{snn_result['conditional']['ece_ci'][1]:.3f}]")
    if snn_result['conditional']['temperature'] is not None:
        print(f"Temperature (conditional): {snn_result['conditional']['temperature']:.3f}")
        print(f"ECE post-T (conditional): {snn_result['conditional']['ece_post_temperature']:.3f}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    results = {
        "metadata": {
            "description": "Fink calibration analysis with conditional/unconditional ECE",
            "reference": "Geifman & El-Yaniv 2017 (selective classification)",
            "classifier_type": "binary (SN Ia vs not SN Ia)",
        },
        "random_forest": {
            "zero_diagnostic": rf_diagnostic,
            "results": rf_result,
        },
        "supernnova": {
            "zero_diagnostic": snn_diagnostic,
            "results": snn_result,
        },
        "interpretation": {
            "rf": (
                "Random Forest implements implicit selective classification via "
                "minimum-epoch eligibility gate (≥3 detections per filter per Leoni et al. 2022). "
                f"Score=0 means 'no prediction made', not P(SN Ia)=0. "
                f"Diagnosis: {rf_diagnostic['diagnosis']}"
            ),
            "snn": (
                f"SuperNNova zero-origin diagnosis: {snn_diagnostic['diagnosis']}. "
                f"{snn_diagnostic['explanation']}"
            ),
        }
    }

    output_path = "results/fink_conditional_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"\nFigures generated:")
    print(f"  - Raw API reliability diagrams")
    print(f"  - Conditional reliability diagrams")
    print(f"  - Zero-origin diagnostic histograms")
    print(f"\nDone!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()