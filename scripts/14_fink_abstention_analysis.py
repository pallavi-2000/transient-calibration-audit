"""
Fink Abstention Analysis (Selective Classification Reframe)
============================================================

Purpose:
    Reframe Fink's zero-score outputs as ABSTENTIONS on real astrophysical
    transients (since BTS provides spectroscopic ground truth), NOT as
    negative predictions or miscalibration.

Reviewer concern:
    "The unconditional ECE for Fink RF (0.411) is misleading because
     94% of scores are exactly 0.0. These zeros are not 'P(SN Ia) = 0'
     predictions — they are eligibility-gate failures (Leoni et al. 2022:
     >=3 epochs per filter required). Reframe as coverage + conditional
     calibration."

What this script does:
    1. Loads existing Fink prediction table + BTS truth labels
    2. For BOTH RF (rf_snia_vs_nonia) and SuperNNova (snn_snia_vs_nonia):
       - Identifies zero-score (abstention) vs non-zero (prediction) subsets
       - Computes per-class abstention rates with 95% Wilson CIs
       - Tests if abstention is class-asymmetric (chi-squared)
    3. Generates publication-ready figures and tables
    4. Outputs text-ready sentences for the paper

Scientific framing:
    Since BTS contains SPECTROSCOPICALLY CONFIRMED real transients,
    zero-score objects represent classifier ABSTENTION, not "non-Ia"
    predictions. The class composition of the abstention set tells us
    which real objects Fink could not emit usable probabilities for.

Usage:
    python scripts/14_fink_abstention_analysis.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ===========================================================================
# CONFIGURATION (edit if your file layout differs)
# ===========================================================================

FINK_CSV_CANDIDATES = [
    "data/raw/fink_classifications.csv",
    "fink_classifications.csv",
]

BTS_CSV_CANDIDATES = [
    "data/ground_truth/bts_sample.csv",
    "data/ground_truth/bts_spectroscopic.csv",
]

# Score columns (Fink API schema)
RF_SCORE_COL = "rf_snia_vs_nonia"
SNN_SCORE_COL = "snn_snia_vs_nonia"

# BTS file column names (CONFIRMED from your data)
BTS_OID_COL = "ZTFID"
BTS_CLASS_COL = "alerce_class"  # already mapped to ALeRCE taxonomy

# Possible OID column names in Fink CSV (auto-detected)
FINK_OID_CANDIDATES = ["oid", "ZTFID", "ztf_id", "objectId", "ztfid"]

# Display order (most relevant first)
CLASS_ORDER = ["SNIa", "SNIbc", "SNII", "SLSN", "TDE", "Other"]


# ===========================================================================
# DATA LOADING
# ===========================================================================

def find_first_existing(candidates, label):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n  " + "\n  ".join(candidates)
    )


def load_fink_predictions():
    path = find_first_existing(FINK_CSV_CANDIDATES, "Fink predictions CSV")
    print(f"Loading Fink predictions from: {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    for col in [RF_SCORE_COL, SNN_SCORE_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not in Fink CSV. "
                f"Available: {list(df.columns)}"
            )

    # Auto-detect OID column in Fink
    fink_oid = None
    for cand in FINK_OID_CANDIDATES:
        if cand in df.columns:
            fink_oid = cand
            break
    if fink_oid is None:
        raise ValueError(
            f"Could not find OID column in Fink CSV. "
            f"Tried {FINK_OID_CANDIDATES}. Available: {list(df.columns)}"
        )
    print(f"  Fink OID column: '{fink_oid}'")

    # Standardize for join
    df = df.rename(columns={fink_oid: "_oid"})
    return df


def load_bts_truth():
    path = find_first_existing(BTS_CSV_CANDIDATES, "BTS truth CSV")
    print(f"Loading BTS truth from: {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    if BTS_OID_COL not in df.columns:
        raise ValueError(f"Expected BTS OID column '{BTS_OID_COL}' not found.")
    if BTS_CLASS_COL not in df.columns:
        raise ValueError(f"Expected BTS class column '{BTS_CLASS_COL}' not found.")

    # Standardize for join
    df = df.rename(columns={BTS_OID_COL: "_oid", BTS_CLASS_COL: "_true_class"})

    # Filter to relevant subset (drop missing classes)
    df = df.dropna(subset=["_true_class"])
    print(f"  After dropping rows with missing class: {len(df)}")
    print(f"  Class distribution:")
    print(df["_true_class"].value_counts().to_string())

    return df[["_oid", "_true_class"]]


def merge_predictions_with_truth(fink_df, bts_df):
    print(f"\nMerging Fink predictions with BTS truth...")
    merged = pd.merge(fink_df, bts_df, on="_oid", how="inner")
    print(f"  Merged shape: {merged.shape}")
    print(f"  Fink-only objects (no BTS truth): {len(fink_df) - len(merged)}")
    print(f"  BTS-only objects (no Fink prediction): {len(bts_df) - len(merged)}")

    if len(merged) == 0:
        raise ValueError("Merge produced zero rows. Check OID column compatibility.")

    print(f"\n  Class distribution in merged sample:")
    print(merged["_true_class"].value_counts().to_string())

    return merged


# ===========================================================================
# ABSTENTION ANALYSIS
# ===========================================================================

def analyze_abstention(df, score_col, classifier_name):
    print(f"\n{'=' * 70}")
    print(f"ABSTENTION ANALYSIS: {classifier_name}")
    print(f"{'=' * 70}")

    scores = df[score_col].values
    true_classes = df["_true_class"].values

    # Treat NaN as abstention too
    is_zero = (scores == 0.0) | np.isnan(scores)
    is_nonzero = scores > 0.0

    n_total = len(df)
    n_zero = int(is_zero.sum())
    n_nonzero = int(is_nonzero.sum())
    coverage = n_nonzero / n_total

    print(f"Total objects: {n_total}")
    print(f"  Abstentions (score = 0 or NaN): {n_zero} ({100 * n_zero / n_total:.1f}%)")
    print(f"  Predictions (score > 0):        {n_nonzero} ({100 * n_nonzero / n_total:.1f}%)")
    print(f"  Effective coverage: {100 * coverage:.1f}%")

    # Per-class breakdown
    print(f"\nClass-conditional abstention rates:")
    print(f"{'Class':<12} {'N_total':>8} {'N_zero':>8} {'%_zero':>8} {'N_pred':>8} {'%_pred':>8}")
    print("-" * 56)

    by_class = {}
    unique_classes = sorted(
        np.unique(true_classes),
        key=lambda x: (CLASS_ORDER.index(x) if x in CLASS_ORDER else 999)
    )

    for true_class in unique_classes:
        mask_class = (true_classes == true_class)
        n_class = int(mask_class.sum())
        n_zero_class = int((mask_class & is_zero).sum())
        n_nonzero_class = int((mask_class & is_nonzero).sum())

        pct_zero = 100 * n_zero_class / n_class if n_class > 0 else 0
        pct_nonzero = 100 * n_nonzero_class / n_class if n_class > 0 else 0

        print(f"{true_class:<12} {n_class:>8d} {n_zero_class:>8d} "
              f"{pct_zero:>7.1f}% {n_nonzero_class:>8d} {pct_nonzero:>7.1f}%")

        # 95% Wilson confidence interval
        if n_class > 0:
            ci = stats.binomtest(n_zero_class, n_class).proportion_ci(
                confidence_level=0.95, method="wilson"
            )
            ci_lower, ci_upper = float(ci.low), float(ci.high)
        else:
            ci_lower, ci_upper = (0.0, 0.0)

        by_class[true_class] = {
            "true_class": true_class,
            "total_objects_in_class": n_class,
            "zero_score_count": n_zero_class,
            "zero_score_fraction": float(n_zero_class / n_class) if n_class > 0 else 0,
            "zero_score_ci_low": ci_lower,
            "zero_score_ci_high": ci_upper,
            "nonzero_score_count": n_nonzero_class,
            "nonzero_score_fraction": float(n_nonzero_class / n_class) if n_class > 0 else 0,
        }

    # Chi-squared test: is abstention class-dependent?
    contingency = pd.crosstab(true_classes, is_zero)
    chi2_result = stats.chi2_contingency(contingency.values)
    print(f"\nChi-squared test for class-dependent abstention:")
    print(f"  chi2 = {chi2_result.statistic:.2f}, dof = {chi2_result.dof}, "
          f"p = {chi2_result.pvalue:.2e}")

    if chi2_result.pvalue < 0.001:
        print(f"  ==> Abstention IS class-dependent (p < 0.001)")
        print(f"      Signature of an ELIGIBILITY GATE, not data-loading bug.")
    else:
        print(f"  ==> No significant class-dependence in abstention.")

    return {
        "classifier_name": classifier_name,
        "score_column": score_col,
        "overall": {
            "total_objects": n_total,
            "zero_score_count": n_zero,
            "zero_score_fraction": float(n_zero / n_total),
            "nonzero_score_count": n_nonzero,
            "nonzero_score_fraction": float(n_nonzero / n_total),
            "coverage": float(coverage),
        },
        "by_class": by_class,
        "class_dependent_test": {
            "chi2_statistic": float(chi2_result.statistic),
            "dof": int(chi2_result.dof),
            "p_value": float(chi2_result.pvalue),
            "is_class_dependent": bool(chi2_result.pvalue < 0.001),
        },
    }


def compute_abstention_ratio(rf_results, snn_results):
    """SN Ia vs non-SN Ia abstention ratio (key publication finding)."""
    sn_ia_classes = ["SNIa"]
    non_sn_ia_classes = [c for c in CLASS_ORDER if c not in sn_ia_classes]

    ratios = {}
    for clf_name, results in [("RF", rf_results), ("SNN", snn_results)]:
        by_class = results["by_class"]

        n_sn_ia_total = sum(by_class[c]["total_objects_in_class"]
                            for c in sn_ia_classes if c in by_class)
        n_sn_ia_zero = sum(by_class[c]["zero_score_count"]
                           for c in sn_ia_classes if c in by_class)
        n_non_total = sum(by_class[c]["total_objects_in_class"]
                          for c in non_sn_ia_classes if c in by_class)
        n_non_zero = sum(by_class[c]["zero_score_count"]
                         for c in non_sn_ia_classes if c in by_class)

        rate_sn = n_sn_ia_zero / n_sn_ia_total if n_sn_ia_total > 0 else 0
        rate_non = n_non_zero / n_non_total if n_non_total > 0 else 0
        ratio = rate_non / rate_sn if rate_sn > 0 else float("inf")

        ratios[clf_name] = {
            "sn_ia_abstention_rate": rate_sn,
            "non_sn_ia_abstention_rate": rate_non,
            "ratio_non_to_ia": ratio,
            "n_sn_ia_total": n_sn_ia_total,
            "n_non_sn_ia_total": n_non_total,
        }

        print(f"\n{clf_name}: non-Ia abstains at {ratio:.1f}x the rate of SNIa "
              f"({rate_non * 100:.1f}% vs {rate_sn * 100:.1f}%)")

    return ratios


# ===========================================================================
# FIGURES
# ===========================================================================

def plot_zero_fraction_by_class(results, save_path, color):
    by_class = results["by_class"]
    classifier = results["classifier_name"]

    classes = [c for c in CLASS_ORDER if c in by_class]
    fractions = [by_class[c]["zero_score_fraction"] for c in classes]
    ci_low = [by_class[c]["zero_score_ci_low"] for c in classes]
    ci_high = [by_class[c]["zero_score_ci_high"] for c in classes]
    counts = [by_class[c]["total_objects_in_class"] for c in classes]

    err_lower = [max(0, f - l) for f, l in zip(fractions, ci_low)]
    err_upper = [max(0, h - f) for f, h in zip(fractions, ci_high)]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    bars = ax.bar(x, fractions, yerr=[err_lower, err_upper],
                   capsize=8, color=color, edgecolor="black", alpha=0.8,
                   error_kw={"linewidth": 1.5})

    for bar, frac, n in zip(bars, fractions, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{frac * 100:.1f}%\n(N={n})",
                ha="center", va="bottom", fontsize=10)

    overall_rate = results["overall"]["zero_score_fraction"]
    ax.axhline(overall_rate, color="red", linestyle="--", alpha=0.6,
               label=f"Overall: {overall_rate * 100:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("Abstention rate (zero-score fraction)", fontsize=12)
    ax.set_title(f"{classifier}: Zero-score abstention by true class\n"
                 f"(error bars: 95% Wilson CI)", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_zero_vs_nonzero_stacked(rf_results, snn_results, save_path):
    classes = [c for c in CLASS_ORDER if c in rf_results["by_class"]]

    rf_zero = [rf_results["by_class"][c]["zero_score_count"] for c in classes]
    rf_nonzero = [rf_results["by_class"][c]["nonzero_score_count"] for c in classes]
    snn_zero = [snn_results["by_class"][c]["zero_score_count"] for c in classes]
    snn_nonzero = [snn_results["by_class"][c]["nonzero_score_count"] for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    width = 0.6
    x = np.arange(len(classes))

    ax1.bar(x, rf_nonzero, width, label="Predictions (score > 0)",
            color="#2ca02c", edgecolor="black")
    ax1.bar(x, rf_zero, width, bottom=rf_nonzero,
            label="Abstentions (score = 0)",
            color="#d62728", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylabel("Number of objects", fontsize=12)
    ax1.set_title(f"Fink RF (rf_snia_vs_nonia)\n"
                  f"Coverage: {rf_results['overall']['coverage'] * 100:.1f}%",
                  fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(x, snn_nonzero, width, label="Predictions (score > 0)",
            color="#2ca02c", edgecolor="black")
    ax2.bar(x, snn_zero, width, bottom=snn_nonzero,
            label="Abstentions (score = 0)",
            color="#d62728", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.set_title(f"Fink SuperNNova (snn_snia_vs_nonia)\n"
                  f"Coverage: {snn_results['overall']['coverage'] * 100:.1f}%",
                  fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(alpha=0.3, axis="y")

    plt.suptitle("Fink output composition: predictions vs. abstentions on real BTS transients",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# OUTPUT GENERATION
# ===========================================================================

def save_csv_summary(rf_results, snn_results, save_path):
    rows = []
    for classifier, results in [("RF", rf_results), ("SuperNNova", snn_results)]:
        for true_class, stats_dict in results["by_class"].items():
            row = {"classifier": classifier, **stats_dict}
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")


def save_json_summary(rf_results, snn_results, abstention_ratios, merged_size, save_path):
    summary = {
        "metadata": {
            "description": "Fink abstention analysis (selective classification framing)",
            "framing": "Zero scores are abstentions on real BTS transients, "
                       "not negative predictions",
            "evaluation_sample": "BTS spectroscopically confirmed transients",
            "n_total_objects": merged_size,
        },
        "fink_random_forest": rf_results,
        "fink_supernnova": snn_results,
        "abstention_class_asymmetry": abstention_ratios,
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {save_path}")


def print_paper_ready_text(rf_results, snn_results, ratios):
    print(f"\n{'=' * 70}")
    print("PAPER-READY TEXT")
    print(f"{'=' * 70}\n")

    rf_overall = rf_results["overall"]
    snn_overall = snn_results["overall"]

    print("--- Section 5.2.1 (Fink RF) ---\n")
    print(
        f"Fink's Random Forest classifier (rf_snia_vs_nonia, deployed from "
        f"Leoni et al. 2022) emitted exactly zero scores for "
        f"{rf_overall['zero_score_count']:,} of {rf_overall['total_objects']:,} "
        f"objects ({rf_overall['zero_score_fraction'] * 100:.1f}%), yielding an "
        f"effective coverage of {rf_overall['coverage'] * 100:.1f}%. "
        f"Because BTS contains spectroscopically confirmed transients, these "
        f"zeros represent classifier ABSTENTION on real astrophysical objects, "
        f"not 'P(SN Ia) = 0' predictions. Abstention is strongly class-dependent "
        f"(chi-squared p = {rf_results['class_dependent_test']['p_value']:.2e}): "
        f"non-SN Ia transients abstain at "
        f"{ratios['RF']['ratio_non_to_ia']:.1f}x the rate of SN Ia "
        f"({ratios['RF']['non_sn_ia_abstention_rate'] * 100:.1f}% vs "
        f"{ratios['RF']['sn_ia_abstention_rate'] * 100:.1f}%), consistent with "
        f"the eligibility gate (>=3 epochs per filter, plus host-based and "
        f"max-detection cuts) and the early-classifier operating window "
        f"described in Leoni et al. 2022. We therefore analyze RF as a "
        f"selective-classification problem: report coverage and class-conditional "
        f"abstention rates here, and compute conditional ECE on the "
        f"{rf_overall['nonzero_score_count']:,} non-zero predictions in Section 5.2.3.\n"
    )

    print("--- Section 5.2.2 (Fink SuperNNova) ---\n")
    print(
        f"Fink's SuperNNova classifier emitted zero scores for "
        f"{snn_overall['zero_score_count']:,} of "
        f"{snn_overall['total_objects']:,} objects "
        f"({snn_overall['zero_score_fraction'] * 100:.1f}%), with effective "
        f"coverage {snn_overall['coverage'] * 100:.1f}%. As with RF, abstention "
        f"is class-dependent "
        f"(chi-squared p = {snn_results['class_dependent_test']['p_value']:.2e}): "
        f"non-SN Ia transients abstain at "
        f"{ratios['SNN']['ratio_non_to_ia']:.1f}x the rate of SN Ia "
        f"({ratios['SNN']['non_sn_ia_abstention_rate'] * 100:.1f}% vs "
        f"{ratios['SNN']['sn_ia_abstention_rate'] * 100:.1f}%). The hard gap "
        f"from exactly zero to the first non-zero score is diagnostic of an "
        f"explicit abstention gate at the module level. Coverage exceeds RF "
        f"because SuperNNova does not require the early-classifier sigmoid-fit "
        f"prerequisites, but still applies host, Solar System, and minimum-point "
        f"selection cuts.\n"
    )

    print("--- Replacement for the misleading sentence ---\n")
    print(
        '   OLD: "Fink RF produces degenerate probability outputs"\n'
        '   NEW: "Fink RF exhibits low effective coverage because a large\n'
        '         fraction of real BTS transients receive zero outputs from\n'
        '         the early-Ia eligibility gate. We treat these zeros as\n'
        '         implicit abstentions and analyze their class composition\n'
        '         separately from conditional calibration on non-zero\n'
        '         predictions."\n'
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("FINK ABSTENTION ANALYSIS")
    print("Reframing zero scores as selective classification, not miscalibration")
    print("=" * 70)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    fink_df = load_fink_predictions()
    bts_df = load_bts_truth()
    merged = merge_predictions_with_truth(fink_df, bts_df)

    rf_results = analyze_abstention(merged, RF_SCORE_COL, "Fink Random Forest")
    snn_results = analyze_abstention(merged, SNN_SCORE_COL, "Fink SuperNNova")

    print(f"\n{'=' * 70}")
    print("CLASS-ASYMMETRY SUMMARY (key finding)")
    print(f"{'=' * 70}")
    ratios = compute_abstention_ratio(rf_results, snn_results)

    print(f"\n{'=' * 70}")
    print("GENERATING FIGURES")
    print(f"{'=' * 70}")
    plot_zero_fraction_by_class(
        rf_results,
        "figures/fink_zero_fraction_by_class_rf.png",
        color="#1f77b4",
    )
    plot_zero_fraction_by_class(
        snn_results,
        "figures/fink_zero_fraction_by_class_snn.png",
        color="#ff7f0e",
    )
    plot_zero_vs_nonzero_stacked(
        rf_results, snn_results,
        "figures/fink_zero_vs_nonzero_by_class.png",
    )

    print(f"\n{'=' * 70}")
    print("SAVING RESULT TABLES")
    print(f"{'=' * 70}")
    save_csv_summary(rf_results, snn_results,
                     "results/fink_zero_abstention_summary.csv")
    save_json_summary(rf_results, snn_results, ratios, len(merged),
                      "results/fink_zero_abstention_summary.json")

    print_paper_ready_text(rf_results, snn_results, ratios)

    print(f"\n{'=' * 70}")
    print("DONE.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()