"""
Class-Dependent Missingness Analysis
=====================================

Purpose:
    Quantify whether broker-prediction missingness (objects in the BTS
    evaluation sample for which no API response was retrieved) is correlated
    with class, brightness, or epoch. Directly addresses reviewer concern:

    "The manuscript reports that ALeRCE predictions were retrieved for 1,149
     of 1,436 sampled objects and Fink predictions for 1,237 of 1,436, but
     it does not quantify whether the missing objects differ systematically
     by class, phase, brightness, or alert-history quality."

What this script does:
    1. Identifies BTS objects that are MISSING from each broker's predictions
    2. Computes class-conditional missingness rates with Wilson 95% CIs
    3. Tests for class-dependent missingness via chi-squared
    4. Compares missing-vs-retained on observable BTS properties (peakmag,
       duration, redshift) where available
    5. Generates a missingness table for paper

Outputs:
    results/missingness_analysis.json
    results/missingness_table.csv
    figures/missingness_by_class.pdf

Usage:
    python3 scripts/18_missingness_analysis.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ===========================================================================
# CONFIGURATION
# ===========================================================================

ALERCE_CSV_CANDIDATES = [
    "data/raw/alerce_classifications.csv",
]

FINK_CSV_CANDIDATES = [
    "data/raw/fink_classifications.csv",
]

BTS_CSV_CANDIDATES = [
    "data/ground_truth/bts_sample.csv",
]

CLASS_ORDER = ["SNIa", "SNIbc", "SNII", "SLSN", "TDE", "Other"]


# ===========================================================================
# LOADING
# ===========================================================================

def find_first_existing(candidates, label):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {label}: tried {candidates}")


def load_bts_full():
    """Load full BTS evaluation sample with observable properties."""
    path = find_first_existing(BTS_CSV_CANDIDATES, "BTS CSV")
    print(f"Loading BTS: {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Standardize column names
    df = df.rename(columns={"ZTFID": "_oid", "alerce_class": "_true_class"})
    df = df.dropna(subset=["_true_class"])
    print(f"  After dropping missing class: {len(df)}")
    return df


def get_broker_oids(broker_csv_candidates, broker_name):
    """Get the set of OIDs for which the broker returned predictions."""
    path = find_first_existing(broker_csv_candidates, f"{broker_name} CSV")
    print(f"\nLoading {broker_name}: {path}")
    df = pd.read_csv(path)
    
    # Auto-detect OID column
    oid_candidates = ["oid", "ZTFID", "ztf_id", "objectId"]
    oid_col = next((c for c in oid_candidates if c in df.columns), None)
    if oid_col is None:
        raise ValueError(f"No OID column in {broker_name} CSV")
    
    oids = set(df[oid_col].dropna().astype(str).tolist())
    print(f"  {broker_name} returned predictions for: {len(oids)} unique OIDs")
    return oids


# ===========================================================================
# MISSINGNESS ANALYSIS
# ===========================================================================

def per_class_missingness(bts_df, broker_oids, broker_name):
    """
    For each true class, compute:
      - n_total: BTS objects of that class
      - n_retrieved: objects with broker predictions
      - n_missing: objects without broker predictions
      - missing_rate with Wilson 95% CI
    """
    bts_df = bts_df.copy()
    bts_df["_oid_str"] = bts_df["_oid"].astype(str)
    bts_df["_retrieved"] = bts_df["_oid_str"].isin(broker_oids)

    print(f"\n{'='*80}")
    print(f"MISSINGNESS BY CLASS: {broker_name}")
    print(f"{'='*80}")
    print(f"{'Class':<10} {'n_total':>9} {'n_retrieved':>12} {'n_missing':>10} "
          f"{'missing_rate':>14} {'95% CI':>22}")
    print("-" * 90)

    by_class = {}
    for cls in CLASS_ORDER:
        sub = bts_df[bts_df["_true_class"] == cls]
        n_total = len(sub)
        if n_total == 0:
            continue
        n_retrieved = int(sub["_retrieved"].sum())
        n_missing = n_total - n_retrieved
        missing_rate = n_missing / n_total

        # Wilson 95% CI for missing_rate
        ci_obj = stats.binomtest(n_missing, n_total).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        ci_low, ci_high = float(ci_obj.low), float(ci_obj.high)

        ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]"
        print(f"{cls:<10} {n_total:>9d} {n_retrieved:>12d} {n_missing:>10d} "
              f"{missing_rate:>14.3f} {ci_str:>22}")

        by_class[cls] = {
            "n_total": n_total,
            "n_retrieved": n_retrieved,
            "n_missing": n_missing,
            "missing_rate": float(missing_rate),
            "missing_rate_ci": [ci_low, ci_high],
        }

    # Chi-squared test for class-dependent missingness
    contingency = pd.crosstab(bts_df["_true_class"], bts_df["_retrieved"])
    chi2_result = stats.chi2_contingency(contingency.values)
    print(f"\nChi-squared test for class-dependent missingness:")
    print(f"  chi2 = {chi2_result.statistic:.2f}, dof = {chi2_result.dof}, "
          f"p = {chi2_result.pvalue:.2e}")
    if chi2_result.pvalue < 0.001:
        print(f"  ==> Missingness IS class-dependent (p < 0.001)")
    elif chi2_result.pvalue < 0.05:
        print(f"  ==> Missingness is statistically class-dependent (p < 0.05) but weakly")
    else:
        print(f"  ==> No significant class-dependence in missingness")

    return {
        "by_class": by_class,
        "chi2_statistic": float(chi2_result.statistic),
        "dof": int(chi2_result.dof),
        "p_value": float(chi2_result.pvalue),
        "is_class_dependent": bool(chi2_result.pvalue < 0.001),
    }


def compare_observable_properties(bts_df, broker_oids, broker_name):
    """
    Compare BTS observable properties (peakmag, duration, redshift) between
    retrieved and missing objects.
    """
    bts_df = bts_df.copy()
    bts_df["_oid_str"] = bts_df["_oid"].astype(str)
    bts_df["_retrieved"] = bts_df["_oid_str"].isin(broker_oids)

    print(f"\n{'='*80}")
    print(f"OBSERVABLE PROPERTIES: retrieved vs missing ({broker_name})")
    print(f"{'='*80}")

    # Properties to test
    candidate_columns = ["peakmag", "duration", "redshift", "peakabs"]
    available = [c for c in candidate_columns if c in bts_df.columns]
    print(f"Available observable columns: {available}")

    out = {}
    for col in available:
        # Coerce to numeric (strings like '0.073' may be present)
        x = pd.to_numeric(bts_df[col], errors="coerce")
        retrieved_vals = x[bts_df["_retrieved"]].dropna()
        missing_vals = x[~bts_df["_retrieved"]].dropna()

        if len(retrieved_vals) < 10 or len(missing_vals) < 10:
            print(f"  {col}: too few values (retrieved {len(retrieved_vals)}, missing {len(missing_vals)})")
            continue

        # Mann-Whitney U test (non-parametric)
        stat, p = stats.mannwhitneyu(retrieved_vals, missing_vals, alternative="two-sided")
        med_retrieved = float(retrieved_vals.median())
        med_missing = float(missing_vals.median())

        print(f"  {col}: median retrieved = {med_retrieved:.3f} (n={len(retrieved_vals)}), "
              f"median missing = {med_missing:.3f} (n={len(missing_vals)}), "
              f"Mann-Whitney p = {p:.3e}")

        out[col] = {
            "median_retrieved": med_retrieved,
            "median_missing": med_missing,
            "n_retrieved": int(len(retrieved_vals)),
            "n_missing": int(len(missing_vals)),
            "mannwhitney_statistic": float(stat),
            "mannwhitney_p": float(p),
            "significant_at_0.05": bool(p < 0.05),
        }

    return out


# ===========================================================================
# FIGURES
# ===========================================================================

def plot_missingness_comparison(alerce_results, fink_results, save_path):
    """Side-by-side bar chart of missingness rates by class for both brokers."""
    classes = [c for c in CLASS_ORDER if c in alerce_results["by_class"] or c in fink_results["by_class"]]

    alerce_rates = []
    alerce_err = []
    fink_rates = []
    fink_err = []
    for c in classes:
        a = alerce_results["by_class"].get(c, {"missing_rate": 0, "missing_rate_ci": [0, 0]})
        f = fink_results["by_class"].get(c, {"missing_rate": 0, "missing_rate_ci": [0, 0]})
        alerce_rates.append(a["missing_rate"])
        alerce_err.append([
            max(0, a["missing_rate"] - a["missing_rate_ci"][0]),
            max(0, a["missing_rate_ci"][1] - a["missing_rate"]),
        ])
        fink_rates.append(f["missing_rate"])
        fink_err.append([
            max(0, f["missing_rate"] - f["missing_rate_ci"][0]),
            max(0, f["missing_rate_ci"][1] - f["missing_rate"]),
        ])

    alerce_err = np.array(alerce_err).T
    fink_err = np.array(fink_err).T

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(classes))
    width = 0.35
    ax.bar(x - width/2, alerce_rates, width, yerr=alerce_err, capsize=6,
           label=f"ALeRCE (chi² p = {alerce_results['p_value']:.2e})",
           color="#1f77b4", edgecolor="black", alpha=0.85,
           error_kw={"linewidth": 1.5})
    ax.bar(x + width/2, fink_rates, width, yerr=fink_err, capsize=6,
           label=f"Fink (chi² p = {fink_results['p_value']:.2e})",
           color="#ff7f0e", edgecolor="black", alpha=0.85,
           error_kw={"linewidth": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("Class-conditional missing rate", fontsize=12)
    ax.set_title("Missingness in broker predictions by true BTS class\n"
                 "(error bars: 95% Wilson CI)", fontsize=12)
    ax.set_ylim(0, max(max(alerce_rates), max(fink_rates)) * 1.3 + 0.05)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 80)
    print("CLASS-DEPENDENT MISSINGNESS ANALYSIS")
    print("=" * 80)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load BTS
    bts = load_bts_full()
    n_total = len(bts)

    # Get broker OIDs (sets of objects with predictions)
    alerce_oids = get_broker_oids(ALERCE_CSV_CANDIDATES, "ALeRCE")
    fink_oids = get_broker_oids(FINK_CSV_CANDIDATES, "Fink")

    # Per-class missingness
    alerce_miss = per_class_missingness(bts, alerce_oids, "ALeRCE")
    fink_miss = per_class_missingness(bts, fink_oids, "Fink")

    # Observable property tests
    alerce_obs = compare_observable_properties(bts, alerce_oids, "ALeRCE")
    fink_obs = compare_observable_properties(bts, fink_oids, "Fink")

    # Save
    out = {
        "metadata": {
            "description": "Class-dependent missingness in broker predictions",
            "addresses": "Reviewer concern about retrieval rates and bias direction",
            "n_bts_total": n_total,
            "n_alerce_retrieved": len(alerce_oids & set(bts["_oid"].astype(str))),
            "n_fink_retrieved": len(fink_oids & set(bts["_oid"].astype(str))),
        },
        "alerce": {
            **alerce_miss,
            "observable_property_tests": alerce_obs,
        },
        "fink": {
            **fink_miss,
            "observable_property_tests": fink_obs,
        },
    }
    with open("results/missingness_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/missingness_analysis.json")

    # CSV
    rows = []
    for broker, miss in [("ALeRCE", alerce_miss), ("Fink", fink_miss)]:
        for cls, stats_dict in miss["by_class"].items():
            rows.append({
                "broker": broker,
                "class": cls,
                **stats_dict,
                "ci_low": stats_dict["missing_rate_ci"][0],
                "ci_high": stats_dict["missing_rate_ci"][1],
            })
    df = pd.DataFrame(rows)
    df = df.drop(columns=["missing_rate_ci"])
    df.to_csv("results/missingness_table.csv", index=False)
    print(f"Saved: results/missingness_table.csv")

    # Figure
    plot_missingness_comparison(alerce_miss, fink_miss,
                                  "figures/missingness_by_class.pdf")

    # Paper-ready text
    print(f"\n{'=' * 80}")
    print("PAPER-READY TEXT (insert as §3.4 Missingness Analysis)")
    print(f"{'=' * 80}\n")
    print(
        f"Of the {n_total} BTS objects in our evaluation sample, broker predictions "
        f"were retrieved for "
        f"{len(alerce_oids & set(bts['_oid'].astype(str)))} (ALeRCE) and "
        f"{len(fink_oids & set(bts['_oid'].astype(str)))} (Fink). "
        f"Missingness was tested for class-dependence via chi-squared. "
    )
    if alerce_miss["is_class_dependent"]:
        print(
            f"For ALeRCE, missingness is significantly class-dependent "
            f"(chi² p = {alerce_miss['p_value']:.2e}); per-class missing "
            f"rates and 95% Wilson CIs are reported in Table X."
        )
    else:
        print(
            f"For ALeRCE, missingness is not significantly class-dependent "
            f"(chi² p = {alerce_miss['p_value']:.2e})."
        )
    if fink_miss["is_class_dependent"]:
        print(
            f"For Fink, missingness is significantly class-dependent "
            f"(chi² p = {fink_miss['p_value']:.2e})."
        )
    else:
        print(
            f"For Fink, missingness is not significantly class-dependent "
            f"(chi² p = {fink_miss['p_value']:.2e})."
        )

    print(f"\n{'=' * 80}")
    print("DONE.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
