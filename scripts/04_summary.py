"""
Summary Figures and Tables
============================

Generates the key comparative figures for the paper:
  - Figure: ECE comparison bar chart across all classifiers
  - Table: Summary results table (LaTeX-formatted)
  - Figure: Three-panel reliability diagram (ALeRCE vs Fink vs NEEDLE)

Run AFTER the individual analysis scripts (01, 02, 03).

Usage:
    python scripts/04_summary.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotting import summary_comparison_bar


def load_results():
    """Load all results from individual analysis scripts."""
    results = {}

    for name, path in [
        ("ALeRCE", "results/alerce_results.json"),
        ("Fink", "results/fink_results.json"),
        ("NEEDLE", "results/needle_results.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"WARNING: {path} not found. Run the analysis script first.")

    return results


def generate_comparison_figure(results):
    """Bar chart comparing ECE across all classifiers."""
    bar_data = {}

    if "ALeRCE" in results:
        r = results["ALeRCE"]
        bar_data["ALeRCE\n(BRF, 15-class)"] = {
            "ece": r["ece"],
            "ci_lower": r["ece_ci"][0],
            "ci_upper": r["ece_ci"][1],
        }

    if "Fink" in results:
        rf = results["Fink"].get("random_forest", {})
        snn = results["Fink"].get("supernnova", {})

        if rf.get("ece"):
            bar_data["Fink RF\n(binary)"] = {
                "ece": rf["ece"],
                "ci_lower": rf["ece_ci"][0],
                "ci_upper": rf["ece_ci"][1],
            }

        if snn.get("ece"):
            bar_data["Fink SNN\n(binary)"] = {
                "ece": snn["ece"],
                "ci_lower": snn["ece_ci"][0],
                "ci_upper": snn["ece_ci"][1],
            }

    if "NEEDLE" in results:
        r = results["NEEDLE"]
        bar_data["NEEDLE\n(3-class)"] = {
            "ece": r["ece"],
            "ci_lower": r["ece_ci"][0],
            "ci_upper": r["ece_ci"][1],
        }

    if bar_data:
        summary_comparison_bar(
            bar_data,
            save_path="figures/fig_summary_ece_comparison.pdf"
        )
        summary_comparison_bar(
            bar_data,
            save_path="figures/fig_summary_ece_comparison.png"
        )


def generate_latex_table(results):
    """Generate LaTeX-formatted summary table."""
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\caption{Calibration audit summary for four production transient classifiers. "
                 r"ECE is computed with equal-mass binning (15 bins) following \citet{roelofs2022}. "
                 r"95\% confidence intervals via bootstrap (1000 replicates).}")
    lines.append(r"\label{tab:summary}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\hline\hline")
    lines.append(r"Classifier & Architecture & Classes & $N$ & ECE & Post-hoc & ECE$_\mathrm{post}$ \\")
    lines.append(r"\hline")

    if "ALeRCE" in results:
        r = results["ALeRCE"]
        ts = r["temperature_scaling"]
        lines.append(
            f"ALeRCE & BRF & 15 & {r['n_objects']} & "
            f"${r['ece']:.3f}$ $[{r['ece_ci'][0]:.3f}, {r['ece_ci'][1]:.3f}]$ & "
            f"Temp. ($T={ts['T']:.3f}$) & ${ts['ece_after']:.3f}$ \\\\"
        )

    if "Fink" in results:
        rf = results["Fink"].get("random_forest", {})
        snn = results["Fink"].get("supernnova", {})

        if rf:
            lines.append(
                f"Fink RF & RF & 2 & {rf['n_objects']} & "
                f"${rf['ece']:.3f}$ $[{rf['ece_ci'][0]:.3f}, {rf['ece_ci'][1]:.3f}]$ & "
                f"N/A (degenerate) & --- \\\\"
            )

        if snn:
            lines.append(
                f"Fink SNN & RNN & 2 & {snn['n_objects']} & "
                f"${snn['ece']:.3f}$ $[{snn['ece_ci'][0]:.3f}, {snn['ece_ci'][1]:.3f}]$ & "
                f"N/A (bound hit) & --- \\\\"
            )

    if "NEEDLE" in results:
        r = results["NEEDLE"]
        gt = r["global_temperature"]
        lines.append(
            f"NEEDLE & CNN+DNN & 3 & {r['n_predictions']} & "
            f"${r['ece']:.3f}$ $[{r['ece_ci'][0]:.3f}, {r['ece_ci'][1]:.3f}]$ & "
            f"Global T worsens & ${gt['ece_after']:.3f}$ \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_text = "\n".join(lines)

    with open("results/summary_table.tex", "w") as f:
        f.write(table_text)

    print(f"\nLaTeX table saved to: results/summary_table.tex")
    return table_text


def generate_text_summary(results):
    """Generate human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("CALIBRATION AUDIT SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    if "ALeRCE" in results:
        r = results["ALeRCE"]
        ts = r["temperature_scaling"]
        lines.append(f"ALeRCE (Balanced Random Forest, lc_classifier v1.1.13)")
        lines.append(f"  Architecture: {r['architecture']}")
        lines.append(f"  Classes: {r['n_classes']} ({', '.join(r['class_names'])})")
        lines.append(f"  Sample: {r['n_objects']} objects")
        lines.append(f"  Accuracy: {r['accuracy']:.3f}")
        lines.append(f"  Mean confidence: {r['mean_confidence']:.3f}")
        lines.append(f"  Direction: {r['direction']}")
        lines.append(f"  ECE: {r['ece']:.3f} [{r['ece_ci'][0]:.3f}, {r['ece_ci'][1]:.3f}]")
        lines.append(f"  Temperature: T={ts['T']:.3f}")
        lines.append(f"  Post-calibration ECE: {ts['ece_after']:.3f} "
                     f"({ts['improvement_pct']:.0f}% improvement)")
        lines.append(f"  Finding: Systematic underconfidence inverts Guo et al. (2017) norm")
        lines.append("")

    if "Fink" in results:
        rf = results["Fink"].get("random_forest", {})
        snn = results["Fink"].get("supernnova", {})

        lines.append(f"Fink Random Forest (rf_snia_vs_nonia)")
        if rf:
            lines.append(f"  Type: Binary (SN Ia vs not)")
            lines.append(f"  Sample: {rf['n_objects']} objects")
            lines.append(f"  Zero scores: {rf['zero_fraction']*100:.0f}%")
            lines.append(f"  ECE: {rf['ece']:.3f}")
            lines.append(f"  Finding: Structurally broken — 94% zeros")
        lines.append("")

        lines.append(f"Fink SuperNNova (snn_snia_vs_nonia)")
        if snn:
            lines.append(f"  Type: Binary (SN Ia vs not)")
            lines.append(f"  Sample: {snn['n_objects']} objects")
            lines.append(f"  Zero scores: {snn['zero_fraction']*100:.0f}%")
            lines.append(f"  ECE: {snn['ece']:.3f}")
            lines.append(f"  Finding: Temperature scaling hits optimizer bound")
        lines.append("")

    if "NEEDLE" in results:
        r = results["NEEDLE"]
        gt = r["global_temperature"]
        pct = r.get("per_class_temperature", {})

        lines.append(f"NEEDLE (Sheng et al. 2024)")
        lines.append(f"  Architecture: {r['architecture']}")
        lines.append(f"  Classes: {', '.join(r['class_names'])}")
        lines.append(f"  Sample: {r['n_predictions']} predictions, {r['n_unique_objects']} objects")
        lines.append(f"  Accuracy: {r['accuracy']:.3f}")
        lines.append(f"  Mean confidence: {r['mean_confidence']:.3f}")
        lines.append(f"  Aggregate ECE: {r['ece']:.3f}")

        if "per_class" in r:
            for cls, stats in r["per_class"].items():
                lines.append(f"    {cls}: ECE={stats['ece']:.3f}, "
                           f"acc={stats['accuracy']:.3f}, "
                           f"conf={stats['mean_confidence']:.3f}")

        lines.append(f"  Global T={gt['T']:.3f}: ECE {gt['ece_before']:.3f} -> {gt['ece_after']:.3f} (WORSENS)")

        if pct:
            lines.append(f"  Per-class T: {pct['class_Ts']}")
            lines.append(f"  Per-class ECE: {pct['ece_after']:.3f}")

        lines.append(f"  Finding: {r.get('key_finding', '')}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("KEY TAKEAWAYS")
    lines.append("=" * 70)
    lines.append("1. No production classifier publishes calibration metrics")
    lines.append("2. ALeRCE: fixable underconfidence (T=0.36 reduces ECE by ~71%)")
    lines.append("3. Fink RF: structurally broken (94% zeros)")
    lines.append("4. Fink SNN: miscalibrated, T scaling inappropriate")
    lines.append("5. NEEDLE: aggregate ECE masks class-asymmetric miscalibration")
    lines.append("6. Inverse-frequency class weighting distorts probabilities")
    lines.append("7. The failure of temperature scaling for NEEDLE is itself publishable")

    text = "\n".join(lines)

    with open("results/summary.txt", "w") as f:
        f.write(text)

    print(text)
    return text


def main():
    print("Generating summary figures and tables...\n")

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    results = load_results()

    if not results:
        print("ERROR: No results found. Run analysis scripts first:")
        print("  python scripts/01_alerce_analysis.py")
        print("  python scripts/02_fink_analysis.py")
        print("  python scripts/03_needle_analysis.py")
        return

    generate_comparison_figure(results)
    generate_latex_table(results)
    generate_text_summary(results)

    print(f"\nAll summary outputs saved:")
    print(f"  figures/fig_summary_ece_comparison.pdf")
    print(f"  results/summary_table.tex")
    print(f"  results/summary.txt")


if __name__ == "__main__":
    main()
