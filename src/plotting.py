"""
Publication-Quality Calibration Plots
======================================

Reliability diagrams, confidence histograms, and comparison
figures for The Astronomical Journal.

All figures follow AJ guidelines:
  - PDF/EPS format, >= 300 DPI
  - Single-column width (~3.5 inches) or double-column (~7 inches)
  - Colorblind-safe palette
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
    "black": "#000000",
}

# Default figure settings for AJ
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def reliability_diagram(bin_data, title="", save_path=None,
                        figsize=(3.5, 4.5), show_histogram=True,
                        show_gap=True, color=None):
    """
    Single reliability diagram with confidence histogram.

    Parameters
    ----------
    bin_data : list of dict
        Output from compute_ece() — each dict has
        'confidence', 'accuracy', 'count', 'gap'.
    title : str
        Figure title.
    save_path : str or None
        Path to save figure (PDF recommended for AJ).
    show_histogram : bool
        Show sample count histogram below the diagram.
    show_gap : bool
        Show gap bars between accuracy and confidence.
    color : str or None
        Bar color. Defaults to blue.
    """
    if color is None:
        color = COLORS["blue"]

    confs = [b["confidence"] for b in bin_data]
    accs = [b["accuracy"] for b in bin_data]
    counts = [b["count"] for b in bin_data]
    gaps = [b["gap"] for b in bin_data]

    # Compute ECE for annotation
    total = sum(counts)
    ece = sum(abs(g) * c / total for g, c in zip(gaps, counts))

    if show_histogram:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    # Bar width based on bin edges
    bar_width = 0.8 / len(bin_data) if len(bin_data) > 10 else 0.06

    # Accuracy bars
    ax1.bar(confs, accs, width=bar_width, alpha=0.7, color=color,
            edgecolor="white", linewidth=0.5, label="Observed")

    # Gap visualization
    if show_gap:
        for c, a, g in zip(confs, accs, gaps):
            if g > 0:  # Underconfident
                ax1.bar(c, g, bottom=a - g, width=bar_width,
                        alpha=0.3, color=COLORS["red"], edgecolor="none")
            else:  # Overconfident
                ax1.bar(c, abs(g), bottom=a, width=bar_width,
                        alpha=0.3, color=COLORS["orange"], edgecolor="none")

    ax1.set_ylabel("Accuracy")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left", framealpha=0.9)

    # ECE annotation
    ax1.text(0.95, 0.05, f"ECE = {ece:.3f}",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="grey", alpha=0.9))

    if title:
        ax1.set_title(title)

    # Histogram
    if ax2 is not None:
        ax2.bar(confs, counts, width=bar_width, color=color,
                alpha=0.5, edgecolor="white", linewidth=0.5)
        ax2.set_xlabel("Confidence")
        ax2.set_ylabel("Count")
        ax2.set_xlim(0, 1)
        plt.setp(ax1.get_xticklabels(), visible=False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def reliability_diagram_comparison(bin_data_before, bin_data_after,
                                   T_value=None, title="",
                                   save_path=None, figsize=(7, 4.5)):
    """
    Side-by-side reliability diagrams: before and after calibration.

    Used for ALeRCE temperature scaling figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, bin_data, label, color in [
        (ax1, bin_data_before, "Before", COLORS["red"]),
        (ax2, bin_data_after, "After", COLORS["green"]),
    ]:
        confs = [b["confidence"] for b in bin_data]
        accs = [b["accuracy"] for b in bin_data]
        counts = [b["count"] for b in bin_data]
        gaps = [b["gap"] for b in bin_data]

        total = sum(counts)
        ece = sum(abs(g) * c / total for g, c in zip(gaps, counts))

        bar_width = 0.06
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.bar(confs, accs, width=bar_width, alpha=0.7, color=color,
               edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label} (ECE = {ece:.3f})")
        ax.set_aspect("equal")

    if T_value:
        fig.suptitle(f"{title} — Temperature Scaling (T = {T_value:.3f})",
                     fontsize=11, fontweight="bold")
    elif title:
        fig.suptitle(title, fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def per_class_reliability_grid(class_results, class_bin_data,
                               title="", save_path=None,
                               figsize=None):
    """
    Grid of per-class reliability diagrams.

    Parameters
    ----------
    class_results : dict
        Output from compute_classwise_ece().
    class_bin_data : dict
        {class_name: bin_data_list} for each class.
    """
    class_names = list(class_results.keys())
    n_classes = len(class_names)

    if figsize is None:
        cols = min(n_classes, 3)
        rows = (n_classes + cols - 1) // cols
        figsize = (3.5 * cols, 3.5 * rows)

    fig, axes = plt.subplots(
        (n_classes + 2) // 3, min(3, n_classes),
        figsize=figsize, squeeze=False
    )
    axes = axes.flatten()

    colors_list = [COLORS["blue"], COLORS["orange"], COLORS["green"],
                   COLORS["red"], COLORS["purple"]]

    for i, cls_name in enumerate(class_names):
        ax = axes[i]
        stats = class_results[cls_name]
        color = colors_list[i % len(colors_list)]

        if cls_name in class_bin_data:
            bins = class_bin_data[cls_name]
            confs = [b["confidence"] for b in bins]
            accs = [b["accuracy"] for b in bins]

            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.bar(confs, accs, width=0.06, alpha=0.7, color=color,
                   edgecolor="white", linewidth=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{cls_name}\nECE={stats['ece']:.3f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_aspect("equal")

    # Hide unused axes
    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def summary_comparison_bar(results_dict, save_path=None, figsize=(7, 4)):
    """
    Bar chart comparing ECE across all classifiers.

    Parameters
    ----------
    results_dict : dict
        {classifier_name: {"ece": float, "ci_lower": float, "ci_upper": float}}
    """
    names = list(results_dict.keys())
    eces = [results_dict[n]["ece"] for n in names]

    # Error bars if CI available
    has_ci = all("ci_lower" in results_dict[n] for n in names)
    if has_ci:
        lower = [results_dict[n]["ece"] - results_dict[n]["ci_lower"] for n in names]
        upper = [results_dict[n]["ci_upper"] - results_dict[n]["ece"] for n in names]
        yerr = [lower, upper]
    else:
        yerr = None

    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"],
              COLORS["red"], COLORS["purple"]]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(range(len(names)), eces, yerr=yerr,
                  color=[colors[i % len(colors)] for i in range(len(names))],
                  alpha=0.8, edgecolor="white", linewidth=0.5,
                  capsize=4)

    # Reference lines
    ax.axhline(y=0.05, color=COLORS["grey"], linestyle=":", linewidth=1,
               label="Well-calibrated (ECE < 0.05)")
    ax.axhline(y=0.10, color=COLORS["grey"], linestyle="--", linewidth=1,
               label="Acceptable (ECE < 0.10)")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("Calibration Comparison Across Classifiers")
    ax.legend(loc="upper right", fontsize=8)

    # Value labels on bars
    for bar, ece in zip(bars, eces):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{ece:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig
