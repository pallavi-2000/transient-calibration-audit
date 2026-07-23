"""Poster figures for the A1 NAM poster (paper/poster/claude_design).

Generates large-font, print-ready figures from the audited results:
  1. Multi-broker ECE before/after global temperature scaling, with the
     fitted temperature and its direction (under- vs overconfident).
  2. Discrimination-calibration map: each audited score placed by its
     target-class ROC-AUC and its raw ECE, with the isotonic result for
     Fink RF shown as a vertical move (ECE collapses, ranking unchanged).
  3. Reliability diagram for ALeRCE, raw vs out-of-fold calibrated.

All values are read from results/tables/ rather than hard-coded, except
the audited constants below that are reported in the corresponding
result files (and cited in the poster caption).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "poster" / "claude_design" / "assets"

TEAL = "#1D9E75"
TEAL_DK = "#0F6E56"
NAVY = "#0C447C"
NAVY_DK = "#042C53"
CORAL = "#D85A30"
CORAL_DK = "#993C1D"
GRAY = "#B4B2A9"
INK = "#2C2C2A"

plt.rcParams.update(
    {
        "font.size": 17,
        "axes.labelsize": 18,
        "axes.titlesize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "axes.edgecolor": "#5F5E5A",
        "figure.facecolor": "white",
    }
)


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def broker_rows() -> list[dict]:
    """Audited calibration summary per broker score."""
    cv = json.loads((TABLES / "temperature_scaling_cv.json").read_text())
    fink_ts = json.loads((TABLES / "fink_temperature_scaling.json").read_text())

    # NEEDLE values come from the study's text summary (5-model pooled run).
    needle = {"ece_before": 0.0504, "ece_after": 0.1278, "T": 1.553}

    return [
        {
            "label": "ALeRCE\n15-class",
            "before": cv["baseline"]["ece"],
            "after": cv["global_T"]["ece"],
            "T": cv["global_T"]["T_mean"],
            "direction": "underconfident",
            "auc": 0.9155,
            "auc_note": "SNIa vs rest",
        },
        {
            "label": "NEEDLE\n3-class",
            "before": needle["ece_before"],
            "after": needle["ece_after"],
            "T": needle["T"],
            "direction": "overconfident",
            "auc": 0.9081,
            "auc_note": "macro one-vs-rest",
        },
        {
            "label": "Fink\nSuperNNova",
            "before": fink_ts["snn"]["ece_before"],
            "after": fink_ts["snn"]["ece_after"],
            "T": fink_ts["snn"]["T_optimal"],
            "direction": "no ranking signal",
            "auc": 0.5246,
            "auc_note": "SNIa vs rest",
        },
        {
            "label": "Fink\nRandom Forest",
            "before": fink_ts["rf"]["ece_before"],
            "after": fink_ts["rf"]["ece_after"],
            "T": fink_ts["rf"]["T_optimal"],
            "direction": "no ranking signal",
            "auc": 0.5021,
            "auc_note": "SNIa vs rest",
        },
    ]


def fig_broker_ece(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.7))
    x = np.arange(len(rows))
    w = 0.36
    before = [r["before"] for r in rows]
    after = [r["after"] for r in rows]

    ax.bar(x - w / 2, before, w, color=GRAY, label="raw probabilities")
    bars_after = ax.bar(x + w / 2, after, w, color=TEAL, label="after global temperature scaling")

    for i, r in enumerate(rows):
        worse = r["after"] > r["before"]
        if worse:
            bars_after[i].set_color(CORAL)
        elif r["T"] > 9:
            # Fink: T pinned at the optimiser bound. The nominal ECE drop is not
            # a usable gain because the underlying score carries no ranking, so
            # hatch it rather than let it read as a success.
            bars_after[i].set_color("white")
            bars_after[i].set_edgecolor(GRAY)
            bars_after[i].set_hatch("//")
            bars_after[i].set_linewidth(1.6)

    for i, r in enumerate(rows):
        ax.text(i - w / 2, r["before"] + 0.012, f"{r['before']:.3f}", ha="center", fontsize=15)
        worse = r["after"] > r["before"]
        ax.text(
            i + w / 2,
            r["after"] + 0.012,
            f"{r['after']:.3f}" + (" ✗" if worse else ""),
            ha="center",
            fontsize=15,
            color=CORAL_DK if worse else (GRAY if r["T"] > 9 else TEAL_DK),
            fontweight="bold" if worse else "normal",
        )
        tag = f"T = {r['T']:.2f}"
        if r["T"] > 9:
            tag += " (bound)"
        ax.annotate(tag, (i, 0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, -58),
                    ha="center", fontsize=14, color=INK, annotation_clip=False)
        ax.annotate(r["direction"], (i, 0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, -78),
                    ha="center", fontsize=13.5, color="#5F5E5A", style="italic",
                    annotation_clip=False)

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows])
    ax.tick_params(axis="x", length=0, pad=9)
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, 0.55)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.95)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_broker_ece.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_discrimination_map(rows: list[dict]) -> None:
    fink_extras = json.loads((TABLES / "fink_extras.json").read_text())
    iso_ece = fink_extras["rf"]["full"]["isotonic_cv"]["ece"]
    iso_auc = fink_extras["rf"]["full"]["roc_auc"]

    fig, ax = plt.subplots(figsize=(10.2, 6.6))

    # Left band: without usable ranking, no monotonic recalibration can help.
    ax.axvspan(0.44, 0.65, color=CORAL, alpha=0.07)
    ax.text(
        0.545, 0.625, "no usable ranking\ncalibration cannot help",
        ha="center", va="top", fontsize=13.5, color=CORAL_DK, style="italic",
    )

    styles = {
        "ALeRCE\n15-class": (NAVY, "o"),
        "NEEDLE\n3-class": (TEAL_DK, "s"),
        "Fink\nSuperNNova": (CORAL, "^"),
        "Fink\nRandom Forest": (CORAL_DK, "v"),
    }
    label_offsets = {
        "Fink\nRandom Forest": (8, 20, "left"),
        "Fink\nSuperNNova": (14, 2, "left"),
        "ALeRCE\n15-class": (-10, 16, "right"),
        "NEEDLE\n3-class": (-10, -12, "right"),
    }
    for r in rows:
        c, m = styles[r["label"]]
        ax.scatter(r["auc"], r["before"], s=260, color=c, marker=m, zorder=6,
                   edgecolor="white", linewidth=1.5)
        dx, dy, ha = label_offsets[r["label"]]
        ax.annotate(
            r["label"].replace("\n", " "), (r["auc"], r["before"]),
            textcoords="offset points", xytext=(dx, dy), ha=ha,
            fontsize=14.5, fontweight="bold", color=c,
        )

    # ALeRCE: temperature scaling drives ECE down; ranking is untouched.
    al = rows[0]
    ax_x = al["auc"] + 0.016
    ax.annotate("", xy=(ax_x, al["after"]), xytext=(ax_x, al["before"]),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=3.0))
    ax.scatter([al["auc"]], [al["after"]], s=150, color=TEAL, marker="o", zorder=6,
               edgecolor="white", linewidth=1.3)
    ax.text(ax_x + 0.008, (al["before"] + al["after"]) / 2,
            "T = 0.36\nrepaired\n→ 0.015", fontsize=13, color=TEAL_DK, va="center")

    # NEEDLE: the same global correction moves it the wrong way.
    nd = rows[1]
    nd_x = nd["auc"] - 0.016
    ax.annotate("", xy=(nd_x, nd["after"]), xytext=(nd_x, nd["before"]),
                arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=3.0))
    ax.text(nd_x - 0.008, nd["after"] + 0.012,
            "T = 1.55\nglobal scaling\nmakes it worse → 0.128",
            fontsize=13, color=CORAL_DK, ha="right", va="bottom")

    # Fink RF under isotonic regression: ECE collapses, ranking unchanged.
    rf = rows[3]
    ax.annotate("", xy=(iso_auc, iso_ece), xytext=(rf["auc"], rf["before"]),
                arrowprops=dict(arrowstyle="-|>", color=CORAL_DK, lw=2.6,
                                linestyle=(0, (5, 3))))
    ax.scatter([iso_auc], [iso_ece], s=150, color=CORAL_DK, marker="v", zorder=6,
               edgecolor="white", linewidth=1.3)
    ax.annotate(
        f"isotonic: ECE {rf['before']:.2f} → {iso_ece:.3f},\n"
        "but ROC-AUC still 0.50\n“calibrated yet uninformative”",
        (iso_auc, iso_ece), textcoords="offset points", xytext=(18, 14),
        ha="left", fontsize=13.5, color=CORAL_DK, fontweight="bold",
    )

    ax.axvline(0.65, color="#5F5E5A", lw=1.1, linestyle=":")
    ax.set_xlabel("Discrimination — ROC-AUC of the score for its target class")
    ax.set_ylabel("Expected calibration error (raw)")
    ax.set_xlim(0.44, 1.02)
    ax.set_ylim(-0.03, 0.64)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_discrimination_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_reliability() -> None:
    oof = pd.read_csv(TABLES / "alerce_deferral_oof_predictions.csv")
    correct = oof["correct_recomputed"].to_numpy(float)
    raw = oof["confidence_recomputed"].to_numpy(float)
    cal = oof["score_calibrated_confidence"].to_numpy(float)

    def binned(conf, n_bins=10):
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
        xs, ys, ns = [], [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() >= 8:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
                ns.append(int(m.sum()))
        return np.array(xs), np.array(ys), np.array(ns)

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.plot([0, 1], [0, 1], "--", color=INK, lw=1.8, label="perfect calibration")

    xr, yr, nr = binned(raw)
    xc, yc, nc = binned(cal)
    ax.plot(xr, yr, "-o", color=CORAL, lw=3.0, ms=11, label="raw ALeRCE probabilities")
    ax.plot(xc, yc, "-o", color=TEAL, lw=3.0, ms=11, label="after temperature scaling (out-of-fold)")

    # Highlight the underconfidence gap on the raw curve.
    for x, y in zip(xr, yr):
        ax.plot([x, x], [x, y], color=CORAL, alpha=0.28, lw=2.4)
    k = max(1, len(xr) // 3)
    ax.annotate(
        "raw curve lies\nabove the diagonal\n→ underconfident",
        xy=(xr[k], yr[k]), xytext=(0.04, 0.97), textcoords="axes fraction",
        ha="left", va="top", fontsize=15, color=CORAL_DK,
        arrowprops=dict(arrowstyle="-|>", color=CORAL_DK, lw=1.8),
    )

    ax.set_xlabel("Reported confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=14)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = broker_rows()
    fig_broker_ece(rows)
    fig_discrimination_map(rows)
    fig_reliability()
    print("Wrote:")
    for name in ("fig_broker_ece", "fig_discrimination_map", "fig_reliability"):
        print(" ", OUT / f"{name}.png")


if __name__ == "__main__":
    main()
