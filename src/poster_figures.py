"""Poster figures for the A1 NAM poster (paper/poster/claude_design).

Two audited systems:
  * ALeRCE  — the production alert broker's 15-class light-curve classifier;
  * NEEDLE  — a rare-transient classifier (not a broker), 3 classes.

Figures generated:
  1. Aggregate ECE before/after global temperature scaling, annotated with the
     fitted temperature and its direction (under- vs overconfident).
  2. NEEDLE per class: discrimination, and the per-class fitted temperature,
     which points in opposite directions across its own classes.
  3. ALeRCE reliability diagram, raw vs out-of-fold calibrated.

Values come from results/tables/ and data/processed/ rather than being
hard-coded, except the audited NEEDLE constants below, which are reported in
results/tables/needle_*.txt and cited in the poster captions.
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
DATA = ROOT / "data" / "processed"
OUT = ROOT / "paper" / "poster" / "claude_design" / "assets"

TEAL = "#1D9E75"
TEAL_DK = "#0F6E56"
NAVY = "#0C447C"
CORAL = "#D85A30"
CORAL_DK = "#993C1D"
GRAY = "#B4B2A9"
INK = "#2C2C2A"

plt.rcParams.update(
    {
        "font.size": 17,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "axes.edgecolor": "#5F5E5A",
        "figure.facecolor": "white",
    }
)

# NEEDLE aggregate calibration, from results/tables/needle_*.txt.
NEEDLE = {"ece_before": 0.0504, "ece_after": 0.1278, "T": 1.553}
# NEEDLE per-class temperatures from the vector-scaling run.
NEEDLE_T = {"SN": 1.5779, "SLSN-I": 1.8793, "TDE": 0.4249}
# NEEDLE per-class ECE from the calibration study.
NEEDLE_ECE = {"SN": 0.1532, "SLSN-I": 0.0858, "TDE": 0.0726}


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def system_rows() -> list[dict]:
    cv = json.loads((TABLES / "temperature_scaling_cv.json").read_text())
    return [
        {
            "label": "ALeRCE\n15-class broker classifier",
            "before": cv["baseline"]["ece"],
            "after": cv["global_T"]["ece"],
            "T": cv["global_T"]["T_mean"],
            "direction": "underconfident  (T < 1)",
        },
        {
            "label": "NEEDLE\n3-class rare-transient classifier",
            "before": NEEDLE["ece_before"],
            "after": NEEDLE["ece_after"],
            "T": NEEDLE["T"],
            "direction": "overconfident  (T > 1)",
        },
    ]


def needle_per_class() -> pd.DataFrame:
    """Object-level (deduplicated) NEEDLE performance per class."""
    from sklearn.metrics import roc_auc_score

    z = np.load(DATA / "needle_predictions.npz", allow_pickle=True)
    names = list(z["class_names"])
    df = pd.DataFrame(z["probs"], columns=names)
    df["y"] = z["labels"]
    df["oid"] = z["ztf_ids"]
    # One row per object: NEEDLE predictions are pooled over 5 models.
    g = df.groupby("oid").agg({**{c: "mean" for c in names}, "y": "first"})
    probs = g[names].to_numpy()
    y = g["y"].to_numpy()
    pred = probs.argmax(axis=1)

    rows = []
    for i, cls in enumerate(names):
        m = y == i
        rows.append(
            {
                "cls": cls,
                "n": int(m.sum()),
                "recall": float((pred[m] == i).mean()),
                "auc": float(roc_auc_score((y == i).astype(int), probs[:, i])),
                "ece": NEEDLE_ECE[cls],
                "T": NEEDLE_T[cls],
            }
        )
    return pd.DataFrame(rows)


def fig_system_ece(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    x = np.arange(len(rows))
    w = 0.30
    ax.bar(x - w / 2, [r["before"] for r in rows], w, color=GRAY, label="raw probabilities")
    bars = ax.bar(x + w / 2, [r["after"] for r in rows], w, color=TEAL,
                  label="after global temperature scaling")

    for i, r in enumerate(rows):
        worse = r["after"] > r["before"]
        if worse:
            bars[i].set_color(CORAL)
        ax.text(i - w / 2, r["before"] + 0.010, f"{r['before']:.3f}", ha="center", fontsize=16)
        ax.text(i + w / 2, r["after"] + 0.010, f"{r['after']:.3f}" + (" ✗" if worse else " ✓"),
                ha="center", fontsize=16, fontweight="bold",
                color=CORAL_DK if worse else TEAL_DK)
        ax.annotate(f"T = {r['T']:.2f} — {r['direction']}", (i, 0),
                    xycoords=("data", "axes fraction"), textcoords="offset points",
                    xytext=(0, -60), ha="center", fontsize=14, color=INK,
                    annotation_clip=False)

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows])
    ax.tick_params(axis="x", length=0, pad=9)
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, 0.32)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=14)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_system_ece.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_needle_per_class(pc: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 3.9))
    order = ["TDE", "SLSN-I", "SN"]
    pc = pc.set_index("cls").loc[order].reset_index()
    y = np.arange(len(pc))

    # Left: discrimination per class.
    ax1.barh(y, pc["auc"], 0.6, color=[TEAL if a >= 0.9 else NAVY for a in pc["auc"]])
    for i, r in pc.iterrows():
        ax1.text(r["auc"] - 0.012, i, f"{r['auc']:.3f}", va="center", ha="right",
                 color="white", fontweight="bold", fontsize=15)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{r.cls}\nn = {r.n}" for r in pc.itertuples()])
    ax1.set_xlim(0.5, 1.0)
    ax1.set_ylim(-0.8, 2.9)
    ax1.set_xlabel("ROC-AUC (one-vs-rest)")
    ax1.set_title("Discrimination is strong,\nbest on the rarest class", fontsize=16)
    ax1.grid(axis="x", alpha=0.25)
    ax1.set_axisbelow(True)
    _despine(ax1)

    # Right: per-class fitted temperature, diverging about T = 1.
    ax2.axvline(1.0, color=INK, lw=2.0)
    for i, r in pc.iterrows():
        over = r["T"] > 1.0
        ax2.barh(i, r["T"] - 1.0, 0.6, left=1.0, color=CORAL if over else NAVY)
        if over:
            ax2.text(r["T"] + 0.06, i, f"T = {r['T']:.2f}", va="center", ha="left",
                     fontsize=15, fontweight="bold", color=CORAL_DK)
        else:
            # Label inside the bar: outside-left would collide with the tick label.
            ax2.text(r["T"] + 0.06, i, f"T = {r['T']:.2f}", va="center", ha="left",
                     fontsize=15, fontweight="bold", color="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels(pc["cls"])
    ax2.set_xlim(0.0, 2.45)
    ax2.set_ylim(-0.8, 2.9)
    ax2.set_xlabel("Fitted per-class temperature")
    ax2.set_title("…but the classes need corrections\nin opposite directions", fontsize=16)
    ax2.text(1.72, 2.55, "overconfident", fontsize=14.5, color=CORAL_DK,
             ha="center", style="italic")
    ax2.text(0.55, -0.55, "underconfident", fontsize=14.5, color=NAVY,
             ha="center", style="italic")
    ax2.grid(axis="x", alpha=0.25)
    ax2.set_axisbelow(True)
    _despine(ax2)

    fig.tight_layout()
    fig.savefig(OUT / "fig_needle_per_class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_reliability() -> None:
    oof = pd.read_csv(TABLES / "alerce_deferral_oof_predictions.csv")
    correct = oof["correct_recomputed"].to_numpy(float)
    raw = oof["confidence_recomputed"].to_numpy(float)
    cal = oof["score_calibrated_confidence"].to_numpy(float)

    def binned(conf, n_bins=10):
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() >= 8:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        return np.array(xs), np.array(ys)

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.plot([0, 1], [0, 1], "--", color=INK, lw=1.8, label="perfect calibration")

    xr, yr = binned(raw)
    xc, yc = binned(cal)
    ax.plot(xr, yr, "-o", color=CORAL, lw=3.0, ms=11, label="raw ALeRCE probabilities")
    ax.plot(xc, yc, "-o", color=TEAL, lw=3.0, ms=11,
            label="after temperature scaling (out-of-fold)")

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
    rows = system_rows()
    pc = needle_per_class()
    fig_system_ece(rows)
    fig_needle_per_class(pc)
    fig_reliability()
    print(pc.to_string(index=False))
    print("\nWrote fig_system_ece / fig_needle_per_class / fig_reliability to", OUT)


if __name__ == "__main__":
    main()
