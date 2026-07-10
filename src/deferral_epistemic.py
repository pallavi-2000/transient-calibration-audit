"""
src/deferral_epistemic.py

Layer-2 preview: how much of the out-of-taxonomy TDE leak could an
uncertainty channel recover, and how good would that channel have to be?
The deferral prototype showed that expected-cost deferral, working from the
probability vector alone, still lets ~77% of TDEs through at a 10% budget
(src/deferral.py, OOD sidebar). Here we ask what a second signal buys —
without yet building the ensemble.

(C) GROUNDED — single-model predictive entropy.
    Entropy H[p] of the deployed classifier's own 15-class vector is a
    signal we already have. It is a WEAK proxy for epistemic uncertainty
    (it conflates aleatoric ambiguity with model ignorance), but for an
    out-of-taxonomy object the classifier tends to spread mass, so entropy
    is elevated. We measure the real TDE recall from adding entropy to the
    expected-cost signal. This is a lower bound on what a true ensemble
    epistemic-variance channel would provide.

(D) FORWARD-LOOKING — an OOD-channel requirement curve. ILLUSTRATIVE, NOT
    A MEASURED RESULT. We ask: IF an ensemble supplied an OOD/epistemic
    score separating TDEs from in-distribution objects with a given AUC q,
    what TDE recall would a combined policy reach at a 10% budget? Epistemic
    scores are simulated as in-dist ~ N(0,1), TDE ~ N(mu,1) with
    q = Phi(mu/sqrt(2)); this translates "how good must Layer-2's novelty
    channel be" into "how many TDEs you catch within budget". It assumes the
    epistemic signal is independent of the probability vector and says
    nothing about whether any specific ensemble attains a given q.

Outputs
-------
results/figures/deferral_epistemic.png
results/tables/deferral_epistemic.json
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deferral import (ALL_CLASSES, NAMEABLE, COST, T_CALIB,
                      temperature_scale, project_to_decision_space,
                      expected_cost_risk)

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATASET = os.path.join(ROOT, "data", "processed", "alerce_dataset.csv")
FIG = os.path.join(ROOT, "results", "figures", "deferral_epistemic.png")
OUT = os.path.join(ROOT, "results", "tables", "deferral_epistemic.json")

BUDGET_FRACS = [0.05, 0.10, 0.20]
AUC_GRID = np.round(np.arange(0.50, 0.991, 0.02), 3)
SEED = 42


def predictive_entropy(probs, eps=1e-12):
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def pct_rank(x):
    """Map values to [0,1] percentile ranks (ties broken by position)."""
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return order / (len(x) - 1)


def tde_recall_at_budget(score_all, is_tde, frac):
    B = int(round(frac * len(score_all)))
    defer = np.argsort(-score_all, kind="mergesort")[:B]
    mask = np.zeros(len(score_all), bool); mask[defer] = True
    return float(mask[is_tde].mean())


def main():
    df = pd.read_csv(DATASET)
    is_tde = (df["alerce_class"] == "TDE").values

    p15 = df[ALL_CLASSES].values.astype(float)
    p15 = p15 / p15.sum(axis=1, keepdims=True)
    p15_cal = temperature_scale(p15, T_CALIB)
    cal4 = project_to_decision_space(p15_cal)
    w = np.array([COST[c] for c in NAMEABLE])

    # signals over ALL objects (in-dist + TDE), ranked jointly
    sig_cost = expected_cost_risk(cal4, w)          # expected-cost (probability only)
    sig_ent = predictive_entropy(p15_cal)           # grounded epistemic proxy
    r_cost, r_ent = pct_rank(sig_cost), pct_rank(sig_ent)
    sig_combined = np.maximum(r_cost, r_ent)         # "defer if EITHER flags it"

    # ── (C) grounded: entropy's real contribution ────────────────────────────
    print("(C) GROUNDED — TDE recall by signal (real data, single model):")
    print(f"  {'budget':>7} {'exp-cost':>9} {'entropy':>8} {'combined':>9}")
    grounded = []
    for frac in BUDGET_FRACS:
        rc = tde_recall_at_budget(sig_cost, is_tde, frac)
        re = tde_recall_at_budget(sig_ent, is_tde, frac)
        rk = tde_recall_at_budget(sig_combined, is_tde, frac)
        grounded.append({"budget_frac": frac, "recall_expcost": round(rc, 4),
                         "recall_entropy": round(re, 4), "recall_combined": round(rk, 4)})
        print(f"  {frac*100:>5.0f}%  {rc:>9.3f} {re:>8.3f} {rk:>9.3f}")
    print("  → single-model entropy adds a little, but the deployed classifier")
    print("    cannot itself separate an object it has no class for; a genuine")
    print("    ensemble epistemic-variance channel is needed (quantified below).")

    # ── (D) forward-looking: OOD-channel requirement curve ───────────────────
    rng = np.random.default_rng(SEED)
    n = len(df)
    print(f"\n(D) ILLUSTRATIVE requirement curve — TDE recall vs assumed OOD-AUC")
    req = {f"{int(f*100)}pct": [] for f in BUDGET_FRACS}
    # average over repeats to smooth the Monte-Carlo epistemic draw
    for q in AUC_GRID:
        mu = np.sqrt(2.0) * _probit(q)              # Gaussian-separation → AUC
        recs = {f: [] for f in BUDGET_FRACS}
        for _ in range(40):
            epi = rng.normal(0.0, 1.0, n)
            epi[is_tde] = rng.normal(mu, 1.0, is_tde.sum())
            combined = np.maximum(r_cost, pct_rank(epi))
            for f in BUDGET_FRACS:
                recs[f].append(tde_recall_at_budget(combined, is_tde, f))
        for f in BUDGET_FRACS:
            req[f"{int(f*100)}pct"].append(round(float(np.mean(recs[f])), 4))

    # what OOD-AUC is needed to recover 80% of TDEs at a 10% budget?
    r10 = req["10pct"]
    target = next((float(q) for q, r in zip(AUC_GRID, r10) if r >= 0.80), None)
    print(f"  OOD-AUC needed for ≥80% TDE recall @ 10% budget: "
          f"{target if target else '> 0.99'}")

    results = {
        "grounded_entropy": grounded,
        "requirement_curve": {
            "assumption": "in-dist ~ N(0,1), TDE ~ N(mu,1), q = Phi(mu/sqrt2); "
                          "epistemic signal independent of probability vector; "
                          "combined = max(pct_rank(expected_cost), pct_rank(epistemic)). "
                          "ILLUSTRATIVE design target, not a measured ensemble result.",
            "auc_grid": AUC_GRID.tolist(),
            "tde_recall_by_budget": req,
            "auc_for_80pct_recall_at_10pct_budget": target,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {OUT}")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    x = np.arange(len(BUDGET_FRACS))
    width = 0.26
    ax1.bar(x - width, [g["recall_expcost"] for g in grounded], width,
            color="#c0714a", label="expected-cost (prob. only)")
    ax1.bar(x, [g["recall_entropy"] for g in grounded], width,
            color="#8a8f98", label="predictive entropy")
    ax1.bar(x + width, [g["recall_combined"] for g in grounded], width,
            color="#2f6d4f", label="combined")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{int(f*100)}%" for f in BUDGET_FRACS])
    ax1.set_xlabel("Spectroscopic budget", fontsize=11.5)
    ax1.set_ylabel("TDE recall  (fraction flagged)", fontsize=11.5)
    ax1.set_title("(C) GROUNDED: single-model signals on real data\n"
                  "cap out well below full recovery", fontsize=11)
    ax1.grid(alpha=0.25, axis="y"); ax1.legend(fontsize=9.5, frameon=False)

    for f, col in zip(BUDGET_FRACS, ["#9aa7ba", "#b0741b", "#2f6d4f"]):
        ax2.plot(AUC_GRID, req[f"{int(f*100)}pct"], lw=2.2, color=col,
                 label=f"{int(f*100)}% budget")
    ax2.axhline(0.80, color="#a63a2b", ls=":", lw=1.4)
    ax2.text(0.51, 0.82, "80% recall", color="#a63a2b", fontsize=9)
    if target:
        ax2.axvline(target, color="#a63a2b", ls=":", lw=1.4)
    ax2.set_xlabel("Assumed OOD-detection AUC of an ensemble channel", fontsize=11.5)
    ax2.set_ylabel("TDE recall", fontsize=11.5)
    ax2.set_title("(D) ILLUSTRATIVE: the novelty-channel quality\n"
                  "Layer 2 must reach to recover TDEs within budget", fontsize=11)
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.25); ax2.legend(fontsize=9.5, frameon=False, loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    plt.savefig(FIG, dpi=200, bbox_inches="tight")
    print(f"Saved figure → {FIG}")


def _probit(p):
    """Inverse standard-normal CDF (Acklam approximation, no SciPy dep)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


if __name__ == "__main__":
    main()
