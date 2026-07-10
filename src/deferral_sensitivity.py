"""
src/deferral_sensitivity.py

Robustness checks for the deferral prototype (src/deferral.py). Two hand-set
parameters govern its behaviour — the rare-class cost weight and the
spectroscopic cost c_spec — so a reviewer will (rightly) ask whether the
headline results are artefacts of those choices. We sweep both.

(A) Cost-weight sweep.  Vary w(SLSN) over 1–50 (common-class weights fixed
    at SNIa 1 / SNII 2 / SNIbc 3) and track, at a fixed 20% budget:
      * SLSN recall for cost-aware calibrated deferral vs the
        weight-independent confidence baseline (a flat anchor), and
      * the calibration improvement factor AURC(raw)/AURC(calibrated).
    A robust result looks like: cost-aware recall rises with w(SLSN) and
    overtakes the confidence baseline across a wide range, and calibration
    helps (ratio > 1) at every weight.

(B) Budget-threshold sweep.  Vary c_spec and count how many objects each
    signal defers under the rule "defer iff E[cost] > c_spec". Because
    under-confidence inflates E[cost] roughly uniformly, RAW should defer
    at least as many as CALIBRATED for EVERY threshold — the over-deferral
    direction is a structural property, not a c_spec=1 artefact.

Outputs
-------
results/figures/deferral_sensitivity.png
results/tables/deferral_sensitivity.json
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deferral import (ALL_CLASSES, NAMEABLE, RARE, T_CALIB,
                      temperature_scale, project_to_decision_space,
                      expected_cost_risk, confidence_risk,
                      realised_autoaccept_cost, risk_coverage_curve,
                      budget_report)

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATASET = os.path.join(ROOT, "data", "processed", "alerce_dataset.csv")
FIG = os.path.join(ROOT, "results", "figures", "deferral_sensitivity.png")
OUT = os.path.join(ROOT, "results", "tables", "deferral_sensitivity.json")

W_SLSN_GRID = [1, 2, 3, 5, 8, 10, 15, 20, 35, 50]
CSPEC_GRID = np.linspace(0.2, 4.0, 25)
BUDGET_FRAC = 0.20
BASE_COMMON = {"SNIa": 1.0, "SNII": 2.0, "SNIbc": 3.0}


def main():
    df = pd.read_csv(DATASET)
    df_in = df[df["alerce_class"] != "TDE"].reset_index(drop=True)
    n = len(df_in)
    true_class = df_in["alerce_class"].values
    B = int(round(BUDGET_FRAC * n))

    p15 = df_in[ALL_CLASSES].values.astype(float)
    p15 = p15 / p15.sum(axis=1, keepdims=True)
    raw4 = project_to_decision_space(p15)
    cal4 = project_to_decision_space(temperature_scale(p15, T_CALIB))

    # confidence baseline is weight-independent — compute once
    sig_conf = confidence_risk(cal4)
    # its realised cost needs *some* weights; use the default for the anchor
    conf_recall_by_w = {}

    # ── (A) cost-weight sweep ────────────────────────────────────────────────
    print(f"(A) Cost-weight sweep  (SLSN recall @ {int(BUDGET_FRAC*100)}% budget, "
          f"n={n}, B={B})")
    print(f"  {'w_SLSN':>7} {'cost-aware(cal)':>16} {'confidence':>11} "
          f"{'AURC raw':>9} {'AURC cal':>9} {'improve×':>9}")
    sweep_A = []
    for wS in W_SLSN_GRID:
        cost = {**BASE_COMMON, RARE: float(wS)}
        w = np.array([cost[c] for c in NAMEABLE])
        cost_cal, wrong_cal = realised_autoaccept_cost(cal4, true_class, cost)
        cost_raw, wrong_raw = realised_autoaccept_cost(raw4, true_class, cost)

        sig_cal = expected_cost_risk(cal4, w)
        sig_raw = expected_cost_risk(raw4, w)

        rec_cal = budget_report(sig_cal, cost_cal, wrong_cal, true_class, B)["rare_recall"]
        rec_conf = budget_report(sig_conf, cost_cal, wrong_cal, true_class, B)["rare_recall"]
        conf_recall_by_w[wS] = rec_conf

        _, _, aurc_raw = risk_coverage_curve(sig_raw, cost_raw)
        _, _, aurc_cal = risk_coverage_curve(sig_cal, cost_cal)
        improve = aurc_raw / aurc_cal

        sweep_A.append({"w_slsn": wS, "recall_costaware_cal": rec_cal,
                        "recall_confidence": rec_conf,
                        "aurc_raw": round(aurc_raw, 4), "aurc_cal": round(aurc_cal, 4),
                        "improvement_factor": round(float(improve), 3)})
        print(f"  {wS:>7} {rec_cal:>16.3f} {rec_conf:>11.3f} "
              f"{aurc_raw:>9.3f} {aurc_cal:>9.3f} {improve:>8.2f}×")

    # crossover: smallest w_SLSN at which cost-aware >= confidence
    cross = next((r["w_slsn"] for r in sweep_A
                  if r["recall_costaware_cal"] >= r["recall_confidence"]), None)
    print(f"  → cost-aware overtakes confidence baseline at w_SLSN ≈ {cross}")

    # ── (B) budget-threshold sweep ───────────────────────────────────────────
    w_def = np.array([{**BASE_COMMON, RARE: 10.0}[c] for c in NAMEABLE])
    sig_raw_def = expected_cost_risk(raw4, w_def)
    sig_cal_def = expected_cost_risk(cal4, w_def)
    sweep_B = []
    raw_always_ge = True
    for cs in CSPEC_GRID:
        nr = int((sig_raw_def > cs).sum())
        nc = int((sig_cal_def > cs).sum())
        raw_always_ge &= nr >= nc
        sweep_B.append({"c_spec": round(float(cs), 3), "n_defer_raw": nr,
                        "n_defer_cal": nc})
    print(f"\n(B) Budget-threshold sweep: RAW defers ≥ CALIBRATED at every "
          f"c_spec?  {raw_always_ge}")

    results = {
        "config": {"n": n, "budget_frac": BUDGET_FRAC, "budget": B,
                   "base_common_weights": BASE_COMMON, "T_calib": T_CALIB},
        "cost_weight_sweep": sweep_A,
        "crossover_w_slsn": cross,
        "threshold_sweep": sweep_B,
        "raw_over_defers_at_every_cspec": bool(raw_always_ge),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {OUT}")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    wg = [r["w_slsn"] for r in sweep_A]
    ax1.plot(wg, [r["recall_costaware_cal"] for r in sweep_A], "o-",
             color="#2f6d4f", lw=2.2, label="cost-aware, calibrated")
    ax1.plot(wg, [r["recall_confidence"] for r in sweep_A], "s--",
             color="#8a8f98", lw=1.8, label="confidence-only (weight-independent)")
    if cross:
        ax1.axvline(cross, color="#b0741b", ls=":", lw=1.5)
        ax1.text(cross * 1.05, 0.02, f"crossover\nw≈{cross}", color="#b0741b",
                 fontsize=9, va="bottom")
    ax1.set_xscale("log")
    ax1.set_xlabel("Rare-class cost weight  w(SLSN)", fontsize=11.5)
    ax1.set_ylabel(f"SLSN recall @ {int(BUDGET_FRAC*100)}% budget", fontsize=11.5)
    ax1.set_title("(A) Cost-aware advantage is robust across\nthe weight range, "
                  "not a knife-edge choice", fontsize=11)
    ax1.grid(alpha=0.25); ax1.legend(fontsize=9.5, frameon=False, loc="upper left")

    cs = [r["c_spec"] for r in sweep_B]
    ax2.plot(cs, [r["n_defer_raw"] for r in sweep_B], color="#c0714a", lw=2.2,
             label="RAW probabilities")
    ax2.plot(cs, [r["n_defer_cal"] for r in sweep_B], color="#2f6d4f", lw=2.4,
             label="CALIBRATED probabilities")
    ax2.fill_between(cs, [r["n_defer_cal"] for r in sweep_B],
                     [r["n_defer_raw"] for r in sweep_B], color="#c0714a", alpha=0.12)
    ax2.set_xlabel("Spectroscopic cost threshold  c_spec", fontsize=11.5)
    ax2.set_ylabel("Objects deferred  (defer iff E[cost] > c_spec)", fontsize=11.5)
    ax2.set_title("(B) Raw over-defers at every threshold\n"
                  "(under-confidence inflates E[cost] uniformly)", fontsize=11)
    ax2.grid(alpha=0.25); ax2.legend(fontsize=9.5, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    plt.savefig(FIG, dpi=200, bbox_inches="tight")
    print(f"Saved figure → {FIG}")


if __name__ == "__main__":
    main()
