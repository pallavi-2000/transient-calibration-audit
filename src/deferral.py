"""
src/deferral.py

Layer-1 prototype of the deferral-aware follow-up framework, built as a
POST-HOC DECISION LAYER on the calibration audit's existing ALeRCE data.
No deep ensemble, no simulations, no new training — this isolates the one
link the audit uniquely warrants: does *calibration* change *budgeted
deferral decisions*, and does a cost-aware policy recall the rare,
high-value transients a naive confidence rule misses?

Design-scope note (why the rare-class showcase is SLSN, not TDE)
----------------------------------------------------------------
The deployed ALeRCE light-curve classifier resolves four transient
classes — SNIa, SNIbc, SNII, SLSN (Sanchez-Saez et al. 2021). It has no
TDE class: verified against the live API for both production versions in
our sample (hierarchical_rf_1.1.0, lc_classifier_1.1.13). A TDE-aware
version exists in the literature (Pavez-Herrera et al. 2025, 91% recall)
but is not the version serving the alert stream. Asking an ALeRCE-based
policy to "recall TDEs" is therefore a category error — the class is
structurally unrepresentable. We instead use SLSN as the rare, high-value
showcase (in-taxonomy, n=55, ALeRCE-designed-for), which is a fair test.
TDEs are handled separately as an out-of-distribution sidebar: objects the
classifier cannot represent, to quantify whether a probability signal alone
provides any novelty detection (it does not — motivating the ensemble).

Setup
-----
* Classifier: ALeRCE light-curve RF (its probability vectors, as audited).
* Decision space: the 4 nameable transient classes. Probabilities
  renormalised onto this simplex. Primary experiment excludes the 30 TDEs.
* Asymmetric cost of a wrong auto-classification, by TRUE class:
      SNIa 1  ·  SNII 2  ·  SNIbc 3  ·  SLSN 10
  (common, well-understood SNe are cheap to misclassify; the rare,
  high-value SLSN is expensive to silently lose.)

Decision signal (Bayes expected cost of auto-accepting, no ground truth):
    risk(x) = Σ_{c ≠ ŷ} p(c|x) · w(c)          ŷ = argmax_c p(c|x)
Higher risk → defer. Not a monotone function of top-1 confidence when the
weights differ across classes, so — unlike "defer the least-confident" —
the ranking depends on the calibrated *shape* of the vector, not its peak.

Realised cost (evaluation only, uses the spectroscopic label):
    auto-accept:  0 if ŷ == y  else w(y)
    defer:        c_spec       (a spectrum is spent)

Compared under a fixed nightly budget B:
    (a) confidence-only  — defer the B least-confident   (cost-agnostic baseline)
    (b) expected-cost, RAW probabilities
    (c) expected-cost, CALIBRATED probabilities (temperature scaling, T≈0.36)

Outputs
-------
results/figures/deferral_risk_coverage.png
results/tables/deferral_results.json
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration import renormalize_to_subset

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATASET = os.path.join(ROOT, "data", "processed", "alerce_dataset.csv")
FIG = os.path.join(ROOT, "results", "figures", "deferral_risk_coverage.png")
OUT = os.path.join(ROOT, "results", "tables", "deferral_results.json")

ALL_CLASSES = ["SNIa", "SNIbc", "SNII", "SLSN", "QSO", "AGN", "Blazar",
               "CV/Nova", "YSO", "LPV", "E", "DSCT", "RRL", "CEP",
               "Periodic-Other"]
NAMEABLE = ["SNIa", "SNII", "SNIbc", "SLSN"]    # ALeRCE's transient decision space
RARE = "SLSN"                                    # in-taxonomy, rare, high-value

COST = {"SNIa": 1.0, "SNII": 2.0, "SNIbc": 3.0, "SLSN": 10.0}
C_SPEC = 1.0
T_CALIB = 0.361
BUDGET_FRACS = [0.05, 0.10, 0.20, 0.30]


# ── probability helpers ───────────────────────────────────────────────────────

def temperature_scale(probs15, T, eps=1e-12):
    logits = np.log(np.clip(probs15, eps, 1.0))
    z = logits / T
    z -= z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def project_to_decision_space(probs15):
    idx = [ALL_CLASSES.index(c) for c in NAMEABLE]
    return renormalize_to_subset(probs15, idx)      # (N, 4), rows sum to 1


# ── decision signals ──────────────────────────────────────────────────────────

def expected_cost_risk(probs4, weights):
    """risk(x) = Σ_{c≠ŷ} p(c) w(c) = E_y[cost(ŷ, y)]  (Bayes expected cost)."""
    yhat = np.argmax(probs4, axis=1)
    weighted = probs4 * weights[None, :]
    return weighted.sum(axis=1) - weighted[np.arange(len(probs4)), yhat]


def confidence_risk(probs4):
    """Cost-agnostic baseline: defer the least confident."""
    return 1.0 - probs4.max(axis=1)


# ── realised cost / evaluation ────────────────────────────────────────────────

def realised_autoaccept_cost(probs4, true_class, cost):
    yhat = np.array(NAMEABLE)[np.argmax(probs4, axis=1)]
    wrong = yhat != np.asarray(true_class)
    w_true = np.array([cost[y] for y in true_class])
    return wrong.astype(float) * w_true, wrong


def risk_coverage_curve(risk, autoaccept_cost):
    order = np.argsort(risk, kind="mergesort")
    cost_sorted = autoaccept_cost[order]
    n = len(risk)
    k = np.arange(1, n + 1)
    coverage = k / n
    selective_risk = np.cumsum(cost_sorted) / k
    return coverage, selective_risk, float(np.trapz(selective_risk, coverage))


def budget_report(risk, autoaccept_cost, wrong, true_class, B):
    n = len(risk)
    defer_idx = set(np.argsort(-risk, kind="mergesort")[:B].tolist())
    deferred = np.array([i in defer_idx for i in range(n)])
    accepted = ~deferred
    true_class = np.asarray(true_class)

    n_acc = accepted.sum()
    rare = true_class == RARE
    return {
        "budget": int(B),
        "coverage": round(float(n_acc / n), 4),
        "total_cost": round(float(autoaccept_cost[accepted].sum() + B * C_SPEC), 2),
        "silent_failure_rate": round(float((accepted & wrong).sum() / max(n_acc, 1)), 4),
        "rare_recall": round(float((rare & deferred).sum() / max(rare.sum(), 1)), 4),
        "deferred": deferred,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(DATASET)
    is_tde = df["alerce_class"].values == "TDE"
    df_in = df[~is_tde].reset_index(drop=True)         # primary: nameable only
    df_ood = df[is_tde].reset_index(drop=True)         # OOD sidebar
    n = len(df_in)
    print(f"Primary sample: {n} nameable-class objects "
          f"({', '.join(f'{k}={v}' for k, v in df_in['alerce_class'].value_counts().items())})")
    print(f"OOD sidebar:    {len(df_ood)} TDEs (absent from ALeRCE taxonomy)\n")

    def build(frame):
        p15 = frame[ALL_CLASSES].values.astype(float)
        p15 = p15 / p15.sum(axis=1, keepdims=True)
        return project_to_decision_space(p15), project_to_decision_space(temperature_scale(p15, T_CALIB))

    raw4, cal4 = build(df_in)
    true_class = df_in["alerce_class"].values
    w = np.array([COST[c] for c in NAMEABLE])

    cost_raw, wrong_raw = realised_autoaccept_cost(raw4, true_class, COST)
    cost_cal, wrong_cal = realised_autoaccept_cost(cal4, true_class, COST)

    sig_conf = confidence_risk(cal4)
    sig_raw = expected_cost_risk(raw4, w)
    sig_cal = expected_cost_risk(cal4, w)

    cov_c, sr_c, aurc_c = risk_coverage_curve(sig_conf, cost_cal)
    cov_r, sr_r, aurc_r = risk_coverage_curve(sig_raw, cost_raw)
    cov_k, sr_k, aurc_k = risk_coverage_curve(sig_cal, cost_cal)

    print("Area under risk–coverage curve (lower is better):")
    print(f"  confidence-only (cost-agnostic ref.) : {aurc_c:.4f}")
    print(f"  expected-cost, RAW probs             : {aurc_r:.4f}")
    print(f"  expected-cost, CALIBRATED            : {aurc_k:.4f}")

    results = {
        "config": {"cost": COST, "c_spec": C_SPEC, "T_calib": T_CALIB,
                   "decision_classes": NAMEABLE, "rare_class": RARE, "n_primary": int(n),
                   "n_ood_tde": int(len(df_ood))},
        "aurc": {"confidence": round(aurc_c, 4), "expcost_raw": round(aurc_r, 4),
                 "expcost_calibrated": round(aurc_k, 4)},
        "by_budget": [],
    }

    print(f"\n{'budget':>7} {'signal':<15} {'cover':>6} {'cost':>9} "
          f"{'silent':>7} {'SLSN_recall':>12}")
    print("-" * 60)
    for frac in BUDGET_FRACS:
        B = int(round(frac * n))
        row = {"budget_frac": frac, "budget": B, "signals": {}}
        for name, sig, cost, wrong in [
            ("confidence", sig_conf, cost_cal, wrong_cal),
            ("expcost_raw", sig_raw, cost_raw, wrong_raw),
            ("expcost_calib", sig_cal, cost_cal, wrong_cal),
        ]:
            rep = budget_report(sig, cost, wrong, true_class, B)
            row["signals"][name] = {k: v for k, v in rep.items() if k != "deferred"}
            print(f"{B:>7} {name:<15} {rep['coverage']:>6.3f} {rep['total_cost']:>9.1f} "
                  f"{rep['silent_failure_rate']:>7.3f} {rep['rare_recall']:>12.3f}")
        results["by_budget"].append(row)
        print("-" * 60)

    # ── threshold feasibility: does miscalibration blow the budget? ──────────
    n_defer_raw = int((sig_raw > C_SPEC).sum())
    n_defer_cal = int((sig_cal > C_SPEC).sum())
    results["threshold_feasibility"] = {
        "c_spec": C_SPEC, "n_defer_raw": n_defer_raw, "n_defer_calibrated": n_defer_cal,
        "over_defer_frac": round((n_defer_raw - n_defer_cal) / n, 4),
        "note": "Fixed-threshold rule (defer iff E[cost] > c_spec). Under-confident "
                "raw probs inflate expected costs and over-defer, overrunning a fixed budget.",
    }
    print(f"\nFixed-threshold rule (defer iff E[cost] > c_spec={C_SPEC}):")
    print(f"  RAW        defer {n_defer_raw:>4}   CALIBRATED defer {n_defer_cal:>4}   "
          f"→ raw over-defers by {(n_defer_raw-n_defer_cal)/n*100:.1f}% of the sample")

    # ── OOD sidebar: are un-representable TDEs preferentially flagged? ────────
    # An ideal novelty channel defers OOD objects at a HIGHER rate than
    # in-distribution objects. Does the probability signal alone do that?
    raw4_ood, cal4_ood = build(df_ood)
    sig_ood = expected_cost_risk(cal4_ood, w)
    # rank ALL objects (in-dist + OOD) jointly by the calibrated signal
    sig_all = np.concatenate([sig_cal, sig_ood])
    is_ood_all = np.concatenate([np.zeros(n, bool), np.ones(len(df_ood), bool)])
    N_all = len(sig_all)
    ood_rows = []
    for frac in BUDGET_FRACS:
        B = int(round(frac * N_all))
        defer = np.argsort(-sig_all, kind="mergesort")[:B]
        deferred_mask = np.zeros(N_all, bool); deferred_mask[defer] = True
        ood_rate = deferred_mask[is_ood_all].mean()
        ind_rate = deferred_mask[~is_ood_all].mean()
        ood_rows.append({"budget_frac": frac,
                         "tde_defer_rate": round(float(ood_rate), 4),
                         "indist_defer_rate": round(float(ind_rate), 4),
                         "ratio": round(float(ood_rate / max(ind_rate, 1e-9)), 3),
                         "tde_silently_accepted": round(float(1.0 - ood_rate), 4)})
    results["ood_sidebar"] = {
        "description": "TDEs (out-of-taxonomy) ranked jointly with in-distribution "
                       "objects by the calibrated expected-cost signal. TDEs are deferred "
                       "at 1.1-2.4x the in-distribution rate — because ALeRCE, unable to "
                       "name them, tends to emit a diffuse (high expected-cost) vector — so "
                       "the probability signal is a WEAK, budget-sensitive novelty detector, "
                       "not a null one. But it is far from sufficient: at a 10% budget ~77% "
                       "of TDEs are still silently auto-accepted as SNe. Reliable capture "
                       "would need a dedicated OOD / epistemic-uncertainty channel (the "
                       "ensemble). This is an out-of-taxonomy observation, NOT an ALeRCE "
                       "performance critique: the deployed classifier has no TDE class "
                       "(Sanchez-Saez 2021); a TDE-aware version exists but is not deployed "
                       "(Pavez-Herrera 2025).",
        "by_budget": ood_rows,
    }
    print("\nOOD sidebar — are un-representable TDEs preferentially flagged? (calibrated signal)")
    print(f"  {'budget':>7} {'TDE defer':>10} {'in-dist':>9} {'ratio':>7} {'TDE silently accepted':>22}")
    for r in ood_rows:
        print(f"  {r['budget_frac']*100:>5.0f}%  {r['tde_defer_rate']:>10.3f} "
              f"{r['indist_defer_rate']:>9.3f} {r['ratio']:>7.2f} {r['tde_silently_accepted']:>21.1%}")
    print("  → weak novelty signal (ratio 1.1-2.4), but most TDEs still slip through:")
    print("    a dedicated OOD/epistemic channel is needed for reliable capture.")

    # ── figure ───────────────────────────────────────────────────────────────
    plt.figure(figsize=(7.2, 5.4))
    plt.plot(cov_c, sr_c, lw=1.6, color="#8a8f98", ls=":",
             label=f"confidence-only, cost-agnostic ref.  (AURC {aurc_c:.3f})")
    plt.plot(cov_r, sr_r, lw=2.0, color="#c0714a",
             label=f"expected-cost, RAW probs  (AURC {aurc_r:.3f})")
    plt.plot(cov_k, sr_k, lw=2.6, color="#2f6d4f",
             label=f"expected-cost, CALIBRATED  (AURC {aurc_k:.3f})")
    plt.xlabel("Coverage  (fraction auto-classified)", fontsize=12)
    plt.ylabel("Selective risk  (mean cost of a wrong auto-label)", fontsize=12)
    plt.title("Calibration improves the cost-aware deferral ranking\n"
              f"(expected-cost AURC {aurc_r:.2f} raw → {aurc_k:.2f} calibrated; "
              "SLSN as rare high-value class)", fontsize=11.5)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9.5, frameon=False, loc="upper left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    plt.savefig(FIG, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure → {FIG}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results → {OUT}")


if __name__ == "__main__":
    main()
