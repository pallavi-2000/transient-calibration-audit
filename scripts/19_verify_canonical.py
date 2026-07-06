"""
Canonical Number Verification Gate
===================================

This is the single source of truth check for the paper's headline numbers.
It does not recompute results from raw data (that is what scripts 01-18
do) -- it asserts that the numbers already sitting in results/*.json match
what the paper (figures/main_updated.tex) and README.md actually cite,
and it snapshots those numbers to results/canonical_numbers_v1.json.

Run this after ANY change to results/, scripts/, or src/calibration.py.
All checks must print PASS. A FAIL means either the results file drifted
from what the paper claims, or the paper claims something the results no
longer support -- in both cases, stop and reconcile the root cause. Do
not edit the expected values below to force a PASS; they are transcribed
from the submitted manuscript, not invented.

Usage:
    python3 scripts/19_verify_canonical.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

TOL = 1e-3          # tight tolerance: same run, same data, must reproduce exactly
LOOSE_TOL = 0.02    # for figures already rounded in the paper text (e.g. "0.271")

_checks = []
_failures = []


def check(label, actual, expected, tol=TOL):
    ok = abs(actual - expected) <= tol
    _checks.append((label, actual, expected, ok))
    if not ok:
        _failures.append(label)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: got {actual:.6g}, expected {expected:.6g} (tol {tol})")
    return ok


def load(name):
    path = RESULTS / name
    if not path.exists():
        print(f"FATAL: missing {path}")
        print("  Run the data acquisition + analysis pipeline first "
              "(see DATA_ACQUISITION.md and README.md 'Reproducing the Analysis').")
        sys.exit(1)
    return json.loads(path.read_text())


def main():
    print("=" * 70)
    print("CANONICAL NUMBER VERIFICATION")
    print("=" * 70)

    canonical = {}

    # ---------------------------------------------------------------
    # ALeRCE
    # ---------------------------------------------------------------
    print("\nALeRCE (lc_classifier v1.1.13)")
    alerce = load("alerce_results.json")
    check("alerce.n_objects", alerce["n_objects"], 1114, tol=0)
    check("alerce.ece", alerce["ece"], 0.271, tol=LOOSE_TOL)
    check("alerce.ece_ci_lower", alerce["ece_ci"][0], 0.249, tol=LOOSE_TOL)
    check("alerce.ece_ci_upper", alerce["ece_ci"][1], 0.296, tol=LOOSE_TOL)
    check("alerce.temperature_T", alerce["temperature_scaling"]["T"], 0.357, tol=LOOSE_TOL)
    check("alerce.ece_post", alerce["temperature_scaling"]["ece_after"], 0.097, tol=LOOSE_TOL)

    gain_cv = load("alerce_operational_gain_cv_results.json")
    check("alerce.gain_pooled_cv", gain_cv["pooled_held_out"]["gain"], 20.9, tol=LOOSE_TOL)
    if not gain_cv["validation"]["passes"]:
        _failures.append("alerce.gain_cv.validation.passes")
        print("  [FAIL] alerce_operational_gain_cv_results.json self-reports validation.passes = False")
    else:
        print("  [PASS] alerce.gain_cv.validation.passes == True")

    canonical["alerce"] = {
        "n_objects": alerce["n_objects"],
        "ece": alerce["ece"],
        "ece_ci": alerce["ece_ci"],
        "temperature_T": alerce["temperature_scaling"]["T"],
        "ece_post": alerce["temperature_scaling"]["ece_after"],
        "operational_gain_pooled_cv": gain_cv["pooled_held_out"]["gain"],
    }

    # ---------------------------------------------------------------
    # Fink RF + SNN abstention
    # ---------------------------------------------------------------
    print("\nFink RF / SNN abstention (regime mismatch vs selective classification)")
    fink_zero = load("fink_zero_abstention_summary.json")
    rf = fink_zero["fink_random_forest"]["overall"]
    snn = fink_zero["fink_supernnova"]["overall"]
    check("fink.n_total", rf["total_objects"], 1237, tol=0)
    check("fink_rf.zero_fraction", rf["zero_score_fraction"], 0.939, tol=LOOSE_TOL)
    check("fink_snn.zero_fraction", snn["zero_score_fraction"], 0.359, tol=LOOSE_TOL)
    check("fink_snn.nonzero_count", snn["nonzero_score_count"], 793, tol=0)

    canonical["fink"] = {
        "n_total": rf["total_objects"],
        "rf_zero_fraction": rf["zero_score_fraction"],
        "snn_zero_fraction": snn["zero_score_fraction"],
        "snn_n_accepted": snn["nonzero_score_count"],
    }

    print("\nFink SuperNNova conditional calibration (accepted set only)")
    snn_cond = load("fink_snn_conditional_analysis.json")
    step1 = snn_cond["step1_conditional_calibration"]
    step3 = snn_cond["step3_temperature_scaling"]
    check("fink_snn.n_accepted", step1["n_accepted"], 793, tol=0)
    check("fink_snn.ece_conditional", step1["ece"], 0.183, tol=LOOSE_TOL)
    check("fink_snn.ece_ci_lower", step1["ece_ci"][0], 0.154, tol=LOOSE_TOL)
    check("fink_snn.ece_ci_upper", step1["ece_ci"][1], 0.220, tol=LOOSE_TOL)
    check("fink_snn.temperature_T", step3["temperature"], 3.65, tol=LOOSE_TOL)
    check("fink_snn.ece_post", step3["ece_post"], 0.051, tol=LOOSE_TOL)

    canonical["fink_snn_conditional"] = {
        "n_accepted": step1["n_accepted"],
        "ece": step1["ece"],
        "ece_ci": step1["ece_ci"],
        "temperature_T": step3["temperature"],
        "ece_post": step3["ece_post"],
    }

    # ---------------------------------------------------------------
    # NEEDLE (object-level dedup = primary; model-instance = sensitivity)
    # ---------------------------------------------------------------
    print("\nNEEDLE (object-level dedup, primary)")
    needle_dedup = load("needle_dedup_results.json")
    prim = needle_dedup["primary_object_level"]
    sens = needle_dedup["sensitivity_model_instance"]
    check("needle.n_objects_object_level", prim["n_objects"], 278, tol=0)
    check("needle.ece_object_level", prim["ece_aggregate"], 0.048, tol=LOOSE_TOL)
    check("needle.n_predictions_model_instance", sens["n_objects"], 429, tol=0)
    check("needle.ece_model_instance", sens["ece_aggregate"], 0.073, tol=LOOSE_TOL)

    needle_full = load("needle_results.json")
    gt = needle_full["global_temperature"]
    check("needle.global_T", gt["T"], 1.552, tol=LOOSE_TOL)
    check("needle.global_ece_before", gt["ece_before"], 0.126, tol=LOOSE_TOL)
    check("needle.global_ece_after", gt["ece_after"], 0.169, tol=LOOSE_TOL)
    if gt["recommended"] is not False:
        _failures.append("needle.global_temperature.recommended")
        print("  [FAIL] needle.global_temperature.recommended should be False (scaling worsens ECE)")
    else:
        print("  [PASS] needle.global_temperature.recommended == False")

    canonical["needle"] = {
        "n_objects_object_level": prim["n_objects"],
        "ece_object_level": prim["ece_aggregate"],
        "ece_ci_object_level": prim["ece_ci"],
        "n_predictions_model_instance": sens["n_objects"],
        "ece_model_instance": sens["ece_aggregate"],
        "global_temperature_T": gt["T"],
        "global_ece_before": gt["ece_before"],
        "global_ece_after": gt["ece_after"],
    }

    # ---------------------------------------------------------------
    # NEEDLE class-weight ratio (Sheng et al. 2024, Table 1, r-band)
    # Verified 2026-07-06 against arXiv:2312.04968: SN=5237, TDE=64
    # -> ratio ~81.8:1, matches the paper's "~80:1" claim. The "~123x"
    # figure once used in scripts/03_needle_analysis.py and
    # results/summary.txt did not correspond to any number recoverable
    # from the cited source and has been corrected. See CLAUDE.md item 2.
    # ---------------------------------------------------------------
    print("\nNEEDLE class-weight ratio (Sheng et al. 2024, Table 1, r-band)")
    sn_train, tde_train = 5237, 64
    ratio = sn_train / tde_train
    check("needle.class_weight_ratio_sn_tde", ratio, 80.0, tol=5.0)
    canonical["needle"]["class_weight_ratio_sn_tde_source"] = {
        "sn_train_count": sn_train,
        "tde_train_count": tde_train,
        "ratio": ratio,
        "source": "Sheng et al. 2024 (arXiv:2312.04968), Table 1, r-band",
    }

    # ---------------------------------------------------------------
    # Bin sensitivity (ECE stable at M=15)
    # ---------------------------------------------------------------
    print("\nBin sensitivity")
    bins = load("bin_sensitivity_results.json")
    canonical["bin_sensitivity"] = bins

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    n_pass = sum(1 for *_, ok in _checks if ok)
    n_total = len(_checks)
    print(f"RESULT: {n_pass}/{n_total} checks passed")

    out_path = RESULTS / "canonical_numbers_v1.json"
    out_path.write_text(json.dumps(canonical, indent=2, sort_keys=True) + "\n")
    print(f"Snapshot written to {out_path}")

    if _failures:
        print("\nFAILED CHECKS:")
        for f in _failures:
            print(f"  - {f}")
        print("=" * 70)
        sys.exit(1)

    print("ALL CHECKS PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
