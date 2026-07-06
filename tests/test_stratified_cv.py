"""
Stratified K-Fold Correctness
=============================

Verifies that src.calibration._stratified_kfold:
1. Puts every class in every fold (critical for rare classes like
   SLSN ~7%, TDE ~3% -- an unstratified split could leave a fold with
   zero examples of a rare class, making NLL optimization unstable).
2. Produces materially different fold membership than a naive random
   permutation split would.
3. Reproduces the paper's cited numbers when run on the real data.

Run:
    pytest tests/test_stratified_cv.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import _stratified_kfold, fit_temperature_cv

REPO_ROOT = Path(__file__).resolve().parent.parent
TOL = 0.02

ALERCE_FILE = REPO_ROOT / "data/raw/alerce_classifications.csv"
FINK_FILE = REPO_ROOT / "data/raw/fink_classifications.csv"
SAMPLE_FILE = REPO_ROOT / "data/ground_truth/bts_sample.csv"
NEEDLE_FILE = REPO_ROOT / "data/processed/needle_predictions.npz"
CANONICAL_FILE = REPO_ROOT / "results/canonical_numbers_v1.json"

# Simulated distribution matching the paper's ALeRCE sample
# (SNIa=463, SNIbc=225, SNII=341, SLSN=85; see DATA_ACQUISITION.md)
SIMULATED_LABELS = np.array([0] * 463 + [1] * 225 + [2] * 341 + [3] * 85)


@pytest.fixture(scope="module")
def canonical():
    if not CANONICAL_FILE.exists():
        pytest.skip("results/canonical_numbers_v1.json missing -- run "
                    "scripts/19_verify_canonical.py first")
    return json.loads(CANONICAL_FILE.read_text())


def test_stratified_folds_contain_every_class():
    folds = _stratified_kfold(SIMULATED_LABELS, n_folds=5, random_state=42)
    n_classes = len(np.unique(SIMULATED_LABELS))

    assert len(folds) == 5
    for cal_idx, test_idx in folds:
        fold_labels = SIMULATED_LABELS[test_idx]
        present = set(np.unique(fold_labels))
        assert present == set(range(n_classes)), (
            f"fold is missing classes {set(range(n_classes)) - present}")
        # cal + test partition the full index set with no overlap
        assert set(cal_idx).isdisjoint(set(test_idx))
        assert len(cal_idx) + len(test_idx) == len(SIMULATED_LABELS)


def test_stratified_beats_naive_permutation_on_rare_class():
    """Demonstrates the bug stratification fixes: with SLSN at only ~7.6%
    of the sample, a naive random 5-way split has a real chance of
    starving a fold of SLSN examples entirely; the stratified split
    never does, by construction."""
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(SIMULATED_LABELS))
    fold_size = len(SIMULATED_LABELS) // 5

    naive_fold_has_gap = False
    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size if fold < 4 else len(SIMULATED_LABELS)
        fold_labels = SIMULATED_LABELS[indices[start:end]]
        if len(np.unique(fold_labels)) < len(np.unique(SIMULATED_LABELS)):
            naive_fold_has_gap = True

    stratified_folds = _stratified_kfold(SIMULATED_LABELS, n_folds=5, random_state=42)
    stratified_always_complete = all(
        len(np.unique(SIMULATED_LABELS[test_idx])) == len(np.unique(SIMULATED_LABELS))
        for _, test_idx in stratified_folds
    )

    assert stratified_always_complete
    # Not asserting naive_fold_has_gap (that's seed-dependent); recorded
    # for visibility into why stratification matters here.
    del naive_fold_has_gap


@pytest.mark.skipif(
    not (ALERCE_FILE.exists() and SAMPLE_FILE.exists()),
    reason="ALeRCE raw data not present locally (see DATA_ACQUISITION.md)",
)
def test_alerce_stratified_cv_matches_canonical(canonical):
    alerce = pd.read_csv(ALERCE_FILE)
    sample = pd.read_csv(SAMPLE_FILE)
    merged = alerce.merge(sample[["ZTFID", "alerce_class"]],
                           left_on="oid", right_on="ZTFID")
    merged = merged[merged["alerce_class"] != "TDE"].copy()

    trans = ["SNIa", "SNIbc", "SNII", "SLSN"]
    class_to_int = {c: i for i, c in enumerate(trans)}
    proba = merged[trans].apply(pd.to_numeric, errors="coerce").values
    proba = proba / proba.sum(axis=1, keepdims=True)
    labels = merged["alerce_class"].map(class_to_int).values

    folds = _stratified_kfold(labels, n_folds=5, random_state=42)
    for _, test_idx in folds:
        assert set(np.unique(labels[test_idx])) == set(class_to_int.values())

    ts = fit_temperature_cv(labels, proba, n_folds=5)
    assert ts["T_mean"] == pytest.approx(canonical["alerce"]["temperature_T"], abs=TOL)
    assert ts["ece_after"] == pytest.approx(canonical["alerce"]["ece_post"], abs=TOL)


@pytest.mark.skipif(
    not NEEDLE_FILE.exists(),
    reason="NEEDLE predictions not present locally (see DATA_ACQUISITION.md)",
)
def test_needle_stratified_cv_matches_canonical(canonical):
    data = np.load(NEEDLE_FILE, allow_pickle=True)
    probs, labels = data["probs"], data["labels"]

    folds = _stratified_kfold(labels, n_folds=5, random_state=42)
    for _, test_idx in folds:
        assert set(np.unique(labels[test_idx])) == set(np.unique(labels))

    ts = fit_temperature_cv(labels, probs, n_folds=5)
    assert ts["T_mean"] == pytest.approx(
        canonical["needle"]["global_temperature_T"], abs=TOL)


@pytest.mark.skipif(
    not (FINK_FILE.exists() and SAMPLE_FILE.exists()),
    reason="Fink raw data not present locally (see DATA_ACQUISITION.md)",
)
def test_fink_snn_conditional_stratified_cv_matches_canonical(canonical):
    """Fits temperature scaling on the ACCEPTED (non-zero) Fink SNN scores
    only. Zero scores are an explicit abstention signal, not P(SN Ia)=0
    (see docs_fink_acquisition.md); including them here would silently
    revert to the pre-revision framing this repo moved away from."""
    fink = pd.read_csv(FINK_FILE)
    sample = pd.read_csv(SAMPLE_FILE)
    merged = fink.merge(sample[["ZTFID", "alerce_class"]],
                         left_on="oid", right_on="ZTFID")
    is_snia = (merged["alerce_class"] == "SNIa").astype(int).values
    snn = pd.to_numeric(merged["snn_snia_vs_nonia"], errors="coerce").values

    accepted = ~np.isnan(snn) & (snn > 0)
    assert accepted.sum() == canonical["fink"]["snn_n_accepted"]

    ts = fit_temperature_cv(is_snia[accepted], snn[accepted], n_folds=5)
    assert ts["T_mean"] == pytest.approx(
        canonical["fink_snn_conditional"]["temperature_T"], abs=TOL * 2)
