"""
Recomputes ALeRCE and NEEDLE calibration metrics directly from the raw
data files and asserts they match results/canonical_numbers_v1.json
within tolerance.

This is a regression test on src/calibration.py itself: if someone edits
the ECE binning, the stratified-fold logic, or the temperature-scaling
optimizer, this is what catches the drift before it reaches a results
file. scripts/19_verify_canonical.py checks the opposite direction --
that results/*.json still matches what the paper cites -- and does not
recompute from raw data.

Raw data files (data/raw/, data/processed/) are not tracked in git (see
DATA_ACQUISITION.md); tests here skip gracefully when they are absent
rather than failing, since a missing frozen data snapshot is not a code
regression.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import bootstrap_ece, fit_temperature_cv

REPO_ROOT = Path(__file__).resolve().parent.parent
TOL = 0.02  # matches scripts/19_verify_canonical.py's LOOSE_TOL

ALERCE_FILE = REPO_ROOT / "data/raw/alerce_classifications.csv"
SAMPLE_FILE = REPO_ROOT / "data/ground_truth/bts_sample.csv"
NEEDLE_FILE = REPO_ROOT / "data/processed/needle_predictions.npz"
CANONICAL_FILE = REPO_ROOT / "results/canonical_numbers_v1.json"

needs_alerce_data = pytest.mark.skipif(
    not (ALERCE_FILE.exists() and SAMPLE_FILE.exists()),
    reason="ALeRCE raw data not present locally (not tracked in git; "
           "see DATA_ACQUISITION.md to reacquire)",
)
needs_needle_data = pytest.mark.skipif(
    not NEEDLE_FILE.exists(),
    reason="NEEDLE predictions not present locally (not tracked in git; "
           "see DATA_ACQUISITION.md to reacquire)",
)


@pytest.fixture(scope="module")
def canonical():
    if not CANONICAL_FILE.exists():
        pytest.skip("results/canonical_numbers_v1.json missing -- run "
                    "scripts/19_verify_canonical.py first")
    return json.loads(CANONICAL_FILE.read_text())


@pytest.fixture(scope="module")
def alerce_data():
    alerce = pd.read_csv(ALERCE_FILE)
    sample = pd.read_csv(SAMPLE_FILE)
    merged = alerce.merge(sample[["ZTFID", "alerce_class"]],
                           left_on="oid", right_on="ZTFID")
    merged = merged[merged["alerce_class"] != "TDE"]

    trans = ["SNIa", "SNIbc", "SNII", "SLSN"]
    class_to_int = {c: i for i, c in enumerate(trans)}
    proba = merged[trans].apply(pd.to_numeric, errors="coerce").values
    proba = proba / proba.sum(axis=1, keepdims=True)
    labels = merged["alerce_class"].map(class_to_int).values
    return labels, proba


@pytest.fixture(scope="module")
def needle_data():
    needle = np.load(NEEDLE_FILE, allow_pickle=True)
    return needle["labels"], needle["probs"]


@needs_alerce_data
def test_alerce_ece_matches_canonical(alerce_data, canonical):
    labels, proba = alerce_data
    boot = bootstrap_ece(labels, proba)
    assert boot["ece"] == pytest.approx(canonical["alerce"]["ece"], abs=TOL)


@needs_alerce_data
def test_alerce_temperature_scaling_matches_canonical(alerce_data, canonical):
    labels, proba = alerce_data
    ts = fit_temperature_cv(labels, proba)
    assert ts["T_mean"] == pytest.approx(
        canonical["alerce"]["temperature_T"], abs=TOL)
    assert ts["ece_after"] == pytest.approx(
        canonical["alerce"]["ece_post"], abs=TOL)


@needs_needle_data
def test_needle_model_instance_ece_matches_canonical(needle_data, canonical):
    labels, proba = needle_data
    boot = bootstrap_ece(labels, proba)
    assert boot["ece"] == pytest.approx(
        canonical["needle"]["ece_model_instance"], abs=TOL)


@needs_needle_data
def test_needle_global_temperature_worsens_ece(needle_data, canonical):
    """Confirms the paper's structural finding: global T-scaling worsens
    NEEDLE's ECE because SN and SLSN-I/TDE need opposite corrections."""
    labels, proba = needle_data
    ts = fit_temperature_cv(labels, proba)
    assert ts["T_mean"] == pytest.approx(
        canonical["needle"]["global_temperature_T"], abs=TOL)
    assert ts["recommended"] is False
    assert ts["ece_after"] > ts["ece_before"]
