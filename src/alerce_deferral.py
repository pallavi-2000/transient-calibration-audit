"""
ALeRCE calibration-aware deferral prototype.

This script evaluates post-hoc selective classification on the existing
spectroscopically labelled ZTF BTS × ALeRCE probability dataset. It does not
train or modify the ALeRCE classifier. It asks whether information already
present in the latest 15-class ALeRCE probability vector can identify
in-taxonomy predictions that should be accepted automatically or deferred.

Scientific scope
----------------
* Primary cohort: SNIa, SNII, SNIbc and SLSN objects represented in the
  deployed 15-class ALeRCE taxonomy.
* TDE objects are excluded from model fitting because TDE is absent from that
  taxonomy. They are retained for a separate exploratory taxonomy-gap stress
  test.
* All learned deferral scores and calibrated probabilities used for primary
  evaluation are out-of-fold.
* Results are conditioned on the deliberately constructed, bright BTS sample;
  they are not estimates for the natural ZTF/LSST alert-stream prevalence.

Methodological grounding
------------------------
* Rejection/deferral with a fixed cost: Chow (1970), IEEE Trans. IT 16(1) —
  accept when P(error) <= deferral cost (the rule used in
  evaluate_deferral_costs).
* Selective classification, risk-coverage curves and AURC: El-Yaniv & Wiener
  (2010), JMLR 11; Geifman & El-Yaniv (2017), NeurIPS. AURC and excess AURC
  (E-AURC, Geifman et al. 2019, ICLR) are reported as threshold-free
  summaries of each policy's full risk-coverage curve.
* Maximum softmax probability / margin / entropy as error- and OOD-detection
  baselines: Hendrycks & Gimpel (2017), ICLR.
* Temperature scaling: Guo et al. (2017), ICML — fitted on log-probabilities
  here because the ALeRCE API exposes probabilities, not logits (same
  convention as src/calibration.py and the published audit).
* This is *selective prediction*, not learning-to-defer in the sense of
  Madras et al. (2018) / Mozannar & Sontag (2020): there is no model of the
  downstream expert (spectroscopic follow-up is assumed always correct), so
  the optimal policy reduces to Chow's rule on P(top-1 correct).

Run from the repository root:
    python src/alerce_deferral.py

Optional:
    python src/alerce_deferral.py --seed 42 --n-splits 5 --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ALERCE_PROB_COLS: List[str] = [
    "SNIa", "SNIbc", "SNII", "SLSN", "QSO",
    "AGN", "Blazar", "CV/Nova", "YSO", "LPV",
    "E", "DSCT", "RRL", "CEP", "Periodic-Other",
]

IN_TAXONOMY_CLASSES: List[str] = ["SNIa", "SNII", "SNIbc", "SLSN"]
TRANSIENT_CLASSES: List[str] = ["SNIa", "SNIbc", "SNII", "SLSN"]
TRANSIENT_INDICES = np.array([ALERCE_PROB_COLS.index(c) for c in TRANSIENT_CLASSES])
CLASS_TO_INDEX = {name: i for i, name in enumerate(ALERCE_PROB_COLS)}

DEFAULT_COVERAGES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)
DEFAULT_THRESHOLDS = (0.50, 0.60, 0.70, 0.80)
DEFAULT_DEFERRAL_COSTS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
EPS = 1e-12


@dataclass
class DataAudit:
    input_rows: int
    valid_rows: int
    duplicate_oids: int
    invalid_probability_rows: int
    in_taxonomy_rows: int
    tde_rows: int
    other_ground_truth_rows: int
    base_accuracy: float
    class_counts: Dict[str, int]
    predicted_class_counts: Dict[str, int]
    probability_sum_min: float
    probability_sum_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/alerce_dataset.csv"),
        help="Processed ALeRCE dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Root output directory containing figures/ and tables/.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap replicates over fixed out-of-fold predictions. Set 0 to skip.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=500,
        help="Random-policy repetitions used for the random baseline.",
    )
    return parser.parse_args()


def ensure_directories(output_dir: Path) -> Tuple[Path, Path]:
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def validate_and_load(path: Path) -> Tuple[pd.DataFrame, DataAudit]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run src/build_dataset.py first."
        )

    df = pd.read_csv(path)
    input_rows = len(df)

    required = {"oid", "alerce_class", *ALERCE_PROB_COLS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    duplicate_oids = int(df["oid"].duplicated().sum())
    if duplicate_oids:
        raise ValueError(
            f"Found {duplicate_oids} duplicate oid rows. The ALeRCE analysis requires "
            "one row per ZTF object; resolve duplicates before running deferral analysis."
        )

    probs = df[ALERCE_PROB_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = np.isfinite(probs).all(axis=1)
    nonnegative = (probs >= -1e-10).all(axis=1)
    row_sums = probs.sum(axis=1)
    valid_sum = np.isclose(row_sums, 1.0, atol=1e-2)
    valid_mask = finite & nonnegative & valid_sum
    invalid_probability_rows = int((~valid_mask).sum())

    if invalid_probability_rows:
        print(
            f"WARNING: excluding {invalid_probability_rows} rows with missing, negative, "
            "or non-normalised probabilities.",
            file=sys.stderr,
        )
        df = df.loc[valid_mask].copy().reset_index(drop=True)
        probs = probs[valid_mask]
        row_sums = row_sums[valid_mask]

    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    df.loc[:, ALERCE_PROB_COLS] = probs

    pred_idx = probs.argmax(axis=1)
    pred_class = np.asarray(ALERCE_PROB_COLS, dtype=object)[pred_idx]
    df["predicted_class_recomputed"] = pred_class
    df["confidence_recomputed"] = probs[np.arange(len(df)), pred_idx]
    df["correct_recomputed"] = (pred_class == df["alerce_class"].astype(str).to_numpy()).astype(int)

    in_tax = df["alerce_class"].isin(IN_TAXONOMY_CLASSES)
    tde = df["alerce_class"].eq("TDE")
    other = ~(in_tax | tde)

    audit = DataAudit(
        input_rows=input_rows,
        valid_rows=len(df),
        duplicate_oids=duplicate_oids,
        invalid_probability_rows=invalid_probability_rows,
        in_taxonomy_rows=int(in_tax.sum()),
        tde_rows=int(tde.sum()),
        other_ground_truth_rows=int(other.sum()),
        base_accuracy=float(df.loc[in_tax, "correct_recomputed"].mean()),
        class_counts={
            str(k): int(v)
            for k, v in df["alerce_class"].value_counts(dropna=False).to_dict().items()
        },
        predicted_class_counts={
            str(k): int(v)
            for k, v in df.loc[in_tax, "predicted_class_recomputed"].value_counts().to_dict().items()
        },
        probability_sum_min=float(row_sums.min()),
        probability_sum_max=float(row_sums.max()),
    )
    return df, audit


def probabilities_to_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Temperature must be positive and finite, got {temperature}")
    logp = np.log(np.clip(np.asarray(probs, dtype=float), EPS, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    exp_logp = np.exp(logp)
    return exp_logp / exp_logp.sum(axis=1, keepdims=True)


def multiclass_nll(probs: np.ndarray, y_true_idx: np.ndarray) -> float:
    chosen = np.clip(probs[np.arange(len(y_true_idx)), y_true_idx], EPS, 1.0)
    return float(-np.log(chosen).mean())


def fit_temperature(
    probs: np.ndarray,
    y_true_idx: np.ndarray,
    lower: float = 0.1,
    upper: float = 10.0,
) -> float:
    """Fit T by NLL minimisation. Optimising log(T) enforces positivity."""

    def objective(log_t: float) -> float:
        scaled = probabilities_to_temperature(probs, float(np.exp(log_t)))
        return multiclass_nll(scaled, y_true_idx)

    result = minimize_scalar(
        objective,
        bounds=(math.log(lower), math.log(upper)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimisation failed: {result.message}")
    return float(np.exp(result.x))


def normalised_entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    return -(probs * np.log(probs)).sum(axis=1) / math.log(probs.shape[1])


def shape_features(probs: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Probability-shape features that do not use follow-up metadata."""
    probs = np.asarray(probs, dtype=float)
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    margin = top1 - top2
    entropy = normalised_entropy(probs)

    transient_mass = probs[:, TRANSIENT_INDICES].sum(axis=1)
    transient_cond = probs[:, TRANSIENT_INDICES] / np.clip(
        transient_mass[:, None], EPS, None
    )
    transient_cond_max = transient_cond.max(axis=1)
    transient_cond_entropy = normalised_entropy(transient_cond)

    nontransient_mask = np.ones(probs.shape[1], dtype=bool)
    nontransient_mask[TRANSIENT_INDICES] = False
    nontransient_max = probs[:, nontransient_mask].max(axis=1)

    features = np.column_stack(
        [
            top1,
            top2,
            margin,
            entropy,
            transient_mass,
            transient_cond_max,
            transient_cond_entropy,
            nontransient_max,
        ]
    )
    names = [
        "top1_probability",
        "top2_probability",
        "top1_top2_margin",
        "normalised_entropy",
        "transient_branch_mass",
        "conditional_transient_max",
        "conditional_transient_entropy",
        "nontransient_max",
    ]
    return features, names


def direct_policy_scores(probs: np.ndarray) -> Dict[str, np.ndarray]:
    features, names = shape_features(probs)
    feature_map = {name: features[:, i] for i, name in enumerate(names)}
    return {
        "raw_confidence": feature_map["top1_probability"],
        "raw_margin": feature_map["top1_top2_margin"],
        "raw_negative_entropy": 1.0 - feature_map["normalised_entropy"],
        "raw_transient_mass": feature_map["transient_branch_mass"],
        "raw_conditional_transient_max": feature_map["conditional_transient_max"],
    }


def make_logistic_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    class_weight=None,
                    random_state=seed,
                ),
            ),
        ]
    )


def smoothed_predicted_class_rate(
    train_predicted: np.ndarray,
    train_correct: np.ndarray,
    test_predicted: np.ndarray,
    prior_strength: float = 10.0,
) -> np.ndarray:
    global_rate = float(np.mean(train_correct))
    table: Dict[str, float] = {}
    for cls in np.unique(train_predicted):
        mask = train_predicted == cls
        count = int(mask.sum())
        successes = float(train_correct[mask].sum())
        table[str(cls)] = (successes + prior_strength * global_rate) / (
            count + prior_strength
        )
    return np.asarray([table.get(str(cls), global_rate) for cls in test_predicted])


def choose_stratification_labels(
    true_class: np.ndarray,
    correct: np.ndarray,
    n_splits: int,
) -> np.ndarray:
    composite = np.char.add(
        np.char.add(true_class.astype(str), "__correct_"), correct.astype(str)
    )
    counts = pd.Series(composite).value_counts()
    if len(counts) and int(counts.min()) >= n_splits:
        return composite
    print(
        "WARNING: at least one true-class × correctness stratum is too small; "
        "falling back to true-class stratification.",
        file=sys.stderr,
    )
    return true_class.astype(str)


def cross_validated_scores(
    probs: np.ndarray,
    true_class: np.ndarray,
    y_true_idx: np.ndarray,
    correct: np.ndarray,
    seed: int,
    n_splits: int,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, List[str], List[str], np.ndarray]:
    n = len(correct)
    scores = direct_policy_scores(probs)
    scores.update(
        {
            "calibrated_confidence": np.full(n, np.nan),
            "calibrated_margin": np.full(n, np.nan),
            "calibrated_negative_entropy": np.full(n, np.nan),
            "calibrated_transient_mass": np.full(n, np.nan),
            "predicted_class_rate": np.full(n, np.nan),
            "learned_shape": np.full(n, np.nan),
            "learned_full": np.full(n, np.nan),
        }
    )
    fold_id = np.full(n, -1, dtype=int)
    fold_records: List[Dict[str, float]] = []

    strat_labels = choose_stratification_labels(true_class, correct, n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    shape_feature_names: List[str] = []
    full_feature_names: List[str] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(n), strat_labels), start=1
    ):
        train_probs = probs[train_idx]
        test_probs = probs[test_idx]
        t = fit_temperature(train_probs, y_true_idx[train_idx])
        calibrated_train = probabilities_to_temperature(train_probs, t)
        calibrated_test = probabilities_to_temperature(test_probs, t)

        raw_pred_train = np.asarray(ALERCE_PROB_COLS, dtype=object)[
            train_probs.argmax(axis=1)
        ]
        raw_pred_test = np.asarray(ALERCE_PROB_COLS, dtype=object)[
            test_probs.argmax(axis=1)
        ]
        scaled_pred_test = np.asarray(ALERCE_PROB_COLS, dtype=object)[
            calibrated_test.argmax(axis=1)
        ]
        if not np.array_equal(raw_pred_test, scaled_pred_test):
            raise AssertionError("Temperature scaling changed at least one top-1 class.")

        cal_direct = direct_policy_scores(calibrated_test)
        scores["calibrated_confidence"][test_idx] = cal_direct["raw_confidence"]
        scores["calibrated_margin"][test_idx] = cal_direct["raw_margin"]
        scores["calibrated_negative_entropy"][test_idx] = cal_direct[
            "raw_negative_entropy"
        ]
        scores["calibrated_transient_mass"][test_idx] = cal_direct[
            "raw_transient_mass"
        ]

        scores["predicted_class_rate"][test_idx] = smoothed_predicted_class_rate(
            raw_pred_train,
            correct[train_idx],
            raw_pred_test,
        )

        x_shape_train, shape_feature_names = shape_features(calibrated_train)
        x_shape_test, _ = shape_features(calibrated_test)
        shape_model = make_logistic_pipeline(seed + fold)
        shape_model.fit(x_shape_train, correct[train_idx])
        scores["learned_shape"][test_idx] = shape_model.predict_proba(x_shape_test)[:, 1]

        x_full_train = np.column_stack([calibrated_train, x_shape_train])
        x_full_test = np.column_stack([calibrated_test, x_shape_test])
        full_feature_names = [f"p_{c}" for c in ALERCE_PROB_COLS] + shape_feature_names
        full_model = make_logistic_pipeline(seed + 100 + fold)
        full_model.fit(x_full_train, correct[train_idx])
        scores["learned_full"][test_idx] = full_model.predict_proba(x_full_test)[:, 1]

        fold_id[test_idx] = fold
        fold_records.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "temperature": t,
                "train_accuracy": float(correct[train_idx].mean()),
                "test_accuracy": float(correct[test_idx].mean()),
                "train_nll_before": multiclass_nll(train_probs, y_true_idx[train_idx]),
                "train_nll_after": multiclass_nll(
                    calibrated_train, y_true_idx[train_idx]
                ),
            }
        )

    if (fold_id < 0).any():
        raise AssertionError("Some objects did not receive an out-of-fold prediction.")
    for name, values in scores.items():
        if not np.isfinite(values).all():
            raise AssertionError(f"Policy {name} contains missing/non-finite scores.")

    return scores, pd.DataFrame(fold_records), shape_feature_names, full_feature_names, fold_id


def binary_ece(values: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    targets = np.asarray(targets, dtype=float)
    bin_ids = np.minimum((values * n_bins).astype(int), n_bins - 1)
    result = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            result += mask.mean() * abs(values[mask].mean() - targets[mask].mean())
    return float(result)


def aurc_metrics(score: np.ndarray, correct: np.ndarray) -> Tuple[float, float]:
    """AURC and excess AURC over the full risk-coverage curve.

    AURC (Geifman et al. 2019): mean selective risk over all coverage levels
    k/n, accepting the k highest-scored objects. E-AURC subtracts the AURC of
    the oracle ordering (all correct before all incorrect), isolating the
    ranking quality of the score from the base error rate.
    """
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    incorrect_sorted = 1.0 - correct[order]
    k = np.arange(1, n + 1)
    selective_risk = np.cumsum(incorrect_sorted) / k
    aurc = float(selective_risk.mean())

    oracle_incorrect = np.sort(1.0 - correct)  # all zeros first
    oracle_risk = np.cumsum(oracle_incorrect) / k
    e_aurc = aurc - float(oracle_risk.mean())
    return aurc, e_aurc


def evaluate_score_quality(scores: Mapping[str, np.ndarray], correct: np.ndarray) -> pd.DataFrame:
    records = []
    probability_like = {
        "raw_confidence",
        "calibrated_confidence",
        "predicted_class_rate",
        "learned_shape",
        "learned_full",
    }
    for name, values in scores.items():
        aurc, e_aurc = aurc_metrics(values, correct)
        record = {
            "policy": name,
            "roc_auc_correctness": float(roc_auc_score(correct, values)),
            "average_precision_correctness": float(average_precision_score(correct, values)),
            "aurc": aurc,
            "excess_aurc": e_aurc,
        }
        if name in probability_like:
            record["brier_correctness"] = float(brier_score_loss(correct, values))
            record["ece_correctness"] = binary_ece(values, correct)
        else:
            # Margin, entropy and branch-mass scores are ranking signals, not
            # estimates of P(correct); Brier/ECE would be conceptually invalid.
            record["brier_correctness"] = np.nan
            record["ece_correctness"] = np.nan
        records.append(record)
    return pd.DataFrame(records).sort_values("roc_auc_correctness", ascending=False)


def selection_mask_at_coverage(score: np.ndarray, coverage: float) -> np.ndarray:
    n = len(score)
    n_accept = int(round(float(coverage) * n))
    n_accept = min(max(n_accept, 1), n)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:n_accept]] = True
    return mask


def selection_metrics(
    correct: np.ndarray,
    accept: np.ndarray,
) -> Dict[str, float]:
    correct = np.asarray(correct, dtype=int)
    accept = np.asarray(accept, dtype=bool)
    n = len(correct)
    incorrect = 1 - correct
    n_accept = int(accept.sum())
    n_defer = n - n_accept
    total_errors = int(incorrect.sum())
    errors_deferred = int(incorrect[~accept].sum())
    correct_deferred = int(correct[~accept].sum())
    errors_accepted = int(incorrect[accept].sum())

    return {
        "n": n,
        "n_auto_classified": n_accept,
        "n_deferred": n_defer,
        "coverage": n_accept / n,
        "deferral_rate": n_defer / n,
        "auto_accuracy": float(correct[accept].mean()) if n_accept else np.nan,
        "selective_risk": float(incorrect[accept].mean()) if n_accept else np.nan,
        "total_errors": total_errors,
        "errors_accepted": errors_accepted,
        "errors_deferred": errors_deferred,
        "error_capture_fraction": errors_deferred / total_errors if total_errors else np.nan,
        "correct_unnecessarily_deferred": correct_deferred,
        "deferred_set_error_precision": errors_deferred / n_defer if n_defer else np.nan,
    }


def evaluate_coverages(
    scores: Mapping[str, np.ndarray],
    correct: np.ndarray,
    coverages: Sequence[float],
    seed: int,
    n_random: int,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for policy, score in scores.items():
        for coverage in coverages:
            accept = selection_mask_at_coverage(score, coverage)
            records.append(
                {"policy": policy, "target_coverage": coverage, **selection_metrics(correct, accept)}
            )

    oracle_score = correct.astype(float)
    for coverage in coverages:
        records.append(
            {
                "policy": "oracle_reference",
                "target_coverage": coverage,
                **selection_metrics(correct, selection_mask_at_coverage(oracle_score, coverage)),
            }
        )

    rng = np.random.default_rng(seed)
    for coverage in coverages:
        metrics = []
        for _ in range(n_random):
            random_score = rng.random(len(correct))
            metrics.append(
                selection_metrics(correct, selection_mask_at_coverage(random_score, coverage))
            )
        random_record = {"policy": "random_mean", "target_coverage": coverage}
        for key in metrics[0]:
            vals = np.asarray([m[key] for m in metrics], dtype=float)
            finite_vals = vals[np.isfinite(vals)]
            random_record[key] = (
                float(finite_vals.mean()) if len(finite_vals) else np.nan
            )
            if key in {"auto_accuracy", "selective_risk", "error_capture_fraction"}:
                random_record[f"{key}_sd"] = (
                    float(finite_vals.std(ddof=1)) if len(finite_vals) > 1 else np.nan
                )
        records.append(random_record)

    return pd.DataFrame(records)


def evaluate_thresholds(
    scores: Mapping[str, np.ndarray],
    correct: np.ndarray,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    records = []
    for policy in ("raw_confidence", "calibrated_confidence", "learned_shape", "learned_full"):
        score = scores[policy]
        for threshold in thresholds:
            accept = score >= threshold
            records.append(
                {
                    "policy": policy,
                    "threshold": threshold,
                    **selection_metrics(correct, accept),
                }
            )
    return pd.DataFrame(records)


def evaluate_classwise(
    scores: Mapping[str, np.ndarray],
    correct: np.ndarray,
    true_class: np.ndarray,
    coverages: Sequence[float] = (0.80, 0.90),
) -> pd.DataFrame:
    records = []
    for policy, score in scores.items():
        for coverage in coverages:
            accept = selection_mask_at_coverage(score, coverage)
            for cls in IN_TAXONOMY_CLASSES:
                mask = true_class == cls
                metrics = selection_metrics(correct[mask], accept[mask])
                records.append(
                    {
                        "policy": policy,
                        "target_global_coverage": coverage,
                        "true_class": cls,
                        **metrics,
                    }
                )
    return pd.DataFrame(records)


def evaluate_deferral_costs(
    scores: Mapping[str, np.ndarray],
    correct: np.ndarray,
    costs: Sequence[float],
) -> pd.DataFrame:
    """Decision rule: defer when estimated error probability exceeds defer cost."""
    records = []
    probabilistic_policies = [
        "raw_confidence",
        "calibrated_confidence",
        "predicted_class_rate",
        "learned_shape",
        "learned_full",
    ]
    n = len(correct)
    incorrect = 1 - correct
    for cost in costs:
        for policy in probabilistic_policies:
            score = scores[policy]
            accept = (1.0 - score) <= cost
            metrics = selection_metrics(correct, accept)
            realised_cost = (incorrect[accept].sum() + cost * (~accept).sum()) / n
            records.append(
                {
                    "policy": policy,
                    "deferral_cost": cost,
                    "realised_cost_per_object": float(realised_cost),
                    **metrics,
                }
            )
        records.extend(
            [
                {
                    "policy": "all_accept",
                    "deferral_cost": cost,
                    "realised_cost_per_object": float(incorrect.mean()),
                    **selection_metrics(correct, np.ones(n, dtype=bool)),
                },
                {
                    "policy": "all_defer",
                    "deferral_cost": cost,
                    "realised_cost_per_object": float(cost),
                    **selection_metrics(correct, np.zeros(n, dtype=bool)),
                },
            ]
        )
        oracle_accept = correct.astype(bool)
        oracle_cost = (incorrect[oracle_accept].sum() + cost * (~oracle_accept).sum()) / n
        records.append(
            {
                "policy": "oracle_reference",
                "deferral_cost": cost,
                "realised_cost_per_object": float(oracle_cost),
                **selection_metrics(correct, oracle_accept),
            }
        )
    return pd.DataFrame(records)


def stratified_bootstrap_indices(
    true_class: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    pieces = []
    for cls in np.unique(true_class):
        idx = np.where(true_class == cls)[0]
        pieces.append(rng.choice(idx, size=len(idx), replace=True))
    result = np.concatenate(pieces)
    rng.shuffle(result)
    return result


def bootstrap_coverage_intervals(
    scores: Mapping[str, np.ndarray],
    correct: np.ndarray,
    true_class: np.ndarray,
    n_bootstrap: int,
    seed: int,
    policies: Sequence[str] = (
        "raw_confidence",
        "calibrated_confidence",
        "predicted_class_rate",
        "learned_shape",
        "learned_full",
    ),
    coverages: Sequence[float] = (0.80, 0.90),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    samples: Dict[Tuple[str, float, str], List[float]] = {}
    for _ in range(n_bootstrap):
        idx = stratified_bootstrap_indices(true_class, rng)
        for policy in policies:
            boot_score = scores[policy][idx]
            boot_correct = correct[idx]
            for coverage in coverages:
                metrics = selection_metrics(
                    boot_correct, selection_mask_at_coverage(boot_score, coverage)
                )
                for metric in ("auto_accuracy", "selective_risk", "error_capture_fraction"):
                    samples.setdefault((policy, coverage, metric), []).append(metrics[metric])

    records = []
    for (policy, coverage, metric), values in samples.items():
        arr = np.asarray(values, dtype=float)
        records.append(
            {
                "policy": policy,
                "target_coverage": coverage,
                "metric": metric,
                "bootstrap_mean": float(np.nanmean(arr)),
                "ci_low": float(np.nanpercentile(arr, 2.5)),
                "ci_high": float(np.nanpercentile(arr, 97.5)),
                "n_bootstrap": n_bootstrap,
                "note": "Stratified bootstrap over fixed out-of-fold scores; does not refit models.",
            }
        )
    return pd.DataFrame(records)


PAIRED_COMPARISONS: Tuple[Tuple[str, str], ...] = (
    ("learned_shape", "predicted_class_rate"),
    ("learned_full", "predicted_class_rate"),
    ("learned_full", "learned_shape"),
    ("calibrated_confidence", "raw_confidence"),
    ("learned_full", "calibrated_confidence"),
)


def paired_bootstrap_comparisons(
    scores: Mapping[str, np.ndarray],
    correct: np.ndarray,
    true_class: np.ndarray,
    n_bootstrap: int,
    seed: int,
    comparisons: Sequence[Tuple[str, str]] = PAIRED_COMPARISONS,
    coverages: Sequence[float] = (0.80, 0.90),
) -> pd.DataFrame:
    """Paired stratified bootstrap for differences between deferral policies.

    Marginal confidence intervals cannot establish that one policy beats
    another, because both are evaluated on the same objects and their metrics
    are strongly correlated. Both policies are therefore evaluated on the SAME
    bootstrap resample and the difference is recorded. A 95% interval
    excluding zero indicates a resampling-stable difference (over fixed
    out-of-fold scores; models are not refitted per replicate).
    """
    rng = np.random.default_rng(seed)
    diffs: Dict[Tuple[str, str, str], List[float]] = {}
    for _ in range(n_bootstrap):
        idx = stratified_bootstrap_indices(true_class, rng)
        boot_correct = correct[idx]
        if boot_correct.min() == boot_correct.max():
            continue  # degenerate resample: AUC undefined
        for policy_a, policy_b in comparisons:
            score_a = scores[policy_a][idx]
            score_b = scores[policy_b][idx]
            key = (policy_a, policy_b, "delta_roc_auc")
            diffs.setdefault(key, []).append(
                roc_auc_score(boot_correct, score_a)
                - roc_auc_score(boot_correct, score_b)
            )
            aurc_a, _ = aurc_metrics(score_a, boot_correct)
            aurc_b, _ = aurc_metrics(score_b, boot_correct)
            diffs.setdefault((policy_a, policy_b, "delta_aurc"), []).append(
                aurc_a - aurc_b
            )
            for coverage in coverages:
                risk_a = selection_metrics(
                    boot_correct, selection_mask_at_coverage(score_a, coverage)
                )["selective_risk"]
                risk_b = selection_metrics(
                    boot_correct, selection_mask_at_coverage(score_b, coverage)
                )["selective_risk"]
                diffs.setdefault(
                    (policy_a, policy_b, f"delta_selective_risk_at_{coverage:.2f}"),
                    [],
                ).append(risk_a - risk_b)

    records = []
    for (policy_a, policy_b, metric), values in diffs.items():
        arr = np.asarray(values, dtype=float)
        lo = float(np.nanpercentile(arr, 2.5))
        hi = float(np.nanpercentile(arr, 97.5))
        records.append(
            {
                "policy_a": policy_a,
                "policy_b": policy_b,
                "metric": metric,
                "point_delta": float(np.nanmean(arr)),
                "ci_low": lo,
                "ci_high": hi,
                "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                "n_bootstrap": len(arr),
                "note": (
                    "Paired stratified bootstrap over fixed out-of-fold scores. "
                    "For delta_aurc and delta_selective_risk, negative favours "
                    "policy_a; for delta_roc_auc, positive favours policy_a."
                ),
            }
        )
    return pd.DataFrame(records)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def fit_all_and_score_tde(
    in_probs: np.ndarray,
    in_y_idx: np.ndarray,
    in_correct: np.ndarray,
    tde_probs: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], float]:
    t = fit_temperature(in_probs, in_y_idx)
    cal_in = probabilities_to_temperature(in_probs, t)
    cal_tde = probabilities_to_temperature(tde_probs, t)

    tde_scores = direct_policy_scores(tde_probs)
    cal_tde_direct = direct_policy_scores(cal_tde)
    for suffix in ("confidence", "margin", "negative_entropy", "transient_mass"):
        tde_scores[f"calibrated_{suffix}"] = cal_tde_direct[f"raw_{suffix}"]

    pred_in = np.asarray(ALERCE_PROB_COLS, dtype=object)[in_probs.argmax(axis=1)]
    pred_tde = np.asarray(ALERCE_PROB_COLS, dtype=object)[tde_probs.argmax(axis=1)]
    tde_scores["predicted_class_rate"] = smoothed_predicted_class_rate(
        pred_in, in_correct, pred_tde
    )

    x_shape_in, _ = shape_features(cal_in)
    x_shape_tde, _ = shape_features(cal_tde)
    shape_model = make_logistic_pipeline(seed)
    shape_model.fit(x_shape_in, in_correct)
    tde_scores["learned_shape"] = shape_model.predict_proba(x_shape_tde)[:, 1]

    x_full_in = np.column_stack([cal_in, x_shape_in])
    x_full_tde = np.column_stack([cal_tde, x_shape_tde])
    full_model = make_logistic_pipeline(seed + 1)
    full_model.fit(x_full_in, in_correct)
    tde_scores["learned_full"] = full_model.predict_proba(x_full_tde)[:, 1]
    return tde_scores, t


def tde_taxonomy_gap_analysis(
    in_probs: np.ndarray,
    in_y_idx: np.ndarray,
    in_correct: np.ndarray,
    oof_in_scores: Mapping[str, np.ndarray],
    tde_probs: np.ndarray,
    tde_predicted: np.ndarray,
    seed: int,
    coverages: Sequence[float] = (0.80, 0.90),
) -> Tuple[pd.DataFrame, Dict[str, int], float]:
    if len(tde_probs) == 0:
        return pd.DataFrame(), {}, np.nan

    tde_scores, fitted_t = fit_all_and_score_tde(
        in_probs, in_y_idx, in_correct, tde_probs, seed
    )
    policies = [
        "raw_confidence",
        "calibrated_confidence",
        "raw_margin",
        "calibrated_margin",
        "raw_negative_entropy",
        "predicted_class_rate",
        "learned_shape",
        "learned_full",
    ]
    records = []
    for policy in policies:
        for coverage in coverages:
            # Operating cutoffs come from the primary out-of-fold in-taxonomy
            # scores. The all-data fit is used only to score the unseen TDE
            # category, avoiding optimistic in-sample cutoffs.
            accepted_in = selection_mask_at_coverage(oof_in_scores[policy], coverage)
            accepted_scores = oof_in_scores[policy][accepted_in]
            cutoff = float(np.min(accepted_scores))
            tde_deferred = tde_scores[policy] < cutoff
            successes = int(tde_deferred.sum())
            lo, hi = wilson_interval(successes, len(tde_deferred))
            records.append(
                {
                    "policy": policy,
                    "target_in_taxonomy_coverage": coverage,
                    "in_taxonomy_acceptance_cutoff": cutoff,
                    "n_tde": len(tde_deferred),
                    "n_tde_deferred": successes,
                    "tde_deferral_fraction": successes / len(tde_deferred),
                    "tde_deferral_wilson_low": lo,
                    "tde_deferral_wilson_high": hi,
                    "tde_score_median": float(np.median(tde_scores[policy])),
                    "tde_score_min": float(np.min(tde_scores[policy])),
                    "tde_score_max": float(np.max(tde_scores[policy])),
                    "note": "Exploratory taxonomy-gap stress test; not formal OOD validation.",
                }
            )
    pred_counts = {
        str(k): int(v)
        for k, v in pd.Series(tde_predicted).value_counts().to_dict().items()
    }
    return pd.DataFrame(records), pred_counts, fitted_t


def save_oof_predictions(
    df_in: pd.DataFrame,
    scores: Mapping[str, np.ndarray],
    fold_ids: np.ndarray,
    path: Path,
) -> None:
    columns = [
        "oid",
        "spectroscopic_type",
        "alerce_class",
        "predicted_class_recomputed",
        "confidence_recomputed",
        "correct_recomputed",
    ]
    present = [c for c in columns if c in df_in.columns]
    out = df_in[present].copy()
    out["cv_fold"] = fold_ids
    for policy, values in scores.items():
        out[f"score_{policy}"] = values
    out.to_csv(path, index=False)


def plot_risk_coverage(coverage_df: pd.DataFrame, path_base: Path) -> None:
    preferred = [
        "raw_confidence",
        "calibrated_confidence",
        "raw_margin",
        "predicted_class_rate",
        "learned_shape",
        "learned_full",
        "random_mean",
        "oracle_reference",
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    for policy in preferred:
        subset = coverage_df[coverage_df["policy"] == policy].sort_values("coverage")
        if not subset.empty:
            ax.plot(subset["coverage"], subset["selective_risk"], marker="o", label=policy)
    ax.set_xlabel("Coverage (fraction automatically classified)")
    ax.set_ylabel("Selective risk (error rate among accepted objects)")
    ax.set_title("ALeRCE post-hoc deferral: risk–coverage comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=240)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_error_capture(coverage_df: pd.DataFrame, path_base: Path) -> None:
    preferred = [
        "raw_confidence",
        "calibrated_confidence",
        "raw_margin",
        "predicted_class_rate",
        "learned_shape",
        "learned_full",
        "random_mean",
        "oracle_reference",
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    for policy in preferred:
        subset = coverage_df[coverage_df["policy"] == policy].sort_values("deferral_rate")
        if not subset.empty:
            ax.plot(
                subset["deferral_rate"],
                subset["error_capture_fraction"],
                marker="o",
                label=policy,
            )
    ax.set_xlabel("Deferral rate")
    ax.set_ylabel("Fraction of all classification errors deferred")
    ax.set_title("Error capture by ALeRCE deferral policy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=240)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_thresholds(threshold_df: pd.DataFrame, path_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for policy in ("raw_confidence", "calibrated_confidence", "learned_full"):
        subset = threshold_df[threshold_df["policy"] == policy].sort_values("threshold")
        if not subset.empty:
            ax.plot(
                subset["coverage"],
                subset["auto_accuracy"],
                marker="o",
                label=policy,
            )
            for _, row in subset.iterrows():
                ax.annotate(
                    f"{row['threshold']:.1f}",
                    (row["coverage"], row["auto_accuracy"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
    ax.set_xlabel("Coverage at fixed numerical threshold")
    ax.set_ylabel("Accuracy among automatically classified objects")
    ax.set_title("Fixed-threshold operating points")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=240)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_costs(cost_df: pd.DataFrame, path_base: Path) -> None:
    preferred = [
        "raw_confidence",
        "calibrated_confidence",
        "predicted_class_rate",
        "learned_shape",
        "learned_full",
        "all_accept",
        "all_defer",
        "oracle_reference",
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    for policy in preferred:
        subset = cost_df[cost_df["policy"] == policy].sort_values("deferral_cost")
        if not subset.empty:
            ax.plot(
                subset["deferral_cost"],
                subset["realised_cost_per_object"],
                marker="o",
                label=policy,
            )
    ax.set_xlabel("Assumed cost of deferring one object")
    ax.set_ylabel("Realised cost per object")
    ax.set_title("Decision-theoretic post-hoc deferral")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=240)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def pythonise(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [pythonise(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): pythonise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pythonise(v) for v in value]
    return value


def selected_rows_as_dict(
    frame: pd.DataFrame,
    policies: Iterable[str],
    coverage_col: str = "target_coverage",
    coverages: Sequence[float] = (0.80, 0.90),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for policy in policies:
        result[policy] = {}
        for coverage in coverages:
            row = frame[
                (frame["policy"] == policy)
                & np.isclose(frame[coverage_col].astype(float), coverage)
            ]
            if not row.empty:
                result[policy][f"coverage_{coverage:.2f}"] = pythonise(
                    row.iloc[0].to_dict()
                )
    return result


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    figures_dir, tables_dir = ensure_directories(args.output_dir)

    df, audit = validate_and_load(args.data)
    in_mask = df["alerce_class"].isin(IN_TAXONOMY_CLASSES).to_numpy()
    tde_mask = df["alerce_class"].eq("TDE").to_numpy()
    df_in = df.loc[in_mask].reset_index(drop=True)
    df_tde = df.loc[tde_mask].reset_index(drop=True)

    in_probs = df_in[ALERCE_PROB_COLS].to_numpy(float)
    true_class = df_in["alerce_class"].astype(str).to_numpy()
    y_true_idx = np.asarray([CLASS_TO_INDEX[c] for c in true_class], dtype=int)
    correct = df_in["correct_recomputed"].to_numpy(int)

    print("=" * 72)
    print("ALERCE CALIBRATION-AWARE DEFERRAL PROTOTYPE")
    print("=" * 72)
    print(json.dumps(asdict(audit), indent=2))

    scores, folds_df, shape_names, full_names, fold_ids = cross_validated_scores(
        probs=in_probs,
        true_class=true_class,
        y_true_idx=y_true_idx,
        correct=correct,
        seed=args.seed,
        n_splits=args.n_splits,
    )

    score_quality_df = evaluate_score_quality(scores, correct)
    coverage_df = evaluate_coverages(
        scores,
        correct,
        DEFAULT_COVERAGES,
        seed=args.seed,
        n_random=args.n_random,
    )
    threshold_df = evaluate_thresholds(scores, correct, DEFAULT_THRESHOLDS)
    classwise_df = evaluate_classwise(scores, correct, true_class)
    cost_df = evaluate_deferral_costs(scores, correct, DEFAULT_DEFERRAL_COSTS)

    if args.n_bootstrap > 0:
        bootstrap_df = bootstrap_coverage_intervals(
            scores,
            correct,
            true_class,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 1000,
        )
        paired_df = paired_bootstrap_comparisons(
            scores,
            correct,
            true_class,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 2000,
        )
    else:
        bootstrap_df = pd.DataFrame()
        paired_df = pd.DataFrame()

    tde_df = pd.DataFrame()
    tde_pred_counts: Dict[str, int] = {}
    tde_global_temperature = np.nan
    if len(df_tde):
        tde_probs = df_tde[ALERCE_PROB_COLS].to_numpy(float)
        tde_pred = df_tde["predicted_class_recomputed"].astype(str).to_numpy()
        tde_df, tde_pred_counts, tde_global_temperature = tde_taxonomy_gap_analysis(
            in_probs,
            y_true_idx,
            correct,
            scores,
            tde_probs,
            tde_pred,
            seed=args.seed,
        )

    folds_df.to_csv(tables_dir / "alerce_deferral_cv_folds.csv", index=False)
    score_quality_df.to_csv(tables_dir / "alerce_deferral_score_quality.csv", index=False)
    coverage_df.to_csv(tables_dir / "alerce_deferral_coverage.csv", index=False)
    threshold_df.to_csv(tables_dir / "alerce_deferral_thresholds.csv", index=False)
    classwise_df.to_csv(tables_dir / "alerce_deferral_classwise.csv", index=False)
    cost_df.to_csv(tables_dir / "alerce_deferral_costs.csv", index=False)
    if not bootstrap_df.empty:
        bootstrap_df.to_csv(tables_dir / "alerce_deferral_bootstrap_ci.csv", index=False)
    if not paired_df.empty:
        paired_df.to_csv(tables_dir / "alerce_deferral_paired_comparisons.csv", index=False)
    if not tde_df.empty:
        tde_df.to_csv(tables_dir / "alerce_tde_taxonomy_gap.csv", index=False)

    save_oof_predictions(
        df_in,
        scores,
        fold_ids,
        tables_dir / "alerce_deferral_oof_predictions.csv",
    )

    plot_risk_coverage(coverage_df, figures_dir / "alerce_deferral_risk_coverage")
    plot_error_capture(coverage_df, figures_dir / "alerce_deferral_error_capture")
    plot_thresholds(threshold_df, figures_dir / "alerce_deferral_threshold_operating_points")
    plot_costs(cost_df, figures_dir / "alerce_deferral_cost_curve")

    key_policies = [
        "raw_confidence",
        "calibrated_confidence",
        "predicted_class_rate",
        "learned_shape",
        "learned_full",
    ]
    summary = {
        "analysis_label": "Post-hoc selective-classification / deferral prototype",
        "scope": (
            "Latest available ALeRCE light-curve probabilities on a deliberately "
            "constructed, bright BTS sample. Not an early-alert or LSST-stream evaluation."
        ),
        "audit": asdict(audit),
        "cross_validation": {
            "n_splits": args.n_splits,
            "seed": args.seed,
            "stratification": "true class × correctness when feasible; otherwise true class",
            "temperatures": folds_df["temperature"].tolist(),
            "temperature_mean": float(folds_df["temperature"].mean()),
            "temperature_std": float(folds_df["temperature"].std(ddof=1)),
        },
        "feature_sets": {
            "learned_shape": shape_names,
            "learned_full": full_names,
            "excluded_from_features": [
                "spectroscopic label",
                "redshift",
                "peak magnitude",
                "correctness at inference time",
                "follow-up metadata",
            ],
        },
        "score_quality": pythonise(score_quality_df.to_dict(orient="records")),
        "paired_policy_comparisons": pythonise(
            paired_df.to_dict(orient="records") if not paired_df.empty else []
        ),
        "key_coverage_results": selected_rows_as_dict(coverage_df, key_policies),
        "tde_taxonomy_gap": {
            "n_tde": len(df_tde),
            "global_temperature_fit_on_all_in_taxonomy": pythonise(tde_global_temperature),
            "tde_predicted_class_counts": tde_pred_counts,
            "interpretation": (
                "Exploratory stress test only. TDE is absent from the deployed taxonomy; "
                "failure to defer TDE motivates explicit novelty/OOD detection."
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn_version,
        },
        "outputs": {
            "coverage_table": "results/tables/alerce_deferral_coverage.csv",
            "threshold_table": "results/tables/alerce_deferral_thresholds.csv",
            "classwise_table": "results/tables/alerce_deferral_classwise.csv",
            "cost_table": "results/tables/alerce_deferral_costs.csv",
            "tde_table": "results/tables/alerce_tde_taxonomy_gap.csv",
        },
    }
    with (tables_dir / "alerce_deferral_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pythonise(summary), handle, indent=2)

    print("\nTop score-quality diagnostics:")
    print(score_quality_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nKey 80% and 90% coverage results:")
    key_rows = coverage_df[
        coverage_df["policy"].isin(key_policies)
        & coverage_df["target_coverage"].isin([0.8, 0.9])
    ]
    print(
        key_rows[
            [
                "policy",
                "target_coverage",
                "auto_accuracy",
                "selective_risk",
                "error_capture_fraction",
                "correct_unnecessarily_deferred",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    if not paired_df.empty:
        print("\nPaired policy comparisons (95% bootstrap CI on the difference):")
        print(
            paired_df[
                ["policy_a", "policy_b", "metric", "point_delta", "ci_low", "ci_high", "ci_excludes_zero"]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
    print(f"\nSaved tables to {tables_dir}")
    print(f"Saved figures to {figures_dir}")
    print("\nInterpret results as a post-hoc deferral prototype, not a deployed LSST system.")


if __name__ == "__main__":
    main()
