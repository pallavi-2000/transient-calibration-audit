# Transient Calibration Audit

**A systematic calibration audit of production astronomical transient classifiers.**

> *Are the probability outputs of broker classifiers trustworthy enough to drive follow-up decisions?*

This repository contains the full analysis pipeline, data, and manuscript for a systematic calibration study (to our knowledge, the first against independent spectroscopic ground truth) of three production transient classification systems: **ALeRCE** (Automatic Learning for the Rapid Classification of Events), **Fink**, and **NEEDLE** (Novel Efficient Detection of Extragalactic Events with Light-curve and Environment). All three are deployed in production and routinely used to prioritise spectroscopic and photometric follow-up of ZTF alerts.

---

## Key Results

| Classifier | N | ECE (raw) | Direction | T (optimal) | ECE (post-calib) |
|---|---|---|---|---|---|
| **ALeRCE** | 1,576 | 0.259 [0.237, 0.280] | Underconfident | 0.36 ± 0.01 | 0.015 |
| **Fink (RF)**¹ | 1,368 | 0.464 [0.437, 0.490] | Severely underconfident | ≫10 (bound) | 0.353 |
| **Fink (SNN)** | 1,368 | 0.304 [0.278, 0.332] | Underconfident | ≫10 (bound) | 0.229 |
| **NEEDLE-TH** | 429 | **0.050** | Class-asymmetric | — | — |

**ECE** = Expected Calibration Error (lower is better; 0 = perfectly calibrated).
All CIs are 95% bootstrap intervals. Post-calibration uses 5-fold CV temperature scaling.

¹ `rf_snia_vs_nonia` is documented by Fink as targeting *rising*, early-phase
SN Ia candidates, not general-purpose posteriors — it's evaluated here
against the full BTS sample (which includes late-phase discoveries outside
its intended regime), so its ECE reflects both miscalibration and a
task/target mismatch. Don't read it as a straight apples-to-apples
comparison against ALeRCE LC or Fink SNN.

### NEEDLE per-class temperatures (vector scaling)

| Class | T (mean ± std) | Interpretation |
|---|---|---|
| SN | 1.58 ± 0.10 | Overconfident — soften |
| SLSN-I | **1.88 ± 0.29** | Severely overconfident — inverse-freq weighting artefact |
| TDE | **0.42 ± 0.04** | Underconfident — model is conservative on the hardest class |

Folds are grouped by ZTF object ID (`GroupKFold`), not by row. NEEDLE-TH is a
5-model ensemble evaluated on 429 pooled predictions covering only 278 unique
objects — 151 rows are repeat appearances of an object held out by more than
one ensemble member. A naive row-wise split could put one appearance of an
object in the fitting fold and another appearance of the *same* object in the
"held-out" fold, leaking its label across the split. `GroupKFold` keeps every
appearance of an object in one fold; the wider fold-to-fold spread above
(e.g. SLSN-I std more than doubled vs. a row-wise split) reflects genuine
sampling uncertainty that the earlier, leakier CV was masking. The point
estimates themselves barely moved — the qualitative finding (global scaling
fails; SLSN-I overconfident / TDE underconfident) is robust to the fix.

### ALeRCE version-stability check

The ALeRCE sample was collected by querying each object's most recent
classification at query time, which spans whatever classifier version was
live in production. Re-querying the API for the `classifier_version` tag
(`src/collect_alerce_versions.py`, `data/raw/alerce_versions.csv`, 1,601/1,606
objects recovered) found two distinct versions in the collection window —
`hierarchical_rf_1.1.0` (n=1,346) and a newer `lc_classifier_1.1.13` (n=225).
Stratifying 15-class top-1 ECE by version (`src/alerce_version_check.py`)
gives nearly identical values (0.258 [0.235, 0.281] vs. 0.260 [0.200, 0.317])
with fitted temperatures in the same sharpening regime (0.38 vs. 0.33) —
pooling across versions does not appear to distort the headline
ECE=0.259/T≈0.36 result. Caveats: version tags come from a later re-query
(current metadata may not match the originally retrieved vectors), and the
equivalent check has not been done for Fink.

---

## Scientific Context

Production transient classifiers assign class probabilities to millions of nightly ZTF alerts. These probabilities are consumed directly by follow-up schedulers, survey strategies, and downstream science analyses — yet their reliability as *calibrated* probability estimates has never been systematically tested across multiple production systems.

A well-calibrated classifier satisfies: among all events assigned probability *p* to class *k*, approximately fraction *p* truly belong to class *k*. Miscalibration in either direction has real scientific cost: underconfident systems waste follow-up resources on events with artificially low scores; overconfident systems produce silent failures — high-confidence wrong classifications that go unquestioned.

**What we find:**

- **ALeRCE** (Carrasco-Davis et al. 2021) is strongly underconfident (ECE = 0.259), with mean confidence 45% against 71% accuracy. A single temperature parameter T = 0.36 reduces ECE by 94% to 0.015, indicating the underlying discriminative model is sound but systematically hedges its predictions — consistent with the inverse-frequency class weighting used during training.

- **Fink** (Möller et al. 2021) is more severely miscalibrated: the Random Forest classifier reaches ECE = 0.464 and the SuperNNova model ECE = 0.304. More fundamentally, **both scores carry no measurable ranking signal on this sample** (ROC-AUC = 0.502 and 0.525) — since monotone recalibration preserves ranking, no post-hoc correction (temperature, Platt, isotonic, beta) can make them informative posteriors. CV isotonic regression *does* reach ECE ≈ 0.01, but only by collapsing to the base rate — calibrated and useless, a caution against reading ECE alone. Early-phase proxy subsets (ndethist ≤ 5/3) do not rescue the RF, weakening the "evaluated outside its regime" explanation.

- **NEEDLE** (Sheng et al. 2024, MNRAS 531, 2474) appears the best-calibrated of the three (ECE = 0.050), but its residual miscalibration is *class-asymmetric* in a way that resists a global scalar correction. SLSN-I — the rarest class and the one most aggressively upweighted by inverse-frequency training — is badly overconfident (T_SLSNI ≈ 1.88), while TDE is underconfident (T_TDE ≈ 0.43). This asymmetry is a direct, quantifiable consequence of class-imbalance handling via frequency weighting.

**Practical implication:** All three classifiers benefit from post-hoc temperature scaling, but the method's effectiveness varies enormously (ECE reduction: ALeRCE 94%, Fink 11–25%, NEEDLE negative for global T). Practitioners using these systems for follow-up decisions should apply per-broker calibration corrections; raw softmax outputs should not be treated as calibrated probabilities.

---

## Repository Structure

```
transient-calibration-audit/
├── notebooks/
│   ├── 01_alerce_data_collection.ipynb   # BTS cross-match + ALeRCE API queries
│   ├── 02_alerce_calibration.ipynb       # ECE, reliability diagrams, temp scaling
│   ├── 03_fink_data_collection.ipynb     # Fink broker API collection
│   └── 04_fink_calibration.ipynb         # Fink RF + SNN calibration analysis
│
├── src/
│   ├── calibration.py          # ECE, reliability diagram, temperature/isotonic scaling
│   ├── collect_alerce.py       # ALeRCE broker API client + prediction extraction
│   ├── collect_fink.py         # Fink broker API client
│   ├── build_dataset.py        # BTS cross-match, label unification, dataset assembly
│   ├── needle_01_explore.py    # NEEDLE HDF5 + model structure exploration
│   ├── needle_02_extract.py    # Softmax extraction from 5 NEEDLE-TH models
│   ├── needle_03_calibration.py # NEEDLE ECE + per-class reliability analysis
│   ├── needle_04_temperature.py # Global temperature scaling (5-fold CV)
│   └── needle_05_vector_scaling.py  # Per-class vector temperature scaling
│
├── data/
│   ├── raw/
│   │   ├── bts_catalog.csv           # ZTF BTS spectroscopic classifications
│   │   ├── alerce_predictions.csv    # ALeRCE softmax outputs (N=1,606)
│   │   ├── alerce_checkpoint.csv     # ALeRCE API checkpoint
│   │   ├── fink_predictions.csv      # Fink RF + SNN scores (N=1,368)
│   │   └── fink_checkpoint.csv       # Fink API checkpoint
│   └── processed/
│       ├── alerce_dataset.csv        # Cleaned, cross-matched dataset
│       └── needle_predictions.npz    # NEEDLE softmax arrays (5 models × test sets)
│
├── results/
│   ├── figures/
│   │   ├── fig1_reliability_diagram.{png,pdf}      # ALeRCE aggregate reliability
│   │   ├── fig2_perclass_reliability.{png,pdf}     # ALeRCE per-class
│   │   ├── fig3_fink_reliability.{png,pdf}         # Fink RF + SNN
│   │   ├── fig4_temperature_scaling.{png,pdf}      # ALeRCE before/after T scaling
│   │   ├── needle_reliability_diagram.png          # NEEDLE aggregate
│   │   ├── needle_per_class_reliability.png        # NEEDLE per-class
│   │   ├── needle_temperature_scaling.png          # Global T (fails)
│   │   ├── needle_vector_scaling_comparison.png    # Raw vs global T vs vector T
│   │   └── needle_vector_scaling_per_class.png     # Per-class before/after
│   └── tables/
│       ├── aggregate_ece.json                  # ALeRCE aggregate ECE + CIs
│       ├── per_class_ece.json                  # ALeRCE per-class breakdown
│       ├── temperature_scaling_cv.json         # ALeRCE CV temperature scaling
│       ├── fink_ece.json                       # Fink RF + SNN ECE
│       ├── fink_temperature_scaling.json       # Fink temperature scaling
│       ├── needle_calibration_summary.txt      # NEEDLE ECE summary
│       ├── needle_temperature_scaling_results.txt
│       └── needle_vector_scaling_results.txt
│
└── paper/
    ├── main.tex              # LaTeX manuscript
    ├── main.pdf              # Compiled paper
    ├── references.bib
    └── sections/
        ├── data.tex
        ├── methods.tex
        ├── results.tex
        ├── discussion.tex
        └── related.tex
```

---

## Methods

### Calibration metric
We use Expected Calibration Error (ECE) with equal-width confidence bins:

```
ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|
```

where *B_b* is the set of predictions in bin *b*, *acc* is the fraction correct, and *conf* is the mean predicted probability. 95% confidence intervals are computed via 1,000-sample bootstrap.

### Post-hoc calibration
**Temperature scaling** divides all logits by a scalar T before the final softmax. T is fitted by minimising negative log-likelihood on a held-out calibration set (5-fold stratified CV throughout). T < 1 sharpens (fixes underconfidence); T > 1 softens (fixes overconfidence).

**Vector scaling** extends temperature scaling to per-class temperatures [T_0, T_1, T_2], fitted jointly by minimising NLL. This addresses class-asymmetric miscalibration that a global scalar cannot correct.

### Data
- **ALeRCE / Fink**: ZTF BTS (Bright Transient Survey) spectroscopic classifications cross-matched against broker probability outputs, 2018–2022. N = 1,606 (ALeRCE), 1,368 (Fink).
- **NEEDLE**: Pre-trained NEEDLE-TH models from Sheng et al. (2024), evaluated on held-out test sets recorded in each model's `testset_obj.json`. N = 429 predictions across 5 models (278 unique objects).

---

## Reproducing the Analysis

```bash
git clone https://github.com/pallavi-2000/transient-calibration-audit
cd transient-calibration-audit
pip install -r requirements.txt

# ALeRCE calibration
jupyter notebook notebooks/02_alerce_calibration.ipynb

# NEEDLE calibration (requires TF 2.12 + NEEDLE models — see src/needle_01_explore.py)
python src/needle_02_extract.py
python src/needle_03_calibration.py
python src/needle_04_temperature.py
python src/needle_05_vector_scaling.py
```

**Note on NEEDLE models:** The pre-trained NEEDLE-TH models (~3 MB each) and the ZTF BTS HDF5 dataset (143 MB) are not included due to size. Download from [Kaggle (sherrysheng97/needle-repo-dataset)](https://www.kaggle.com/datasets/sherrysheng97/needle-repo-dataset) and place at `NEEDLE/needle_th_models/`.

---

## References

- Carrasco-Davis et al. (2021). *Alert Classification for the ALeRCE Broker System: The Light Curve Classifier.* AJ 162, 231.
- Möller et al. (2021). *SuperNNova: an open-source framework for Bayesian, neural network-based supernova classification.* MNRAS 501, 3272.
- Sheng et al. (2024). *NEEDLE: identifying new extragalactic transients in real time.* MNRAS 531, 2474.
- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML 2017.
- Lakshminarayanan et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017.

---

## Citation

If you use this code or results, please cite:

```bibtex
@misc{sati2026calibration,
  author  = {Sati, Pallavi},
  title   = {Transient Calibration Audit: A Systematic Study of Probability
             Calibration in Production Astronomical Transient Classifiers},
  year    = {2026},
  url     = {https://github.com/pallavi-2000/transient-calibration-audit}
}
```

---

*This project is part of an ongoing effort to establish calibration as a standard evaluation criterion for astronomical classifier deployments, alongside accuracy, AUC, and confusion matrices.*
