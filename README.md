# Calibration Audit of Production Astronomical Transient Classifiers

**A systematic calibration audit of ML classifiers deployed on the ZTF alert stream.**

This repository contains the full analysis pipeline, paper source, and reproducibility materials for:

> *Calibration Audit of Production Astronomical Transient Classifiers on ZTF Alert Streams*  
> Pallavi Sati (2026)

We evaluate four classifiers across three production broker systems — **ALeRCE**, **Fink**, and **NEEDLE** — using spectroscopically confirmed transients from the ZTF Bright Transient Survey as ground truth.

---

## Key Results

| Classifier | Architecture | N | ECE (pre) | Post-hoc | ECE (post) | Primary finding |
|---|---|---|---|---|---|---|
| ALeRCE | Balanced RF (15→4 class) | 1114 | 0.271 [0.249, 0.296] | Temp. T=0.357 | 0.097 | Underconfidence; 21× operational gain at p>0.8 |
| Fink RF | RF binary | 1237 | — | N/A | — | Regime mismatch: 93.9% abstain, near-uniform across classes |
| Fink SNN | SuperNNova RNN | 793 (cond.) | 0.183 [0.154, 0.220] | Temp. T=3.65 | 0.051 | Selective classification; T>1 eliminates p≥0.8 selections |
| NEEDLE | CNN+DNN (obj-level) | 278 | 0.048 [0.048, 0.111] | Global T worsens | 0.169 | Aggregate masks severe per-class miscalibration |

### Key Findings

1. **ALeRCE underconfidence (ECE=0.271) is fully fixable.** T=0.357 sharpens compressed confidences [0.3, 0.75] → [0.35, 1.0], reducing ECE by 65% and increasing high-confidence (p>0.8) candidates from 20 to 418 at 91% precision — a **21× operational gain** confirmed by held-out 5-fold CV (pooled gain 20.9×, T stable at 0.357±0.012, no data leakage).

2. **Fink RF is an operational regime mismatch, not a miscalibration failure.** Designed for early-time alert classification (≤9 days), it abstains on 93.9% of well-evolved BTS objects with near-uniform abstention across all transient classes (90–99%). ECE is non-representative and excluded.

3. **Fink SNN implements implicit selective classification** with class-dependent abstention (χ², p=8.4×10⁻³⁷): non-Ia transients abstain at 2.6× the SN Ia rate (monotonic increase from SNIa 19% → SLSN 68%). On accepted predictions, T=3.65 reduces conditional ECE from 0.183 to 0.051 (72%). However, T>1 compresses scores to [0.191, 0.708], **eliminating all p≥0.8 predictions** — the calibration gain is operationally null at high thresholds. Pre-calibration scores (209 objects at p≥0.8, 73% precision) remain more useful for triage.

4. **NEEDLE's aggregate ECE (0.048) is at the well-calibrated boundary but masks severe per-class miscalibration** driven by inverse-frequency class weighting (~80:1 SN:TDE). SLSN-I: 73% mean confidence, 31% accuracy (gap=−0.423). Three true SLSN-I objects classified as SN at >0.998 confidence. Global T-scaling worsens ECE because SN (T<1 needed) and SLSN-I/TDE (T>1 needed) require opposite corrections.

5. **Calibration and operational utility diverge for overconfident classifiers.** For underconfident classifiers (ALeRCE), calibration unlocks high-confidence selection. For overconfident classifiers (Fink SNN), calibration eliminates it. Both ECE and threshold-curve diagnostics are required.

---

## Repository Structure

```
transient-calibration-audit/
├── src/                              # Reusable modules
│   ├── calibration.py                # ECE, temperature scaling, bootstrap CI
│   ├── plotting.py                   # Reliability diagrams, per-class plots
│   ├── data_acquisition.py           # ALeRCE/Fink API queries, BTS download
│   └── needle_extraction.py          # NEEDLE local inference pipeline
│
├── scripts/                          # Analysis scripts (run in order)
│   ├── 01_alerce_analysis.py         # ALeRCE full calibration audit
│   ├── 02_fink_analysis.py           # Fink RF + SNN conditional/unconditional
│   ├── 03_needle_analysis.py         # NEEDLE per-class analysis
│   ├── 04_summary.py                 # Comparative figures and LaTeX tables
│   ├── 05_renormalization_audit.py   # ALeRCE 15→4 class robustness check
│   ├── 06_needle_duplicates.py       # NEEDLE duplicate audit
│   ├── 07_reviewer_fixes.py          # Initial reviewer validation
│   ├── 08_needle_dedup.py            # Object-level deduplication (primary)
│   ├── 09_alerce_dual_task.py        # 15-class vs 4-class ECE validation
│   ├── 10_alerce_operational_gain_cv.py  # Held-out CV for 21× gain
│   ├── 11_fink_rf_calibration_methods.py # T-scaling vs Platt vs isotonic
│   ├── 12_bin_sensitivity_analysis.py    # ECE stability: 5/10/15/20 bins
│   ├── 13_fink_snn_abstention_verification.py # SNN zero-score characterisation
│   ├── 14_fink_abstention_analysis.py    # RF/SNN chi-sq abstention tests
│   └── 15_fink_snn_conditional_calibration.py # SNN conditional T=3.65
│
├── figures/                          # Figures + paper source
│   ├── main_updated.tex              # Paper source (AASTeX 6.3.1)
│   ├── main_updated.pdf              # Compiled paper (18 pages)
│   ├── references.bib                # Bibliography
│   └── fig_*.pdf / *.png             # Publication figures
│
├── results/                          # JSON + CSV outputs (canonical numbers)
│   ├── alerce_results.json
│   ├── alerce_dual_task_results.json
│   ├── alerce_operational_gain_cv_results.json
│   ├── bin_sensitivity_results.json
│   ├── fink_conditional_results.json
│   ├── fink_rf_calibration_comparison.json
│   ├── fink_snn_abstention_analysis.json
│   ├── fink_snn_conditional_analysis.json
│   ├── fink_snn_operational_threshold_table.csv
│   ├── fink_zero_abstention_summary.json
│   ├── needle_dedup_results.json
│   └── needle_results.json
│
├── data/
│   ├── raw/           # API outputs (not tracked — see Data section)
│   ├── ground_truth/  # BTS spectroscopic labels (not tracked)
│   └── processed/     # needle_predictions.npz (not tracked)
│
├── docs_alerce_acquisition.md
├── docs_fink_acquisition.md
└── requirements.txt
```

---

## Reproducing the Analysis

### 1. Setup

```bash
git clone https://github.com/pallavi-2000/transient-calibration-audit.git
cd transient-calibration-audit
pip install -r requirements.txt
```

### 2. Data Acquisition

```bash
# BTS ground truth + ALeRCE/Fink API queries (~10 min)
python3 src/data_acquisition.py

# NEEDLE local inference (requires models from Sheng et al. 2024)
python3 src/needle_extraction.py --needle-dir /path/to/NEEDLE
```

The NEEDLE pre-trained models (`lasair_th_r` family) are available from [Sheng et al. 2024](https://doi.org/10.1093/mnras/stae123). The resulting `data/processed/needle_predictions.npz` is required for scripts 08+.

### 3. Core Analysis

```bash
python3 scripts/01_alerce_analysis.py
python3 scripts/02_fink_analysis.py
python3 scripts/03_needle_analysis.py
python3 scripts/04_summary.py
python3 scripts/05_renormalization_audit.py
python3 scripts/06_needle_duplicates.py
python3 scripts/07_reviewer_fixes.py
```

### 4. Revision Validation

```bash
python3 scripts/08_needle_dedup.py            # Object-level dedup (primary)
python3 scripts/09_alerce_dual_task.py        # 4-class restriction check
python3 scripts/10_alerce_operational_gain_cv.py  # 21× gain held-out CV
python3 scripts/11_fink_rf_calibration_methods.py # Method comparison
python3 scripts/12_bin_sensitivity_analysis.py    # Bin-count sensitivity
python3 scripts/13_fink_snn_abstention_verification.py
python3 scripts/14_fink_abstention_analysis.py
python3 scripts/15_fink_snn_conditional_calibration.py
```

### 5. Compile the Paper

```bash
cd figures
pdflatex main_updated.tex && bibtex main_updated
pdflatex main_updated.tex && pdflatex main_updated.tex
# Output: figures/main_updated.pdf (18 pages)
```

---

## Methodology Notes

### ECE Computation
ECE uses equal-mass (quantile) binning with 15 bins (Roelofs et al. 2022).
**Important:** For multi-class classifiers (ALeRCE 4-class, NEEDLE 3-class), ECE must be computed via the 2D `(labels, proba_matrix)` calling convention in `src/calibration.py`. The 1D `(correct, max_confidence)` path gives incorrect results when max confidence < 0.5 — which occurs for 722/1114 ALeRCE objects — and must not be used for multi-class problems.

### NEEDLE Deduplication
Five pre-trained models each have a different held-out test set, producing 429 model-instance predictions across 278 unique objects. Primary analysis uses object-level deduplication (mean-of-model-probabilities, renormalised). The 43 multi-model objects show 100% inter-model class agreement, validating averaging. Object-level ECE (0.048) is primary; model-instance ECE (0.075) is a sensitivity analysis.

### Fink Selective Classification
Score=0 in the Fink API represents classifier abstention, not P(SN Ia)=0. The two classifiers implement different acceptance gates: RF is a data-quality gate that is near-uniform across classes (regime mismatch with BTS); SNN is a photometry-quality gate that is strongly class-dependent (rarer/fainter classes abstain more). Conditional calibration metrics are computed on non-zero predictions only.

---

## Data Availability

Raw data files are not tracked in git:

| File | How to obtain |
|---|---|
| `data/raw/alerce_classifications.csv` | `python3 src/data_acquisition.py` (ALeRCE API) |
| `data/raw/fink_classifications.csv` | `python3 src/data_acquisition.py` (Fink API) |
| `data/ground_truth/bts_sample.csv` | `python3 src/data_acquisition.py` (ZTF BTS) |
| `data/processed/needle_predictions.npz` | `python3 src/needle_extraction.py` (NEEDLE models) |

All `results/*.json` files are tracked and contain the canonical numbers used in the paper.

---

## References

| Reference | Role in paper |
|---|---|
| Guo et al. 2017 | Temperature scaling, ECE |
| Nixon et al. 2019 | Classwise ECE (SCE/ACE) |
| Roelofs et al. 2022 | Equal-mass ECE binning |
| Niculescu-Mizil & Caruana 2005 | RF underconfidence mechanism |
| van den Goorbergh et al. 2022 | Class-weighting harms calibration |
| Geifman & El-Yaniv 2017 | Selective classification framework |
| Förster et al. 2021 | ALeRCE broker |
| Möller et al. 2020 | Fink SuperNNova |
| Leoni et al. 2022 | Fink Random Forest |
| Sheng et al. 2024 | NEEDLE classifier |
| Fremling et al. 2020 | ZTF Bright Transient Survey |

---

## Author

**Pallavi Sati** — pallavisati23@gmail.com

## License

MIT
