# Calibration Audit of Production Astronomical Transient Classifiers

**The first systematic calibration study of ML classifiers deployed on ZTF transient alert streams.**

> When a broker says "80% probability of SN Ia," is it actually correct 80% of the time?

This repository contains the complete analysis pipeline for auditing the calibration of four production classifiers across three brokers: **ALeRCE**, **Fink**, and **NEEDLE**. All three are deployed in production on the Zwicky Transient Facility (ZTF) alert stream and routinely used to prioritize spectroscopic follow-up.

Paper in preparation for *The Astronomical Journal*.

---

## Key Results

| Classifier | Architecture | ECE (pre) | Post-hoc Method | ECE (post) | Finding |
|---|---|---|---|---|---|
| ALeRCE | Balanced Random Forest | ~0.27 | Temperature scaling (T≈0.36) | ~0.08 | Fixable underconfidence |
| Fink RF | Random Forest (binary) | ~0.41 | N/A (degenerate) | — | 94% zeros; structurally broken |
| Fink SNN | SuperNNova RNN (binary) | ~0.20 | N/A (bound hit) | — | T scaling inappropriate |
| NEEDLE | CNN+DNN hybrid | ~0.05 | Global T **worsens** | ~0.13 | Class-asymmetric miscalibration |

### Most Distinctive Findings

1. **ALeRCE's underconfidence inverts the Guo et al. (2017) overconfidence norm.** The Balanced Random Forest architecture pushes predictions away from 0 and 1, consistent with Niculescu-Mizil and Caruana (2005).

2. **NEEDLE's aggregate ECE masks class-asymmetric miscalibration.** SLSN-I is overconfident (conf ~95%, acc ~80%) while TDE is underconfident (conf ~86%, acc ~100%). Global temperature scaling cannot fix opposing miscalibration directions — this is driven by inverse-frequency class weighting (~80:1 ratio).

3. **Fink RF is not a probability estimator.** With 94% of scores at exactly 0.0, it functions as a detection filter, not a calibrated classifier.

---

## Repository Structure

```
transient-calibration-audit/
├── src/                          # Reusable Python modules
│   ├── data_acquisition.py       # BTS download, ALeRCE/Fink API queries
│   ├── calibration.py            # ECE, temperature scaling, auto-calibrate
│   ├── plotting.py               # Publication-quality reliability diagrams
│   └── needle_extraction.py      # NEEDLE model inference
├── scripts/                      # Analysis scripts (run in order)
│   ├── 01_alerce_analysis.py     # ALeRCE full calibration audit
│   ├── 02_fink_analysis.py       # Fink RF + SNN audit
│   ├── 03_needle_analysis.py     # NEEDLE per-class analysis
│   └── 04_summary.py             # Comparative figures and LaTeX table
├── data/
│   ├── raw/                      # API outputs (not tracked in git)
│   ├── ground_truth/             # BTS spectroscopic labels (not tracked)
│   └── processed/                # Cleaned datasets (not tracked)
├── figures/                      # Publication figures
├── results/                      # JSON results and LaTeX tables
└── requirements.txt
```

## Reproducing the Analysis

### 1. Setup

```bash
git clone https://github.com/pallavi-2000/transient-calibration-audit.git
cd transient-calibration-audit
pip install -r requirements.txt
```

### 2. Data Acquisition (~10 minutes)

```bash
python3 src/data_acquisition.py
python3 src/needle_extraction.py --needle-dir /path/to/NEEDLE
```

### 3. Run Analysis (in order)

```bash
python3 scripts/01_alerce_analysis.py
python3 scripts/02_fink_analysis.py
python3 scripts/03_needle_analysis.py
python3 scripts/04_summary.py
```

## Methodology

### Calibration Metrics
- **ECE**: Equal-mass binning, 15 bins (Roelofs et al. 2022)
- **Classwise ECE / ACE** (Nixon et al. 2019)
- **Brier Score**: Proper scoring rule
- **Bootstrap CIs**: 1000 replicates, 95% confidence

### Post-hoc Calibration
- **Temperature scaling** (Guo et al. 2017): NLL minimization, 5-fold CV
- **Per-class temperature scaling** (Frenkel 2021): For asymmetric miscalibration
- **Auto-calibrate**: Selects best strategy or identifies when none works

### Ground Truth
- **ZTF Bright Transient Survey** (Fremling et al. 2020)
- Stratified sampling for rare-class representation

## Key References

- Guo et al. 2017 — Temperature scaling, ECE
- Nixon et al. 2019 — SCE/ACE metrics
- Kull et al. 2019 — Dirichlet calibration
- Roelofs et al. 2022 — ECE binning recommendations
- Forster et al. 2021 — ALeRCE broker
- Moller et al. 2021 — Fink broker
- Sheng et al. 2024 — NEEDLE classifier
- Fremling et al. 2020 — ZTF Bright Transient Survey

## Author

Pallavi Kailas — Independent researcher

## License

MIT
