# Data Acquisition Manifest

This file documents the exact API queries and snapshot timestamps for the
broker prediction data used in this audit. Reproduction of the exact
results requires either re-querying the same API endpoints (which may
return different scores if classifier versions evolve) or using the
archived raw outputs in `data/raw/`.

## ZTF Bright Transient Survey ground truth

- **Source:** https://sites.astro.caltech.edu/ztf/bts/explorer.php
- **Snapshot date:** 2026-04-01
- **Sample size at snapshot:** 7,167 spectroscopically classified objects
- **Class mapping:** subtype → ALeRCE 4-class taxonomy (see §3.1 of paper
  and `src/data_acquisition.py`)
- **Stratified sample:** 1,436 objects
  (600 SN Ia / 400 SN II / 300 SN Ibc / 97 SLSN / 39 TDE)
- **Local file:** `data/ground_truth/bts_sample.csv`
  (modification date: 2026-04-01 13:16)

## ALeRCE classifications

- **Endpoint:** `https://api.alerce.online/ztf/v1/objects/{oid}`
- **Classifier audited:** ALeRCE light curve classifier (`lc_classifier`)
- **Classifier version at snapshot:** v1.1.13 (Sánchez-Sáez et al. 2021)
- **Output format:** 15-class probability vector
- **Snapshot date:** 2026-04-01
- **Retrieval rate:** 1,149 of 1,436 (80.0%; missingness analyzed in §3.4)
- **Restricted analysis sample:** 1,114 objects after dropping TDE
  (no TDE class in v1.1.13 taxonomy)
- **Local file:** `data/raw/alerce_classifications.csv`
  (modification date: 2026-04-01 14:08)
- **Acquisition script:** `src/data_acquisition.py`

## Fink classifications

- **Endpoint:** `https://api.fink-portal.org`
- **Classifiers audited:**
  - `rf_snia_vs_nonia` (Random Forest, Leoni et al. 2022)
  - `snn_snia_vs_nonia` (SuperNNova, Möller & de Boissière 2020,
    deployed in non-Bayesian form)
- **Output format:** binary score [0, 1] per classifier
- **Query mode:** latest alert per object (acknowledged caveat: see §3.3
  of paper; latest-alert is non-neutral for the Fink RF early-classifier
  window)
- **SSL note:** acquisition code uses `verify=False` due to intermittent
  certificate issues at Fink portal; documented in
  `docs_fink_acquisition.md`
- **Snapshot date:** 2026-04-01
- **Retrieval rate:** 1,237 of 1,436 (86.1%; missingness in §3.4 shows
  no significant class dependence, p = 0.061)
- **Local file:** `data/raw/fink_classifications.csv`
  (modification date: 2026-04-01 16:13)
- **Acquisition script:** `src/data_acquisition.py`

## NEEDLE predictions

- **Source:** Local inference using publicly available pre-trained models
- **Model family:** `lasair_th_r` (Sheng et al. 2024)
- **Number of model instances:** 5 (each with different held-out test
  set per `testset_obj.json`)
- **Output format:** 3-class probability vector (SN, SLSN-I, TDE)
- **Inference script:** `src/needle_extraction.py`
- **Snapshot date:** 2026-04-01
- **Predictions:** 429 model-instance predictions across 278 unique
  ZTF objects (43 multi-model overlaps; 100% inter-model class agreement)
- **Local file:** `data/processed/needle_predictions.npz`
  (modification date: 2026-04-01 16:36)

## Reproducibility caveats

- Broker classifier versions may have evolved since the snapshot dates
  above; re-querying may yield different scores.
- The pseudo-logit transformation (`z_k = log p_k`) applied to RF
  classifiers is heuristic; see §4.2 of paper.
- Stratified sampling is reproducible from the BTS public catalogue +
  `src/data_acquisition.py` (random seed declared in script).
- All bootstrap analyses use `random_state=42`
  (declared in `scripts/16_needle_per_class_bootstrap.py`).
- ALeRCE TDE support was added in a beta classifier (Pavez-Herrera et al.
  2025) after our snapshot date; our results apply to v1.1.13 only.
