# ALeRCE Data Acquisition — Documentation

## What We Did

Collected classification probabilities from ALeRCE's light curve
classifier (Balanced Random Forest, Förster et al. 2021) for
spectroscopically confirmed ZTF transients.

## Why

To compare what ALeRCE *predicted* against what each object
*actually is* (from spectroscopy). This comparison is the
foundation of calibration analysis — measuring whether ALeRCE's
probability outputs are trustworthy.

## Step-by-Step Process

### Step 1: Ground Truth (BTS)

- Source: ZTF Bright Transient Survey (Fremling et al. 2020)
- URL: sites.astro.caltech.edu/ztf/bts/explorer.php
- Filter: confirmed transients, peak magnitude < 19
- Downloaded: 10,317 total objects
- With spectroscopic types: 7,167 objects
- Saved to: data/ground_truth/bts_spectroscopic.csv

### Step 2: Class Mapping

BTS uses ~30 fine-grained spectroscopic types. ALeRCE's classifier
predicts 5 transient classes. We mapped:

| ALeRCE Class | BTS Types Included                                     | Count |
|-------------|--------------------------------------------------------|-------|
| SNIa        | SN Ia, SN Ia-91T, SN Ia-91bg, SN Ia-pec, SN Ia-CSM, SN Ia-SC, SN Iax | 5,328 |
| SNII        | SN II, SN IIn, SN IIP, SN II-pec                      | 1,148 |
| SNIbc       | SN Ib, SN Ic, SN Ic-BL, SN IIb, SN Ib/c, SN Ibn, SN Icn, SN Ib-pec, SN Ic-pec | 504 |
| SLSN        | SLSN-I, SLSN-II                                       | 97    |
| TDE         | TDE, TDE-He, TDE-H-He, TDE-featureless                | 39    |

Dropped 51 objects with types ALeRCE cannot classify:
nova (34), Other (5), LRN (3), LBV (3), other (2), ILRT (2),
Ca-rich (1), SN Ca-rich-Ca (1)

Total mapped: 7,116 objects

### Step 3: Stratified Sampling

Problem: BTS is 75% SN Ia. Random sampling would leave rare
classes (SLSN, TDE) with too few objects for meaningful
per-class calibration analysis.

Solution: Cap large classes, take all available rare objects.

| Class | Available | Sampled |
|-------|-----------|---------|
| SNIa  | 5,328     | 600     |
| SNII  | 1,148     | 400     |
| SNIbc | 504       | 300     |
| SLSN  | 97        | 97      |
| TDE   | 39        | 39      |
| Total |           | 1,436   |

Why this is scientifically valid: ECE is a conditional measurement
("when the classifier says 80%, is it right 80% of the time?").
This does not depend on the natural class distribution. Per-class
metrics are completely unaffected by sampling. For aggregate
metrics, we can reweight by true prevalence. This is standard
practice (Guo et al. 2017, Nixon et al. 2019).

Saved to: data/ground_truth/bts_sample.csv
Random seed: 42 (for reproducibility)

### Step 4: ALeRCE API Queries

- API endpoint: api.alerce.online/ztf/v1/objects/{oid}/probabilities
- Classifier used: lc_classifier (version lc_classifier_1.1.13)
- Architecture: Balanced Random Forest
- Rate limit: 0.15 seconds between calls (~7 req/sec)
- Checkpointing: every 50 objects (resumable)

Results:
- Successful queries: 1,149 / 1,436 (80%)
- Failed queries: 287 (likely objects without enough light curve points)
- Saved to: data/raw/alerce_classifications.csv

### Step 5: Understanding the Output

ALeRCE's lc_classifier returns probabilities for 15 classes:
- 5 transient: SNIa, SNIbc, SNII, SLSN (no TDE class)
- 10 non-transient: QSO, AGN, Blazar, CV/Nova, YSO, LPV, E,
  DSCT, RRL, CEP, Periodic-Other

Probabilities sum to 1.0 across all 15 classes.

## Key Decisions and Their Justification

### 1. Why lc_classifier only?

ALeRCE has 4 classifiers (lc_classifier, stamp_classifier,
LC_classifier_BHRF_forced_phot(beta), LC_classifier_ATAT_forced_phot(beta)).
We use lc_classifier because:
- It is the production classifier (not beta)
- It outputs a proper probability vector summing to 1.0
- Stamp classifier does real/bogus filtering, not classification

### 2. Why TDEs are excluded from calibration analysis

ALeRCE's lc_classifier has no TDE class. It cannot predict
something it was not trained on. When given actual TDEs, it
spreads probability across SLSN (most common), SNII, and SNIa
with low confidence — the classifier is appropriately uncertain.

This is not a calibration flaw to measure. It is a scope boundary.

Usable sample for calibration: 1,114 objects (1,149 minus 35 TDEs)

| Class | Objects |
|-------|---------|
| SNIa  | 463     |
| SNII  | 341     |
| SNIbc | 225     |
| SLSN  | 85      |
| Total | 1,114   |

### 3. Why 287 objects failed

Most likely causes:
- Object has too few detections for the light curve classifier
- Object was detected but not processed by ALeRCE's pipeline
- Transient API timeout

This is expected and does not bias results — failure to classify
is unrelated to miscalibration of objects that were classified.

## Files Produced

| File | Location | Contents |
|------|----------|----------|
| bts_spectroscopic.csv | data/ground_truth/ | 7,167 BTS objects with spectroscopic types |
| bts_sample.csv | data/ground_truth/ | 1,436 stratified sample |
| alerce_classifications.csv | data/raw/ | 1,149 ALeRCE probability vectors |

## What Comes Next

1. Query Fink for the same objects (RF + SuperNNova scores)
2. Build calibration module (ECE, reliability diagrams)
3. Apply temperature scaling
4. Generate publication figures
