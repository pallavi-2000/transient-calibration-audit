# Fink Data Acquisition — Documentation

## What We Did

Collected classification scores from Fink's two binary classifiers
(Random Forest and SuperNNova) for spectroscopically confirmed ZTF
transients from the BTS sample.

## Understanding Fink's Output: Selective Classification

Both Fink classifiers return score = 0 for objects that fail eligibility
criteria. **Score = 0 does not mean P(SN Ia) = 0.** It means the
classifier abstained — the object did not pass the acceptance gate.

This distinction matters for calibration. Treating zeros as probabilities
produces meaningless ECE values and conflates two different regimes:
objects where the classifier ran (score > 0) and objects where it didn't.

We evaluate three properties separately:
1. **Coverage**: what fraction of objects receive a non-zero score?
2. **Conditional calibration**: are the emitted scores reliable?
3. **Abstention bias**: does the abstention pattern bias the calibration estimate?

## The Two Classifiers

### rf_snia_vs_nonia (Random Forest, Leoni et al. 2022)

Designed as an **early-time SN Ia detector** for the ZTF alert stream,
typically activated within ~9 days of first detection. Requires at least
3 observed epochs per filter, fewer than 20 prior detections, and a
host-galaxy cross-match.

**In our BTS sample:** 93.9% abstention (1,161 / 1,237 objects).
Abstention is near-uniform across classes:
- SNIa: 93.0%, SNIbc: 90.2%, SNII: 96.2%, SLSN: 98.8%, TDE: 97.3%

**Interpretation:** Regime mismatch. BTS contains well-evolved,
spectroscopically confirmed transients — the opposite of the early
alert stream for which RF was designed. The high abstention rate is
expected and does not represent a calibration failure. ECE on the 76
non-zero predictions is not population-representative and is excluded
from the paper's calibration analysis.

### snn_snia_vs_nonia (SuperNNova, Möller et al. 2020)

Recurrent neural network classifier for SN Ia vs non-SN Ia. Uses a
photometric data-quality gate that produces score = 0 for objects with
insufficient light-curve data.

**In our BTS sample:** 35.9% abstention (444 / 1,237 objects).
Abstention is strongly class-dependent (χ², p = 8.4×10⁻³⁷):
- SNIa: 19.0%, SNIbc: 29.8%, SNII: 55.7%, TDE: 62.2%, SLSN: 67.5%

Non-Ia transients abstain at 2.6× the SN Ia rate, consistent with
sparser light curves for fainter, rarer classes. The first non-zero
score is 0.0051 (hard gap diagnostic of an explicit threshold gate).

**On the 793 accepted objects:**
- Mean score: 0.564, median: 0.664
- Conditional ECE: 0.183 [0.154, 0.220]
- Optimal temperature: T = 3.65 (no bound hit)
- Post-T conditional ECE: 0.051 (72% reduction)
- Post-T score range: [0.191, 0.708] — all p≥0.8 predictions eliminated
- Pre-T at p≥0.8: 209 objects at 73.2% precision

**Abstention bias:** The accepted set is enriched in SNIa (+11.2%) and
depleted in SNII (−8.6%) relative to BTS. BTS-reweighted calibration gap
is 0.460 vs conditional gap 0.436 — abstention bias of +0.024.

## Data Acquisition Details

### API

- Endpoint: `api.fink-portal.org/api/v1/objects`
- Method: POST with JSON body
- Columns: `i:objectId`, `d:rf_snia_vs_nonia`, `d:snn_snia_vs_nonia`, `i:jd`
- Rate limit: 0.15 s between calls; checkpointing every 50 objects

Note: API URL changed in January 2025 from `fink-portal.org/api/v1/`.
If SSL certificate errors occur, use `verify=False` (public data only).

### Per-alert vs per-object

Fink returns one row per alert. We take the **latest alert** per object
(most complete light curve = most informed classification). This evaluates
the classifier at its best — if it is miscalibrated with complete data,
that is a stronger finding than measuring calibration on early partial data.

### Sample composition

| Class | Objects queried |
|-------|----------------|
| SNIa  | 527            |
| SNII  | 345            |
| SNIbc | 245            |
| SLSN  | 83             |
| TDE   | 37             |
| **Total** | **1,237** |

Unlike ALeRCE, Fink binary classifiers evaluate TDE objects (the question
"Is this SN Ia?" is well-defined for TDEs — the answer is no).

## Output File

| File | Contents |
|---|---|
| `data/raw/fink_classifications.csv` | 1,237 objects × {oid, rf_snia_vs_nonia, snn_snia_vs_nonia} |
