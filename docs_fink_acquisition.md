# Fink Data Acquisition — Documentation

## What We Did

Collected classification scores from Fink's two binary classifiers
(Random Forest and SuperNNova) for the same spectroscopically
confirmed ZTF transients used in the ALeRCE audit.

## Why

Fink takes a fundamentally different approach from ALeRCE.
Instead of multi-class probabilities ("which of 15 classes?"),
Fink provides binary scores ("is this a SN Ia? yes/no").
Auditing both architectures lets us compare how different
classifier designs affect calibration.

## How Fink Differs from ALeRCE

| | ALeRCE | Fink |
|---|---|---|
| Question asked | "Which of 15 classes?" | "Is this SN Ia?" |
| Output | Probability vector summing to 1.0 | Single score per classifier |
| Architecture | Balanced Random Forest (multi-class) | RF + SuperNNova (binary) |
| Classifiers | 1 production (lc_classifier) | 2 independent (RF, SNN) |
| Data per object | 1 row (final classification) | 1 row per alert (we take latest) |

## Step-by-Step Process

### Step 1: API Endpoint Discovery

The Fink API URL changed in January 2025:
- Old (dead): `fink-portal.org/api/v1/`
- New (working): `api.fink-portal.org/api/v1/`

This is documented in Fink's blog post (2025-01-30).
The API uses POST requests with `json=` parameters, not `data=`.

### Step 2: Understanding the Response

Fink returns one row per alert — every time ZTF detected the
object, Fink scored it again. For calibration, we use the
**latest alert** (most complete light curve = most informed
classification).

Example for ZTF25aagezfh (actual: SN Ia):
- 43 alerts total
- RF score across all alerts: mostly 0.0, briefly 0.358-0.678 during early alerts
- SNN score across all alerts: starts at 0.48, rises to ~0.91, latest is 0.69

### Step 3: SSL Certificate Issue

Fink's SSL certificate had expired at time of data collection.
Workaround: `verify=False` in requests.post(). This is safe for
our use case — we're downloading public classification scores,
not sending sensitive data.

### Step 4: Binary Classifier Scores

Two classifiers queried:

**rf_snia_vs_nonia (Random Forest)**
- Binary score: P(SN Ia) vs P(not SN Ia)
- Range: 0.0 to 1.0
- Finding: 94% of scores are exactly 0.0
- Mean score: 0.011
- Interpretation: Essentially non-functional for our sample.
  Returns zero for almost everything, including confirmed SN Ia.
  This is the classifier with ECE = 0.464.

**snn_snia_vs_nonia (SuperNNova)**
- Binary score: P(SN Ia) vs P(not SN Ia)
- Architecture: Recurrent Neural Network (Möller & de Boissière 2020)
- Range: 0.0 to 1.0
- Finding: 36% of scores are exactly 0.0
- Mean score: 0.361
- Interpretation: Has discriminative power but is underconfident.
  For a sample that is 43% SN Ia, a well-calibrated classifier
  should average closer to 0.43. ECE = 0.304.

### Step 5: API Query

- Endpoint: api.fink-portal.org/api/v1/objects
- Method: POST with JSON body
- Columns requested: i:objectId, d:rf_snia_vs_nonia, d:snn_snia_vs_nonia, i:jd
- Rate limit: 0.15 seconds between calls
- Checkpointing: every 50 objects (resumable)

Results:
- Successful queries: 1,237 / 1,436 (86%)
- Failed queries: 199
- Saved to: data/raw/fink_classifications.csv

## Key Observation: The RF Zero Problem

The RF classifier returning 0.0 for 94% of objects is not a data
error. It is a real property of the classifier. The RF was
originally designed for the early ZTF alert stream and appears
to require specific alert properties that many objects in our
sample do not trigger. This is itself a calibration failure —
a score of 0.0 for a confirmed SN Ia is maximally wrong.

When we compute ECE for the RF:
- Bin 0.0-0.1 has ~1161 objects, 94% of the sample
- Of these, ~450+ are actual SN Ia (ground truth)
- The classifier says "~0% chance of SN Ia" for objects that ARE SN Ia
- This single bin drives most of the ECE = 0.464

## Key Observation: Binary vs Multi-class Calibration

Calibrating binary scores is conceptually simpler than
multi-class. For Fink, calibration asks: "When the SNN says
0.7, is the object actually SN Ia 70% of the time?"

But there is a subtlety: binary calibration treats "not SN Ia"
as a single class, lumping SN II, SN Ibc, SLSN, TDE, and
everything else together. A score of 0.3 for a SN Ibc and
0.3 for a SLSN mean the same thing to the binary classifier —
both are "not SN Ia." This limits the operational utility
compared to ALeRCE's multi-class approach.

## Key Decision: Why We Take the Latest Alert

Fink provides scores for every alert (detection) of an object.
Early alerts have less light curve data, so classifications are
less informed. By taking the latest alert, we evaluate the
classifier at its best — when it has the most data to work with.

If the classifier is miscalibrated even with complete light
curves, that is a stronger finding than measuring calibration
on early, noisy predictions.

Alternative approach (not taken): evaluate calibration as a
function of light curve completeness. This is interesting but
out of scope — it would show how calibration evolves over time,
which is a separate paper.

## Files Produced

| File | Location | Contents |
|------|----------|----------|
| fink_classifications.csv | data/raw/ | 1,237 objects with RF and SNN scores |

## Objects Per True Class

| Class | Objects |
|-------|---------|
| SNIa  | 527     |
| SNII  | 345     |
| SNIbc | 245     |
| SLSN  | 83      |
| TDE   | 37      |
| Total | 1,237   |

Note: Unlike ALeRCE, Fink's binary classifiers CAN evaluate TDE
objects. The question "Is this SN Ia?" has a valid answer for
TDEs (no, it is not). So TDEs are included in Fink calibration.

## What Comes Next

1. Build calibration module (ECE, reliability diagrams)
2. Apply to ALeRCE multi-class predictions
3. Apply to Fink binary scores
4. Temperature scaling and post-hoc calibration

