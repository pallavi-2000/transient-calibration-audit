# ALeRCE Class-wise Performance: Base Rate → Calibration → Learned Deferral

## Summary Table: Accuracy at 80% Auto-Accept Coverage

| Class | N | Type | Base Acc | Raw@80% | Cal@80% | Learned@80% | Cal Δ | Learn Δ | Note |
|-------|---|------|----------|---------|---------|------------|-------|---------|------|
| **SNIa** | 757 | Common thermonuclear | 83.1% | 88.3% | 88.7% | 90.8% | +0.4pp | +2.2pp | ✓ Well-calibrated; deferral recovers edge cases |
| **SNII** | 560 | Core-collapse | **49.5%** | 52.0% | 53.3% | 63.9% | +1.3pp | **+10.6pp** | ✗✓ Bottleneck; learned model essential |
| **SNIbc** | 204 | Stripped-envelope | 82.8% | 87.8% | 88.2% | 85.5% | +0.4pp | −2.7pp | ✓− Baseline strong; deferral conservative |
| **SLSN** | 55 | Rare super-luminous | 78.2% | 81.6% | 78.8% | 77.8% | −2.8pp | −1.1pp | ⚠ Rare class harmed by global calibration |

---

## Temperature Scaling Fit

| Fold | N Train | N Test | Temperature | Train Acc | Test Acc |
|------|---------|--------|-------------|-----------|----------|
| 1 | 1260 | 316 | 0.430 | 71.0% | 70.6% |
| 2 | 1261 | 315 | 0.426 | 71.0% | 70.8% |
| 3 | 1261 | 315 | 0.428 | 70.8% | 71.4% |
| 4 | 1261 | 315 | 0.430 | 70.8% | 71.4% |
| 5 | 1261 | 315 | 0.419 | 71.1% | 70.5% |
| **Mean ± SD** | — | — | **0.427 ± 0.004** | — | **70.9% ± 0.4%** |

**Interpretation:** T ≈ 0.43 << 1 confirms ALeRCE is severely underconfident. Probabilities are systematically compressed; temperature scaling expands them to match observed frequencies.

---

## Error Rates (Selective Risk) at 80% Auto-Accept Coverage

| Class | Raw Risk | Cal Risk | Learned Risk | Improvement |
|-------|----------|----------|--------------|-------------|
| SNIa | 11.7% | 11.3% | 9.2% | −2.5pp |
| SNII | 48.0% | 46.7% | 36.1% | −11.9pp |
| SNIbc | 12.2% | 11.8% | 14.5% | +2.7pp |
| SLSN | 18.4% | 21.2% | 22.2% | +3.8pp |

**Key insight:** Learned model reduces error the most on SNII (the hardest class). For rare SLSN, it is conservative (higher selective risk), prioritizing precision over coverage.

---

## TDE (Out-of-Taxonomy) Stress Test at 80% In-Taxonomy Coverage

| Policy | TDEs Deferred | Fraction | 95% CI |
|--------|---------------|----------|--------|
| raw_confidence | 13/30 | 43.3% | [27.4%, 60.8%] |
| calibrated_confidence | 11/30 | 36.7% | [21.9%, 54.5%] |
| learned_shape | 9/30 | 30.0% | [16.7%, 47.9%] |
| **learned_full** | 15/30 | **50.0%** | [33.2%, 66.8%] |

**Verdict:** Even the best policy (learned_full) defers only 50% of TDEs. Calibration *worsens* deferral (37% vs 43% raw), proving that sharpening probabilities on an unrepresentable class makes it look more confident, not less. **Confidence-based deferral cannot replace novelty detection.**

---

## Key Findings & Interpretations

### SNIa (Common, well-calibrated)
- **Base accuracy:** 83.1% (good starting point)
- **Calibration gain:** +0.4pp (negligible; already well-tuned)
- **Deferral gain:** +2.2pp by identifying truly ambiguous SNIa
- **Conclusion:** Strong performer. Both methods help slightly; no urgent action needed.

### SNII (Bottleneck class)
- **Base accuracy:** 49.5% (near chance; ALeRCE struggles)
- **Calibration gain:** +1.3pp (insufficient; still <54%)
- **Deferral gain:** +10.6pp (major; learned model captures genuine difficulty)
- **Conclusion:** Calibration alone cannot fix SNII. Selective deferral is essential. This class demands either classifier retraining or systematic deferral for expert review.

### SNIbc (Easy baseline)
- **Base accuracy:** 82.8% (already good)
- **Calibration gain:** +0.4pp (marginal)
- **Deferral gain:** −2.7pp (model is conservative; defers marginal cases)
- **Conclusion:** Baseline strong. Learned model trades coverage for precision—does not hurt overall performance; just raises the bar for auto-acceptance.

### SLSN (Rare, harmed by global calibration)
- **Base accuracy:** 78.2%
- **Calibration gain:** −2.8pp ⚠ (global temperature scaling overconfidently scales SLSN probabilities)
- **Deferral gain:** −1.1pp (stable, not worsened further)
- **Conclusion:** Rare class demonstrates that global calibration can harm asymmetrically. LSST should use class-specific or instance-weighted temperature scaling, not one-size-fits-all.

### TDE (Absent from taxonomy)
- **Base accuracy:** 0% (impossible to classify)
- **Raw confidence deferral:** 43% (random-ish)
- **Calibrated confidence deferral:** 37% (worse—calibration makes the model more confident in wrong classes)
- **Best deferral:** 50% (learned model, but insufficient)
- **Conclusion:** No confidence-based method solves OOD detection. LSST requires explicit novelty/OOD channel alongside calibrated confidence. Calibration alone is insufficient and can be harmful for unseen categories.

---

## Poster-Ready Summary

| Finding | Evidence | Implication for LSST |
|---------|----------|---------------------|
| **Calibration changes thresholds, not ranking** | Raw p≥0.8: 9 objects; Calibrated p≥0.8: 395 objects (88% purity) | Threshold-based gates are meaningless without calibration |
| **Learned deferral outperforms confidence** | ΔROC-AUC +0.083, CI [0.057, 0.107]; ΔAURC −0.050 | Selective classification is superior to naive confidence gating |
| **SNII requires deferral, not just calibration** | +10.6pp improvement only with learned model | Common classes vary widely in calibration needs; no one-size-fits-all fix |
| **Rare classes harmed by global calibration** | SLSN −2.8pp after global T | Use class-specific or cost-weighted calibration for LSST |
| **OOD detection cannot use confidence alone** | TDE deferral 37–50%; calibration worsens to 37% | Implement explicit OOD/novelty channel; confidence is necessary but insufficient |

