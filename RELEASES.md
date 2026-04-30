# Release History

Each tagged release corresponds to a milestone in addressing reviewer feedback.
The paper source at each tag is `figures/main_updated.tex`.

## v1.0-submitted

**Submission-ready release for The Astronomical Journal.**

Contents:
- Paper: `figures/main_updated.tex` (22 pages, AASTeX631 format)
- 18 analysis scripts (`scripts/01`–`18`) plus `src/` helpers
- Pinned environment: `requirements.txt` (full freeze in
  `requirements_frozen_full.txt`)
- Data acquisition manifest: `DATA_ACQUISITION.md`
- All result files: `results/*.json`, `results/*.csv`
- All figures: `figures/*.pdf`
- Citation metadata: `CITATION.cff`

## v0.4-session3 — Framing and methodological caveats

Reviewer Tier 2 fixes:
- Added §4.1 Terminology subsection (7 definitions)
- Surfaced pseudo-logit caveat into §4.2 (was buried in §6.5)
- Rewrote §6.4 LSST Implications as bounded transferability
  (extrapolations under domain shift, not direct forecasts)
- Softened "first systematic" novelty claims to "to our knowledge"
- Added transferability closing paragraph to §7 Conclusions

## v0.3-session2 — Statistical rigor

Reviewer Tier 1 fixes:
- Wilson + bootstrap 95% CIs added to NEEDLE per-class metrics (§5.3.2)
- §3.4 Missingness Analysis added (chi² tests, observable property
  comparisons; ALeRCE class-dependent p=4.1×10⁻⁴, Fink uniform p=0.061)
- §6.6 Sensitivity to Evaluation-Sample Composition added (three-prior
  comparison shows 21–33× gain robust across priors)
- Categorical NEEDLE claims softened at §5.3.5

## v0.2-session1.5 — Bug fixes and repo sync

- Fixed model-instance NEEDLE ECE throughout paper: 0.075 → 0.073
  (stale value from old 1D bootstrap_ece calling convention)
- Synced README and docs_fink_acquisition.md with selective-classification
  framing
- Fixed leoni2022 author: "Leoni, B." → "Leoni, M. and Ishida et al."

## v0.1 (implicit — pre-rewrite state)

- Pre-rewrite Fink framing (treated zeros as ordinary probabilities)
- Pre-rewrite NEEDLE framing (model-instance as primary)
- Preserved in `figures/main_updated_backup_pre_*.tex` files
