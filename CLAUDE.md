# CLAUDE.md — Rules for working in this repository

This repository backs a submitted journal paper (*Calibration Audit of
Production Astronomical Transient Classifiers on ZTF Alert Streams*, Sati
2026, target: The Astronomical Journal) and a PhD application. Domain
experts may inspect every file, number, and sentence. **Authenticity
outranks tidiness.** When cleanliness and traceability conflict,
traceability wins.

## Terminology (never blur this)

"Calibration" in this repo means **probability calibration of ML
classifiers** — ECE, reliability diagrams, temperature scaling. It is
**never** Rubin Observatory instrumental/photometric calibration
(FGCM/jointcal). No file, comment, or docstring may use "calibration" in
the instrumental sense.

## Absolute rules

1. **Never edit an expected/reference value to make a check pass.** If
   `scripts/19_verify_canonical.py` fails, the code or the number that
   produced it is wrong — fix the root cause, or report and stop. Do not
   adjust the check's tolerance or reference value to paper over a
   mismatch.
2. **Never change a numerical convention** (equal-mass ECE binning with
   M=15 bins; stratified 5-fold CV with `random_state=42`; bootstrap CIs
   with `n_bootstrap=1000`, `random_state=42`; NEEDLE dedup = mean of
   available model probabilities, then renormalize) without the author's
   explicit sign-off, because every cited number in the paper depends on
   these being held fixed.
3. **Never re-query broker APIs or touch `data/`.** The data snapshot
   (2026-04-01) is frozen; re-querying ALeRCE/Fink would return different
   scores as classifier versions evolve (see `DATA_ACQUISITION.md`).
   `data/raw/`, `data/processed/`, `data/ground_truth/` are load-bearing
   and are never regenerated, edited, or deleted.
4. **No fabricated content.** Every number in documentation or the paper
   must be traceable to a file in `results/`. If a number can't be traced
   or verified against its cited source, mark it `[FILL]` and say so
   explicitly rather than guessing or inventing a plausible-looking value.
5. **The canonical manuscript is `figures/main_updated.tex`** (AASTeX631,
   target AJ). `figures/main_final.tex`, `main_AJ_final.tex`,
   `main_MNRAS_final.tex`, and `main2.tex` are dated exploratory drafts —
   confirmed with the author 2026-07-06 — and live in `archive/` with
   full history preserved in git. Do not promote one of them to canonical
   without the author's explicit say-so.

## The test for every file

Does this help a referee or PhD-admissions reader trace a paper claim to
code and data? If yes: keep it working-tree-visible. If it's superseded
but historically meaningful: `archive/`, not deleted (git history keeps
everything, but `archive/` keeps the working tree honest without erasing
context). Only pure build artifacts (`__pycache__/`, `.DS_Store`,
`*.aux`/`*.log`/`*.out`/`*.bbl`/`*.blg` LaTeX byproducts) are deleted.

## Known defect list (found in audit, 2026-07-06)

Real issues found in this repo, in the order they should be fixed:

1. **`results/summary.txt` is stale.** It predates the Fink RF reframing
   and the NEEDLE object-level dedup revision: it calls Fink RF
   "structurally broken" (current framing: operational regime mismatch —
   see README/paper §3.3), and reports NEEDLE aggregate ECE=0.073 /
   T=1.552 worsening to 0.169 / "~123x" class weight (current, correct,
   object-level numbers: ECE=0.048, T=1.552 worsening 0.126→0.169 is the
   *per-class-scaling* comparison, not aggregate — see
   `results/canonical_numbers_v1.json`).
2. **`~80:1` vs `~123x` NEEDLE class-weight ratio.** The paper
   (`figures/main_updated.tex`) says "~80:1 for common supernovae versus
   TDEs," citing Sheng et al. 2024. Verified 2026-07-06 against the
   published paper (arXiv:2312.04968, Table 1): r-band training pool is
   SN=5237, TDE=64 → ratio ≈ 81.8:1. This matches the paper's ~80:1
   claim. `scripts/03_needle_analysis.py` (line ~220) and
   `results/summary.txt` instead say "~123x" — this number does not
   correspond to any ratio recoverable from the cited source and should
   be corrected to ~80:1 with the Table 1 counts cited inline.
3. **Two undocumented tex drafts risk being mistaken for canonical.**
   `main_final.tex`/`main_AJ_final.tex`/`main_MNRAS_final.tex` are newer
   (2026-05-07) than the last commit (2026-04-30) and were never
   integrated into README/RELEASES/CITATION.cff. Confirmed with author:
   `main_updated.tex` remains canonical; the others move to `archive/`.
4. **No `tests/` directory.** `test_full.py` and `test_stratified_cv.py`
   sit at repo root as standalone print-based scripts, not pytest, with
   no assertions against a frozen baseline.
5. **README undercounts the pipeline.** Repository Structure and
   Reproducing sections list only scripts 01–15; scripts 16
   (`needle_per_class_bootstrap`), 17 (`alerce_prior_reweighting`), and 18
   (`missingness_analysis`) exist and are undocumented.
6. **Duplicated aggregation logic.** `scripts/08_needle_dedup.py`
   (`deduplicate_by_object`) and `scripts/16_needle_per_class_bootstrap.py`
   (`deduplicate_to_object_level`) reimplement the same
   mean-then-renormalize object-level aggregation independently.
7. **`.gitignore` doesn't cover LaTeX build byproducts** (`*.aux`,
   `*.bbl`, `*.blg`, `*.log`, `*.out`, `texput.log`), so every
   `pdflatex`/`bibtex` run litters `git status` with untracked noise.
8. **No verification gate existed before this audit.**
   `scripts/19_verify_canonical.py` and `results/canonical_numbers_v1.json`
   were created 2026-07-06 to give every subsequent change a fast,
   objective PASS/FAIL check against the numbers the paper actually cites.

## Reproduction

See `README.md` §Reproducing the Analysis. Gate check before trusting any
change: `python3 scripts/19_verify_canonical.py` must print all PASS.
