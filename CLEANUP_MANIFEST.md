# Cleanup Manifest — 2026-07-06

Every file moved or deleted during the `cleanup-v1` audit, with the reason.
Nothing was deleted from git history — `archive/` items remain fully
tracked and their prior history is visible via `git log --follow`.
Phase 0 baseline (all canonical checks pass, unaffected by this phase):
`results/canonical_numbers_v1.json` sha256
`a518c9e8ce112e5d29d8098d3247e39beffab921a27ac49e858e4ce71b206797`.

## Canonical manuscript decision

Confirmed with the author (2026-07-06): **`figures/main_updated.tex`
remains canonical** (AASTeX631, target *The Astronomical Journal*, matches
CITATION.cff/README/RELEASES.md). Four other manuscript files existed in
the working tree, all superseded or abandoned:

| File | Status | Reason |
|---|---|---|
| `figures/main.tex` | ARCHIVE | Earliest tracked draft (commit `b8824a6`); author field still reads "Pallavi Kailas," a pre-rename placeholder. Superseded by every later draft. |
| `figures/main2.tex` (+`.pdf`) | ARCHIVE | Second tracked draft (commit `0affa6d`), superseded once `main_updated.tex` became primary in the same commit. |
| `figures/main_final.tex` (+`.pdf`) | ARCHIVE | Untracked, dated 2026-05-07 (a week after the last commit). A genuine rewrite (tighter abstract, drops Fink RF from the classifier count) that was never integrated into README/RELEASES/CITATION.cff. Not confirmed as canonical by the author. |
| `figures/main_AJ_final.tex` (+`.pdf`) | ARCHIVE | Same date/status as `main_final.tex`; near-duplicate (674-line diff between the two). |
| `figures/main_MNRAS_final.tex` (+`.pdf`) | ARCHIVE | **Does not compile cleanly** — `main_MNRAS_final.log` shows real LaTeX errors (`! LaTeX Error: File `' not found`, undefined control sequences, misplaced alignment tabs), and every one of its 18 `\includegraphics` calls contains the literal, unexpanded string `\1` where a figure filename should be — the artifact of a botched sed/regex substitution that was never fixed. Not a usable draft in its current state. |
| `figures/main_updated_backup_pre_fink_rewrite.tex` | ARCHIVE | Explicitly named backup, referenced by `RELEASES.md` v0.1 note ("Preserved in `main_updated_backup_pre_*.tex` files"). |
| `figures/main_updated_backup_pre_session2.tex` | ARCHIVE | Same as above. |
| `figures/main_updated_backup_pre_session3.tex` | ARCHIVE | Same as above. |
| `figures/references2.bib` | ARCHIVE | Bibliography used only by `figures/main.tex` (archived above); `main_updated.tex` uses `references.bib` (kept). |

## Orphaned figures (not `\includegraphics`'d by `main_updated.tex`)

Cross-checked every `\plotone`/`\includegraphics` call in `main_updated.tex`
against `figures/*.pdf`. Figures below are exclusively referenced by the
archived tex drafts above (verified — each appears in `main.tex` or
`main2.tex`'s figure calls), so they move to `archive/figures/` alongside
the drafts that use them, preserving a reproducible unit:

- `diagnostic_random_forest_histogram.pdf`/`.png`, `diagnostic_supernnova_histogram.pdf`/`.png`,
  `fig_fink_random_forest_conditional.pdf`/`.png`, `fig_fink_random_forest_raw.pdf`/`.png`,
  `fig_fink_supernnova_conditional.pdf`/`.png`, `fig_fink_supernnova_raw.pdf`/`.png`,
  `fig8_comparison.pdf` — referenced only by `figures/main.tex`.
- `fig_fink_random_forest.pdf`/`.png`, `fig_fink_supernnova.pdf`/`.png`,
  `fig_needle_reliability.pdf`/`.png` — referenced only by `figures/main2.tex`.
- `fig_alerce_15class_reliability.pdf`, `fig_alerce_15class_vs_4class.pdf`/`.png`,
  `fig_alerce_4class_reliability.pdf`, `fink_snn_abstention_bias.pdf`/`.png`,
  `fink_zero_fraction_by_class_rf.pdf`/`.png`, `fink_zero_vs_nonzero_by_class.pdf`/`.png`
  — not referenced by any tex file (tracked or archived); superseded
  exploratory figures from earlier analysis sessions (per-class NEEDLE
  reliability and Fink RF class-breakdown work superseded by the
  conditional/dedup revision figures now in the paper).

All PNGs that ARE the sibling of a currently-cited PDF (e.g.
`fig_alerce_operational_gain_cv.png`, `fink_snn_class_composition_shift.png`,
`fink_zero_fraction_by_class_snn.png`) were **kept** in `figures/` — they
are convenience renders of load-bearing figures, not superseded content.

## Other archived files

| File | Reason |
|---|---|
| `VERIFICATION_GUIDE_FINK_SNN.txt` | Manual verification checklist used to confirm the Fink SNN zero-score abstention mechanism was real (not a data bug) before the paper made claims about it. Historically meaningful (shows the verification was done) but not cited by anything; its conclusion is now stated directly in `docs_fink_acquisition.md` and the paper. |

## Deleted (pure build ephemera only — regenerable, never tracked as science content)

`figures/*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out` for `main`, `main2`,
`main_final`, `main_AJ_final`, `main_MNRAS_final`, `main_updated`;
`figures/texput.log`; `texput.log` (repo root); `src/__pycache__/`;
all `.DS_Store` files. `.gitignore` updated to cover all of these going
forward (LaTeX byproducts + OS junk + `__pycache__`).

## Explicitly NOT touched (confirmed load-bearing)

- `docs_alerce_acquisition.md` / `docs_fink_acquisition.md` — initially
  suspected as superseded by `DATA_ACQUISITION.md`, but on inspection
  they contain unique methodology detail not recorded elsewhere (BTS→ALeRCE
  class-mapping table, stratified-sampling rationale, why TDE is excluded
  from ALeRCE analysis, Fink API query mechanics) that the paper's §3.1
  and §3.3 rely on. `DATA_ACQUISITION.md` is a snapshot/versioning
  manifest; these two are the narrative methodology record. Both are kept.
- All of `data/`, `results/*.json`, `results/*.csv`, `src/`, `scripts/01`–`18`,
  `figures/main_updated.tex`/`.pdf`, `references.bib`, tests, `README.md`,
  `RELEASES.md`, `DATA_ACQUISITION.md`, `CITATION.cff`, requirements files.

## Verification

Rerun after this phase:
```
python3 scripts/19_verify_canonical.py
```
Expected: all 25 checks pass, `results/canonical_numbers_v1.json` output
identical to the Phase 0 baseline (this phase touched no `results/` or
`src/` files).
