"""
src/build_dataset.py

Takes raw ALeRCE output and produces a clean, analysis-ready dataset.
Run once after collect_alerce.py finishes:

    python src/build_dataset.py

Saves to:
    data/processed/alerce_dataset.csv   — clean, mapped, ready for analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CLASS MAPPING
# Maps detailed BTS spectroscopic types → ALeRCE's 5 broad classes
# ─────────────────────────────────────────────────────────────────────────────

SPEC_TO_ALERCE = {
    # SNIa family
    'SN Ia':        'SNIa',
    'SN Ia-91T':    'SNIa',
    'SN Ia-91bg':   'SNIa',
    'SN Ia-pec':    'SNIa',
    'SN Iax':       'SNIa',

    # SNII family
    'SN II':        'SNII',
    'SN IIn':       'SNII',
    'SN IIb':       'SNII',
    'SN IIP':       'SNII',
    'SN IIL':       'SNII',

    # SNIbc family
    'SN Ib':        'SNIbc',
    'SN Ic':        'SNIbc',
    'SN Ib/c':      'SNIbc',
    'SN Ic-BL':     'SNIbc',
    'SN Ibn':       'SNIbc',

    # Rare classes — keep as-is
    'SLSN-I':       'SLSN',
    'SLSN-II':      'SLSN',
    'TDE':          'TDE',
}

# ALeRCE's 15 probability output columns
ALERCE_PROB_COLS = [
    'SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO',
    'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV',
    'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other'
]

# The 5 classes we have ground truth for
EVAL_CLASSES = ['SNIa', 'SNII', 'SNIbc', 'SLSN', 'TDE']


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset():
    print("=" * 60)
    print("BUILDING ANALYSIS DATASET")
    print("=" * 60)

    # ── Load raw predictions ─────────────────────────────────────────────────
    raw_path = Path("data/raw/alerce_predictions.csv")
    assert raw_path.exists(), f"Raw data not found at {raw_path}. Run collect_alerce.py first."

    df = pd.read_csv(raw_path)
    print(f"\nLoaded {len(df):,} raw predictions")

    # ── Map spectroscopic types to ALeRCE classes ────────────────────────────
    df['alerce_class'] = df['spectroscopic_type'].map(SPEC_TO_ALERCE)

    # Report any unmapped types
    unmapped = df[df['alerce_class'].isna()]
    if len(unmapped) > 0:
        print(f"\nUnmapped types (will be dropped):")
        print(unmapped['spectroscopic_type'].value_counts().to_string())

    df = df.dropna(subset=['alerce_class'])
    print(f"After mapping: {len(df):,} objects")

    # ── Check probability columns exist ─────────────────────────────────────
    available_prob_cols = [c for c in ALERCE_PROB_COLS if c in df.columns]
    missing_prob_cols   = [c for c in ALERCE_PROB_COLS if c not in df.columns]

    if missing_prob_cols:
        print(f"\nMissing probability columns (will be set to 0): {missing_prob_cols}")
        for col in missing_prob_cols:
            df[col] = 0.0

    # ── Verify probabilities sum to 1 ────────────────────────────────────────
    prob_sums = df[ALERCE_PROB_COLS].sum(axis=1)
    bad_rows  = (prob_sums < 0.99) | (prob_sums > 1.01)

    if bad_rows.sum() > 0:
        print(f"\nDropping {bad_rows.sum()} rows where probabilities don't sum to 1")
        df = df[~bad_rows]

    # ── Create integer class index for analysis ──────────────────────────────
    class_to_idx = {cls: i for i, cls in enumerate(EVAL_CLASSES)}
    df['class_idx'] = df['alerce_class'].map(class_to_idx)

    # ── Add top-1 prediction and confidence ─────────────────────────────────
    prob_matrix = df[ALERCE_PROB_COLS].values
    df['predicted_class'] = [ALERCE_PROB_COLS[i] for i in prob_matrix.argmax(axis=1)]
    df['confidence']      = prob_matrix.max(axis=1)
    df['correct']         = (df['predicted_class'] == df['alerce_class']).astype(int)

    # ── Save clean dataset ───────────────────────────────────────────────────
    output_path = Path("data/processed/alerce_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"\nTotal objects:  {len(df):,}")
    print(f"Overall accuracy: {df['correct'].mean():.3f}")
    print(f"\nClass distribution:")

    for cls in EVAL_CLASSES:
        n   = (df['alerce_class'] == cls).sum()
        acc = df[df['alerce_class'] == cls]['correct'].mean()
        pct = n / len(df) * 100
        print(f"  {cls:8s}: {n:4d}  ({pct:5.1f}%)  accuracy: {acc:.3f}")

    print(f"\nConfidence stats:")
    print(f"  Min:    {df['confidence'].min():.3f}")
    print(f"  Median: {df['confidence'].median():.3f}")
    print(f"  Mean:   {df['confidence'].mean():.3f}")
    print(f"  Max:    {df['confidence'].max():.3f}")

    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    df = build_dataset()