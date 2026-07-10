"""
src/collect_alerce.py

Downloads BTS catalog and collects ALeRCE LC classifier predictions.
Run once from your project root:

    python src/collect_alerce.py

Saves to:
    data/raw/bts_catalog.csv          — full BTS download
    data/raw/alerce_predictions.csv   — final collected predictions

Checkpoints every 50 objects so you can safely interrupt and resume.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from alerce.core import Alerce
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING STRATEGY
# Get ALL rare objects — every TDE and SLSN available.
# For common classes, sample generously but not exhaustively.
# Target: ~1,500 objects total.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLING = {
    # Common classes — large enough for robust ECE, not so large it takes days
    'SN Ia':     800,   # ~5044 available — ECE anchor class
    'SN II':     300,   # ~899 available
    'SN Ib':     None,  # None = take ALL available (~114)
    'SN Ic':     None,  # None = take ALL available (~153)
    'SN IIb':    None,  # None = take ALL available (~107)
    'SN IIn':    None,  # None = take ALL available (~175)
    'SN IIP':    None,  # None = take ALL available (~62)
    'SN Ia-91T': None,  # None = take ALL available (~151)
    'SN Ia-91bg':None,  # None = take ALL available (~41)
    # Rare classes — take every single one
    'SLSN-I':    None,  # None = take ALL (~63) — critical for paper
    'TDE':       None,  # None = take ALL (~33) — critical for paper
}

BTS_URL = (
    "https://sites.astro.caltech.edu/ztf/bts/explorer.php"
    "?f=s&subsample=cantrans&classstring=&classexclude="
    "&quality=y&ztflink=lasair&lastdet=&startsavedate="
    "&startpeakdate=&startra=&startdec=&startz=&startdur="
    "&startrise=&startfade=&startpeakmag=&startabsmag="
    "&starthostabs=&starthostcol=&startb=&startav="
    "&endsavedate=&endpeakdate=&endra=&enddec=&endz="
    "&enddur=&endrise=&endfade=&endpeakmag=19.0"
    "&endabsmag=&endhostabs=&endhostcol=&endb=&endav=&format=csv"
)

CHECKPOINT = Path("data/raw/alerce_checkpoint.csv")
OUTPUT     = Path("data/raw/alerce_predictions.csv")
BTS_FILE   = Path("data/raw/bts_catalog.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Download BTS catalog
# ─────────────────────────────────────────────────────────────────────────────

def download_bts() -> pd.DataFrame:
    if BTS_FILE.exists():
        print(f"BTS catalog already exists — loading from {BTS_FILE}")
        return pd.read_csv(BTS_FILE)

    print("Downloading BTS catalog...")
    bts = pd.read_csv(BTS_URL)
    bts.to_csv(BTS_FILE, index=False)
    print(f"  Saved {len(bts):,} objects to {BTS_FILE}")
    return bts


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Build stratified sample
# ─────────────────────────────────────────────────────────────────────────────

def build_sample(bts: pd.DataFrame) -> pd.DataFrame:
    # Keep only spectroscopically confirmed objects
    bts_spec = bts[bts['type'] != '-'].copy()
    print(f"\nBTS objects with spectroscopic type: {len(bts_spec):,}")
    print(f"\nAvailable counts for target classes:")

    sampled = []
    total_target = 0

    for cls, n_target in SAMPLING.items():
        available = bts_spec[bts_spec['type'] == cls]
        n_available = len(available)
        n_to_take = n_available if n_target is None else min(n_target, n_available)
        total_target += n_to_take

        label = "ALL" if n_target is None else str(n_target)
        print(f"  {cls:15s}: {n_to_take:4d} / {n_available:4d} available  (target: {label})")

        if n_to_take > 0:
            sample = available.sample(
                n=n_to_take,
                random_state=42
            ) if n_to_take < n_available else available
            sampled.append(sample[['ZTFID', 'type', 'redshift', 'peakmag']])

    sample_df = pd.concat(sampled, ignore_index=True)
    sample_df = sample_df.rename(columns={'type': 'spectroscopic_type'})

    print(f"\nTotal sample: {len(sample_df):,} objects")
    print(f"Estimated collection time: ~{len(sample_df) * 0.6 / 60:.0f} minutes")
    return sample_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Collect ALeRCE predictions with checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def collect_predictions(sample_df: pd.DataFrame) -> pd.DataFrame:
    alerce = Alerce()

    # Resume from checkpoint if it exists
    collected = {}
    if CHECKPOINT.exists():
        existing = pd.read_csv(CHECKPOINT)
        collected = {row['oid']: row for _, row in existing.iterrows()}
        print(f"\nResuming from checkpoint: {len(collected):,} already collected")

    # Only query what we don't have yet
    remaining = sample_df[~sample_df['ZTFID'].isin(collected.keys())]
    print(f"Objects to collect: {len(remaining):,}")
    print("-" * 60)

    failed = []

    for i, (_, row) in enumerate(tqdm(remaining.iterrows(),
                                       total=len(remaining),
                                       desc="Collecting")):
        ztf_id = row['ZTFID']
        spec_type = row['spectroscopic_type']

        try:
            probs = alerce.query_probabilities(ztf_id, format="pandas")
            lc = probs[probs['classifier_name'] == 'lc_classifier']

            if len(lc) > 0:
                result = {
                    'oid':               ztf_id,
                    'spectroscopic_type': spec_type,
                    'redshift':          row.get('redshift', None),
                    'peakmag':           row.get('peakmag', None),
                    # ALeRCE's lc_classifier is versioned and updated in production
                    # (see collect_alerce_versions.py) — captured here so future
                    # analyses can check calibration stability across versions
                    # without a second round of API calls.
                    'classifier_version': lc['classifier_version'].iloc[0],
                }
                for _, prob_row in lc.iterrows():
                    result[prob_row['class_name']] = prob_row['probability']

                collected[ztf_id] = result
            else:
                failed.append(ztf_id)

        except Exception as e:
            failed.append(ztf_id)

        time.sleep(0.5)  # respectful rate limiting

        # Checkpoint every 50 objects
        if (i + 1) % 50 == 0:
            pd.DataFrame(list(collected.values())).to_csv(CHECKPOINT, index=False)
            success_rate = len(collected) / (len(collected) + len(failed)) * 100
            print(f"\n  Checkpoint {i+1}/{len(remaining)} | "
                  f"Collected: {len(collected):,} | "
                  f"Failed: {len(failed)} | "
                  f"Success rate: {success_rate:.1f}%")

    # Final save
    results_df = pd.DataFrame(list(collected.values()))
    results_df.to_csv(OUTPUT, index=False)
    print(f"\n{'='*60}")
    print(f"Collection complete")
    print(f"  Successful:  {len(results_df):,}")
    print(f"  Failed:      {len(failed)}")
    print(f"  Saved to:    {OUTPUT}")

    # Print final class distribution
    print(f"\nFinal class distribution:")
    print(results_df['spectroscopic_type'].value_counts().to_string())

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Make sure output dirs exist
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ALERCE DATA COLLECTION")
    print("Target: ~1,500 spectroscopically confirmed ZTF transients")
    print("=" * 60)

    # Step 1: BTS catalog
    bts = download_bts()

    # Step 2: Build sample
    sample_df = build_sample(bts)

    # Step 3: Collect
    print("\nStarting ALeRCE API collection...")
    print("You can safely interrupt with Ctrl+C — progress is checkpointed.")
    print("Re-run the script to resume from where you left off.\n")

    results = collect_predictions(sample_df)