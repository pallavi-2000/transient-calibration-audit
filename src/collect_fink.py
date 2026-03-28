"""
src/collect_fink.py

Collects Fink broker classification scores for the same ZTF objects
already collected from ALeRCE. Run in a separate terminal and leave
running — it will checkpoint every 50 objects.

    python src/collect_fink.py

Saves to:
    data/raw/fink_predictions.csv

Note: Fink uses POST requests to api.fink-portal.org (changed Jan 2025).
Many BTS objects will not be in Fink — this is expected and reported
separately from API failures.
"""

import requests
import pandas as pd
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

FINK_API   = "https://api.fink-portal.org/api/v1"
CHECKPOINT = Path("data/raw/fink_checkpoint.csv")
OUTPUT     = Path("data/raw/fink_predictions.csv")


def query_fink(ztf_id: str, timeout: int = 20) -> dict | None | str:
    """
    Query Fink for one ZTF object.
    Returns:
        dict          — scores found
        "not_in_fink" — object not in Fink (empty array response)
        None          — API error or timeout
    """
    try:
        r = requests.post(
            f"{FINK_API}/objects",
            json={"objectId": ztf_id, "output-format": "json"},
            timeout=timeout
        )
        if r.status_code == 200:
            data = r.json()
            if data and len(data) > 0:
                latest = data[-1]
                return {
                    "rf_snia_vs_nonia":  latest.get("d:rf_snia_vs_nonia",  np.nan),
                    "snn_snia_vs_nonia": latest.get("d:snn_snia_vs_nonia", np.nan),
                    "ndethist":          latest.get("i:ndethist",           np.nan),
                    "n_alerts":          len(data),
                }
            # Empty array = object simply not in Fink, not an error
            return "not_in_fink"
        return None

    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def collect_fink():
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    # Load ALeRCE dataset to get ZTF IDs
    alerce_path = Path("data/processed/alerce_dataset.csv")
    assert alerce_path.exists(), (
        "Run build_dataset.py first — need data/processed/alerce_dataset.csv"
    )

    alerce_df = pd.read_csv(alerce_path)
    ztf_ids   = alerce_df[['oid', 'spectroscopic_type', 'alerce_class']].copy()

    print("=" * 60)
    print("FINK DATA COLLECTION")
    print(f"Objects to query: {len(ztf_ids):,}")
    print("Checkpoint every 50 objects — safe to interrupt with Ctrl+C")
    print("Re-run script to resume from checkpoint")
    print("=" * 60)

    # Resume from checkpoint if exists and non-empty
    collected = {}
    if CHECKPOINT.exists() and CHECKPOINT.stat().st_size > 0:
        try:
            existing  = pd.read_csv(CHECKPOINT)
            collected = {row["oid"]: row.to_dict() for _, row in existing.iterrows()}
            print(f"\nResuming: {len(collected):,} already collected")
        except Exception:
            print("\nCheckpoint unreadable — starting fresh")

    remaining = ztf_ids[~ztf_ids["oid"].isin(collected.keys())]
    print(f"Remaining: {len(remaining):,}\n")

    failed      = []
    not_in_fink = []

    for i, (_, row) in enumerate(tqdm(remaining.iterrows(),
                                       total=len(remaining),
                                       desc="Fink")):
        ztf_id     = row["oid"]
        spec_type  = row["spectroscopic_type"]
        alerce_cls = row["alerce_class"]

        result = query_fink(ztf_id)

        if result is not None and result != "not_in_fink":
            result["oid"]                = ztf_id
            result["spectroscopic_type"] = spec_type
            result["alerce_class"]       = alerce_cls
            collected[ztf_id]            = result
        elif result == "not_in_fink":
            not_in_fink.append(ztf_id)
        else:
            failed.append(ztf_id)

        time.sleep(0.5)

        # Checkpoint every 50 objects
        if (i + 1) % 50 == 0:
            if collected:
                pd.DataFrame(list(collected.values())).to_csv(CHECKPOINT, index=False)
            print(f"\n  [{i+1}/{len(remaining)}] "
                  f"In Fink: {len(collected):,}  "
                  f"Not in Fink: {len(not_in_fink)}  "
                  f"Errors: {len(failed)}")

    # Final save
    if collected:
        results_df = pd.DataFrame(list(collected.values()))
        results_df.to_csv(OUTPUT, index=False)

        print(f"\n{'='*60}")
        print(f"FINK COLLECTION COMPLETE")
        print(f"  In Fink (saved):  {len(results_df):,}")
        print(f"  Not in Fink:      {len(not_in_fink)}")
        print(f"  API errors:       {len(failed)}")
        print(f"  Saved to:         {OUTPUT}")

        print(f"\nClass breakdown of objects found in Fink:")
        print(results_df["alerce_class"].value_counts().to_string())
    else:
        print("\nNo objects found in Fink.")


if __name__ == "__main__":
    collect_fink()