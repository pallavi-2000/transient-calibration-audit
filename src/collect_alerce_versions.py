"""
src/collect_alerce_versions.py

Re-queries the ALeRCE broker API for the `classifier_version` tag attached
to each object's lc_classifier probabilities. The original collection
script (collect_alerce.py) discarded this field, so the 1,606-object
dataset was analysed as if it came from one stationary classifier, even
though ALeRCE's production `lc_classifier` is versioned and updated over
time. This script fills that gap so calibration can be checked for
stability across classifier versions.

Checkpoints every 50 objects. Safe to interrupt and resume.

Output: data/raw/alerce_versions.csv  (oid, classifier_version)
"""

import time
from pathlib import Path

import pandas as pd
from alerce.core import Alerce
from tqdm import tqdm

PREDICTIONS = Path("data/raw/alerce_predictions.csv")
CHECKPOINT = Path("data/raw/alerce_versions_checkpoint.csv")
OUTPUT = Path("data/raw/alerce_versions.csv")


def main():
    preds = pd.read_csv(PREDICTIONS)
    oids = preds["oid"].tolist()
    print(f"Objects to version-tag: {len(oids)}")

    collected = {}
    if CHECKPOINT.exists():
        existing = pd.read_csv(CHECKPOINT)
        collected = dict(zip(existing["oid"], existing["classifier_version"]))
        print(f"Resuming from checkpoint: {len(collected)} already collected")

    remaining = [o for o in oids if o not in collected]
    print(f"Remaining: {len(remaining)}")

    alerce = Alerce()
    failed = []

    for i, oid in enumerate(tqdm(remaining, desc="Version-tagging")):
        try:
            probs = alerce.query_probabilities(oid, format="pandas")
            lc = probs[probs["classifier_name"] == "lc_classifier"]
            if len(lc) > 0:
                collected[oid] = lc["classifier_version"].iloc[0]
            else:
                failed.append(oid)
        except Exception:
            failed.append(oid)

        time.sleep(0.3)

        if (i + 1) % 50 == 0:
            pd.DataFrame(
                {"oid": list(collected.keys()), "classifier_version": list(collected.values())}
            ).to_csv(CHECKPOINT, index=False)
            print(f"  Checkpoint {i+1}/{len(remaining)} | collected={len(collected)} | failed={len(failed)}")

    out = pd.DataFrame(
        {"oid": list(collected.keys()), "classifier_version": list(collected.values())}
    )
    out.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(out)} version tags -> {OUTPUT}")
    print(f"Failed: {len(failed)}")
    print("\nVersion distribution:")
    print(out["classifier_version"].value_counts().to_string())


if __name__ == "__main__":
    main()
