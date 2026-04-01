"""
Data Acquisition - Step 1: Download BTS Ground Truth
=====================================================
The ZTF Bright Transient Survey gives us objects with KNOWN
spectroscopic types. This is our "answer key" for checking
whether broker predictions are calibrated.
"""

import pandas as pd
import requests
from io import StringIO
from pathlib import Path
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Where to save the downloaded data
GROUND_TRUTH_DIR = Path("data/ground_truth")

# BTS Explorer URL
BTS_URL = (
    "https://sites.astro.caltech.edu/ztf/bts/explorer.php"
    "?f=s&subsample=cantrans&classstring=&classexclude="
    "&quality=y&ztflink=lasair"
    "&lastdet=&startsavedate=&startpeakdate="
    "&startra=&startdec=&startz=&startdur=&startrise=&startfade="
    "&startpeakmag=&startabsmag=&starthostabs=&starthostcol="
    "&startb=&startav="
    "&endsavedate=&endpeakdate=&endra=&enddec=&endz=&enddur="
    "&endrise=&endfade=&endpeakmag=19.0&endabsmag="
    "&endhostabs=&endhostcol=&endb=&endav=&format=csv"
)

# Map BTS spectroscopic types -> ALeRCE's 5 classes
SPEC_TO_ALERCE = {
    # SN Ia (thermonuclear)
    "SN Ia": "SNIa",
    "SN Ia-91T": "SNIa",
    "SN Ia-91bg": "SNIa",
    "SN Ia-pec": "SNIa",
    "SN Ia-CSM": "SNIa",
    "SN Ia-SC": "SNIa",
    "SN Iax": "SNIa",

    # SN II (hydrogen-rich core collapse)
    "SN II": "SNII",
    "SN IIn": "SNII",
    "SN IIP": "SNII",
    "SN II-pec": "SNII",

    # SN Ibc (stripped-envelope)
    "SN Ib": "SNIbc",
    "SN Ic": "SNIbc",
    "SN Ic-BL": "SNIbc",
    "SN IIb": "SNIbc",
    "SN Ib/c": "SNIbc",
    "SN Ibn": "SNIbc",
    "SN Icn": "SNIbc",
    "SN Ib-pec": "SNIbc",
    "SN Ic-pec": "SNIbc",

    # Superluminous
    "SLSN-I": "SLSN",
    "SLSN-II": "SLSN",

    # Tidal disruption events
    "TDE": "TDE",
    "TDE-He": "TDE",
    "TDE-H-He": "TDE",
    "TDE-featureless": "TDE",
}

def download_bts_catalog():
    """Download the BTS catalog and keep only objects with spectroscopic types."""

    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading ZTF Bright Transient Survey catalog...")

    # Use requests instead of pandas for the download (handles SSL better)
    response = requests.get(BTS_URL, timeout=60)
    response.raise_for_status()

    # Parse the CSV text into a dataframe
    bts = pd.read_csv(StringIO(response.text))
    print(f"Downloaded {len(bts):,} total objects")

    # type == '-' means no spectroscopic classification yet
    bts_typed = bts[bts["type"] != "-"].copy()
    print(f"With spectroscopic types: {len(bts_typed):,}")

    # Show what we got
    print(f"\nTop 10 types:")
    print(bts_typed["type"].value_counts().head(10))

    # Save
    save_path = GROUND_TRUTH_DIR / "bts_spectroscopic.csv"
    bts_typed.to_csv(save_path, index=False)
    print(f"\nSaved to: {save_path}")

    return bts_typed

def map_and_sample(bts_df, random_state=42):
    """
    Map BTS types to ALeRCE classes and create a stratified sample.
    
    Why stratified: SN Ia is 75% of BTS. Random sampling would give
    us ~5 TDEs — useless for per-class calibration. Instead we cap
    large classes and take all available rare objects.
    """
    # Map types
    bts_df = bts_df.copy()
    bts_df["alerce_class"] = bts_df["type"].map(SPEC_TO_ALERCE)
    bts_df = bts_df.dropna(subset=["alerce_class"]).reset_index(drop=True)

    print(f"Mapped {len(bts_df):,} objects to ALeRCE classes")

    # Target counts per class
    target_counts = {
        "SNIa": 600,    # 5328 available, 600 is plenty
        "SNII": 400,    # 1148 available
        "SNIbc": 300,   # 504 available
        "SLSN": 97,     # take all
        "TDE": 39,      # take all
    }

    samples = []
    print("\nStratified sampling:")
    for cls, n_target in target_counts.items():
        cls_df = bts_df[bts_df["alerce_class"] == cls]
        n_available = len(cls_df)
        n_sample = min(n_target, n_available)

        sample = cls_df.sample(n=n_sample, random_state=random_state)
        samples.append(sample)

        print(f"  {cls:6s}: {n_available:4d} available, sampled {n_sample:4d}")

    result = pd.concat(samples, ignore_index=True)
    print(f"\nTotal sample: {len(result):,} objects")

    # Save
    save_path = GROUND_TRUTH_DIR / "bts_sample.csv"
    result.to_csv(save_path, index=False)
    print(f"Saved to: {save_path}")

    return result

def query_alerce_single(oid):
    """
    Query ALeRCE for one object's lc_classifier probabilities.
    
    Returns dict with probabilities for the 5 transient classes,
    or None if the query fails.
    """
    url = f"https://api.alerce.online/ztf/v1/objects/{oid}/probabilities"

    try:
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            return None

        data = response.json()

        # Keep only lc_classifier
        lc_probs = [
            entry for entry in data
            if entry["classifier_name"] == "lc_classifier"
        ]

        if not lc_probs:
            return None

        # Build probability dict
        prob_dict = {"oid": oid}
        for entry in lc_probs:
            prob_dict[entry["class_name"]] = entry["probability"]

        return prob_dict

    except Exception:
        return None


def collect_alerce_predictions(ztf_ids, save_every=50):
    """
    Query ALeRCE for all objects. Saves progress every 50 objects
    so you can resume if interrupted.
    """
    import time

    save_path = Path("data/raw/alerce_classifications.csv")
    checkpoint_path = Path("data/raw/alerce_checkpoint.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if it exists
    results = []
    done_oids = set()

    if checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        results = existing.to_dict("records")
        done_oids = set(existing["oid"])
        print(f"Resuming: {len(done_oids)} already done")

    remaining = [oid for oid in ztf_ids if oid not in done_oids]

    print(f"Querying ALeRCE for {len(remaining)} objects...")
    print(f"Estimated time: ~{len(remaining) * 0.15 / 60:.0f} minutes")

    failed = []

    for i, oid in enumerate(remaining):
        prob_dict = query_alerce_single(oid)

        if prob_dict is not None:
            results.append(prob_dict)
        else:
            failed.append(oid)

        # Progress update
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(remaining)} done ({len(failed)} failed)")
            # Save checkpoint
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)

        time.sleep(0.15)  # Rate limit: ~7 requests/sec

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    print(f"\nDone! {len(results_df)} successful, {len(failed)} failed")
    print(f"Saved to: {save_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return results_df
def query_fink_single(oid):
    """
    Query Fink for one object's RF and SuperNNova scores.
    
    Fink returns one row per alert. We take the LATEST alert
    (most complete light curve = most informed classification).
    
    Returns binary scores (SN Ia vs not-SN Ia), not multi-class.
    """
    url = "https://api.fink-portal.org/api/v1/objects"

    try:
        response = requests.post(
            url,
            json={
                "objectId": oid,
                "columns": "i:objectId,d:rf_snia_vs_nonia,d:snn_snia_vs_nonia,i:jd",
                "output-format": "json",
            },
            timeout=30,
            verify=False,
        )

        if response.status_code != 200:
            return None

        data = response.json()
        if not data:
            return None

        # Sort by Julian Date, take the latest alert
        df = pd.DataFrame(data)
        df = df.sort_values("i:jd", ascending=False)
        latest = df.iloc[0]

        return {
            "oid": oid,
            "rf_snia_vs_nonia": float(latest["d:rf_snia_vs_nonia"]),
            "snn_snia_vs_nonia": float(latest["d:snn_snia_vs_nonia"]),
        }

    except Exception:
        return None


def collect_fink_predictions(ztf_ids, save_every=50):
    """
    Query Fink for all objects. Same checkpointing logic as ALeRCE.
    """
    import time

    save_path = Path("data/raw/fink_classifications.csv")
    checkpoint_path = Path("data/raw/fink_checkpoint.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    done_oids = set()

    if checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        results = existing.to_dict("records")
        done_oids = set(existing["oid"])
        print(f"Resuming: {len(done_oids)} already done")

    remaining = [oid for oid in ztf_ids if oid not in done_oids]

    print(f"Querying Fink for {len(remaining)} objects...")
    print(f"Estimated time: ~{len(remaining) * 0.15 / 60:.0f} minutes")

    failed = []

    for i, oid in enumerate(remaining):
        score_dict = query_fink_single(oid)

        if score_dict is not None:
            results.append(score_dict)
        else:
            failed.append(oid)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(remaining)} done ({len(failed)} failed)")
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)

        time.sleep(0.15)

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    print(f"\nDone! {len(results_df)} successful, {len(failed)} failed")
    print(f"Saved to: {save_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return results_df

if __name__ == "__main__":
    bts = download_bts_catalog()
    sample = map_and_sample(bts)

    ztf_ids = sample["ZTFID"].tolist()

    # Query both brokers
    alerce_df = collect_alerce_predictions(ztf_ids)
    fink_df = collect_fink_predictions(ztf_ids)
