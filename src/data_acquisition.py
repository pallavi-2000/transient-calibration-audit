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


if __name__ == "__main__":
    download_bts_catalog()
