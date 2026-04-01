"""
NEEDLE Prediction Extraction
=============================

Extract softmax predictions from NEEDLE's 5 pre-trained models
for calibration analysis.

Unlike ALeRCE and Fink (API queries), NEEDLE requires local
model inference using TensorFlow.

Prerequisites:
    1. Clone NEEDLE: git clone https://github.com/XinyueSheng2019/NEEDLE.git
    2. Download data.hdf5 from Kaggle: sherrysheng97/needle-repo-dataset
    3. Install: pip install tensorflow-macos==2.12.0 h5py numpy

Usage:
    python src/needle_extraction.py --needle-dir /path/to/NEEDLE
"""

import json
import os
import sys
import numpy as np


def load_hdf5(data_path):
    """Load the full NEEDLE dataset."""
    import h5py

    print(f"Loading {data_path}...")
    with h5py.File(data_path, "r") as f:
        imageset = f["imageset"][:]  # (5206, 60, 60, 2)
        metaset = f["metaset"][:]    # (5206, 15)
        labels = f["label"][:]       # (5206,)
        idx_set = f["idx_set"][:]    # (5206,)

    print(f"  Images: {imageset.shape}")
    print(f"  Metadata: {metaset.shape}")
    print(f"  Labels: {np.bincount(labels.astype(int))}")
    return imageset, metaset, labels, idx_set


def get_test_positions(testset_obj, hash_table):
    """
    Map testset_obj ZTF IDs to HDF5 row positions.

    The dataset indices stored in testset_obj.json are STALE —
    the Kaggle data.hdf5 was rebuilt with different row ordering.
    We recover the correct rows by matching ZTF IDs through
    hash_table.json instead.
    """
    ztf_to_row = {v["ztf_id"]: int(k) for k, v in hash_table.items()}
    ztf_to_label = {v["ztf_id"]: int(v["label"]) for k, v in hash_table.items()}

    positions, ztf_ids, true_labels = [], [], []

    for cls_name, obj_dict in testset_obj.items():
        for ztf_id in obj_dict:
            row = ztf_to_row.get(ztf_id)
            if row is None:
                print(f"  WARNING: {ztf_id} not in hash_table — skipping")
                continue
            positions.append(row)
            ztf_ids.append(ztf_id)
            true_labels.append(ztf_to_label[ztf_id])

    return np.array(positions), ztf_ids, np.array(true_labels)


def apply_scaling(meta_subset, scaling_path):
    """Standardize metadata using per-model scaling_data.json."""
    with open(scaling_path) as f:
        sd = json.load(f)
    mean = np.array(sd["mean"], dtype=np.float32)
    std = np.array(sd["std"], dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
    return (meta_subset - mean) / std


def run_inference(model_dir, images, meta):
    """Load SavedModel and predict. Returns (N, 3) softmax array."""
    import tensorflow as tf

    print(f"  Loading model...", end="", flush=True)
    model = tf.keras.models.load_model(model_dir, compile=False)
    print(" done")

    probs = model.predict(
        {"image_input": images, "meta_input": meta},
        batch_size=64,
        verbose=0,
    )
    return probs


def extract_all(needle_dir, output_path):
    """
    Main extraction pipeline.

    Parameters
    ----------
    needle_dir : str
        Path to the NEEDLE directory (containing lasair_th_r/ and needle_th_models/)
    output_path : str
        Where to save the .npz file
    """
    data_path = os.path.join(needle_dir, "needle_th_models", "data.hdf5")
    hash_path = os.path.join(needle_dir, "needle_th_models", "hash_table.json")
    model_family_dir = os.path.join(needle_dir, "lasair_th_r")

    # Check paths exist
    for path, name in [(data_path, "data.hdf5"), (hash_path, "hash_table.json"),
                       (model_family_dir, "lasair_th_r/")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            print("See docs_needle_acquisition.md for setup instructions.")
            return

    # Load dataset
    imageset, metaset, labels_hdf5, idx_set = load_hdf5(data_path)
    with open(hash_path) as f:
        hash_table = json.load(f)

    # Discover models
    model_dirs = sorted(
        os.path.join(model_family_dir, d)
        for d in os.listdir(model_family_dir)
        if os.path.isdir(os.path.join(model_family_dir, d))
        and os.path.exists(os.path.join(model_family_dir, d, "testset_obj.json"))
    )
    print(f"\nFound {len(model_dirs)} models")

    # Extract predictions
    all_probs, all_labels, all_model_ids, all_ztf_ids = [], [], [], []
    class_names = ["SN", "SLSN-I", "TDE"]

    for model_idx, model_dir in enumerate(model_dirs):
        model_name = os.path.basename(model_dir)
        print(f"\n{'='*50}")
        print(f"Model {model_idx}: {model_name}")

        # Load test split
        with open(os.path.join(model_dir, "testset_obj.json")) as f:
            testset_obj = json.load(f)

        positions, ztf_ids, true_labels = get_test_positions(testset_obj, hash_table)
        print(f"  Test objects: {len(positions)}")
        print(f"    SN: {np.sum(true_labels==0)}, "
              f"SLSN-I: {np.sum(true_labels==1)}, "
              f"TDE: {np.sum(true_labels==2)}")

        # Prepare inputs
        imgs = np.nan_to_num(imageset[positions])
        meta = np.nan_to_num(metaset[positions])
        meta = apply_scaling(meta, os.path.join(model_dir, "scaling_data.json"))

        # Run model
        probs = run_inference(model_dir, imgs, meta)

        # Sanity check
        pred_cls = np.argmax(probs, axis=1)
        accuracy = np.mean(pred_cls == true_labels)
        mean_conf = np.mean(np.max(probs, axis=1))
        print(f"  Accuracy: {accuracy:.3f}  Mean confidence: {mean_conf:.3f}")

        all_probs.append(probs)
        all_labels.append(true_labels)
        all_model_ids.append(np.full(len(true_labels), model_idx, dtype=np.int32))
        all_ztf_ids.extend(ztf_ids)

    # Combine and save
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_model_ids = np.concatenate(all_model_ids, axis=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        probs=all_probs,
        labels=all_labels,
        model_ids=all_model_ids,
        ztf_ids=np.array(all_ztf_ids),
        class_names=np.array(class_names),
    )

    print(f"\n{'='*50}")
    print(f"SAVED: {output_path}")
    print(f"  Total predictions: {len(all_probs)}")
    print(f"  Unique objects: {len(set(all_ztf_ids))}")
    print(f"  Overall accuracy: {np.mean(np.argmax(all_probs,1)==all_labels):.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract NEEDLE predictions")
    parser.add_argument(
        "--needle-dir",
        default=os.path.expanduser("~/Downloads/needle-calibration/NEEDLE"),
        help="Path to NEEDLE directory",
    )
    args = parser.parse_args()

    output = os.path.join("data", "processed", "needle_predictions.npz")
    extract_all(args.needle_dir, output)
