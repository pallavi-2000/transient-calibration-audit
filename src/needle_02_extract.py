"""
Step 2: Extract softmax predictions from the 5 NEEDLE-TH (lasair_th_r) models.

For each model:
1. Read testset_obj.json  → ZTF IDs + dataset indices actually held out
2. Map those indices to rows in data.hdf5 via idx_set
3. Apply per-model metadata normalisation (scaling_data.json)
4. Run inference with the SavedModel
5. Collect (softmax probs, true labels, ZTF IDs, model index)

Output: outputs/needle_predictions.npz
"""

import json
import os
import sys

import h5py
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
NEEDLE_DIR  = os.path.join(os.path.dirname(__file__), '..', 'NEEDLE')
DATA_PATH   = os.path.join(NEEDLE_DIR, 'needle_th_models', 'data.hdf5')
HASH_PATH   = os.path.join(NEEDLE_DIR, 'needle_th_models', 'hash_table.json')
MODEL_FAMILY_DIR = os.path.join(NEEDLE_DIR, 'lasair_th_r')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# label mapping used during NEEDLE training (from label_dict_equal_test.json)
# classes 3 (Non-SN) and 4 (Other) were excluded at training time
LABEL_MAP = {0: 'SN', 1: 'SLSN-I', 2: 'TDE'}
CLASS_NAMES = ['SN', 'SLSN-I', 'TDE']

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_hdf5(data_path):
    """Load full dataset into memory (≈1 GB — fits comfortably on modern hardware)."""
    print(f"Loading {data_path} …")
    with h5py.File(data_path, 'r') as f:
        imageset = f['imageset'][:]   # (N, 60, 60, 2)
        metaset  = f['metaset'][:]    # (N, 15)
        labels   = f['label'][:]      # (N,)  int32
        idx_set  = f['idx_set'][:]    # (N,)  float32  — object IDs (0..N-1)
    print(f"  imageset {imageset.shape}  metaset {metaset.shape}  "
          f"labels {labels.shape}  idx_set {idx_set.shape}")
    return imageset, metaset, labels, idx_set


def load_testset_obj(model_dir):
    """Return {class_name: {ztf_id: dataset_idx_str}} from testset_obj.json."""
    path = os.path.join(model_dir, 'testset_obj.json')
    with open(path) as f:
        return json.load(f)


def get_test_positions(testset_obj, hash_table):
    """
    Convert testset_obj ZTF IDs into row positions in the HDF5 arrays.

    NOTE: the dataset indices stored in testset_obj.json are stale — the
    Kaggle data.hdf5 was rebuilt with different row ordering.  We recover
    the correct rows by matching ZTF IDs through hash_table.json instead.

    Returns
    -------
    positions  : np.ndarray  shape (N_test,)   row indices into HDF5 arrays
    ztf_ids    : list[str]   ZTF IDs in the same order
    true_labels: np.ndarray  shape (N_test,)   ground-truth class (0/1/2)
                             taken from hash_table (authoritative for this HDF5)
    """
    # Build ZTF-ID → (row, label) from hash_table
    ztf_to_row   = {v['ztf_id']: int(k)       for k, v in hash_table.items()}
    ztf_to_label = {v['ztf_id']: int(v['label']) for k, v in hash_table.items()}

    # label mapping used during NEEDLE training
    label_name_to_int = {'SN': 0, 'SLSN-I': 1, 'TDE': 2}

    positions, ztf_ids, true_labels = [], [], []
    for cls_name, obj_dict in testset_obj.items():
        for ztf_id in obj_dict:
            row = ztf_to_row.get(ztf_id)
            if row is None:
                print(f"  WARNING: {ztf_id} not found in hash_table — skipping")
                continue
            positions.append(row)
            ztf_ids.append(ztf_id)
            # Use hash_table label as ground truth (matches the HDF5)
            true_labels.append(ztf_to_label[ztf_id])

    return np.array(positions), ztf_ids, np.array(true_labels)


def apply_scaling(meta_subset, scaling_path):
    """Standardise metadata using per-model scaling_data.json (method 1: z-score)."""
    with open(scaling_path) as f:
        sd = json.load(f)
    mean = np.array(sd['mean'], dtype=np.float32)
    std  = np.array(sd['std'],  dtype=np.float32)
    # Guard against zero std (shouldn't happen but be safe)
    std  = np.where(std == 0, 1.0, std)
    return (meta_subset - mean) / std


def run_inference(model_dir, images, meta):
    """Load Keras SavedModel and predict. Returns (N, 3) softmax array."""
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow not installed.")
        sys.exit(1)

    print(f"  Loading model from {os.path.basename(model_dir)} …", end='', flush=True)
    model = tf.keras.models.load_model(model_dir, compile=False)
    print(" done")

    probs = model.predict(
        {'image_input': images, 'meta_input': meta},
        batch_size=64,
        verbose=0,
    )
    return probs  # (N, 3)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("NEEDLE Calibration Study — Prediction Extraction")
    print("=" * 60)

    # 1. Load full dataset and hash table
    imageset, metaset, labels_hdf5, idx_set = load_hdf5(DATA_PATH)
    with open(HASH_PATH) as f:
        hash_table = json.load(f)

    # 2. Discover models (sorted for reproducibility)
    model_dirs = sorted(
        os.path.join(MODEL_FAMILY_DIR, d)
        for d in os.listdir(MODEL_FAMILY_DIR)
        if os.path.isdir(os.path.join(MODEL_FAMILY_DIR, d))
           and os.path.exists(os.path.join(MODEL_FAMILY_DIR, d, 'testset_obj.json'))
    )
    print(f"\nFound {len(model_dirs)} models in {MODEL_FAMILY_DIR}:")
    for d in model_dirs:
        print(f"  {os.path.basename(d)}")

    # 3. Extract predictions from each model
    all_probs, all_labels, all_model_ids, all_ztf_ids, all_positions = [], [], [], [], []

    for model_idx, model_dir in enumerate(model_dirs):
        model_name = os.path.basename(model_dir)
        print(f"\n{'─'*60}")
        print(f"Model {model_idx} / {len(model_dirs)-1}:  {model_name}")

        # Load test split from the JSON saved at training time
        # Positions are resolved by ZTF ID via hash_table (stored indices are stale)
        testset_obj = load_testset_obj(model_dir)
        positions, ztf_ids, true_labels = get_test_positions(testset_obj, hash_table)
        print(f"  Test objects: {len(positions)}  "
              f"(SN={np.sum(true_labels==0)}, "
              f"SLSN-I={np.sum(true_labels==1)}, "
              f"TDE={np.sum(true_labels==2)})")

        # Prepare inputs
        imgs  = np.nan_to_num(imageset[positions])   # (N, 60, 60, 2)
        meta  = np.nan_to_num(metaset[positions])    # (N, 15)
        meta  = apply_scaling(meta, os.path.join(model_dir, 'scaling_data.json'))

        # Inference
        probs = run_inference(model_dir, imgs, meta)
        print(f"  Probs shape: {probs.shape}   "
              f"sum-to-1 check: {np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)}")

        # Quick accuracy / confidence report
        pred_cls  = np.argmax(probs, axis=1)
        accuracy  = np.mean(pred_cls == true_labels)
        mean_conf = np.mean(np.max(probs, axis=1))
        print(f"  Accuracy: {accuracy:.3f}   Mean confidence: {mean_conf:.3f}")

        all_probs.append(probs)
        all_labels.append(true_labels)
        all_model_ids.append(np.full(len(true_labels), model_idx, dtype=np.int32))
        all_ztf_ids.extend(ztf_ids)
        all_positions.extend(positions.tolist())

    # 4. Combine and save
    all_probs     = np.concatenate(all_probs,     axis=0)
    all_labels    = np.concatenate(all_labels,    axis=0)
    all_model_ids = np.concatenate(all_model_ids, axis=0)

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"  Total predictions : {len(all_probs)}")
    print(f"  Models used       : {len(model_dirs)}")
    print(f"  Unique objects    : {len(set(all_ztf_ids))}")
    print(f"  Overall accuracy  : {np.mean(np.argmax(all_probs,1)==all_labels):.3f}")
    print(f"  Mean confidence   : {np.mean(np.max(all_probs,1)):.3f}")
    print(f"  Class breakdown   : SN={np.sum(all_labels==0)}, "
          f"SLSN-I={np.sum(all_labels==1)}, TDE={np.sum(all_labels==2)}")

    output_path = os.path.join(OUTPUT_DIR, 'needle_predictions.npz')
    np.savez(
        output_path,
        probs        = all_probs,        # (N_total, 3)  softmax outputs
        labels       = all_labels,       # (N_total,)    true class (0/1/2)
        model_ids    = all_model_ids,    # (N_total,)    which model (0-4)
        ztf_ids      = np.array(all_ztf_ids),
        hdf5_rows    = np.array(all_positions),
        class_names  = np.array(CLASS_NAMES),
    )
    print(f"\nSaved → {output_path}")
    print("Next step: Run 03_calibration_analysis.py")


if __name__ == '__main__':
    main()
