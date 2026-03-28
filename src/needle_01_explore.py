"""
Step 1: Explore the NEEDLE dataset structure.

This script inspects the HDF5 file and hash_table.json to understand:
- What data is stored (images, metadata, labels)
- How objects are indexed
- How train/test splits are defined
- The class distribution

Run this FIRST before anything else.
"""

import h5py
import json
import numpy as np
import os
import sys

# === CONFIGURATION ===
# Adjust these paths to match your local setup
NEEDLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'NEEDLE')
DATA_PATH = os.path.join(NEEDLE_DIR, 'needle_th_models', 'data.hdf5')
HASH_TABLE_PATH = os.path.join(NEEDLE_DIR, 'needle_th_models', 'hash_table.json')
MODEL_DIR = os.path.join(NEEDLE_DIR, 'needle_th_models')


def explore_hdf5(filepath):
    """Recursively print HDF5 file structure."""
    print(f"\n{'='*60}")
    print(f"HDF5 FILE: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        print(f"Make sure you've downloaded the dataset:")
        print(f"  kaggle datasets download -d sherrysheng97/needle-repo-dataset")
        return None
    
    with h5py.File(filepath, 'r') as f:
        print(f"\nTop-level keys: {list(f.keys())}")
        
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f"\n  Dataset: '{key}'")
                print(f"    Shape: {item.shape}")
                print(f"    Dtype: {item.dtype}")
                print(f"    Min/Max: {np.min(item[:10]):.4f} / {np.max(item[:10]):.4f}")
                if len(item.shape) == 1 and item.shape[0] < 100:
                    print(f"    Values: {item[:]}")
                elif len(item.shape) == 1:
                    print(f"    First 10: {item[:10]}")
                    unique = np.unique(item[:])
                    print(f"    Unique values: {unique}")
                    for u in unique:
                        count = np.sum(item[:] == u)
                        print(f"      {u}: {count} objects")
            elif isinstance(item, h5py.Group):
                print(f"\n  Group: '{key}'")
                for subkey in item.keys():
                    subitem = item[subkey]
                    if isinstance(subitem, h5py.Dataset):
                        print(f"    Dataset: '{subkey}' — shape={subitem.shape}, dtype={subitem.dtype}")
    
    return True


def explore_hash_table(filepath):
    """Examine the hash table mapping objects to indices."""
    print(f"\n{'='*60}")
    print(f"HASH TABLE: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        hash_table = json.load(f)
    
    print(f"\nType: {type(hash_table)}")
    
    if isinstance(hash_table, dict):
        print(f"Number of entries: {len(hash_table)}")
        keys = list(hash_table.keys())
        print(f"First 10 keys: {keys[:10]}")
        print(f"Sample values:")
        for k in keys[:5]:
            print(f"  '{k}': {hash_table[k]}")
    elif isinstance(hash_table, list):
        print(f"Length: {len(hash_table)}")
        print(f"First 10 entries: {hash_table[:10]}")
    
    return hash_table


def explore_models(model_dir):
    """List available pre-trained models."""
    print(f"\n{'='*60}")
    print(f"MODELS DIRECTORY: {model_dir}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_dir):
        print(f"ERROR: Directory not found at {model_dir}")
        return
    
    all_files = os.listdir(model_dir)
    print(f"\nAll files: {sorted(all_files)}")
    
    # Look for model files (Keras .h5 or SavedModel directories)
    model_files = [f for f in all_files if f.endswith('.h5') or f.endswith('.keras')]
    model_dirs = [f for f in all_files if os.path.isdir(os.path.join(model_dir, f)) 
                  and ('model' in f.lower() or 'needle' in f.lower() or f.isdigit())]
    
    print(f"\nModel files (.h5/.keras): {sorted(model_files)}")
    print(f"Model directories: {sorted(model_dirs)}")
    
    # Check for subdirectories that might contain models
    for d in sorted(all_files):
        full_path = os.path.join(model_dir, d)
        if os.path.isdir(full_path):
            subfiles = os.listdir(full_path)
            if any(f.endswith('.h5') or f == 'saved_model.pb' or f.endswith('.keras') 
                   for f in subfiles):
                print(f"\n  Found model in: {d}/")
                print(f"    Contents: {subfiles[:10]}")


def explore_source_code(needle_dir):
    """Find key source files for understanding model loading and prediction."""
    print(f"\n{'='*60}")
    print(f"SOURCE CODE STRUCTURE")
    print(f"{'='*60}")
    
    key_files = ['Config.py', 'Makefile', 'requirements.txt']
    
    for fname in key_files:
        fpath = os.path.join(needle_dir, fname)
        if os.path.exists(fpath):
            print(f"\n--- {fname} ---")
            with open(fpath, 'r') as f:
                content = f.read()
            # Print first 50 lines
            lines = content.split('\n')
            for line in lines[:50]:
                print(f"  {line}")
            if len(lines) > 50:
                print(f"  ... ({len(lines) - 50} more lines)")
    
    # Find Python files
    print(f"\n--- Python files ---")
    for root, dirs, files in os.walk(needle_dir):
        for f in sorted(files):
            if f.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, f), needle_dir)
                print(f"  {rel_path}")


if __name__ == '__main__':
    print("NEEDLE Calibration Study — Data Exploration")
    print("=" * 60)
    
    # Step 1: Explore HDF5
    explore_hdf5(DATA_PATH)
    
    # Step 2: Explore hash table
    explore_hash_table(HASH_TABLE_PATH)
    
    # Step 3: Explore models
    explore_models(MODEL_DIR)
    
    # Step 4: Source code structure
    explore_source_code(NEEDLE_DIR)
    
    print(f"\n{'='*60}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*60}")
    print("\nNext step: Run 02_extract_predictions.py")
    print("But first, review the output above and adjust paths in Config.py if needed.")
