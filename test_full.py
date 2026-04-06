import pandas as pd
import numpy as np
from src.calibration import (
    compute_ece, compute_per_class_ece,
    fit_temperature_cv, bootstrap_ece
)

# ====== ALeRCE ======
print("=" * 50)
print("ALeRCE — Full Calibration")
print("=" * 50)

alerce = pd.read_csv('data/raw/alerce_classifications.csv')
sample = pd.read_csv('data/ground_truth/bts_sample.csv')
merged = alerce.merge(sample[['ZTFID', 'alerce_class']], left_on='oid', right_on='ZTFID')
merged = merged[merged['alerce_class'] != 'TDE']

trans = ['SNIa', 'SNIbc', 'SNII', 'SLSN']
class_to_int = {c: i for i, c in enumerate(trans)}
proba = merged[trans].apply(pd.to_numeric, errors='coerce').values
proba = proba / proba.sum(axis=1, keepdims=True)
labels = merged['alerce_class'].map(class_to_int).values

# ECE with bootstrap CI
boot = bootstrap_ece(labels, proba)
print(f"ECE: {boot['ece']:.3f} [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")

# Per-class
per_class = compute_per_class_ece(labels, proba, class_names=trans)
print("\nPer-class:")
for cls, stats in per_class.items():
    print(f"  {cls:6s}: ECE={stats['ece']:.3f}  acc={stats['accuracy']:.3f}  "
          f"conf={stats['mean_confidence']:.3f}  gap={stats['gap']:+.3f}")

# Temperature scaling with CV
print("\nTemperature scaling (5-fold CV):")
ts = fit_temperature_cv(labels, proba)
print(f"  T = {ts['T_mean']:.3f} +/- {ts['T_std']:.3f}")
print(f"  ECE before: {ts['ece_before_mean']:.3f}")
print(f"  ECE after:  {ts['ece_after_mean']:.3f}")
print(f"  Improvement: {ts['improvement']:.3f}")

# ====== NEEDLE ======
print(f"\n{'=' * 50}")
print("NEEDLE — Full Calibration")
print("=" * 50)

needle = np.load('data/processed/needle_predictions.npz', allow_pickle=True)
n_labels = needle['labels']
n_probs = needle['probs']
n_names = list(needle['class_names'])

boot_n = bootstrap_ece(n_labels, n_probs)
print(f"ECE: {boot_n['ece']:.3f} [{boot_n['ci_lower']:.3f}, {boot_n['ci_upper']:.3f}]")

per_class_n = compute_per_class_ece(n_labels, n_probs, class_names=n_names)
print("\nPer-class:")
for cls, stats in per_class_n.items():
    print(f"  {cls:6s}: ECE={stats['ece']:.3f}  acc={stats['accuracy']:.3f}  "
          f"conf={stats['mean_confidence']:.3f}  gap={stats['gap']:+.3f}")

ts_n = fit_temperature_cv(n_labels, n_probs)
print(f"\nTemperature scaling (5-fold CV):")
print(f"  T = {ts_n['T_mean']:.3f} +/- {ts_n['T_std']:.3f}")
print(f"  ECE before: {ts_n['ece_before_mean']:.3f}")
print(f"  ECE after:  {ts_n['ece_after_mean']:.3f}")
