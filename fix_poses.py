"""
Run this once to clean up old 75-feature .npy files.
Deletes any .npy that doesn't match the expected (n, 63) shape.
Run: python fix_poses.py
"""
import os
import numpy as np
from pathlib import Path

POSES_DIR    = r"J:\Agent My Learning\agent 5\data\poses"
EXPECTED_DIM = 63

deleted = kept = errors = 0

for npy_path in Path(POSES_DIR).rglob("*.npy"):
    try:
        arr = np.load(str(npy_path))
        if arr.shape[1] != EXPECTED_DIM:
            os.remove(str(npy_path))
            deleted += 1
        else:
            kept += 1
    except Exception:
        os.remove(str(npy_path))
        errors += 1

print(f"Deleted : {deleted}  (wrong shape — old 75-feature files)")
print(f"Kept    : {kept}     (correct 63-feature files)")
print(f"Errors  : {errors}   (unreadable, also deleted)")
print(f"\nNow run: python 4_train_model.py")