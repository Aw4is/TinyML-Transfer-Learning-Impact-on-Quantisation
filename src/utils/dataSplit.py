"""
Create a small PlantVillage subset split (per-class cap).

Assumes source_root contains one folder per class:
    source_root/
        Class_1/
        Class_2/
        ...

For each class folder:
- take up to n per class images
- split into train/val/test by split ratios (remainder goes to test)
- copy into:
    TARGET_ROOT/train/<class_name>/
    TARGET_ROOT/val/<class_name>/
    TARGET_ROOT/test/<class_name>/
"""

from pathlib import Path
import random
import shutil

# ---- Config ----
SOURCE_ROOT = Path(r"")
TARGET_ROOT = Path(r"")

N_PER_CLASS = 100
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
RNG_SEED = 0

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Use a local RNG so we don't affect other random calls in the same Python session
rng = random.Random(RNG_SEED)

for class_dir in SOURCE_ROOT.iterdir():
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name
    print(f"\nProcessing class: {class_name}")

    # Collect image files, sort for deterministic ordering, then shuffle with seeded RNG
        #class_dir.iter returns an iterator over everything inside that folder (image files)
        # p.suffix -> returns file extension (jpeg which we convert to lower)
        # sorted -> forces a consistent, alphabetical order
    # So path will contain list of path objects each one pointing to an image file inside that class folder
    files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    # shuffle with fixed seed
    rng.shuffle(files)

    # Select up to N_PER_CLASS
    selected = files[:N_PER_CLASS] # will now contain path of n per class rather than all
    n = len(selected) # total number of images selected for that class
    print(f"  Using {n} images")

    # Compute split sizes (test gets the remainder to ensure totals match exactly)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val   = int(n * SPLIT_RATIOS["val"])

    # We slice arrays:
        #array[stard:end] -> train does from 0 to n_train (69), then val does 70 to 89, remainder goes to test
    splits = {
        "train": selected[:n_train],
        "val":   selected[n_train:n_train + n_val],
        "test":  selected[n_train + n_val:],
    }

    # Copy images into TARGET_ROOT/<split>/<class_name>/
    for split_name, split_files in splits.items(): # returns (train, list of train paths), (val, list of val paths)...)
        out_dir = TARGET_ROOT / split_name / class_name # create output directory
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  {split_name}: {len(split_files)} -> {out_dir}")
        for src_path in split_files: #get splitName list of file path it has and copy them
            shutil.copy2(src_path, out_dir / src_path.name)

print("\nDone. Splits written to:", TARGET_ROOT)
