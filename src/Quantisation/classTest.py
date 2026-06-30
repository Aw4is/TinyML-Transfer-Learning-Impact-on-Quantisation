# Computes Per-Class test results 
# Builds on Global.py - only change being the addition of class-level evaluation
# Purpose -> Check if TL Regimes broadly find the same classes hard

import os
import sys
import re
import copy
import random
from pathlib import Path
from time import time
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.ao.quantization.observer import ObserverBase


from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


from src.MobileNetV2.mobilenetv2 import MobileNetV2



# --- Config --- #
DATA_ROOT    = Path(r"")
IMG_SIZE     = 96
BATCH_SIZE   = 50
NUM_WORKERS  = 0
WIDTH_MULT   = 0.35

WEIGHTS_DIR  = Path(r"")

DROPOUT_P    = 0.5

SEEDS        = [0, 2, 5, 42, 1337]

# PTQ settings
# Note: Calibration uses train only, sampled class-balanced (N images per class).
# Val/Test are used entirely for evaluation (no leakage).
CALIB_PER_CLASS = 15
CALIB_SEED      = 0     # fixed so every method/checkpoint sees the same calib subset
CALIB_BATCHES   = None  # None = run through entire calibration subset

# Devices:
# - Evaluate FP32 on CPU so FP32 -> INT8 drops are backend-matched (CPU vs CPU)
# - Quantization (FX PTQ) runs on CPU (required for PTQ/quantized kernels)
FP32_DEVICE   = "cpu"
PTQ_DEVICE    = "cpu"

# Root folder for logs
SAVE_DIR = Path(r"")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------- Reproducability ------------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------- Logging ------------------------------- #

# Lines store original terminal output streams
ORIG_STDOUT = sys.stdout
ORIG_STDERR = sys.stderr

# Class that accepts multiple file-like objects
    # Duplicates everything printed to terminal to a file
    # Takes terminal and log file as input
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    # Whenever something is printed, it is written to all streams an flushed immediately for real-time logs
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
            # Prevents output buffering issues during long runs
    def flush(self):
        for s in self.streams:
            s.flush()

# Compute mean and standard deviation...
def mean_std(x):
    x = np.array(x, dtype=np.float64)
    # edge case handling
        # 0 seeds => run 0.0,0.0
        # 1 seed => return mean but not std
    if len(x) <= 1:
        return float(x.mean()) if len(x) == 1 else 0.0, 0.0
    return float(x.mean()), float(x.std(ddof=1)) # ddof = 1 for sample std


def write_summary_header(path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("PTQ SEED SWEEP SUMMARY\n")
        f.write(f"Config: Img={IMG_SIZE}, Batch={BATCH_SIZE}, CalibPerClass={CALIB_PER_CLASS}, CalibSeed={CALIB_SEED}\n")
        f.write(f"Devices: FP32={FP32_DEVICE}, INT8={PTQ_DEVICE}\n")
        f.write(f"Weights dir: {WEIGHTS_DIR}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write("=" * 80 + "\n\n")


def append_seed_line(path: str,
                     seed: int,
                     ckpt_name: str,
                     fp32_val_acc: float, fp32_test_acc: float, fp32_val_f1: float, fp32_test_f1: float,
                     int8_val_acc: float, int8_test_acc: float, int8_val_f1: float, int8_test_f1: float,
                     drop_val_acc: float, drop_test_acc: float, drop_val_f1: float, drop_test_f1: float):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"Seed {seed} | ckpt={ckpt_name}\n")
        f.write(f"  FP32: VAL acc={fp32_val_acc:.6f}, TEST acc={fp32_test_acc:.6f}, VAL f1={fp32_val_f1:.6f}, TEST f1={fp32_test_f1:.6f}\n")
        f.write(f"  INT8: VAL acc={int8_val_acc:.6f}, TEST acc={int8_test_acc:.6f}, VAL f1={int8_val_f1:.6f}, TEST f1={int8_test_f1:.6f}\n")
        f.write(f"  DROP (FP32-INT8): VAL acc={drop_val_acc:.6f}, TEST acc={drop_test_acc:.6f}, VAL f1={drop_val_f1:.6f}, TEST f1={drop_test_f1:.6f}\n\n")


# ------------------------------- Data ------------------------------- #

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms():
    # Keep everything deterministic (Resize + Normalize)
    # Train augmentations are not used here because:
    #   - this script is evaluation + PTQ calibration only
    #   - calibration needs inference-style data
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])



def get_dataloaders(data_root: Path):
    splits = ["train", "val", "test"]
    datasets_dict = {}
    loaders = {}

    for split in splits:
        ds = datasets.ImageFolder(
            root=str(data_root / split),
            transform=get_transforms(),
        )
        datasets_dict[split] = ds
        loaders[split] = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,  # deterministic evaluation (we are not training here)
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

    class_names = datasets_dict["train"].classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Num classes:", num_classes)
    print("Sizes:", {s: len(datasets_dict[s]) for s in splits})

    return datasets_dict, loaders, class_names, num_classes


# ------------------------------- Model Building ------------------------------- #

def build_model(num_classes: int) -> nn.Module:
   # Match training model

    base = MobileNetV2(num_classes=1000, width_mult=WIDTH_MULT)
    in_features = base.classifier.in_features
    base.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, num_classes),
    )
    return base


def load_weights(model: nn.Module, weights_path: Path, device: str):
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model


# ------------------------------- Eval ------------------------------- #

# Evaluate model on specific dataset split (train,val,test...)
def evaluate_split(model, dataloaders, split_name, device, desc="Model"):
    model.eval() # place eval mode

    # initalise lists to collect results across batches
    all_labels = []
    all_preds  = []

    t0 = time()
    with torch.inference_mode():
        # iterates over batches from selected dataset split
        for inputs, labels in dataloaders[split_name]: # inputs = image tensors, labels = ground-truth class indidices
            # move data to CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs) # run forward passs, logits
            preds = torch.argmax(outputs, dim=1) # convert logits to predicted class labels

            # Store prediction and labels on CPU
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    # Merge batch tensors into one long vector
    all_labels = torch.cat(all_labels).numpy()
    all_preds  = torch.cat(all_preds).numpy()

    # Compute acc, f1
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    elapsed = time() - t0
    print(f"\n=== {desc} | {split_name.upper()} ===")
    print(f"Time: {elapsed:.2f} s")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Macro F1: {f1:.4f}")

    return {"acc": acc, "f1_macro": f1, "time_s": elapsed}


def evaluate_split_per_class_acc(model, dataloaders, split_name, device, num_classes: int):
    model.eval()

    # Create two counters -> how many samples truly belong to class c
        #correct[c] -> how many of those were predicted correctly
    correct = np.zeros((num_classes,), dtype=np.int64)
    total   = np.zeros((num_classes,), dtype=np.int64)

    with torch.inference_mode():
        for inputs, labels in dataloaders[split_name]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # run forward pass to get logits
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1) # get predicted class index per sample

            # Move to CPU numpy for counting
            y = labels.detach().cpu().numpy()
            p = preds.detach().cpu().numpy()

            # for each class c, mask selects samples in this batch whose true label is c
            for c in range(num_classes):
                mask = (y == c)
                if mask.any():
                    total[c] += int(mask.sum())  # how many such samples are present in this patch add to total[c]
                    correct[c] += int((p[mask] == c).sum()) # checks which of those were prediced as c -> counts correct ones -> add to correct[c]

    # avoid divide-by-zero (shouldn't happen if split has all classes)
    acc = np.zeros((num_classes,), dtype=np.float64)
    for c in range(num_classes):
        acc[c] = (correct[c] / total[c]) if total[c] > 0 else 0.0 # compute accuracy

    return acc, total #acc[c] = accuracy for class c, total[c] = total samples of class c in that split


# ------------------------------- Calibration Subset ------------------------------- #

def make_class_balanced_subset(ds: datasets.ImageFolder, per_class: int, seed: int) -> Subset:
    # Builds deterministic class set for reproducability
    rng = random.Random(seed)

    targets = ds.targets  # list[int]
    num_classes = len(ds.classes)

    # Collect indices per class
    class_to_indices = {c: [] for c in range(num_classes)}
    for idx, y in enumerate(targets):
        class_to_indices[y].append(idx)

    # Sample per class deterministically
    chosen = []
    for c in range(num_classes):
        idxs = class_to_indices[c]
        rng.shuffle(idxs)
        take = min(per_class, len(idxs))
        chosen.extend(idxs[:take])

    # Shuffle final list deterministically so batches are mixed
    rng.shuffle(chosen)

    return Subset(ds, chosen)



# ------------------------------- PTQ Quantise ------------------------------- #




def quantize_model_fx(model_fp32: nn.Module,
                      img_size: int,
                      calib_loader: DataLoader,
                      backend: str,
                      calib_batches: int = None) -> nn.Module:
    # Work on a copy so keep the FP32 model untouched
    model_fp32 = copy.deepcopy(model_fp32).to("cpu").eval()

    qconfig_mapping = get_default_qconfig_mapping(backend)

    example_inputs = (torch.randn(1, 3, img_size, img_size),)

    # Prepare inserts observers + does fusion where applicable
    prepared = prepare_fx(model_fp32, qconfig_mapping, example_inputs)



    prepared.eval()
    print("Prepared model for FX PTQ. Calibrating...")

    # Calibration: run representative data through prepared model
    with torch.inference_mode():
        for i, (x, _) in enumerate(calib_loader):
            prepared(x.cpu())
            if calib_batches is not None and (i + 1) >= calib_batches:
                break

    # Convert to quantized model
    quantized = convert_fx(prepared)

    # check_quantized_types(quantized)


    return quantized



# ------------------------------- Helpers ------------------------------- #
    #So can discover and match 5 checkpoint files to 5 seeds
    #Expected filename format: MethodName-SeedX-... (e.g., KD-Seed0-FP32_bestVal0.9645.pth)
def find_checkpoints(weights_dir: Path) -> List[Path]:
    ckpts = sorted(list(weights_dir.glob("*.pth")))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No .pth files found in: {weights_dir}")
    return ckpts


def parse_seed_from_name(path: Path) -> int:
    # matches: Method-Seed0-..., method_seed2..., etc.
    m = re.search(r"(seed)[\s_\-]*([0-9]+)", path.name, flags=re.IGNORECASE)
    if m:
        return int(m.group(2))
    return -1


def assign_ckpts_to_seeds(ckpts: List[Path], seeds: List[int]) -> Dict[int, Path]:
    # Simple mapping: exactly one checkpoint per seed must exist
    mapping: Dict[int, Path] = {}

    for p in ckpts:
        s = parse_seed_from_name(p)
        if s in seeds:
            if s in mapping:
                raise RuntimeError(f"Duplicate checkpoints for seed {s}: {mapping[s].name} and {p.name}")
            mapping[s] = p

    missing = [s for s in seeds if s not in mapping]
    if len(missing) > 0:
        raise RuntimeError(
            f"Missing checkpoints for seeds: {missing}. "
            f"Expected filenames like MethodName-SeedX-... (e.g., KD-Seed0-FP32_bestVal0.9645.pth)"
        )

    return mapping


# ------------------------------- Run ------------------------------- #

def run_ptq_seed_sweep():
    sweep_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = SAVE_DIR / f"PTQ_seed_sweep_{sweep_stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # creates path for files to store
    summary_txt = str(sweep_dir / "seeds_summary.txt")
    runs_root   = sweep_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    write_summary_header(summary_txt)

    # Global seed for python/numpy/torch
    set_seed(0)

    # Build dataloaders
    image_datasets, dataloaders, class_names, num_classes = get_dataloaders(DATA_ROOT)

    # Precompute TEST class counts once (for reporting)
    test_targets = np.array(image_datasets["test"].targets, dtype=np.int64)
    test_class_counts = np.bincount(test_targets, minlength=num_classes)

    # ---- Choose backend (x86: fbgemm, ARM: qnnpack) ----
    engines = torch.backends.quantized.supported_engines
    backend = "fbgemm" if "fbgemm" in engines else "qnnpack"
    torch.backends.quantized.engine = backend
    print("Supported quantized engines:", engines)
    print("Using quantized backend:", backend)

    print(f"\nFP32 eval device: {FP32_DEVICE}")
    print(f"PTQ/INT8 device : {PTQ_DEVICE}\n")

    # --- Setup Clean Calibration Loader (train subset, class-balanced) ---
    print("Setting up calibration loader from TRAIN (class-balanced, unaugmented)...")
    calib_train_full = datasets.ImageFolder(
        root=str(DATA_ROOT / "train"),
        transform=get_transforms()
    )

    calib_subset = make_class_balanced_subset(
        ds=calib_train_full,
        per_class=CALIB_PER_CLASS,
        seed=CALIB_SEED
    )

    calib_loader = DataLoader(
        calib_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # deterministic; subset already mixed deterministically
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Calibration subset: {len(calib_subset)} images "
          f"({CALIB_PER_CLASS} per class, seed={CALIB_SEED}).")

    # Collect checkpoints and assign to seeds
    ckpts = find_checkpoints(WEIGHTS_DIR)
    seed_to_ckpt = assign_ckpts_to_seeds(ckpts, SEEDS)

    print("\nAssigned checkpoints:")
    for s in SEEDS:
        print(f"  seed {s}: {seed_to_ckpt[s].name}")

    # Store arrays for summary stats
    fp32_val_acc, fp32_val_f1, fp32_test_acc, fp32_test_f1 = [], [], [], []
    int8_val_acc, int8_val_f1, int8_test_acc, int8_test_f1 = [], [], [], []
    drop_val_acc, drop_val_f1, drop_test_acc, drop_test_f1 = [], [], [], []

    # Store INT8 TEST per-class accuracies per seed (for mean±std across seeds)
    int8_test_perclass_all_seeds = []

    for seed in SEEDS:
        ckpt_path = seed_to_ckpt[seed]

        run_tag = f"seed{seed}"
        run_dir = runs_root / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        log_txt = run_dir / "train_log.txt"
        _log_fh = open(log_txt, "w", encoding="utf-8")
        sys.stdout = Tee(ORIG_STDOUT, _log_fh)
        sys.stderr = Tee(ORIG_STDERR, _log_fh)

        try:
            # write basic header per run
            print("\n" + "="*80)
            print(f"PTQ RUN | seed={seed}")
            print(f"ckpt: {ckpt_path.name}")
            print(f"Devices: FP32={FP32_DEVICE}, INT8={PTQ_DEVICE}")
            print(f"Calib: per_class={CALIB_PER_CLASS}, calib_seed={CALIB_SEED}, calib_size={len(calib_subset)}")
            print("="*80)

            seed_start = time()

            # ---- Build + load FP32 + evaluate on val,test --- #
            model_fp32 = build_model(num_classes)
            model_fp32 = load_weights(model_fp32, ckpt_path, device="cpu")
            model_fp32 = model_fp32.to(FP32_DEVICE).eval()

            fp32_val  = evaluate_split(model_fp32, dataloaders, "val",  device=FP32_DEVICE, desc=f"FP32 (seed {seed})")
            fp32_test = evaluate_split(model_fp32, dataloaders, "test", device=FP32_DEVICE, desc=f"FP32 (seed {seed})")

            # ---- INT8 PTQ  ----
            model_int8 = quantize_model_fx(
                model_fp32=model_fp32.to("cpu").eval(), # move model to vpu on eval mode
                img_size=IMG_SIZE,
                calib_loader=calib_loader, # calib data to load forward pass to collect statistics
                backend=backend,
                calib_batches=CALIB_BATCHES,
            )

            # INT8 model runs on CPU
            int8_val  = evaluate_split(model_int8, dataloaders, "val",  device=PTQ_DEVICE, desc=f"INT8 PTQ (seed {seed})")
            int8_test = evaluate_split(model_int8, dataloaders, "test", device=PTQ_DEVICE, desc=f"INT8 PTQ (seed {seed})")

            # INT8 TEST per-class accuracy (store; no need to print per seed)
            perclass_acc, _perclass_counts = evaluate_split_per_class_acc(
                model_int8,
                dataloaders,
                "test",
                device=PTQ_DEVICE,
                num_classes=num_classes
            )
            int8_test_perclass_all_seeds.append(perclass_acc)

            # Drops (FP32 - INT8)
            d_val_acc  = fp32_val["acc"] - int8_val["acc"]
            d_val_f1   = fp32_val["f1_macro"] - int8_val["f1_macro"]
            d_test_acc = fp32_test["acc"] - int8_test["acc"]
            d_test_f1  = fp32_test["f1_macro"] - int8_test["f1_macro"]

            seed_elapsed = time() - seed_start
            print(f"\nSeed runtime: {seed_elapsed:.2f} s")

            # Append to human-readable summary for summary_text
            append_seed_line(
                summary_txt,
                seed=seed,
                ckpt_name=ckpt_path.name,
                fp32_val_acc=fp32_val["acc"], fp32_test_acc=fp32_test["acc"],
                fp32_val_f1=fp32_val["f1_macro"], fp32_test_f1=fp32_test["f1_macro"],
                int8_val_acc=int8_val["acc"], int8_test_acc=int8_test["acc"],
                int8_val_f1=int8_val["f1_macro"], int8_test_f1=int8_test["f1_macro"],
                drop_val_acc=d_val_acc, drop_test_acc=d_test_acc,
                drop_val_f1=d_val_f1, drop_test_f1=d_test_f1
            )

            # Store for overall stats -> allows for mean,stf across seeds
            fp32_val_acc.append(fp32_val["acc"])
            fp32_val_f1.append(fp32_val["f1_macro"])
            fp32_test_acc.append(fp32_test["acc"])
            fp32_test_f1.append(fp32_test["f1_macro"])

            int8_val_acc.append(int8_val["acc"])
            int8_val_f1.append(int8_val["f1_macro"])
            int8_test_acc.append(int8_test["acc"])
            int8_test_f1.append(int8_test["f1_macro"])

            drop_val_acc.append(d_val_acc)
            drop_val_f1.append(d_val_f1)
            drop_test_acc.append(d_test_acc)
            drop_test_f1.append(d_test_f1)

        finally:
            # Restore stdout/stderr and close per-seed log -> avoids seed mixing
            sys.stdout = ORIG_STDOUT
            sys.stderr = ORIG_STDERR
            _log_fh.close()

            # Cleanup
            try:
                del model_fp32
            except Exception:
                pass
            try:
                del model_int8
            except Exception:
                pass
            torch.cuda.empty_cache()

    # ---- Overall mean±std after completing loop----
    fp32_vmu, fp32_vsd = mean_std(fp32_val_acc)
    fp32_tmu, fp32_tsd = mean_std(fp32_test_acc)
    fp32_vfmu, fp32_vfsd = mean_std(fp32_val_f1)
    fp32_tfmu, fp32_tfsd = mean_std(fp32_test_f1)

    int8_vmu, int8_vsd = mean_std(int8_val_acc)
    int8_tmu, int8_tsd = mean_std(int8_test_acc)
    int8_vfmu, int8_vfsd = mean_std(int8_val_f1)
    int8_tfmu, int8_tfsd = mean_std(int8_test_f1)

    drop_vmu, drop_vsd = mean_std(drop_val_acc)
    drop_tmu, drop_tsd = mean_std(drop_test_acc)
    drop_vfmu, drop_vfsd = mean_std(drop_val_f1)
    drop_tfmu, drop_tfsd = mean_std(drop_test_f1)

    # INT8 TEST per-class mean±std across seeds
    int8_test_perclass_all_seeds = np.array(int8_test_perclass_all_seeds, dtype=np.float64)  # [num_seeds, num_classes]
    perclass_mean = int8_test_perclass_all_seeds.mean(axis=0)
    if int8_test_perclass_all_seeds.shape[0] > 1:
        perclass_std = int8_test_perclass_all_seeds.std(axis=0, ddof=1)
    else:
        perclass_std = np.zeros_like(perclass_mean)

    sort_idx = np.argsort(perclass_mean)  # worst -> best

    # Write to summary file for overall mean
    with open(summary_txt, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("OVERALL (mean ± std across seeds)\n\n")

        f.write("FP32:\n")
        f.write(f"  VAL  acc: {fp32_vmu:.6f} ± {fp32_vsd:.6f}\n")
        f.write(f"  TEST acc: {fp32_tmu:.6f} ± {fp32_tsd:.6f}\n")
        f.write(f"  VAL  f1 : {fp32_vfmu:.6f} ± {fp32_vfsd:.6f}\n")
        f.write(f"  TEST f1 : {fp32_tfmu:.6f} ± {fp32_tfsd:.6f}\n\n")

        f.write("INT8 PTQ:\n")
        f.write(f"  VAL  acc: {int8_vmu:.6f} ± {int8_vsd:.6f}\n")
        f.write(f"  TEST acc: {int8_tmu:.6f} ± {int8_tsd:.6f}\n")
        f.write(f"  VAL  f1 : {int8_vfmu:.6f} ± {int8_vfsd:.6f}\n")
        f.write(f"  TEST f1 : {int8_tfmu:.6f} ± {int8_tfsd:.6f}\n\n")

        f.write("FP32 → INT8 Drops (FP32 - INT8):\n")
        f.write(f"  VAL  acc drop: {drop_vmu:.6f} ± {drop_vsd:.6f}\n")
        f.write(f"  TEST acc drop: {drop_tmu:.6f} ± {drop_tsd:.6f}\n")
        f.write(f"  VAL  f1  drop: {drop_vfmu:.6f} ± {drop_vfsd:.6f}\n")
        f.write(f"  TEST f1  drop: {drop_tfmu:.6f} ± {drop_tfsd:.6f}\n\n")

        # ---- INT8 TEST per-class accuracy (sorted worst->best) ----
        f.write("=" * 80 + "\n")
        f.write("INT8 TEST per-class accuracy (mean ± std across seeds), sorted worst → best\n")
        f.write(f"(Computed on TEST split; N_seeds={len(SEEDS)})\n\n")
        f.write(f"{'rank':>4}  {'class':<45}  {'N':>6}  {'mean_acc(%)':>12}  {'std(%)':>10}\n")
        f.write("-" * 80 + "\n")

        for r, c in enumerate(sort_idx, start=1):
            cname = class_names[c]
            n_c = int(test_class_counts[c]) if c < len(test_class_counts) else 0
            mu = perclass_mean[c] * 100.0
            sd = perclass_std[c] * 100.0
            f.write(f"{r:>4}  {cname:<45}  {n_c:>6}  {mu:>12.2f}  {sd:>10.2f}\n")

        f.write("\n")

    print("\nDONE.")
    print(f"Summary: {summary_txt}")
    print(f"Runs:    {runs_root}")


if __name__ == "__main__":
    run_ptq_seed_sweep()