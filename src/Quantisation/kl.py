# Computes KL divergence + bitflip + logs it
# Builds on Global.py - only change being the addition of KL + Bitflip metrics
# Purpose -> Check if the high-performing regimes remain broadly aligned with their FP32 counterpart (Section 4.3 results)


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
import torch.nn.functional as F
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
                     drop_val_acc: float, drop_test_acc: float, drop_val_f1: float, drop_test_f1: float,
                     kl_val: float, kl_test: float, bitflip_val: float, bitflip_test: float,
                     c2w_val: float, w2c_val: float, c2w_test: float, w2c_test: float,
                     wwflip_val: float, wwflip_test: float,
                     delta_acc_val: float, delta_acc_test: float,
                     flip_delta_val: float, flip_delta_test: float):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"Seed {seed} | ckpt={ckpt_name}\n")
        f.write(f"  FP32: VAL acc={fp32_val_acc:.6f}, TEST acc={fp32_test_acc:.6f}, VAL f1={fp32_val_f1:.6f}, TEST f1={fp32_test_f1:.6f}\n")
        f.write(f"  INT8: VAL acc={int8_val_acc:.6f}, TEST acc={int8_test_acc:.6f}, VAL f1={int8_val_f1:.6f}, TEST f1={int8_test_f1:.6f}\n")
        f.write(f"  DROP (FP32-INT8): VAL acc={drop_val_acc:.6f}, TEST acc={drop_test_acc:.6f}, VAL f1={drop_val_f1:.6f}, TEST f1={drop_test_f1:.6f}\n")
        f.write(f"  BEHAVIOR (FP32 vs INT8): VAL KL={kl_val:.6f}, TEST KL={kl_test:.6f}, VAL bitflip={bitflip_val:.6f}, TEST bitflip={bitflip_test:.6f}\n")
        f.write(f"  FLIP BREAKDOWN: VAL c→w={c2w_val:.6f}, VAL w→c={w2c_val:.6f}, TEST c→w={c2w_test:.6f}, TEST w→c={w2c_test:.6f}\n")
        f.write(f"  SANITY: VAL w→w(diff wrong labels)={wwflip_val:.6f}, TEST w→w(diff wrong labels)={wwflip_test:.6f}\n")
        f.write(f"  SANITY: Δacc(INT8-FP32) VAL={delta_acc_val:.6f}, TEST={delta_acc_test:.6f} | (w→c - c→w) VAL={flip_delta_val:.6f}, TEST={flip_delta_test:.6f}\n\n")


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


# ------------------------------- KL + Bit-flip (FP32 vs INT8) ------------------------------- #
def eval_kl_bitflip(fp32_model: nn.Module,
                    int8_model: nn.Module,
                    dataloaders: Dict[str, DataLoader],
                    split_name: str,
                    fp32_device: str,
                    int8_device: str,
                    desc: str = "Behavior"):
    # KL(p_fp32 || p_int8) averaged over samples
    # Bit-flip = fraction of samples where argmax differs between FP32 and INT8
    #
    # Flip breakdown:
    #   c→w = FP32 correct, INT8 wrong
    #   w→c = FP32 wrong, INT8 correct
    #
    # Additional sanity breakdown:
    #   w→w(diff) = FP32 wrong, INT8 wrong, but predicted label changes
    #
    # Sanity identity:
    #   Δacc(INT8-FP32) ≈ (w→c - c→w)
    #   (exact when computed on the same sample set)

    fp32_model.eval()
    int8_model.eval()

    # -------------------- Accumulators -------------------- #
    total_kl = 0.0
    total_n  = 0

    bitflips = 0

    c2w = 0       # correct → wrong
    w2c = 0       # wrong → correct
    ww_diff = 0   # wrong → wrong but different label

    # For sanity identity Δacc = w2c − c2w
    fp32_correct_count = 0
    int8_correct_count = 0

    t0 = time()
    with torch.inference_mode():
        for inputs, labels in dataloaders[split_name]:

            # Move data to appropriate devices (CPU for PTQ)
            x_fp32 = inputs.to(fp32_device)
            x_int8 = inputs.to(int8_device)

            y_fp32 = labels.to(fp32_device)
            y_int8 = labels.to(int8_device)

            # -------------------- Forward pass -------------------- #
            # Each output has shape (B, C):
            #   B = batch size
            #   C = number of classes
            #
            # Each row corresponds to one sample
            # Each column corresponds to a class logit
            out_fp32 = fp32_model(x_fp32)
            out_int8 = int8_model(x_int8)

            # -------------------- KL divergence -------------------- #
            # Convert logits to log-probabilities
            # logp = log p_fp32 (reference distribution)
            # logq = log p_int8  (quantised approximation)
            logp = F.log_softmax(out_fp32, dim=1)
            logq = F.log_softmax(out_int8, dim=1)

            # F.kl_div with reduction='none' returns a (B, C) tensor:
            #   each entry is the per-class contribution to KL
            #
            # Summing over dim=1 gives per-sample KL:
            #   KL(p_fp32 || p_int8) for each sample
            kl_batch = F.kl_div(
                logq,
                logp,
                reduction="none",
                log_target=True
            ).sum(dim=1) # kl[0] = kl divergence for image 0

            # Accumulate total KL over all samples
            total_kl += float(kl_batch.sum().cpu().item())

            # -------------------- Predictions -------------------- #
            # argmax over class dimension (dim=1)
            # gives one predicted class per sample
            pred_fp32 = torch.argmax(out_fp32, dim=1)
            pred_int8 = torch.argmax(out_int8, dim=1)

            # Bit-flip: FP32 and INT8 predict different classes
            diff = (pred_fp32 != pred_int8)
            bitflips += int(diff.sum().cpu().item())

            # Correctness flags (compare to same ground truth)
            fp32_correct = (pred_fp32 == y_fp32)
            int8_correct = (pred_int8 == y_fp32)

            fp32_correct_count += int(fp32_correct.sum().cpu().item())
            int8_correct_count += int(int8_correct.sum().cpu().item())

            # -------------------- Flip breakdown -------------------- #
            # c→w: FP32 correct, INT8 wrong
            c2w_mask = diff & fp32_correct & (~int8_correct)

            # w→c: FP32 wrong, INT8 correct
            w2c_mask = diff & (~fp32_correct) & int8_correct

            c2w += int(c2w_mask.sum().cpu().item())
            w2c += int(w2c_mask.sum().cpu().item())

            # w→w(diff): both wrong, but predicted label changes
            ww_mask = diff & (~fp32_correct) & (~int8_correct)
            ww_diff += int(ww_mask.sum().cpu().item())

            total_n += int(inputs.size(0))

    # -------------------- Final metrics -------------------- #
    elapsed = time() - t0

    mean_kl = total_kl / max(total_n, 1)
    bitflip_rate = bitflips / max(total_n, 1)

    c2w_rate = c2w / max(total_n, 1)
    w2c_rate = w2c / max(total_n, 1)
    ww_rate  = ww_diff / max(total_n, 1)

    # Sanity identity: Δacc vs (w→c − c→w)
    acc_fp32 = fp32_correct_count / max(total_n, 1)
    acc_int8 = int8_correct_count / max(total_n, 1)

    delta_acc = acc_int8 - acc_fp32
    flip_delta = w2c_rate - c2w_rate

    mismatch1 = abs(bitflip_rate - (c2w_rate + w2c_rate + ww_rate))
    mismatch2 = abs(delta_acc - flip_delta)

    # -------------------- Logging -------------------- #
    print(f"\n=== {desc} | {split_name.upper()} ===")
    print(f"Time: {elapsed:.2f} s")
    print(f"Mean KL(FP32 || INT8): {mean_kl:.6f}")
    print(f"Bit-flip rate (argmax mismatch): {bitflip_rate:.6f}")
    print(f"Flip breakdown: c→w={c2w_rate:.6f}, w→c={w2c_rate:.6f}, w→w(diff)={ww_rate:.6f}")
    print(f"Sanity: bitflip ?= c→w + w→c + w→w(diff) | mismatch={mismatch1:.8f}")
    print(f"Sanity: Δacc(INT8-FP32)={delta_acc:.6f} ?= (w→c - c→w)={flip_delta:.6f} | mismatch={mismatch2:.8f}")

    return {
        "kl": mean_kl,
        "bitflip": bitflip_rate,
        "c2w": c2w_rate,
        "w2c": w2c_rate,
        "ww": ww_rate,
        "delta_acc": delta_acc,
        "flip_delta": flip_delta,
        "mismatch1": mismatch1,
        "mismatch2": mismatch2,
        "time_s": elapsed
    }

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

    # behavior metrics
    kl_val_list, kl_test_list = [], []
    bitflip_val_list, bitflip_test_list = [], []

    # flip breakdown metrics
    c2w_val_list, w2c_val_list = [], []
    c2w_test_list, w2c_test_list = [], []

    # sanity breakdown: wrong->wrong different label flips
    ww_val_list, ww_test_list = [], []

    # sanity identity: delta acc vs (w2c-c2w)
    delta_acc_val_list, delta_acc_test_list = [], []
    flip_delta_val_list, flip_delta_test_list = [], []
    mismatch1_val_list, mismatch1_test_list = [], []
    mismatch2_val_list, mismatch2_test_list = [], []

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

            # KL divergence + bit-flip + flip breakdown + sanity checks (FP32 vs INT8) on val/test
            beh_val = eval_kl_bitflip(
                fp32_model=model_fp32,
                int8_model=model_int8,
                dataloaders=dataloaders,
                split_name="val",
                fp32_device=FP32_DEVICE,
                int8_device=PTQ_DEVICE,
                desc=f"BEHAVIOR (seed {seed})"
            )
            beh_test = eval_kl_bitflip(
                fp32_model=model_fp32,
                int8_model=model_int8,
                dataloaders=dataloaders,
                split_name="test",
                fp32_device=FP32_DEVICE,
                int8_device=PTQ_DEVICE,
                desc=f"BEHAVIOR (seed {seed})"
            )

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
                drop_val_f1=d_val_f1, drop_test_f1=d_test_f1,
                kl_val=beh_val["kl"], kl_test=beh_test["kl"],
                bitflip_val=beh_val["bitflip"], bitflip_test=beh_test["bitflip"],
                c2w_val=beh_val["c2w"], w2c_val=beh_val["w2c"],
                c2w_test=beh_test["c2w"], w2c_test=beh_test["w2c"],
                wwflip_val=beh_val["ww"], wwflip_test=beh_test["ww"],
                delta_acc_val=beh_val["delta_acc"], delta_acc_test=beh_test["delta_acc"],
                flip_delta_val=beh_val["flip_delta"], flip_delta_test=beh_test["flip_delta"]
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

            kl_val_list.append(beh_val["kl"])
            kl_test_list.append(beh_test["kl"])
            bitflip_val_list.append(beh_val["bitflip"])
            bitflip_test_list.append(beh_test["bitflip"])

            c2w_val_list.append(beh_val["c2w"])
            w2c_val_list.append(beh_val["w2c"])
            c2w_test_list.append(beh_test["c2w"])
            w2c_test_list.append(beh_test["w2c"])

            ww_val_list.append(beh_val["ww"])
            ww_test_list.append(beh_test["ww"])

            delta_acc_val_list.append(beh_val["delta_acc"])
            delta_acc_test_list.append(beh_test["delta_acc"])
            flip_delta_val_list.append(beh_val["flip_delta"])
            flip_delta_test_list.append(beh_test["flip_delta"])
            mismatch1_val_list.append(beh_val["mismatch1"])
            mismatch1_test_list.append(beh_test["mismatch1"])
            mismatch2_val_list.append(beh_val["mismatch2"])
            mismatch2_test_list.append(beh_test["mismatch2"])

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

    # overall behavior stats
    kl_vmu, kl_vsd = mean_std(kl_val_list)
    kl_tmu, kl_tsd = mean_std(kl_test_list)
    bf_vmu, bf_vsd = mean_std(bitflip_val_list)
    bf_tmu, bf_tsd = mean_std(bitflip_test_list)

    # flip breakdown stats
    c2w_vmu, c2w_vsd = mean_std(c2w_val_list)
    w2c_vmu, w2c_vsd = mean_std(w2c_val_list)
    c2w_tmu, c2w_tsd = mean_std(c2w_test_list)
    w2c_tmu, w2c_tsd = mean_std(w2c_test_list)

    # sanity breakdown: wrong->wrong different label flips
    ww_vmu, ww_vsd = mean_std(ww_val_list)
    ww_tmu, ww_tsd = mean_std(ww_test_list)

    # sanity identity: delta acc vs (w2c-c2w)
    da_vmu, da_vsd = mean_std(delta_acc_val_list)
    da_tmu, da_tsd = mean_std(delta_acc_test_list)
    fd_vmu, fd_vsd = mean_std(flip_delta_val_list)
    fd_tmu, fd_tsd = mean_std(flip_delta_test_list)
    m1_vmu, m1_vsd = mean_std(mismatch1_val_list)
    m1_tmu, m1_tsd = mean_std(mismatch1_test_list)
    m2_vmu, m2_vsd = mean_std(mismatch2_val_list)
    m2_tmu, m2_tsd = mean_std(mismatch2_test_list)

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

        f.write("BEHAVIOR (FP32 vs INT8):\n")
        f.write(f"  VAL  KL(FP32||INT8): {kl_vmu:.6f} ± {kl_vsd:.6f}\n")
        f.write(f"  TEST KL(FP32||INT8): {kl_tmu:.6f} ± {kl_tsd:.6f}\n")
        f.write(f"  VAL  bitflip rate  : {bf_vmu:.6f} ± {bf_vsd:.6f}\n")
        f.write(f"  TEST bitflip rate  : {bf_tmu:.6f} ± {bf_tsd:.6f}\n\n")

        f.write("FLIP BREAKDOWN (rates over all samples):\n")
        f.write(f"  VAL  correct→wrong : {c2w_vmu:.6f} ± {c2w_vsd:.6f}\n")
        f.write(f"  VAL  wrong→correct : {w2c_vmu:.6f} ± {w2c_vsd:.6f}\n")
        f.write(f"  TEST correct→wrong : {c2w_tmu:.6f} ± {c2w_tsd:.6f}\n")
        f.write(f"  TEST wrong→correct : {w2c_tmu:.6f} ± {w2c_tsd:.6f}\n\n")

        f.write("SANITY (rates over all samples):\n")
        f.write(f"  VAL  wrong→wrong (diff label): {ww_vmu:.6f} ± {ww_vsd:.6f}\n")
        f.write(f"  TEST wrong→wrong (diff label): {ww_tmu:.6f} ± {ww_tsd:.6f}\n")
        f.write(f"  VAL  Δacc(INT8-FP32): {da_vmu:.6f} ± {da_vsd:.6f} | (w→c - c→w): {fd_vmu:.6f} ± {fd_vsd:.6f}\n")
        f.write(f"  TEST Δacc(INT8-FP32): {da_tmu:.6f} ± {da_tsd:.6f} | (w→c - c→w): {fd_tmu:.6f} ± {fd_tsd:.6f}\n")
        f.write(f"  VAL  mismatch(bitflip vs breakdown): {m1_vmu:.8f} ± {m1_vsd:.8f}\n")
        f.write(f"  TEST mismatch(bitflip vs breakdown): {m1_tmu:.8f} ± {m1_tsd:.8f}\n")
        f.write(f"  VAL  mismatch(Δacc vs w→c-c→w): {m2_vmu:.8f} ± {m2_vsd:.8f}\n")
        f.write(f"  TEST mismatch(Δacc vs w→c-c→w): {m2_tmu:.8f} ± {m2_tsd:.8f}\n\n")

    print("\nDONE.")
    print(f"Summary: {summary_txt}")
    print(f"Runs:    {runs_root}")


if __name__ == "__main__":
    run_ptq_seed_sweep()
