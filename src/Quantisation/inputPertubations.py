# PTQ (FX Graph Mode) C×Seed sweep for corruption robustness evaluation
# Builds on Global.py - only change being the addition of corruption functions
# Applies JPEG Compression & Guassion Noise (using ImageNet-C implementation) on test set
# Evaluate INT8 accuracy at each severity level among regimes
# Purpose -> To check robustness patterns across TL Regimes (Section 4.4 results)

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

# --- Corruption Sweep Configuration --- #
# Note:
# - Corruptions applied only to test 
# Reference: Hendrycks & Dietterich (2019) - Benchmarking Neural Network Robustness to Common Corruptions and Perturbations

# Choose which corruption types to sweep
SWEEP_JPEG = True
SWEEP_NOISE = False
SWEEP_CLEAN = False  # baseline (no corruption) -> sanity check

# PlantVillage severity levels (adapted from ImageNet-C)
JPEG_SEVERITIES = [1, 2, 3, 4, 5]
NOISE_SEVERITIES = [1, 2, 3, 4, 5]

# PlantVillage corruption parameters
JPEG_QUALITY_MAP = {1: 50, 2: 40, 3: 30, 4: 20, 5: 10}
NOISE_SIGMA_MAP = {1: 0.02, 2: 0.03, 3: 0.04, 4: 0.05, 5: 0.06}

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
# Test is used for evaluation.
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


def write_condition_summary_header(path: str, condition_name: str, corruption_params: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"PTQ EVALUATION: {condition_name}\n")
        f.write(f"{corruption_params}\n")
        f.write(f"Config: Img={IMG_SIZE}, Batch={BATCH_SIZE}, CalibPerClass={CALIB_PER_CLASS}, CalibSeed={CALIB_SEED}\n")
        f.write(f"Devices: FP32={FP32_DEVICE}, INT8={PTQ_DEVICE}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Evaluation: TEST SET ONLY (validation set reserved / optional)\n")
        f.write("=" * 80 + "\n\n")


def append_seed_line(path: str,
                     seed: int,
                     ckpt_name: str,
                     fp32_test_acc: float,
                     fp32_test_f1: float,
                     int8_test_acc: float,
                     int8_test_f1: float,
                     drop_test_acc: float,
                     drop_test_f1: float):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"Seed {seed} | ckpt={ckpt_name}\n")
        f.write(f"  FP32: TEST acc={fp32_test_acc:.6f}, TEST f1={fp32_test_f1:.6f}\n")
        f.write(f"  INT8: TEST acc={int8_test_acc:.6f}, TEST f1={int8_test_f1:.6f}\n")
        f.write(f"  DROP (FP32-INT8): TEST acc={drop_test_acc:.6f}, TEST f1={drop_test_f1:.6f}\n\n")


def append_condition_summary(path: str,
                              fp32_test_acc: List[float],
                              fp32_test_f1: List[float],
                              int8_test_acc: List[float],
                              int8_test_f1: List[float],
                              drop_test_acc: List[float],
                              drop_test_f1: List[float]):
    # Compute mean and std across seeds for this condition
    fp32_tmu, fp32_tsd = mean_std(fp32_test_acc)
    fp32_tfmu, fp32_tfsd = mean_std(fp32_test_f1)

    int8_tmu, int8_tsd = mean_std(int8_test_acc)
    int8_tfmu, int8_tfsd = mean_std(int8_test_f1)

    drop_tmu, drop_tsd = mean_std(drop_test_acc)
    drop_tfmu, drop_tfsd = mean_std(drop_test_f1)

    with open(path, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("OVERALL (mean ± std across seeds)\n\n")

        f.write("FP32:\n")
        f.write(f"  TEST acc: {fp32_tmu:.6f} ± {fp32_tsd:.6f}\n")
        f.write(f"  TEST f1 : {fp32_tfmu:.6f} ± {fp32_tfsd:.6f}\n\n")

        f.write("INT8 PTQ:\n")
        f.write(f"  TEST acc: {int8_tmu:.6f} ± {int8_tsd:.6f}\n")
        f.write(f"  TEST f1 : {int8_tfmu:.6f} ± {int8_tfsd:.6f}\n\n")

        f.write("FP32 → INT8 Drops:\n")
        f.write(f"  TEST acc drop: {drop_tmu:.6f} ± {drop_tsd:.6f}\n")
        f.write(f"  TEST f1  drop: {drop_tfmu:.6f} ± {drop_tfsd:.6f}\n\n")

    return {
        "fp32_test_acc_mean": fp32_tmu, "fp32_test_acc_std": fp32_tsd,
        "fp32_test_f1_mean": fp32_tfmu, "fp32_test_f1_std": fp32_tfsd,
        "int8_test_acc_mean": int8_tmu, "int8_test_acc_std": int8_tsd,
        "int8_test_f1_mean": int8_tfmu, "int8_test_f1_std": int8_tfsd,
        "drop_test_acc_mean": drop_tmu, "drop_test_acc_std": drop_tsd,
        "drop_test_f1_mean": drop_tfmu, "drop_test_f1_std": drop_tfsd,
    }


# ------------------------------- Data ------------------------------- #
# source -> https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_64_c.py

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- Corruptions (ImageNet-C style) taken from their repo --- #
from io import BytesIO
from PIL import Image as PILImage


class JPEGCompression:
    """
    Simulates JPEG compressoin artifacts by taking a clean image, saving to an inmemory JPEG buffer at low quality, reloading it back as an image
    Implementation follows ImageNet-C (Hendrycks & Dietterich, 2019).
    """
    def __init__(self, quality: int = 25): # higher quality better
        self.quality = int(quality)
    # makes class callable so can call it inside transform pipeline
    def __call__(self, x: PILImage.Image) -> PILImage.Image:
        output = BytesIO() # creates in memory byte buffer -> no files written to disk
        x.save(output, 'JPEG', quality=self.quality) # encodes image using JPEG compression, where damage happens
        x = PILImage.open(output) # decoes image back into a PIL aimge -> what model would see after deployment if images are compressed by cameras etc
        # output.seek(0)
        return x # return corrupted image

# 
class GaussianNoise:
    """
    Adds zero-mean gaussain noise to an image -> simulates sensor noise like low-loght capture, transmission noise etc
    Implementation follows ImageNet-C (Hendrycks & Dietterich, 2019).
    """
    def __init__(self, sigma: float = 0.04):
        self.sigma = sigma # sigma = standard deviation of noise

    def __call__(self, x: PILImage.Image) -> PILImage.Image:
        x = np.array(x) / 255.# converts PIL image to NumPY array + normalises pixel values to [0,1]  as noise scale is defined in normal space
        # genertes i.i.d noise per pixel and channel and adds noise
        x = np.clip(x + np.random.normal(size=x.shape, scale=self.sigma), 0, 1) * 255 # clips prevents overflow/underflow + rescale back to 0,225
        return PILImage.fromarray(np.uint8(x)) # convert back to PIL


def get_transforms(corruption_type: str = None, corruption_param: float = None):
    """
    Build transform pipeline with optional corruption.

    Args:
        corruption_type: 'jpeg', 'noise', or None (clean)
        corruption_param: quality (for jpeg) or sigma (for noise)
    """

    tfms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]

    # Apply corruption before ToTensor (ImageNet-C style)
    if corruption_type == 'jpeg' and corruption_param is not None:
        tfms.append(JPEGCompression(quality=int(corruption_param)))
    elif corruption_type == 'noise' and corruption_param is not None:
        tfms.append(GaussianNoise(sigma=corruption_param))

    # ToTensor converts PIL [0,255] -> Tensor [0,1]
    tfms.append(transforms.ToTensor())

    # Normalization last
    tfms.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    return transforms.Compose(tfms)


def get_dataloaders(data_root: Path, corruption_type: str = None, corruption_param: float = None):
    """
    Build dataloaders with specified corruption for test.
    Train always uses clean transform (for calibration).
    """
    splits = ["train", "test"] 
    datasets_dict = {}
    loaders = {}

    for split in splits:
        # Apply corruption only to test (
        use_corruption = (split == "test") and (corruption_type is not None)

        ds = datasets.ImageFolder(
            root=str(data_root / split),
            transform=get_transforms(
                corruption_type=corruption_type if use_corruption else None,
                corruption_param=corruption_param if use_corruption else None
            ),
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

    return datasets_dict, loaders, class_names, num_classes


# ------------------------------- Model Building ------------------------------- #

def build_model(num_classes: int) -> nn.Module:
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

# Evaluate model on specific dataset split (train, test)
def evaluate_split(model, dataloaders, split_name, device, desc="Model"):
    model.eval()  # place eval mode

    # initialise lists to collect results across batches
    all_labels = []
    all_preds  = []

    t0 = time()
    with torch.inference_mode():
        # iterates over batches from selected dataset split
        for inputs, labels in dataloaders[split_name]:  # inputs = image tensors, labels = ground-truth class indices
            # move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # run forward pass, logits
            preds = torch.argmax(outputs, dim=1)  # convert logits to predicted class labels

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
    # So can discover and match 5 checkpoint files to 5 seeds
    # Expected filename format: MethodName-SeedX-... (e.g., KD-Seed0-FP32_bestVal0.9645.pth)
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


# ------------------------------- Build Conditions List ------------------------------- #

def build_conditions():
    """Build list of (condition_name, corruption_type, corruption_param) tuples."""
    conditions = []

    if SWEEP_CLEAN:
        conditions.append(("clean", None, None))

    if SWEEP_JPEG:
        for sev in JPEG_SEVERITIES:
            quality = JPEG_QUALITY_MAP[sev]
            conditions.append((f"jpeg_sev{sev}_q{quality}", "jpeg", quality))

    if SWEEP_NOISE:
        for sev in NOISE_SEVERITIES:
            sigma = NOISE_SIGMA_MAP[sev]
            conditions.append((f"noise_sev{sev}_s{sigma:.2f}", "noise", sigma))

    return conditions


# ------------------------------- Run ------------------------------- #

def run_ptq_c_seed_sweep():
    sweep_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = SAVE_DIR / f"PTQ_CxSeed_sweep_{sweep_stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Global seed for python/numpy/torch
    set_seed(0)

    # Collect checkpoints and assign to seeds
    ckpts = find_checkpoints(WEIGHTS_DIR)
    seed_to_ckpt = assign_ckpts_to_seeds(ckpts, SEEDS)

    print("="*80)
    print("PTQ C×SEED SWEEP (TEST SET ONLY)")
    print("="*80)
    print(f"Assigned checkpoints:")
    for s in SEEDS:
        print(f"  seed {s}: {seed_to_ckpt[s].name}")

    # Build conditions (all combinations of corruption type and severity)
    conditions = build_conditions()
    print(f"\nConditions to sweep: {len(conditions)}")
    for cond_name, ctype, cparam in conditions:
        print(f"  - {cond_name}")

    # ---- Choose backend (x86: fbgemm, ARM: qnnpack) ----
    engines = torch.backends.quantized.supported_engines
    backend = "fbgemm" if "fbgemm" in engines else "qnnpack"
    torch.backends.quantized.engine = backend
    print("Supported quantized engines:", engines)
    print(f"Using quantized backend: {backend}")

    print(f"\nFP32 eval device: {FP32_DEVICE}")
    print(f"PTQ/INT8 device : {PTQ_DEVICE}\n")

    # --- Setup Clean Calibration Loader (train subset, class-balanced) ---
    # This is shared across all conditions and seeds
    print("Setting up calibration loader from TRAIN (class-balanced, unaugmented)...")
    calib_train_full = datasets.ImageFolder(
        root=str(DATA_ROOT / "train"),
        transform=get_transforms(corruption_type=None, corruption_param=None)  # keep calibration CLEAN
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

    # Store overall results for cross-condition comparison
    overall_results = []

    # ==================== MAIN LOOP: C × SEED ==================== #
    for cond_idx, (cond_name, ctype, cparam) in enumerate(conditions, 1):
        print("\n" + "="*80)
        print(f"CONDITION {cond_idx}/{len(conditions)}: {cond_name}")
        print("="*80)

        # Create condition directory
        cond_dir = sweep_dir / cond_name
        cond_dir.mkdir(parents=True, exist_ok=True)

        # creates path for files to store
        summary_txt = str(cond_dir / "seeds_summary.txt")
        runs_root = cond_dir / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        # Write condition header
        corruption_desc = "No corruption (clean baseline)"
        if ctype == 'jpeg':
            corruption_desc = f"JPEG Compression: quality={cparam}"
        elif ctype == 'noise':
            corruption_desc = f"Gaussian Noise: sigma={cparam}"

        write_condition_summary_header(summary_txt, cond_name, corruption_desc)

        # Get dataloaders for this condition (train + test only, with corruption on test)
        _, dataloaders, class_names, num_classes = get_dataloaders(
            DATA_ROOT,
            corruption_type=ctype,
            corruption_param=cparam
        )

        print(f"Classes: {class_names}")
        print(f"Num classes: {num_classes}")
        print(f"Test size: {len(dataloaders['test'].dataset)}")

        if ctype is not None:
            print(f"Corruption applied to TEST: {corruption_desc}")
        else:
            print("No corruption (clean baseline)")

        # Storage for seed results (collect across all seeds in this condition)
        fp32_test_acc, fp32_test_f1 = [], []
        int8_test_acc, int8_test_f1 = [], []
        drop_test_acc, drop_test_f1 = [], []

        # ==================== SEED LOOP ==================== #
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
                print(f"PTQ RUN | CONDITION: {cond_name} | SEED: {seed}")
                print(f"ckpt: {ckpt_path.name}")
                print(f"Devices: FP32={FP32_DEVICE}, INT8={PTQ_DEVICE}")
                print(f"Calib: per_class={CALIB_PER_CLASS}, calib_seed={CALIB_SEED}, calib_size={len(calib_subset)}")
                print(f"Corruption: {corruption_desc}")
                print(f"Evaluation: TEST SET ONLY")
                print("="*80)

                seed_start = time()

                # ---- Build + load FP32 + evaluate on test --- #
                model_fp32 = build_model(num_classes)
                model_fp32 = load_weights(model_fp32, ckpt_path, device="cpu")
                model_fp32 = model_fp32.to(FP32_DEVICE).eval()

                fp32_test = evaluate_split(model_fp32, dataloaders, "test", device=FP32_DEVICE,
                                           desc=f"FP32 (seed {seed})")

                # ---- INT8 PTQ  ----
                model_int8 = quantize_model_fx(
                    model_fp32=model_fp32.to("cpu").eval(),  # move model to cpu on eval mode
                    img_size=IMG_SIZE,
                    calib_loader=calib_loader,  # calib data to load forward pass to collect statistics
                    backend=backend,
                    calib_batches=CALIB_BATCHES,
                )

                # INT8 model runs on CPU
                int8_test = evaluate_split(model_int8, dataloaders, "test", device=PTQ_DEVICE,
                                           desc=f"INT8 PTQ (seed {seed})")

                # Drops (FP32 - INT8)
                d_test_acc = fp32_test["acc"] - int8_test["acc"]
                d_test_f1 = fp32_test["f1_macro"] - int8_test["f1_macro"]

                seed_elapsed = time() - seed_start
                print(f"\nSeed runtime: {seed_elapsed:.2f} s")

                # Append to human-readable summary for summary_text
                append_seed_line(
                    summary_txt,
                    seed=seed,
                    ckpt_name=ckpt_path.name,
                    fp32_test_acc=fp32_test["acc"],
                    fp32_test_f1=fp32_test["f1_macro"],
                    int8_test_acc=int8_test["acc"],
                    int8_test_f1=int8_test["f1_macro"],
                    drop_test_acc=d_test_acc,
                    drop_test_f1=d_test_f1
                )

                # Store for overall stats -> allows for mean, std across seeds
                fp32_test_acc.append(fp32_test["acc"])
                fp32_test_f1.append(fp32_test["f1_macro"])

                int8_test_acc.append(int8_test["acc"])
                int8_test_f1.append(int8_test["f1_macro"])

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

        # ------- CONDITION SUMMARY ------ #
        # After all seeds complete for this condition, compute aggregated stats
        cond_stats = append_condition_summary(
            summary_txt,
            fp32_test_acc, fp32_test_f1,
            int8_test_acc, int8_test_f1,
            drop_test_acc, drop_test_f1
        )

        # Store for overall comparison across all conditions
        overall_results.append({
            "condition": cond_name,
            "type": ctype,
            "param": cparam,
            **cond_stats
        })

        print(f"\n✓ Condition '{cond_name}' complete")
        print(f"  Summary: {summary_txt}")

    # ------- OVERALL SUMMARY --------- #
    # After all conditions complete, write cross-condition comparison
    overall_summary_path = sweep_dir / "overall_summary.txt"
    with open(overall_summary_path, "w", encoding="utf-8") as f:
        f.write("PTQ C×SEED SWEEP - OVERALL SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total conditions: {len(conditions)}\n")
        f.write(f"Seeds per condition: {len(SEEDS)}\n")
        f.write(f"Calibration: {CALIB_PER_CLASS}/class, seed={CALIB_SEED}\n")
        f.write(f"Evaluation: TEST SET ONLY (validation set reserved / optional)\n\n")

        f.write("-"*80 + "\n")
        f.write(f"{'Condition':<30} | FP32 Test Acc   | INT8 Test Acc   | Acc Drop\n")
        f.write("-"*80 + "\n")

        for res in overall_results:
            f.write(f"{res['condition']:<30} | "
                   f"{res['fp32_test_acc_mean']:.4f}±{res['fp32_test_acc_std']:.4f} | "
                   f"{res['int8_test_acc_mean']:.4f}±{res['int8_test_acc_std']:.4f} | "
                   f"{res['drop_test_acc_mean']:.4f}±{res['drop_test_acc_std']:.4f}\n")

        f.write("-"*80 + "\n\n")

        f.write("F1 Score Summary:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Condition':<30} | FP32 Test F1    | INT8 Test F1    | F1 Drop\n")
        f.write("-"*80 + "\n")

        for res in overall_results:
            f.write(f"{res['condition']:<30} | "
                   f"{res['fp32_test_f1_mean']:.4f}±{res['fp32_test_f1_std']:.4f} | "
                   f"{res['int8_test_f1_mean']:.4f}±{res['int8_test_f1_std']:.4f} | "
                   f"{res['drop_test_f1_mean']:.4f}±{res['drop_test_f1_std']:.4f}\n")

        f.write("-"*80 + "\n")

    print("\n" + "="*80)
    print("PTQ C×SEED SWEEP COMPLETE")
    print("="*80)
    print(f"Results directory: {sweep_dir}")
    print(f"Overall summary: {overall_summary_path}")
    print("\nResults structure:")
    print(f"  {sweep_dir}/")
    print(f"    overall_summary.txt  ← Compare all conditions")
    print(f"    clean/")
    print(f"      seeds_summary.txt  ← Stats for clean condition")
    print(f"      runs/seed0/, seed2/, ...")
    print(f"    noise_sev1_s0.03/")
    print(f"      seeds_summary.txt")
    print(f"      runs/...")
    print(f"    ...")


if __name__ == "__main__":
    run_ptq_c_seed_sweep()
