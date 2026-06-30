# =============================================================================
# PTQ (FX Graph Mode) Evaluation: ImageNet Pretrained (ResNet18 OR MobileNetV2)
# - EVAL: HuggingFace ImageNet validation split (streaming -> materialize)
# - CALIB: HuggingFace ImageNet train split, using ONLY CALIB_SAMPLES (streaming -> materialize)
#
# Minimal change from original script:
#    Added a train-streaming dataset for calibration only
#
# MobileNetV2 mode:
#   If USE_MNV2=True, uses MobileNetV2 +  weights
#   Adds minimal HF streaming shuffle(buffer_size=..., seed=...) for calibration only 
# =============================================================================

import os
import sys
import copy
import random
from pathlib import Path
from time import time
from typing import Dict, Optional, List, Any
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from datasets import load_dataset

from sklearn.metrics import f1_score

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

import warnings
warnings.filterwarnings("ignore", message="Please use quant_min and quant_max")


# ------------------------------- Config ------------------------------- #

# ---- Toggle model here ----
USE_MNV2 = True  # False = ResNet18 (torchvision pretrained), True = our MobileNetV2

# HuggingFace ImageNet cache (local path)
HF_CACHE_DIR = Path(r"imagenetcachepath")
HF_DATASET_ID = "ILSVRC/imagenet-1k"

# ImageNet settings
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 for Windows compatibility

# PTQ settings
CALIB_SAMPLES = 2048      # Number of samples for PTQ calibration (from TRAIN split)
EVAL_SAMPLES  = None      # None = use full validation set (~50K val), or set e.g. 5000 for faster eval
CALIB_SEED    = 0         # Seed used for HF shuffle (only if USE_MNV2=True)

# --- Only applied when USE_MNV2=True: minimal shuffle config for train streaming calibration ---
CALIB_SHUFFLE  = True
SHUFFLE_BUFFER = 10000    # training data to hold in buffer

# Devices
FP32_DEVICE = "cpu"
PTQ_DEVICE  = "cpu"

# Where to save logs
SAVE_DIR = Path(r"savelogspath")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------- our MNV2 MODEL (only used if USE_MNV2=True) ------------------------------- #
from src.MobileNetV2.mobilenetv2 import MobileNetV2

MNV2_Weights = Path(r"mnv2ImageNetWeights")
WIDTH_MULT  = 1.00


# ------------------------------- Reproducibility ------------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------- Logging ------------------------------- #

ORIG_STDOUT = sys.stdout
ORIG_STDERR = sys.stderr

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


# ------------------------------- Data (HuggingFace ImageNet - STREAMING) ------------------------------- #

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_imagenet_transforms():
    """Standard ImageNet validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# Follows https://huggingface.co/docs/datasets/stream
class HFImageNetStreamDataset(Dataset):
    """
    HuggingFace streaming ImageNet-1K split ("train" or "validation").
    Materializes only max_samples into a list so __len__/__getitem__ work.
      - This does not download the full split.
      - It streams sequentially and stores only the first max_samples items in memory.

      - HF streaming shuffle(buffer_size=..., seed=...).
        Larger buffer -> closer to a true shuffle, still without downloading full train split.
    """
    def __init__(
        self,
        cache_dir: Path,                    # where we store imagenet cache (faster loading)
        split: str,                         # "train" or "validation"
        transform=None,                     # to transform/augment images
        max_samples: Optional[int] = None,  # max samples to materialize from stream
        shuffle: bool = False,
        shuffle_buffer: int = 10_000,
        shuffle_seed: int = 0,
    ):
        super().__init__()
        self.transform = transform  # saves transform so can be applied later here

        stream_ds = load_dataset(
            HF_DATASET_ID,        # ImageNet dataset id
            split=split,          # load requested split
            streaming=True,       # data not fully downloaded, samples fetched on demand
            cache_dir=str(cache_dir),
        )

        if shuffle:
            stream_ds = stream_ds.shuffle(buffer_size=int(shuffle_buffer), seed=int(shuffle_seed))
            print(f"[HF] Streaming split='{split}'; shuffle=True buffer={shuffle_buffer} seed={shuffle_seed}")
        else:
            print(f"[HF] Streaming split='{split}'; shuffle=False")

        if max_samples is None:
            raise ValueError("max_samples must be provided for streaming->materialize dataset.")

        # set limit to load (we materialize only what we need)
        num_to_load = int(max_samples)
        print(f"[HF] Materializing {num_to_load} samples...")

        # create datalist by looping through streaming dataset
        data_list: List[Dict[str, Any]] = []  # in-memory list to store image + label
        for i, ex in enumerate(stream_ds):    # each ex is a dict that has image and label which we store
            if i >= num_to_load:              # stop after max_samples
                break
            data_list.append({"image": ex["image"], "label": ex["label"]})  # keep image,label from stream
            if (i + 1) % 5000 == 0:           # progress update every 5k samples
                print(f"  [{i+1}/{num_to_load}] loaded...")

        self.data_list = data_list
        print(f"[HF] Materialized {len(self.data_list)} samples from split='{split}'")

    def __len__(self):
        return len(self.data_list)  # sanity check for length of samples

    # defines how we fetch one sample by index
    def __getitem__(self, idx):
        ex = self.data_list[idx]              # retrieves idx-th element dict with image and label
        img = ex["image"].convert("RGB")      # ensure 3 channels, consistent colour space
        label = int(ex["label"])              # convert label to python int
        if self.transform:
            img = self.transform(img)         # apply transformation pipeline here
        return img, label                     # return the image and label


def get_imagenet_val_loader(
    cache_dir: Path,
    max_samples: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """
    Validation loader.
    If max_samples is None -> materialize 50k validation.
    """
    n = 50_000 if max_samples is None else int(max_samples)
    ds = HFImageNetStreamDataset(
        cache_dir=cache_dir,
        split="validation",
        transform=get_imagenet_transforms(),
        max_samples=n,
        shuffle=False,  # keep eval deterministic / ordered
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), # copy batch from CPU to GPU if avaialable, can be faster
    )


def make_calibration_subset_from_train(
    cache_dir: Path,
    num_samples: int,
    seed: int,
    shuffle: bool,
    shuffle_buffer: int,
) -> DataLoader:
    """
    calibration subset from streaming.

    ResNet18 mode:
      - shuffle is forced OFF (streaming first-N) -> consistent with PyTorch implementation

    MobileNetV2 mode:
      - optional shuffle via HF stream shuffle(buffer_size=..., seed=...)
      - still materializes only num_samples
    """
    print(f"[Calib] Using TRAIN split for calibration: {num_samples} samples (seed={seed})")
    print(f"[Calib] shuffle={shuffle} buffer={shuffle_buffer}")

    ds = HFImageNetStreamDataset(
        cache_dir=cache_dir,
        split="train",
        transform=get_imagenet_transforms(),
        max_samples=num_samples,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        shuffle_seed=seed,
    )

    print(f"[Calib] Calibration dataset ready: {len(ds)} samples (TRAIN split)")

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


# ------------------------------- Model ------------------------------- #

def build_model_fp32() -> nn.Module:
    """
    Load either:
      - ResNet18 with ImageNet pretrained weights from torchvision (USE_MNV2=False), OR
      - MobileNetV2 with pretrained weights (USE_MNV2=True).
    """
    if not USE_MNV2:
        print("[Model] Loading ResNet18 with ImageNet pretrained weights...")
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[Model] Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        return model

    print(f"[Model] Loading our MobileNetV2 (width_mult={WIDTH_MULT:.2f}) ImageNet pretrained weights...")
    model = MobileNetV2(num_classes=1000, width_mult=WIDTH_MULT)

    state = torch.load(MNV2_Weights, map_location="cpu")
    model.load_state_dict(state, strict=True)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"[Model] Weights: {MNV2_Weights}")
    return model


# ------------------------------- Evaluation ------------------------------- #

def evaluate_with_top5(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    desc: str = "Model",
) -> Dict:
    """Evaluate model with Top-1 and Top-5 accuracy."""
    model.eval()
    model.to(device)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    all_preds = []
    all_labels = []

    t0 = time()
    with torch.inference_mode():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Top-1
            preds = torch.argmax(outputs, dim=1)
            correct_top1 += (preds == labels).sum().item()

            # Top-5 -> is the true label anywhere in the top-5?
            _, top5_preds = outputs.topk(5, dim=1)
            correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            if (batch_idx + 1) % 50 == 0:
                print(f"  [{desc}] Batch {batch_idx + 1}/{len(dataloader)}")

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    elapsed = time() - t0

    print(f"\n=== {desc} ===")
    print(f"  Samples: {total}")
    print(f"  Time: {elapsed:.2f}s ({total/elapsed:.1f} img/s)")
    print(f"  Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc*100:.2f}%")
    print(f"  Macro F1: {f1:.4f}")

    return {
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "f1_macro": f1,
        "time_s": elapsed,
        "num_samples": total,
    }


# ------------------------------- PTQ (FX Graph Mode) ------------------------------- #

def quantize_model_fx(
    model_fp32: nn.Module,
    calib_loader: DataLoader,
    backend: str,
) -> nn.Module:
    """
    Apply Post-Training Quantization using FX Graph Mode.
    """
    model_fp32 = copy.deepcopy(model_fp32).to("cpu").eval()

    qconfig_mapping = get_default_qconfig_mapping(backend)
    example_inputs = (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),)

    print("[PTQ] Preparing model (inserting observers)...")
    prepared = prepare_fx(model_fp32, qconfig_mapping, example_inputs).eval()

    print(f"[PTQ] Calibrating with {len(calib_loader.dataset)} samples...")
    t0 = time()
    with torch.inference_mode():
        for batch_idx, (inputs, _) in enumerate(calib_loader):
            prepared(inputs.cpu())
            if (batch_idx + 1) % 10 == 0:
                print(f"  [Calib] Batch {batch_idx + 1}/{len(calib_loader)}")
    print(f"[PTQ] Calibration complete in {time() - t0:.2f}s")

    print("[PTQ] Converting to INT8...")
    quantized = convert_fx(prepared)
    return quantized


# ------------------------------- Main ------------------------------- #

def run_ptq_evaluation():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SAVE_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "log.txt"
    log_fh = open(log_file, "w", encoding="utf-8")
    sys.stdout = Tee(ORIG_STDOUT, log_fh)
    sys.stderr = Tee(ORIG_STDERR, log_fh)

    try:
        print("=" * 80)
        print("PTQ Evaluation: ImageNet (HF Streaming)")
        print("=" * 80)

        model_name = "MobileNetV2-1.00" if USE_MNV2 else "ResNet18"

        print("\nConfig:")
        print(f"  Model: {model_name}")
        print(f"  Image size: {IMG_SIZE}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Calibration samples: {CALIB_SAMPLES} (from TRAIN split)")
        print(f"  Eval samples: {EVAL_SAMPLES if EVAL_SAMPLES else 'Full (~50K val)'}")
        print(f"  HF cache dir: {HF_CACHE_DIR}")

        if USE_MNV2:
            print(f"  Calib shuffle: {CALIB_SHUFFLE} | buffer={SHUFFLE_BUFFER} | seed={CALIB_SEED}")
            print(f"  Weights: {MNV2_Weights}")
            print(f"  width_mult: {WIDTH_MULT}")
        else:
            print(f"  Calib shuffle: False (forced OFF for ResNet18)")

        print(f"  Output dir: {run_dir}\n")

        set_seed(CALIB_SEED)

        engines = torch.backends.quantized.supported_engines
        backend = "fbgemm" if "fbgemm" in engines else "qnnpack"
        torch.backends.quantized.engine = backend
        print(f"Quantization backend: {backend}")
        print(f"FP32 device: {FP32_DEVICE}")
        print(f"INT8 device: {PTQ_DEVICE}\n")

        # Load model
        model_fp32 = build_model_fp32().to(FP32_DEVICE).eval()

        # Create dataloaders
        print("\n--- Setting up DataLoaders ---")

        # Calibration from train (stream -> materialize first-N)
        # NOTE: shuffle enabled only in MNV2 mode
        calib_loader = make_calibration_subset_from_train(
            cache_dir=HF_CACHE_DIR,
            num_samples=CALIB_SAMPLES,
            seed=CALIB_SEED,
            shuffle=(USE_MNV2 and CALIB_SHUFFLE),
            shuffle_buffer=SHUFFLE_BUFFER,
        )

        # Evaluation remains on validation (as before)
        eval_loader = get_imagenet_val_loader(
            cache_dir=HF_CACHE_DIR,
            max_samples=EVAL_SAMPLES,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        # FP32 Evaluation
        print("\n" + "=" * 80)
        print("FP32 Evaluation (VAL)")
        print("=" * 80)
        fp32_results = evaluate_with_top5(model_fp32, eval_loader, device=FP32_DEVICE, desc=f"FP32 {model_name}")

        # PTQ
        print("\n" + "=" * 80)
        print("Post-Training Quantization (FX Graph Mode)")
        print("=" * 80)
        model_int8 = quantize_model_fx(model_fp32=model_fp32, calib_loader=calib_loader, backend=backend)

        # INT8 Evaluation
        print("\n" + "=" * 80)
        print("INT8 Evaluation (VAL)")
        print("=" * 80)
        int8_results = evaluate_with_top5(model_int8, eval_loader, device=PTQ_DEVICE, desc=f"INT8 {model_name}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        top1_drop = fp32_results["top1_acc"] - int8_results["top1_acc"]
        top5_drop = fp32_results["top5_acc"] - int8_results["top5_acc"]
        f1_drop   = fp32_results["f1_macro"] - int8_results["f1_macro"]

        print(f"\n{model_name} (ImageNet Pretrained)")
        if USE_MNV2:
            print(f"  Calibration: {CALIB_SAMPLES} samples (TRAIN split, shuffle={CALIB_SHUFFLE}, buffer={SHUFFLE_BUFFER})")
        else:
            print(f"  Calibration: {CALIB_SAMPLES} samples (TRAIN split, streaming first-N)")
        print(f"  Evaluation:  {fp32_results['num_samples']} samples (VAL split)\n")

        print("FP32:")
        print(f"  Top-1 Accuracy: {fp32_results['top1_acc']*100:.2f}%")
        print(f"  Top-5 Accuracy: {fp32_results['top5_acc']*100:.2f}%")
        print(f"  Macro F1:       {fp32_results['f1_macro']:.4f}\n")

        print("INT8 PTQ:")
        print(f"  Top-1 Accuracy: {int8_results['top1_acc']*100:.2f}%")
        print(f"  Top-5 Accuracy: {int8_results['top5_acc']*100:.2f}%")
        print(f"  Macro F1:       {int8_results['f1_macro']:.4f}\n")

        print("Drop (FP32 - INT8):")
        print(f"  Top-1 Accuracy: {top1_drop*100:+.2f}%")
        print(f"  Top-5 Accuracy: {top5_drop*100:+.2f}%")
        print(f"  Macro F1:       {f1_drop:+.4f}\n")

        # ------------------------------- Export ------------------------------- #
        print("\n" + "=" * 80)
        print("EXPORTING MODELS (FP32 + INT8)")
        print("=" * 80)

        prefix = "mnv2_100" if USE_MNV2 else "resnet18"

        fp32_sd_path = run_dir / f"{prefix}_fp32_state_dict.pth"
        torch.save(model_fp32.state_dict(), fp32_sd_path)
        print(f"[SAVE] FP32 state_dict: {fp32_sd_path} ({fp32_sd_path.stat().st_size/1024/1024:.2f} MB)")

        int8_sd_path = run_dir / f"{prefix}_int8_state_dict.pth"
        torch.save(model_int8.state_dict(), int8_sd_path)
        print(f"[SAVE] INT8 state_dict: {int8_sd_path} ({int8_sd_path.stat().st_size/1024/1024:.2f} MB)")

        fp32_ts_path = run_dir / f"{prefix}_fp32_torchscript.pt"
        example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        scripted_fp32 = torch.jit.trace(model_fp32.to("cpu").eval(), example)
        scripted_fp32.save(str(fp32_ts_path))
        print(f"[SAVE] FP32 TorchScript: {fp32_ts_path} ({fp32_ts_path.stat().st_size/1024/1024:.2f} MB)")

        int8_ts_path = run_dir / f"{prefix}_int8_torchscript.pt"
        scripted_int8 = torch.jit.trace(model_int8.to("cpu").eval(), example)
        scripted_int8.save(str(int8_ts_path))
        print(f"[SAVE] INT8 TorchScript: {int8_ts_path} ({int8_ts_path.stat().st_size/1024/1024:.2f} MB)")

        # Save summary file
        summary_file = run_dir / "summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"PTQ Evaluation: {model_name} on ImageNet (HF Streaming)\n")
            f.write("=" * 60 + "\n\n")
            f.write("Config:\n")
            f.write(f"  Model: {model_name}\n")
            f.write(f"  Calibration samples: {CALIB_SAMPLES} (TRAIN split)\n")
            if USE_MNV2:
                f.write(f"  Calib shuffle: {CALIB_SHUFFLE} | buffer={SHUFFLE_BUFFER} | seed={CALIB_SEED}\n")
                f.write(f"  Weights: {MNV2_Weights}\n")
                f.write(f"  width_mult: {WIDTH_MULT}\n")
            else:
                f.write("  Calib shuffle: False (forced OFF for ResNet18)\n")
            f.write(f"  Evaluation samples:  {fp32_results['num_samples']} (VAL split)\n")
            f.write(f"  Backend: {backend}\n\n")

            f.write("FP32:\n")
            f.write(f"  Top-1 Accuracy: {fp32_results['top1_acc']*100:.4f}%\n")
            f.write(f"  Top-5 Accuracy: {fp32_results['top5_acc']*100:.4f}%\n")
            f.write(f"  Macro F1:       {fp32_results['f1_macro']:.6f}\n\n")

            f.write("INT8 PTQ:\n")
            f.write(f"  Top-1 Accuracy: {int8_results['top1_acc']*100:.4f}%\n")
            f.write(f"  Top-5 Accuracy: {int8_results['top5_acc']*100:.4f}%\n")
            f.write(f"  Macro F1:       {int8_results['f1_macro']:.6f}\n\n")

            f.write("Drop (FP32 - INT8):\n")
            f.write(f"  Top-1 Accuracy: {top1_drop*100:+.4f}%\n")
            f.write(f"  Top-5 Accuracy: {top5_drop*100:+.4f}%\n")
            f.write(f"  Macro F1:       {f1_drop:+.6f}\n\n")


        print(f"\nResults saved to: {run_dir}")
        print("DONE.")

    finally:
        sys.stdout = ORIG_STDOUT
        sys.stderr = ORIG_STDERR
        log_fh.close()


if __name__ == "__main__":
    run_ptq_evaluation()
