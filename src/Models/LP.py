'''
Implements Linear Probing - Backbone frozen only classifier trained
Follows Seol's implementation as close as possible with few TinyML changes
'''

import os
import sys
import time
import random
import copy
import csv
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.MobileNetV2.mobilenetv2 import MobileNetV2


# --- Config --- #
DATA_ROOT      = r"PVdataset"
BATCH_SIZE     = 50
LR             = 1e-2
MOMENTUM       = 0.9
WEIGHT_DECAY   = 0
MAX_EPOCHS     = 200
PATIENCE       = 10
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT_P      = 0.5

SEEDS          = [0, 2, 5, 42, 1337]

WEIGHTS_PATH   = r"pathtoimagenetweights"
IMG_SIZE       = 96

# Where to save logs
SAVE_DIR       = r"savelogs"
os.makedirs(SAVE_DIR, exist_ok=True)


# ------------------------------- Reproducability ------------------------------- #

def set_seed(seed: int):
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

# Class that accepts multiple file-like objsects
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
    with open(path, "w", encoding="utf-8") as f: #open file in write mode and writes to file..
        f.write("LP SEED SWEEP SUMMARY\n")
        f.write(f"Config: LR={LR}, Dropout={DROPOUT_P}, Batch={BATCH_SIZE}, Img={IMG_SIZE}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write("=" * 80 + "\n\n")


def append_seed_line(path: str, seed: int, best_epoch: int, best_val_acc: float,
                     val_acc: float, test_acc: float, val_f1: float, test_f1: float,
                     run_dir: str, weights_path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"Seed {seed} results: "
            f"best_epoch={best_epoch}, best_val_during_train={best_val_acc:.6f}, "
            f"VAL acc={val_acc:.6f}, TEST acc={test_acc:.6f}, "
            f"VAL f1={val_f1:.6f}, TEST f1={test_f1:.6f}\n"
        )

# Machine readable results
def append_tsv_row(path: str, row: dict):
    file_exists = os.path.isfile(path) #checks whether this is first write
    fieldnames = list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# ------------------------------- Dataset ------------------------------- #

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    "val": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    "test": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
}

splits = ["train", "val", "test"]


def build_datasets_and_loaders():

    image_datasets = {
        split: datasets.ImageFolder(
            root=os.path.join(DATA_ROOT, split),
            transform=data_transforms[split]
        )
        for split in splits
    }

    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        for split in splits
    }

    dataset_sizes = {s: len(image_datasets[s]) for s in splits}
    class_names   = image_datasets["train"].classes
    num_classes   = len(class_names)

    return image_datasets, dataloaders, dataset_sizes, class_names, num_classes


# ------------------------------- Model ------------------------------- #

def set_bn_eval(module):
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.eval()


def build_lp_model(num_classes: int, dropout_p: float):
    print("Building MobileNetV2 (width_mult=0.35) with Li et al. weights...")
    base = MobileNetV2(num_classes=1000, width_mult=0.35)
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    base.load_state_dict(state, strict=True)
    print("Pretrained weights loaded.")

    in_features = base.classifier.in_features
    print(f"Classifier input features (MNV2-0.35): {in_features}")

    base.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes)
    )

    # Freeze backbone, train only classifier (strict LP)
    for name, param in base.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Parameter stats
    backbone_params  = sum(p.numel() for n, p in base.named_parameters() if "classifier" not in n)
    head_params      = sum(p.numel() for n, p in base.named_parameters() if "classifier" in n)
    trainable_params = sum(p.numel() for p in base.parameters() if p.requires_grad)

    print(f"\n=== Parameter Summary (MNV2-0.35 Linear Probe, SGD) ===")
    print(f"Backbone params (frozen): {backbone_params:,}")
    print(f"Head params (trainable):  {head_params:,}")
    print(f"Total params:             {backbone_params + head_params:,}")
    print(f"Trainable params:         {trainable_params:,}\n")

    return base.to(DEVICE)


# ------------------------------- Metrics ------------------------------- #

def evaluate_split(model, loader, device, split_name: str, out_txt_path: str, class_names):
    model.eval()

    all_labels = []
    all_preds  = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)
    cr   = classification_report(
        all_labels, all_preds,
        target_names=class_names, digits=4, zero_division=0
    )

    print(f"\n--- {split_name.upper()} metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification report:")
    print(cr)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(f"{split_name.upper()} METRICS\n")
        f.write(f"Accuracy : {acc:.6f}\n")
        f.write(f"Precision: {prec:.6f}\n")
        f.write(f"Recall   : {rec:.6f}\n")
        f.write(f"F1-score : {f1:.6f}\n\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write("Classification report:\n")
        f.write(cr + "\n")

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


# ------------------------------- Run ------------------------------- #

# Function that runs the entire LP experiment  across all seeds
def run_lp_seed_sweep():
    sweep_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(SAVE_DIR, f"LP_seed_sweep_{sweep_stamp}") #so every run gets unique folder
    os.makedirs(sweep_dir, exist_ok=True)

    # creates path for files to store
    summary_txt = os.path.join(sweep_dir, "seeds_summary.txt")
    results_tsv = os.path.join(sweep_dir, "seed_results.tsv")
    runs_root   = os.path.join(sweep_dir, "runs")
    os.makedirs(runs_root, exist_ok=True)

    write_summary_header(summary_txt)

    all_val_acc = []
    all_test_acc = []
    all_val_f1 = []
    all_test_f1 = []

    for seed in SEEDS:
        # --- Correct order --- #
        # 1) set_seed
        set_seed(seed)

        # 2) build datasets + dataloaders (before model init)
        image_datasets, dataloaders, dataset_sizes, class_names, num_classes = build_datasets_and_loaders()

        run_tag = f"seed{seed}_lr{LR}_do{DROPOUT_P}"
        run_dir = os.path.join(runs_root, run_tag)
        os.makedirs(run_dir, exist_ok=True)

        log_txt  = os.path.join(run_dir, "train_log.txt")
        val_txt  = os.path.join(run_dir, "val_metrics.txt")
        test_txt = os.path.join(run_dir, "test_metrics.txt")

        # Tee terminal output to per-seed log -> allowing real time logging
        _log_fh = open(log_txt, "w", encoding="utf-8")
        sys.stdout = Tee(ORIG_STDOUT, _log_fh)
        sys.stderr = Tee(ORIG_STDERR, _log_fh)

        try:
            print("=" * 80)
            print(f"LP RUN | seed={seed} | device={DEVICE}")
            print(f"Config: LR={LR} dropout={DROPOUT_P} batch={BATCH_SIZE} img={IMG_SIZE}")
            print("Sizes:", dataset_sizes)
            print("Num classes:", num_classes)
            print("=" * 80)

            # build model and define optimiser
            model = build_lp_model(num_classes=num_classes, dropout_p=DROPOUT_P)

            params_to_update = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.SGD(
                params_to_update,
                lr=LR,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
            )

            scheduler_lp = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=5,
            )

            criterion = nn.CrossEntropyLoss()

            # ------- Training  -------
            best_val_acc      = 0.0
            best_epoch        = 0
            best_model_wts    = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            start_time = time.time()

            for epoch in range(1, MAX_EPOCHS + 1):
                print(f"\nEpoch {epoch}/{MAX_EPOCHS}")

                # ---- Train ----
                model.train()
                model.apply(set_bn_eval)  # keep BN in eval for frozen backbone

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders["train"]:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)

                train_loss = running_loss / dataset_sizes["train"]
                train_acc  = running_corrects.double().item() / dataset_sizes["train"]

                # ---- Validation ----
                model.eval()
                val_loss = 0.0
                val_corrects = 0

                with torch.no_grad():
                    for inputs, labels in dataloaders["val"]:
                        inputs = inputs.to(DEVICE)
                        labels = labels.to(DEVICE)

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        val_loss += loss.item() * inputs.size(0)
                        val_corrects += torch.sum(preds == labels)

                val_loss /= dataset_sizes["val"]
                val_acc  = val_corrects.double().item() / dataset_sizes["val"]

                scheduler_lp.step(val_loss)

                elapsed = time.time() - start_time
                print(f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f}")
                print(f"Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")
                print(f"Elapsed time: {elapsed/60:.1f} min")

                # ---- Early stopping on val_acc ----
                if val_acc > best_val_acc + 1e-4:
                    best_val_acc = val_acc
                    best_epoch   = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f"  -> New best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

             

            # Load best model
            model.load_state_dict(best_model_wts)
            print(f"\nBest model from epoch {best_epoch} with val_acc={best_val_acc:.4f}")

            # Save best fp32 model (per-seed run folder)
            weights_path = os.path.join(
                run_dir,
                f"LP-Seed{seed}-FP32_bestVal{best_val_acc:.4f}.pth"
            )
            torch.save(model.state_dict(), weights_path)
            print(f"Saved best FP32 LP model to: {weights_path}")

            # Evaluate val/test and write clean files
            val_m  = evaluate_split(model, dataloaders["val"],  DEVICE, "val",  val_txt,  class_names)
            test_m = evaluate_split(model, dataloaders["test"], DEVICE, "test", test_txt, class_names)

            # Store for overall stats
            all_val_acc.append(val_m["acc"])
            all_test_acc.append(test_m["acc"])
            all_val_f1.append(val_m["f1"])
            all_test_f1.append(test_m["f1"])

            # Append to human-readable summary
            append_seed_line(
                summary_txt,
                seed=seed,
                best_epoch=best_epoch,
                best_val_acc=best_val_acc,
                val_acc=val_m["acc"],
                test_acc=test_m["acc"],
                val_f1=val_m["f1"],
                test_f1=test_m["f1"],
                run_dir=run_dir,
                weights_path=weights_path
            )

            # Append TSV row 
            append_tsv_row(
                results_tsv,
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "seed": seed,
                    "lr": LR,
                    "dropout": DROPOUT_P,
                    "img_size": IMG_SIZE,
                    "batch_size": BATCH_SIZE,
                    "best_epoch": best_epoch,
                    "best_val_acc_during_train": f"{best_val_acc:.6f}",
                    "val_acc": f"{val_m['acc']:.6f}",
                    "test_acc": f"{test_m['acc']:.6f}",
                    "val_f1": f"{val_m['f1']:.6f}",
                    "test_f1": f"{test_m['f1']:.6f}",
                    "weights_path": weights_path,
                    "run_dir": run_dir,
                }
            )

            print(f"\n[LOG] train_log: {log_txt}")
            print(f"[LOG] val_metrics: {val_txt}")
            print(f"[LOG] test_metrics: {test_txt}")

        finally:
            # Restore stdout/stderr and close per-seed log -> avoids seed mixing
            sys.stdout = ORIG_STDOUT
            sys.stderr = ORIG_STDERR
            _log_fh.close()

            # Cleanup
            try:
                del model, optimizer, scheduler_lp, criterion
            except Exception:
                pass
            torch.cuda.empty_cache()

    # ---- Overall mean±std ----
    vmu, vsd = mean_std(all_val_acc)
    tmu, tsd = mean_std(all_test_acc)
    vfmu, vfsd = mean_std(all_val_f1)
    tfmu, tfsd = mean_std(all_test_f1)

    with open(summary_txt, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("OVERALL (mean ± std across seeds)\n")
        f.write(f"VAL  acc: {vmu:.6f} ± {vsd:.6f}\n")
        f.write(f"TEST acc: {tmu:.6f} ± {tsd:.6f}\n")
        f.write(f"VAL  f1 : {vfmu:.6f} ± {vfsd:.6f}\n")
        f.write(f"TEST f1 : {tfmu:.6f} ± {tfsd:.6f}\n\n")

    print("\nDONE.")
    print(f"Summary: {summary_txt}")
    print(f"TSV:     {results_tsv}")
    print(f"Runs:    {runs_root}")


if __name__ == "__main__":
    run_lp_seed_sweep()
