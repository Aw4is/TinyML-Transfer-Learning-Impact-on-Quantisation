"""
KD sanity validation Script (CIFAR-100)
Uses Sun's Setup as close as possible - Requires Sun's Functions and checkpoints, linked provided where needed
Follows core pipeline of our PV with minor changes to match Sun's configs
For Sun's Config -> https://github.com/sunshangquan/logit-standardization-KD/blob/master/configs/cifar100/kd/ResNet50_MobileNetV2.yaml
"""

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
from pathlib import Path


from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from Models.KD.backbones.resnet import ResNet50  # resnet - https://github.com/xhdr0618/logit-standardization-LS/blob/master/mdistiller/models/cifar/resnet.py
from Models.KD.Distiller.KD import kd_loss
from Models.KD.backbones.mobilenetv2 import mobile_half  # student - https://github.com/xhdr0618/logit-standardization-LS/blob/master/mdistiller/models/cifar/mobilenetv2.py


# ------------------------------- Config ------------------------------- #

DATA_ROOT = Path(r"")
# https://github.com/xhdr0618/logit-standardization-LS/tree/master -> CKPT here
TEACHER_CKPT = Path(r"")


SAVE_DIR = Path(r"")
os.makedirs(SAVE_DIR, exist_ok=True)

# Sun reproduction config
BATCH_SIZE = 64
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

MAX_EPOCHS = 240
PATIENCE = 100000  # Effectively disable it to mimic Sun's setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# KD hyperparameters (Sun)
KD_T = 9.0
CE_WEIGHT = 0.1
KD_WEIGHT = 2.0

# Step LR schedule (Sun)
LR_DECAY_EPOCHS = [150, 180, 210]
LR_DECAY_RATE = 0.1

# Seeds -> only run one for this sanity check
SEEDS = 1


# ------------------------------- Reproducability ------------------------------- #

def set_seed(seed: int):
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


def mean_std(x):
    x = np.array(x, dtype=np.float64)
    if len(x) <= 1:
        return float(x.mean()) if len(x) == 1 else 0.0, 0.0
    return float(x.mean()), float(x.std(ddof=1))


def write_summary_header(path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("KD CIFAR-100 SUMMARY\n")
        f.write(f"Config: LR={LR}, Momentum={MOMENTUM}, WD={WEIGHT_DECAY}, Batch={BATCH_SIZE}\n")
        f.write(f"Schedule: step LR epochs={LR_DECAY_EPOCHS}, decay={LR_DECAY_RATE}\n")
        f.write(f"KD: T={KD_T}, CE_WEIGHT={CE_WEIGHT}, KD_WEIGHT={KD_WEIGHT}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write("=" * 80 + "\n\n")


def append_seed_line(path: str, seed: int, best_epoch: int, best_test_acc: float,
                     test_acc: float, test_f1: float,
                     run_dir: str, weights_path: str):
    # Here, model selection is based on CIFAR test (no val split).
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"Seed {seed} results: "
            f"best_epoch={best_epoch}, best_test_during_train={best_test_acc:.6f}, "
            f"TEST acc={test_acc:.6f}, TEST f1={test_f1:.6f}\n"
        )


def append_tsv_row(path: str, row: dict):
    file_exists = os.path.isfile(path)
    fieldnames = list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# ------------------------------- Dataset ------------------------------- #

# CIFAR-100 mean/std (Sun)
mean = [0.5071, 0.4867, 0.4408]
std  = [0.2675, 0.2565, 0.2761]

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
}

splits = ["train", "test"]


def build_datasets_and_loaders():
    train_ds = datasets.CIFAR100(
        root=DATA_ROOT, train=True, download=True, transform=data_transforms["train"]
    )
    test_ds = datasets.CIFAR100(
        root=DATA_ROOT, train=False, download=True, transform=data_transforms["test"]
    )

    image_datasets = {
        "train": train_ds,
        "test": test_ds,
    }

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            image_datasets["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    dataset_sizes = {s: len(image_datasets[s]) for s in splits}
    class_names = image_datasets["train"].classes
    num_classes = 100
    return image_datasets, dataloaders, dataset_sizes, class_names, num_classes


# ------------------------------- Models ------------------------------- #

# if model returns tuple -> take first element else return as is
    # KD can return intermediate features as some KD implementations like in Sun's can expose it
def _get_logits(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def build_teacher(num_classes: int):
    print("Building CIFAR ResNet50 teacher...")
    teacher = ResNet50(num_classes=num_classes)

    ckpt = torch.load(TEACHER_CKPT, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        best_top1 = ckpt.get("best_top1", None)
        epoch = ckpt.get("epoch", None)
        print(f"Loaded teacher ckpt dict: epoch={epoch}, best_top1={best_top1}")
    else:
        state_dict = ckpt
        print("Loaded teacher checkpoint as raw state_dict")

    teacher.load_state_dict(state_dict, strict=True)

    teacher.to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def build_student(num_classes: int):
    print("Building MobileNetV2-half student...")
    student = mobile_half(num_classes=num_classes).to(DEVICE)

    for p in student.parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"\n=== KD Student Summary (MobileNetV2-half) ===")
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}\n")

    return student


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
            logits = _get_logits(outputs)
            _, preds = torch.max(logits, 1)

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


# ------------------------------- LR schedule helper ------------------------------- #

# takes  pytorch optimiser and current epoch number
def adjust_lr(optimizer, epoch: int):
    if epoch in LR_DECAY_EPOCHS:  # checks if epoch in list where we change LR if so
        for pg in optimizer.param_groups: # get all params and multiply by 0.1 (we access stored dict which has all groups)
            pg["lr"] *= LR_DECAY_RATE
        print(f"[LR] Decayed LR at epoch {epoch}. New LR: {optimizer.param_groups[0]['lr']:.5f}")


# ------------------------------- Run ------------------------------- #

def run_kd():
    sweep_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(SAVE_DIR, f"KD_cifar100_{sweep_stamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    summary_txt = os.path.join(sweep_dir, "seeds_summary.txt")
    results_tsv = os.path.join(sweep_dir, "seed_results.tsv")
    runs_root   = os.path.join(sweep_dir, "runs")
    os.makedirs(runs_root, exist_ok=True)

    write_summary_header(summary_txt)

    all_test_acc = []
    all_test_f1  = []

    # run a single seed deterministically.
    seed = SEEDS
    set_seed(seed)

    image_datasets, dataloaders, dataset_sizes, class_names, num_classes = build_datasets_and_loaders()

    run_tag = f"seed{seed}_lr{LR}_T{KD_T}_a{KD_WEIGHT}_g{CE_WEIGHT}"
    run_dir = os.path.join(runs_root, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    log_txt  = os.path.join(run_dir, "train_log.txt")
    test_txt = os.path.join(run_dir, "test_metrics.txt")

    _log_fh = open(log_txt, "w", encoding="utf-8")
    sys.stdout = Tee(ORIG_STDOUT, _log_fh)
    sys.stderr = Tee(ORIG_STDERR, _log_fh)

    try:
        print("=" * 80)
        print(f"KD RUN | seed={seed} | device={DEVICE}")
        print(f"Config: LR={LR} momentum={MOMENTUM} wd={WEIGHT_DECAY} batch={BATCH_SIZE}")
        print(f"Schedule: epochs={LR_DECAY_EPOCHS}, decay={LR_DECAY_RATE}")
        print(f"KD: T={KD_T} CE_WEIGHT={CE_WEIGHT} KD_WEIGHT={KD_WEIGHT}")
        print("Sizes:", dataset_sizes)
        print("Num classes:", num_classes)
        print("=" * 80)

        teacher = build_teacher(num_classes=num_classes)
        student = build_student(num_classes=num_classes)

        optimizer = optim.SGD(
            student.parameters(),
            lr=LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )

        criterion_ce = nn.CrossEntropyLoss()

        best_test_acc     = 0.0
        best_epoch        = 0
        best_model_wts    = copy.deepcopy(student.state_dict())
        epochs_no_improve = 0
        start_time = time.time()

        for epoch in range(1, MAX_EPOCHS + 1):
            adjust_lr(optimizer, epoch)

            print(f"\nEpoch {epoch}/{MAX_EPOCHS}")

            # ---- Train ----
            student.train()

            running_loss = 0.0
            running_ce   = 0.0
            running_kd   = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders["train"]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.no_grad():
                    t_logits = _get_logits(teacher(inputs))

                s_logits = _get_logits(student(inputs))

                ce_loss = criterion_ce(s_logits, labels)

                kd_loss_value = kd_loss(
                    logits_student_in=s_logits,
                    logits_teacher_in=t_logits,
                    temperature=KD_T,
                    logit_stand=True,
                )

                loss = CE_WEIGHT * ce_loss + KD_WEIGHT * kd_loss_value

                loss.backward()
                optimizer.step()

                _, preds = torch.max(s_logits, 1)
                batch_size = inputs.size(0)

                running_loss     += loss.item() * batch_size
                running_ce       += ce_loss.item() * batch_size
                running_kd       += kd_loss_value.item() * batch_size
                running_corrects += torch.sum(preds == labels)

            train_loss = running_loss / dataset_sizes["train"]
            train_ce   = running_ce   / dataset_sizes["train"]
            train_kd   = running_kd   / dataset_sizes["train"]
            train_acc  = running_corrects.double().item() / dataset_sizes["train"]

            student.eval()
            test_corrects = 0
            test_loss = 0.0

            with torch.no_grad():
                for inputs, labels in dataloaders["test"]:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = student(inputs)
                    logits = _get_logits(outputs)

                    loss_ce = criterion_ce(logits, labels)
                    _, preds = torch.max(logits, 1)

                    test_loss += loss_ce.item() * inputs.size(0)
                    test_corrects += torch.sum(preds == labels)

            test_loss /= dataset_sizes["test"]
            test_acc  = test_corrects.double().item() / dataset_sizes["test"]

            elapsed = time.time() - start_time
            print(f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f}")
            print(f"  CE: {train_ce:.4f}  KD: {train_kd:.4f}")
            print(f"Test  loss: {test_loss:.4f}  acc: {test_acc:.4f}")
            print(f"Elapsed time: {elapsed/60:.1f} min")

            if test_acc > best_test_acc + 1e-4:
                best_test_acc = test_acc
                best_epoch   = epoch
                best_model_wts = copy.deepcopy(student.state_dict())
                epochs_no_improve = 0
                print(f"  -> New best test_acc: {best_test_acc:.4f} at epoch {best_epoch}")
            else:
                epochs_no_improve += 1

            if PATIENCE is not None and epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        student.load_state_dict(best_model_wts)
        print(f"\nBest model from epoch {best_epoch} with test_acc={best_test_acc:.4f}")

        weights_path = os.path.join(
            run_dir,
            f"mnv2_half_KD_cifar100_seed{seed}_e{best_epoch}_bestTest{best_test_acc:.4f}.pth",
        )
        torch.save(student.state_dict(), weights_path)
        print(f"Saved best KD student to: {weights_path}")

        test_m = evaluate_split(student, dataloaders["test"], DEVICE, "test", test_txt, class_names)

        all_test_acc.append(test_m["acc"])
        all_test_f1.append(test_m["f1"])

        append_seed_line(
            summary_txt,
            seed=seed,
            best_epoch=best_epoch,
            best_test_acc=best_test_acc,
            test_acc=test_m["acc"],
            test_f1=test_m["f1"],
            run_dir=run_dir,
            weights_path=weights_path,
        )

        append_tsv_row(
            results_tsv,
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seed": seed,
                "lr": LR,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "kd_T": KD_T,
                "ce_weight": CE_WEIGHT,
                "kd_weight": KD_WEIGHT,
                "lr_decay_epochs": ",".join(map(str, LR_DECAY_EPOCHS)),
                "lr_decay_rate": LR_DECAY_RATE,
                "best_epoch": best_epoch,
                "best_test_acc_during_train": f"{best_test_acc:.6f}",
                "test_acc": f"{test_m['acc']:.6f}",
                "test_f1": f"{test_m['f1']:.6f}",
                "weights_path": weights_path,
                "run_dir": run_dir,
            }
        )

        print(f"\n[LOG] train_log:    {log_txt}")
        print(f"[LOG] test_metrics: {test_txt}")

    finally:
        sys.stdout = ORIG_STDOUT
        sys.stderr = ORIG_STDERR
        _log_fh.close()

        try:
            del teacher, student, optimizer, criterion_ce
        except Exception:
            pass
        torch.cuda.empty_cache()

    tmu, tsd = mean_std(all_test_acc)
    tfmu, tfsd = mean_std(all_test_f1)

    with open(summary_txt, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("OVERALL (mean ± std)\n")
        f.write(f"TEST acc: {tmu:.6f} ± {tsd:.6f}\n")
        f.write(f"TEST f1 : {tfmu:.6f} ± {tfsd:.6f}\n\n")

    print("\nDONE.")
    print(f"Summary: {summary_txt}")
    print(f"TSV:     {results_tsv}")
    print(f"Runs:    {runs_root}")


if __name__ == "__main__":
    run_kd()
