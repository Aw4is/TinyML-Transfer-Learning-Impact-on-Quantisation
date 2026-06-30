'''
Verification script for LP baseline and training pipeline
Follows Seol's et al. Work: https://www.sciencedirect.com/science/article/pii/S2352710225018881?fr=RR-2&ref=pdf_download&rr=9e0793e2efd8f650
'''
import time
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ------- Config ------- #
    # Follows from Paper

DATA_ROOT      = r"concreteDataSet"
BATCH_SIZE     = 50
LR             = 0.001
MOMENTUM       = 0.9
WEIGHT_DECAY   = 0  # none applied
MAX_EPOCHS     = 200
TIME_BUDGET_S  = 60 * 60 * 1 # timer of 1 hour
PATIENCE       = 10
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu" # Checks device availablity - if GPU is available or not

SEED = 55  

# ------- Reproducability ------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed) # controls python random module like shuffling
    torch.manual_seed(seed) # dataset splitting
    torch.cuda.manual_seed_all(seed) # applies above for all CUDA devices
    torch.backends.cudnn.deterministic = True # NVIDIA deep learning library can choose fastest algo for GPU, we make this determinisitc/disable it
    torch.backends.cudnn.benchmark = False # Similar as above

set_seed(SEED)

# ------- Dataset -------
    # https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# Standard normalisation values 
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# No augmentation, simply resize so it fits mobilenetv2 - MATLAB does not accept 256 x 256 unlike in PyTorch
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
}

image_datasets = {
    split: datasets.ImageFolder(
        root=f"{DATA_ROOT}/{split}",
        transform=data_transforms[split]
    )
    for split in ["train", "val"]
}

dataloaders = {
    split: DataLoader(
        image_datasets[split],
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    for split in ["train", "val"]
}

# Checks if dataset is correctly loaded
dataset_sizes = {s: len(image_datasets[s]) for s in ["train", "val"]}
class_names   = image_datasets["train"].classes
num_classes   = len(class_names)

print("Classes:", class_names)
print("Sizes:", dataset_sizes)

# ------- Define Model -------
    # https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze backbone
for p in base.features.parameters():
    p.requires_grad = False

# Replace classifier
in_features = base.classifier[1].in_features
base.classifier = nn.Sequential(
    nn.Linear(in_features, num_classes)
)

# Moves model to desired CPU/GPU
model = base.to(DEVICE)

# Only classifier trainable - pass parameters to optimiser
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(
    params_to_update,
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
)
criterion = nn.CrossEntropyLoss()

# Freeze BN layers
def set_bn_eval(module):
    """Recursively set all BatchNorm layers to eval mode."""
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.eval()

# ------- Training -------
    # https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# Training Loop
best_val_acc = 0.0
best_epoch   = 0
best_model_wts = copy.deepcopy(model.state_dict())  # keep best weights
epochs_no_improve = 0

start_time = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{MAX_EPOCHS}")

    # Sets model in training mode
    model.train()
    model.features.apply(set_bn_eval)  # keep frozen BN in eval

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data
    for inputs, labels in dataloaders["train"]:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # zero gradients -> Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Get predictions and compute losses
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        # Backwards + optimise so update weights
        loss.backward()
        optimizer.step()

        # Average loss * batch size -> Calculate data-set wide average loss
        running_loss += loss.item() * inputs.size(0)

        # Creates boolean tensor that accumlates the total number of correct predictions in the epoch
        running_corrects += torch.sum(preds == labels)

    train_loss = running_loss / dataset_sizes["train"]
    train_acc  = running_corrects.double().item() / dataset_sizes["train"] #total number of correct predictions converted to float, number extracted

    # Sets model in validation mode
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    # Validation step
    with torch.no_grad(): # this tells python not to build graph -> Saves mem and inference faster as we do not compute gradients/update weights
        for inputs, labels in dataloaders["val"]: # loop through validation dataloader
            # Move images/labels to GPU
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward Pass -> Model makes predictions, compute loss to report validation performance
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1) # picks class with highest probability/logit

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels)

    val_loss /= dataset_sizes["val"]
    val_acc  = val_corrects.double().item() / dataset_sizes["val"]

    elapsed = time.time() - start_time
    print(f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f}")
    print(f"Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")
    print(f"Elapsed time: {elapsed/60:.1f} min")

    # Early stopping on val_acc
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

    if elapsed > TIME_BUDGET_S:
        print(f"Time budget exceeded at epoch {epoch}")
        break

# Load best model and compute emtrics
model.load_state_dict(best_model_wts)
print(f"\nBest model from epoch {best_epoch} with val_acc={best_val_acc:.4f}")


# ------- Compute Metrics  ------- #

# Run model in evaluation mode 
model.eval()

# Store prediction, true label for val set
all_labels = []
all_preds  = []

with torch.no_grad():
    for inputs, labels in dataloaders["val"]:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Labels, preds are GPU tensors so we move them to CPU + convert to numpy lists
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Convert to numpy lissts
all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)

val_acc_final = accuracy_score(all_labels, all_preds)
val_prec = precision_score(all_labels, all_preds)   # binary case
val_rec  = recall_score(all_labels, all_preds)
val_f1   = f1_score(all_labels, all_preds)

print("\n--- Validation metrics at best epoch ---")
print(f"Accuracy : {val_acc_final:.4f}")
print(f"Precision: {val_prec:.4f}")
print(f"Recall   : {val_rec:.4f}")
print(f"F1-score : {val_f1:.4f}")

print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
