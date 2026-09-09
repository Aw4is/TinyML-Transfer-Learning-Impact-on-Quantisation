# TinyML: Evaluating Transfer Learning under INT8 Quantisation

**Final Year Project – University of Birmingham**  
**Author:** Awais Aamir  
**Supervisor:** Jizheng Wan

## Overview

TinyML often targets specialised tasks with limited labelled data, where transfer learning can help reduce the need for large task-specific datasets. INT8 quantisation is then commonly used to lower deployment cost on constrained hardware, but it can also affect accuracy and model behaviour. This project investigates a practical question: 

> **Does transfer learning strategy affect accuracy degradation and wider model behaviour after INT8 quantisation?**

Using a fixed MobileNetV2-0.35 architecture, I compared Linear Probing (frozen backbone) against three adaptive approaches:

- Full Fine Tuning
- Batch Normalisation Tuning
- Knowledge Distillation

I then compared how each responded to INT8 quantisation using accuracy degradation, prediction stability, class-level performance and robustness under corrupted inputs.

## Why It Matters

Training strategies differ in how much of the model they adapt, affecting compute and training time. More complex methods like knowledge distillation require more resources, while lighter approaches may still behave similarly after INT8 quantisation. If so, lighter methods could offer a more practical route to deployment when resources are limited.

Accuracy alone may not tell the full story. Prediction stability, class-level behaviour and robustness to corrupted inputs can also matter, so comparing how different training methods degrade and behave after quantisation gives a better basis for deployment decisions.

## Key Findings

- Small differences in post-quantisation accuracy degradation appeared across the adaptive methods (~0.5–1.4%), while Linear Probing showed a noticeably larger drop of ~2.7%.
- Full Fine Tuning outperformed Knowledge Distillation, despite KD using a much larger ResNet50 teacher, showing that extra training complexity did not guarantee better results in this setup.
- BatchNorm tuning performed similarly while updating only a small part of the model, making PEFT-style approaches worth exploring when training compute is limited.
- Differences between adaptive approaches became clearer under harsher corruption, while prediction stability, output distributions and class-level performance were otherwise broadly similar.
  
## Technical Approach

- Python / PyTorch with MobileNetV2-0.35 on PlantVillage
- Static INT8 quantisation using PyTorch FX Graph Mode and FBGEMM
- Evaluation: accuracy degradation, prediction stability, KL divergence, class-level performance and corruption robustness
- Multi-seed experiments for reproducibility

## Repository Structure

- `src/MobileNetV2/` – MobileNetV2 implementation
- `src/Models/` – transfer learning strategies
- `src/Quantisation/` – INT8 quantisation and evaluation
- `src/Verification/` – validation and sanity checks
- `Logs/` – experiment and evaluation logs

## Running an Experiment

### Train a model:

```bash
python -m src.Models.FFT
```

Before running, update the paths in the training script:

```bash
DATA_ROOT = r"/path/to/PlantVillage"
WEIGHTS_PATH = r"/path/to/mobilenetv2_0.35_imagenet.pth"
SAVE_DIR = r"/path/to/log_directory"
```
### Post-Training Quantisation:

First, point the quantisation script (in src/Quantisation/Global) to the trained checkpoints, which should contain the five seed checkpoints generated during training:

```bash
WEIGHTS_DIR = Path(r"UnifiedWeights/FFT")
```
Then, run:

```bash
python -m src.Quantisation.Global
```


