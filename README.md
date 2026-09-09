# TinyML: Evaluating Transfer Learning under INT8 Quantisation

**Final Year Project – University of Birmingham**  
**Author:** Awais Aamir  
**Supervisor:** Jizheng Wan

## Overview

TinyML deployments often rely on INT8 quantisation to reduce the cost of running neural networks on resource-constrained hardware.

This project investigates a practical question:

> **Does transfer learning strategy affect accuracy degradation and wider model behaviour after INT8 quantisation?**

Using a fixed MobileNetV2-0.35 architecture, I compared four transfer learning strategies before and after post-training quantisation:

- Full Fine Tuning
- Linear Probing
- Batch Normalisation Tuning
- Knowledge Distillation

Evaluation covered not only headline accuracy, but also prediction stability, output distribution changes, class-level behaviour and robustness under corrupted inputs.

## Why It Matters

Quantisation is often treated as a post-training optimisation step, but the way a model is trained may also affect how well it responds to INT8 conversion.

This project explores whether transfer learning strategy influences accuracy degradation and wider model behaviour after quantisation.

Understanding that relationship can help inform deployment decisions where accuracy, prediction stability and robustness matter, particularly in resource-constrained or edge environments.

## Key Findings

- Transfer learning strategies did not degrade uniformly after INT8 quantisation.
- Linear Probing showed the largest FP32 → INT8 accuracy drop at approximately **2.7%**.
- Full Fine Tuning, BatchNorm tuning and Knowledge Distillation showed smaller degradation of approximately **0.5–1.4%**.
- High-performing regimes showed broadly similar behaviour overall, although smaller differences remained in prediction stability and output distributions.
- Robustness trends were similar under mild-to-moderate corruption, with clearer differences emerging at more severe levels.

These results suggest that **training strategy can be a relevant consideration when preparing models for quantised deployment**, rather than evaluating deployment suitability from FP32 accuracy alone.

## Technical Approach

The project uses:

- **Python / PyTorch**
- MobileNetV2-0.35 pretrained on ImageNet
- PlantVillage image classification dataset
- PyTorch FX Graph Mode static INT8 quantisation
- FBGEMM backend
- Multi-seed experimentation for reproducibility

Evaluation includes:

- FP32 and INT8 accuracy
- Accuracy degradation
- Prediction flip rate
- KL divergence
- Class-level analysis
- Gaussian noise and JPEG corruption testing

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


