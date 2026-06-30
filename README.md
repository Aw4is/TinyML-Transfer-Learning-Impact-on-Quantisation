# **Behavioural Evaluation of Post-Training Quantisation Across Transfer Learning Regimes in TinyML**

**Author**: Awais Aamir   
**Supervisor**: Jizheng Wan

**Report**: [FYP Report](https://git.cs.bham.ac.uk/projects-2025-26/axa2259/-/blob/main/Report/FYPReport.pdf?ref_type=heads)

## **Project Summary**

This project investigates how different transfer learning regimes interact with post-training quantisation (PTQ) in a TinyML setting. Using MobileNetV2-0.35 pretrained on ImageNet, models are adapted to a subset of the PlantVillage dataset under four transfer learning strategies:

- **Full Fine Tuning (FFT)**
- **Linear Probing (LP)**
- **Batch Normalisation Tuning (BNLP)**
- **Logit Standardisation Knowledge Distillation with Full Fine Tuning (LSKD)**

Beyond standard accuracy metrics, the study evaluates output-level differences between FP32 and INT8 models using KL divergence and prediction flip rates, alongside class-level performance analysis and controlled input corruption tests.


## **Key Findings**

- Transfer learning regimes do not degrade uniformly under PTQ.
- Linear Probing shows the largest FP32→INT8 accuracy drop (~2.7%), which may suggest greater quantisation sensitivity when the backbone remains frozen.
- Adaptive regimes (FFT, BNLP, LSKD) maintain high INT8 accuracy with relatively limited degradation (~0.5–1.4%)
- Output-level behavioural diagnostics between FP32 and INT8 models are broadly aligned among the high-performing regimes, though small differences in output distributions and prediction stability remain.
- Robustness trends under Gaussian noise and JPEG compression are largely similar at low-to-moderate corruption levels, with differences emerging at severe levels.

## **Repository Structure**

This repository contains the code and experimental pipeline for my Final Year Project. The src/ directory contains the core implementation for all experiments conducted in this study:

- **MobileNetV2/** - Implementation of MobileNetV2. Based on the implementation of Li et al. (2019), enabling compatibility with their pretrained weights
- **Models/** - Transfer learning regime implementations, including LP, FFT, BNLP, and LSKD
- **Quantisation/** - FX Graph Mode post training quantisation pipeline using the FBGEMM backend. Includes behavioural evaluation scripts for KL divergence, prediction flip rates, and class-level analysis.
- **Verification/** - Sanity check and validation scripts used to verify training, quantisation and evaluation pipeline prior to applying them to PlantVillage dataset and logging framework.

Additional directories include:
- **Logs/** - Training and evaluation logs for  transfer learning regimes and quantisation experiments
- **Docs/** - Early draft of the project literature review


## Dataset


### PlantVillage
- 38 classes
- Deterministic split: 70 / 20 / 10 (train / val / test)
- Stratified sampling of 100 per class

### Calibration Subset for PTQ
- 15 images per class
- 570 images total
- Class balanced
- Calibration subset is sampled from the training split only, ensuring no overlap with validation or test data.

## Experimental Setup

### Backbone
- MobileNetV2
- width_mult = 0.35
- Pretrained ImageNet weights (Li et al.)

### Training Configuration
- Image size: 96 × 96
- Batch size: 50
- Optimiser: SGD with momentum (0.9)
- Learning Rate: 0.01
- Dropout: 0.5 
- Deterministic seeds: [0, 2, 5, 42, 1337]

### Transfer Learning Regimes
- Linear Probing (LP): Backbone frozen, classifier trained only.
- Full Fine Tuning (FFT): All backbone and classifier parameters are trainable.
- Batch Normalisation Layer Tuning (BNLP): Only BatchNorm parameters and classifier are trainable.
- Knowledge Distillation (LSKD):
    - Student: MobileNetV2-0.35
    - Teacher: ResNet50 backbone
    - Loss: Cross entropy + KL divergence (logit standardised)
    - Training: Full Fine Tuning

## Post Training Quantisation

Quantisation is implemented using:
- PyTorch FX Graph Mode
- Static post training quantisation
- FBGEMM backend
- Histogram observers for activations, MinMax observers for weights
- Calibration on 570 image balanced subset

### Evaluation metrics include:

- INT8 accuracy
- FP32 to INT8 degradation
- KL divergence between FP32 and INT8 logits
- Prediction flip rate

### Corruption Robustness Experiments 

Robustness is evaluated under controlled input corruption levels, including:
- Gaussian noise
- JPEG compression

Corruption severity levels are applied consistently across all transfer learning regimes.

## Reproducibility

To ensure reproducibility:
- Deterministic seeds are fixed
- Exact dataset split available in Google Drive (see below)
- Checkpoints are stored and available in Google Drive 
- Calibration subset is fixed


## Running Experiments

### 1. Training (example): 

> python -m src.Models.FFT

Before execution, update the following path variables at the top of the script:

- DATA_ROOT    = r"/path/to/PlantVillage"
- WEIGHTS_PATH = r"/path/to/mobilenetv2_0.35_imagenet.pth"
- SAVE_DIR     = r"/path/to/log_directory"

These paths must be configured prior to running experiments

### 2. Post Training Quantisation (example):

> python -m src.Quantisation.Global

After training, update the checkpoint directly inside the quantisation script
- WEIGHTS_DIR = Path(r"UnifiedWeights/FFT")

The specified directory should  contain the five seed checkpoints generated during training. 


## Dataset and Model Weights Access

Due to storage constraints, the PlantVillage dataset and trained model checkpoints are hosted externally.

They are available via Google Drive:

[Google Drive - Dataset and Model Weights](https://drive.google.com/drive/u/1/folders/1ftqoSnLfy8quNMfq8hg2TEOrADYY9hAZ)

After downloading:
- Update dataset and checkpoint paths within the training, quantisation, and evaluation scripts as required.

### Pretrained Backbone Weights

MobileNetV2 ImageNet pretrained weights (Li et al., 2019) are publicly available from the original repository:

[Li et al. MobileNetV2-0.35 Weights
](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.35-b2e15951.pth)

These should be downloaded separately and placed in the appropriate directory before training.


