# CNNs on Mini-ImageNet: Architecture Study and Occlusion Sensitivity

**Training convolutional neural networks from scratch on 33 image classes, measuring how each design choice affects accuracy, then probing what the best model actually looks at.**

Part of the *AI & ML for Industry* programme, IIT Delhi (Module 4: Deep Learning).

📄 **[Read the full project report (PDF)](CNN_MiniImageNet_Occlusion_Report.pdf)**

---

## Overview

**Part A: architecture study.** A single configurable PyTorch CNN class is used to train four architectures on a 33-class subset of mini-ImageNet (84×84 colour images; 13,200 train / 3,300 test, perfectly balanced). Together they vary every parameter in the brief: number of conv layers, number of filters, fully connected layers, max pooling vs stride-2 convolution, and training time (epochs), plus kernel size and dropout. All models share the same data, optimiser, seed and training loop, so differences come from the architecture alone.

**Part B: occlusion sensitivity.** Following [Zeiler & Fergus (2014)](https://arxiv.org/abs/1311.2901), a 15×15 gray window is slid over every pixel of 10 test images (7,056 occluded versions each). The model's probability for the true class is recorded at each position and plotted as a heatmap. This shows whether the model relies on the object itself or on background context.

## Results

### Part A: classification accuracy (3,300 test images, random guessing = 3%)

| Model | Architecture | Params | Best test accuracy (epoch) | Train − test gap after 10 epochs |
|---|---|---:|---:|---:|
| 1 | 2 conv (6 → 16), 3 fc | 0.71 M | 33.27% (4) | 62.1 pp |
| **2** | **3 conv (32 → 64 → 128), 3×3 kernels, max pool, 2 fc** | **2.20 M** | **40.55% (4)** | 57.5 pp |
| 3 | 4 conv (16 → 128), stride 2, no pooling, 1 fc | 0.17 M | 38.33% (6) | 50.3 pp |
| 4 | 4 conv AlexNet-style (64 → 384), 3 fc, dropout 0.5 | 4.57 M | 29.18% (9) | 2.5 pp |
| Ref. | Pretrained AlexNet, last layer retrained | 0.14 M trained | 78.79% (3) | 11.8 pp |

**Key findings**
- **Model 2 is the best from-scratch CNN: 40.55% on 33 classes, about 13× better than chance.** Its saved weights are included as `model2_3conv_2fc.pt`.
- **Stride beats pooling on efficiency:** replacing max pooling with stride-2 convolutions (Model 3) gets within 2.2 points of the best model with 13× fewer parameters and half the training time.
- **More capacity is not automatically better:** the widest model (Model 4, 4.6 M parameters, 50% dropout) under-fits and is the least accurate within 10 epochs.
- **Training time matters as much as architecture:** models without regularisation peak after 3–6 epochs and then over-fit badly (train accuracy climbs to 85–93% while test accuracy falls), so early stopping gives the best result.
- **Data is the main bottleneck:** a pretrained AlexNet reaches 78.79%, about twice the best from-scratch model, showing that learning features from only 400 images per class is the hard part. Because AlexNet was pretrained on ImageNet, which includes these classes, it is a reference point only.

### Part B: what the model looks at

- The model stays fully confident for most of the 10 images wherever the window is placed. Its evidence is spread across the image rather than tied to one region.
- Covering backgrounds (sky and water around a catamaran, ground around a lion, leaves around a ladybug) never lowers confidence, so there is no single background cue it depends on.
- Where confidence does drop, the sensitive region is **on the object**: the shell of a rhinoceros beetle, the tip of a lipstick.

## Repository contents

| File | Description |
|---|---|
| `CNN_MiniImageNet_Occlusion.ipynb` | Data loading, configurable CNN class, training loop, architecture study, AlexNet reference, occlusion experiment |
| `model2_3conv_2fc.pt` | Saved weights and config of the best model (Model 2), used in Part B |
| `CNN_MiniImageNet_Occlusion_Report.pdf` | Full report with tables, training curves and occlusion heatmaps |
| `requirements.txt` | Python dependencies |

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook CNN_MiniImageNet_Occlusion.ipynb
```

**Data:** the notebook expects the dataset in a folder named `AI_ML_Assignment_4_Dataset/` next to the notebook, with one sub-folder per class:

```
AI_ML_Assignment_4_Dataset/
├── train/<class_name>/*.jpg   # 400 images per class
└── test/<class_name>/*.jpg    # 100 images per class
```

The images are a 33-class subset of mini-ImageNet (see [mini-imagenet-tools](https://github.com/yaoyao-liu/mini-imagenet-tools)) and are not included in this repo because of their size.

To run only Part B, load `model2_3conv_2fc.pt` (see the occlusion section of the notebook); no retraining is needed.

All results come from a single run per model (seed 42) on an 8-core laptop CPU. Differences of about 1 point are within run-to-run noise.

## Tech stack

Python · PyTorch · torchvision · NumPy · pandas · Matplotlib · Pillow

---

**Author:** Varun Chander · [GitHub](https://github.com/varunchander1)
