# 🧠 ResNet Implementation (Deep Residual Learning for Image Recognition)

This repository provides a clean and complete PyTorch implementation of **ResNet** from the landmark paper:

> **"Deep Residual Learning for Image Recognition"**  
> Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
> *CVPR 2016*  
> [[Paper]](https://arxiv.org/abs/1512.03385)

Residual Networks (ResNets) introduced skip-connections that solve the degradation problem in deep neural networks, enabling effective training of models with **50+, 100+, even 1000+** layers.

This repo includes:
- Full **ResNet-18 / 34 / 50 / 101 / 152** architectures (BasicBlock & Bottleneck)
- **CIFAR-10** training support (default for easy testing)
- **ImageNet-style dataset** support (`ImageFolder`)
- Optimized for **Apple Silicon (M1/M2)** using **Metal (MPS)**
- SGD + Momentum training exactly like original paper

---

## ⚙️ Requirements

| Component | Version |
|----------|---------|
| Python   | ≥ 3.8 |
| PyTorch  | Latest (MPS supported on macOS ≥ 12.3) |
| Torchvision | Latest |

Install dependencies:

```bash
pip install torch torchvision
```

## 🚀 Quick Start

### 📌 Train on CIFAR-10 (Recommended for M1/M2 Macs)

```bash
python train_resnet.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.1
```

This will automatically:
- ✔ Download CIFAR-10
- ✔ Detect and run on MPS if available
- ✔ Save best checkpoint to `best_resnet.pth`

## 🗂 Folder Structure

```
.
├── train_resnet.py      # Main runnable script
├── README.md            # Documentation (this file)
└── data/                # CIFAR-10 or custom dataset storage
```

## 🧬 Supported Models

| Command Option | Depth | Block Type |
|---------------|-------|------------|
| resnet18      | 18 layers | BasicBlock |
| resnet34      | 34 layers | BasicBlock |
| resnet50      | 50 layers | Bottleneck |
| resnet101     | 101 layers | Bottleneck |
| resnet152     | 152 layers | Bottleneck |

Example:

```bash
python train_resnet.py --model resnet50 --dataset cifar10
```

## 📦 Training on Custom Data (ImageNet-style)

Structure your dataset like:

```
/path/to/dataset/
    ├── train/
    │   ├── class1/*.jpg
    │   ├── class2/*.jpg
    └── val/
            ├── class1/*.jpg
            ├── class2/*.jpg
```

Run:

```bash
python train_resnet.py \
    --model resnet50 \
    --dataset imagenet \
    --data-dir /path/to/dataset \
    --epochs 90 \
    --batch-size 64
```

## 📍 Training Settings (Paper-Inspired)

| Setting | Value |
|---------|-------|
| Optimizer | SGD + momentum |
| Momentum | 0.9 |
| Weight Decay | 1e-4 |
| Initial LR | 0.1 |
| LR schedule | Step decay (×0.1 at milestones) |

Example CIFAR-10 defaults:
- Epochs: 200
- LR drops at 100 & 150

## 📊 Expected Results (Approx.)

| Model | CIFAR-10 Top-1 Acc |
|-------|-------------------|
| ResNet-18 | ~94% |
| ResNet-34 | ~95%+ |
| ResNet-50 | ~95%+ (slower on M2) |

*(Training on CPU is not recommended)*

## 💾 Saving Models

Best checkpoint auto-saved to:

```
best_resnet.pth
```

You can modify the save path:

```bash
--save-path my_model.pth
```

## 🧠 What's Next?

Here are some ideas to extend this repo:

- Add TensorBoard/W&B logging
- Add cutmix/mixup for CIFAR-10
- Train deeper variants (ResNet-101/152) on beefier hardware
- Port training logs for paper-style comparison plots
