# 📌 Overview

This repository provides a complete PyTorch reproduction of Stochastic Depth ResNet used in the paper:

**Deep Networks with Stochastic Depth**  
Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger (2016)

The implementation strictly matches the paper’s CIFAR training setup:

- ResNet-110 architecture (3×18 residual blocks = 54 blocks total)
- Stochastic depth with linear survival probability decay
- \( p_\ell = 1 \to p^L = 0.5 \) (default)
- 500 training epochs
- SGD + Nesterov momentum, weight decay, LR schedule
- CIFAR-10 data augmentation identical to the paper

This enables reproducing reported results such as 5.25% test error (stochastic depth ResNet-110).

## 📁 Project Structure

```
.
├── stochastic_resnet.py         # Stochastic Depth ResNet implementation
├── train_cifar_stochastic_depth.py
├── README.md                    # (this file)
├── data/                        # CIFAR-10 automatically downloads here
└── best_stochastic_resnet_cifar10.pth   # saved model (after training)
```

## 🚀 Installation

1. Clone repository  
    ```bash
    git clone <your-repo-url>
    cd stochastic-depth-cifar
    ```

2. Install dependencies  
    ```bash
    pip install torch torchvision tqdm
    ```

    (If using GPU, install the CUDA version of PyTorch from https://pytorch.org.)

## 📦 Dataset: CIFAR-10

CIFAR-10 is automatically downloaded to `./data/`.

Transforms match the paper exactly ([1603.09382v3](https://arxiv.org/abs/1603.09382v3)):

- Random crop 32×32 (padding = 4)
- Random horizontal flip
- Normalize using dataset mean/std

Validation split:
- 45,000 train
- 5,000 validation (no augmentation)
- 10,000 test

## 🧠 Model: Stochastic Depth ResNet

Defined in `stochastic_resnet.py`.

Key characteristics:

- Basic 2-conv ResNet block (CIFAR version)
- Survival probability per block:  
  \( p_\ell = 1 - \frac{\ell}{L} (1 - p_L) \)

During training:
- Each block is dropped with probability \( 1 - p_\ell \)
- Dropping is per mini-batch, matching the paper

During evaluation:
- All blocks active
- Residual outputs scaled by \( p_\ell \) (paper Eq. 5)

Architecture used (ResNet-110):  
Input → Conv → BN → ReLU  
Stage1: 18 residual blocks (16 filters)  
Stage2: 18 residual blocks (32 filters)  
Stage3: 18 residual blocks (64 filters)  
AvgPool → FC

## 🏋️‍♂️ Training

Run:

```bash
python train_cifar_stochastic_depth.py
```

Training hyperparameters (matching paper)

| Parameter | Value |
|-----------|-------|
| Epochs | 500 |
| Batch size | 128 |
| Optimizer | SGD + Nesterov |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| Learning rate schedule | 0.1 → 0.01 at epoch 250 → 0.001 at epoch 375 |
| Augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip |
| Architecture | ResNet-110 |
| Stochastic depth \( p^L \) | 0.5 (default) |

Source: Details from pages 8–9 of the paper ([1603.09382v3](https://arxiv.org/abs/1603.09382v3)).

## 📈 Logging Output

The script prints:

- Epoch number
- Current learning rate
- Training loss & accuracy
- Validation loss & accuracy

Example:

```
Epoch [120/500] LR=0.1000 Train Acc=0.82 Val Acc=0.80
```

The script saves the best validation model to:

`best_stochastic_resnet_cifar10.pth`

## 🧪 Testing

After training, the script automatically evaluates on CIFAR-10 test set:

Test Accuracy: 94.75%

(This number is approximate; exact accuracy depends on hardware & randomness.)

## 📊 Expected Performance

Based on the original paper:

| Model | CIFAR-10 Test Error |
|-------|---------------------|
| ResNet-110 (constant depth) | 6.41% |
| Stochastic Depth ResNet-110 | 5.25% |

Source: Table 1 of the uploaded paper ([1603.09382v3](https://arxiv.org/abs/1603.09382v3)).

## 📚 Citation

If you use this code, cite the original authors:

```bibtex
@article{huang2016deep,
  title={Deep Networks with Stochastic Depth},
  author={Huang, Gao and Sun, Yu and Liu, Zhuang and Sedra, Daniel and Weinberger, Kilian Q},
  journal={arXiv preprint arXiv:1603.09382},
  year={2016}
}
```

## 🔧 Customization

You can easily modify:

- For CIFAR-100:  
  `model = stochastic_resnet_cifar([18,18,18], num_classes=100)`

- For different survival probability:  
  `model = stochastic_resnet_cifar(p_L=0.2)`

- For smaller quick-train models:  
  `model = stochastic_resnet_cifar([3,3,3])`

## 🧩 Troubleshooting

- **Training is too slow**  
  → Reduce depth: [6, 6, 6] blocks  
  → Reduce epochs from 500 → 200

- **Training diverges early**  
  → Use warmup LR (1–5 epochs at LR=0.01)

- **GPU runs out of memory**  
  → Lower batch size or use AMP (torch.cuda.amp)

## 🏁 Final Notes

This repository provides an accurate and faithful reproduction of the CIFAR-10 experiments from:

“Deep Networks with Stochastic Depth”  
Huang et al., 2016

All hyperparameters follow the original paper exactly, enabling reproducibility and high performance
