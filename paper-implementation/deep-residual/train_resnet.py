#!/usr/bin/env python
"""
Train ResNet (He et al. "Deep Residual Learning for Image Recognition")
on CIFAR-10 or an ImageNet-style custom dataset.

- Implements BasicBlock (ResNet-18/34) and Bottleneck (ResNet-50/101/152)
- Training loop mimics original paper: SGD + momentum, weight decay, step LR
- Supports Apple MPS backend for M1/M2 Macs.

Usage examples:

  # Train ResNet-18 on CIFAR-10 (default settings)
  python train_resnet.py --model resnet18 --dataset cifar10 --epochs 200

  # Train ResNet-50 on a custom ImageNet-style dataset
  python train_resnet.py --model resnet50 --dataset imagenet \
      --data-dir /path/to/imagenet/root --batch-size 64 --epochs 90

"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder


# ---------------------------------------------------------------------
# 1. Device selection (M2 Mac with MPS support)
# ---------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple MPS device.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA device.")
        return torch.device("cuda")
    else:
        print("Using CPU.")
        return torch.device("cpu")


# ---------------------------------------------------------------------
# 2. ResNet building blocks (as in the original paper)
# ---------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding (no bias, as BN follows)."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution (projection shortcut / bottleneck)."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34:
    Two 3x3 convs with an identity (or projection) shortcut.

    y = ReLU( F(x) + shortcut(x) )
    where F(x) = conv3x3 -> BN -> ReLU -> conv3x3 -> BN
    """

    expansion = 1  # output_channels = planes * expansion

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample  # projection for shortcut if needed
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if dims changed, use downsample on identity
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet-50/101/152:
    1x1 (reduce) -> 3x3 -> 1x1 (expand) + shortcut

    F(x) = conv1x1 -> BN -> ReLU -> conv3x3 -> BN -> ReLU -> conv1x1 -> BN
    Output channels = planes * expansion (where expansion = 4).
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        # 1x1 reduce
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3x3 conv
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        # 1x1 expand
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 1x1 reduce
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 expand
        out = self.conv3(out)
        out = self.bn3(out)

        # projection shortcut if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Generic ResNet implementation.
    Matches the structure in the original paper:

    conv1: 7x7, stride 2 (for ImageNet-sized inputs)
    maxpool: 3x3, stride 2
    layer1: residual blocks
    layer2: ...
    layer3: ...
    layer4: ...
    global avgpool + FC
    """

    def __init__(self, block, layers, num_classes=1000, in_channels=3):
        """
        block: BasicBlock or Bottleneck
        layers: list with number of blocks per layer, e.g. [2,2,2,2] for ResNet-18
        num_classes: classification head dimension
        """
        super().__init__()
        self.inplanes = 64

        # Initial stem: for ImageNet-like (224x224). For CIFAR we’ll handle differently.
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling + fully-connected classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization (Kaiming normal as in He et al.)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Create one stage (layer1 / layer2 / layer3 / layer4) of ResNet.
        `blocks` = number of residual blocks in this layer.
        """
        downsample = None

        # If we change spatial resolution (stride != 1) OR number of channels,
        # we need a projection shortcut (1x1 conv).
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block may have stride > 1
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        # Remaining blocks (stride 1)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling + FC
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Convenience constructors for common ResNet depths:

def resnet18(num_classes=1000, in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def resnet34(num_classes=1000, in_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


def resnet50(num_classes=1000, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)


def resnet101(num_classes=1000, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_channels)


def resnet152(num_classes=1000, in_channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, in_channels)


# ---------------------------------------------------------------------
# 3. Data: CIFAR-10 & ImageNet-style folder
# ---------------------------------------------------------------------

def get_cifar10_loaders(batch_size, num_workers=4, data_dir="./data"):
    """
    Returns training / test DataLoader for CIFAR-10.

    Transforms approximate those used in the ResNet CIFAR experiments:
    - RandomCrop(32, padding=4)
    - RandomHorizontalFlip
    - Normalize by mean/std
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    testset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # MPS is fine without this; can set True on CUDA
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return trainloader, testloader, 10  # num_classes=10


def get_imagenet_like_loaders(batch_size, num_workers, data_dir, image_size=224):
    """
    ImageNet-style folder:
      data_dir/train/class_x/xxx.png
      data_dir/val/class_x/xxx.png

    For M2 Mac, you'll probably want a smaller subset or reduced resolution.
    """
    # Standard ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    trainset = ImageFolder(train_dir, transform=train_transform)
    valset = ImageFolder(val_dir, transform=val_transform)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    num_classes = len(trainset.classes)
    return trainloader, valloader, num_classes


# ---------------------------------------------------------------------
# 4. Training & evaluation
# ---------------------------------------------------------------------

def train_one_epoch(model, device, dataloader, criterion, optimizer, epoch, print_freq=100):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % print_freq == 0:
            avg_loss = running_loss / total
            acc = 100.0 * correct / total
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {elapsed:.1f}s"
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, device, dataloader, criterion, desc="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(f"{desc}: Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


# ---------------------------------------------------------------------
# 5. Main (argument parsing and training loop)
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ResNet (He et al.) training script")

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="ResNet model variant",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "imagenet"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root directory for dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Initial learning rate (SGD)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=None,
        help="LR decay milestones, e.g. --milestones 100 150",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="LR decay factor for MultiStepLR",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="best_resnet.pth",
        help="Path to save best model",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # Data
    if args.dataset == "cifar10":
        print("Using CIFAR-10 dataset.")
        trainloader, testloader, num_classes = get_cifar10_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
        )
        # CIFAR-10 images are 32x32: in practice, paper used a smaller initial stem (3x3).
        # For simplicity, we keep the standard stem but you could modify for CIFAR.
    else:
        print("Using ImageNet-style dataset (ImageFolder).")
        trainloader, testloader, num_classes = get_imagenet_like_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            image_size=224,
        )

    # Model
    model_map = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
    }
    model_fn = model_map[args.model]
    model = model_fn(num_classes=num_classes, in_channels=3).to(device)

    print(f"Model: {args.model}, num_classes={num_classes}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler (similar spirit to paper)
    if args.milestones is None:
        # Good defaults:
        # - CIFAR-10: 200 epochs, decay at 100 & 150
        # - ImageNet: 90 epochs, could do 30 & 60
        if args.dataset == "cifar10":
            milestones = [100, 150]
        else:
            milestones = [30, 60]
    else:
        milestones = args.milestones

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_one_epoch(
            model,
            device,
            trainloader,
            criterion,
            optimizer,
            epoch,
            print_freq=100,
        )

        val_loss, val_acc = evaluate(
            model,
            device,
            testloader,
            criterion,
            desc="Val/Test",
        )

        scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"Saved new best model with acc={best_acc:.2f}% to {args.save_path}")

    print(f"\nTraining complete. Best Val/Test Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()

# python train_resnet.py \
#   --model resnet18 \
#   --dataset cifar10 \
#   --epochs 200 \
#   --batch-size 128 \
#   --lr 0.1
