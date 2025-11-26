import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from stochastic_resnet import stochastic_resnet_cifar   # <- import the previous model
import time
import os


# ============================================================
# Configuration (matching Huang et al. 2016)
# ============================================================
NUM_EPOCHS = 500
BATCH_SIZE = 128
INITIAL_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 10
P_L = 0.5    # final survival probability (paper default)
VALIDATION_SIZE = 5000


# ============================================================
# Prepare CIFAR-10 Data
# Default augmentation: random crop 32×32 (padding=4), horizontal flip
# Matches paper: Section 4 (CIFAR-10)
# ============================================================

def prepare_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    # Split into train/val (45k / 5k)
    train_dataset, val_dataset = random_split(
        full_train, [len(full_train) - VALIDATION_SIZE, VALIDATION_SIZE]
    )
    val_dataset.dataset.transform = transform_test  # no augmentation in validation

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


# ============================================================
# Learning Rate Schedule (Matching the paper)
#  - 0.1 initially
#  - divide by 10 at epoch 250
#  - divide by 10 at epoch 375
# ============================================================

def lr_schedule(optimizer, epoch):
    if epoch == 250:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    if epoch == 375:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


# ============================================================
# Training loop
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


# ============================================================
# Validation loop
# ============================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / total, correct / total


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data
    train_loader, val_loader, test_loader = prepare_data()

    # Create ResNet-110 with stochastic depth (3×18 blocks)
    model = stochastic_resnet_cifar(layers=[18, 18, 18], num_classes=NUM_CLASSES, p_L=P_L)
    model = model.to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=INITIAL_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )

    best_val_acc = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        lr_schedule(optimizer, epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"LR={optimizer.param_groups[0]['lr']:.4f} "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_stochastic_resnet_cifar10.pth")

    # Total training time
    total_time = (time.time() - start_time) / 3600
    print(f"\nTraining complete in {total_time:.2f} hours.")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

    # Evaluate on test set
    model.load_state_dict(torch.load("best_stochastic_resnet_cifar10.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
