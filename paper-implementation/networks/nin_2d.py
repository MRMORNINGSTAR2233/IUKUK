# nin_2d.py
# Run: python nin_2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---- Model: NIN-style ----
class MLPConvBlock2D(nn.Module):
    """
    mlpconv block: spatial conv (k x k) followed by two 1x1 convs (MLP across channels).
    This mirrors the architecture described in the paper.
    """
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding),  # spatial conv
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),  # 1x1 conv (MLP layer)
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),  # another 1x1 conv (MLP layer)
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class NIN2D(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Example architecture similar flavor to paper:
        self.features = nn.Sequential(
            MLPConvBlock2D(3, 192, kernel=5, padding=2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.5),

            MLPConvBlock2D(192, 160, kernel=5, padding=2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.5),

            MLPConvBlock2D(160, 96, kernel=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.5),
        )
        # Last mlpconv should output num_classes channels for global average pooling -> softmax
        self.classifier = nn.Sequential(
            nn.Conv2d(96, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            # global avg pooling will be applied in forward
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)              # shape: (B, num_classes, H, W)
        x = F.adaptive_avg_pool2d(x, (1,1)) # shape: (B, num_classes, 1, 1)
        x = x.view(x.size(0), -1)           # shape: (B, num_classes)
        return x

# ---- Training setup ----
def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f"Train Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.4f}")

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss_sum += F.cross_entropy(out, target, reduction='sum').item()
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.numel()
    print(f"Test: Avg loss {loss_sum/total:.4f}, Accuracy {100.0*correct/total:.2f}%")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data transforms (basic)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    test_loader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    model = NIN2D(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    for epoch in range(1, 6):  # small quick run
        train_one_epoch(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()
