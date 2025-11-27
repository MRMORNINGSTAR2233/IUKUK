# nin_3d.py
# Run: python nin_3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---- Synthetic 3D dataset (toy) ----
class RandomVolumeDataset(Dataset):
    def __init__(self, n_samples=200, channels=1, D=32, H=32, W=32, n_classes=2):
        self.n = n_samples
        self.n_classes = n_classes
        self.data = torch.randn(n_samples, channels, D, H, W)
        self.labels = torch.randint(0, n_classes, (n_samples,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ---- 3D MLPConv block ----
class MLPConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class NIN3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            MLPConvBlock3D(in_channels, 64),
            nn.MaxPool3d(2),
            nn.Dropout(0.3),

            MLPConvBlock3D(64, 128),
            nn.MaxPool3d(2),
            nn.Dropout(0.4),

            MLPConvBlock3D(128, 256),
            nn.MaxPool3d(2),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Conv3d(256, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # global average pooling over 3D spatial dims -> (B, num_classes, 1,1,1)
        x = F.adaptive_avg_pool3d(x, (1,1,1))
        x = x.view(x.size(0), -1)
        return x

# ---- Training loop (toy) ----
def train_epoch(model, loader, optimizer, device):
    model.train()
    import math
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Batch {i}/{len(loader)} loss {loss.item():.4f}")

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Accuracy: {100.*correct/total:.2f}%")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = RandomVolumeDataset(n_samples=400, channels=1, D=32, H=32, W=32, n_classes=2)
    val_ds = RandomVolumeDataset(n_samples=100, channels=1, D=32, H=32, W=32, n_classes=2)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = NIN3D(in_channels=1, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, 4):
        print(f"Epoch {epoch}")
        train_epoch(model, train_loader, optimizer, device)
        evaluate(model, val_loader, device)

if __name__ == '__main__':
    main()
