import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# -------------------------
# Utility conv layers
# -------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# -------------------------
# Stochastic Basic Block
# -------------------------
class StochasticBasicBlock(nn.Module):
    """
    Basic ResNet block with stochastic depth.
    - p_survive: survival probability p_l for this block.
    Behavior:
      - train(): sample b ~ Bernoulli(p_survive).
          if b==1: out = x + f(x)
          if b==0: out = x  (skip f(x) entirely)
      - eval(): deterministic: out = x + p_survive * f(x)
        (matches Eq. (5) of the paper).
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, p_survive: float = 1.0):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        # survival probability for this block (float between 0 and 1)
        self.p_survive = float(p_survive)

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        # compute residual function f(x) (we will sometimes skip using it)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.training:
            # sample once per forward pass (per mini-batch)
            rand = torch.rand(1, device=out.device).item()
            if rand < self.p_survive:
                # block is active
                out = identity + out
                out = self.relu(out)
            else:
                # block is bypassed -> identity path only
                out = identity
        else:
            # evaluation: deterministic, scale residual by p_survive (paper Eq. (5))
            out = identity + self.p_survive * out
            out = self.relu(out)

        return out


# -------------------------
# Stochastic ResNet
# -------------------------
class StochasticResNet(nn.Module):
    """
    A lightweight ResNet-like model using StochasticBasicBlock.
    layers: list with number of blocks in each stage, e.g. [n1, n2, n3]
    p_L: final survival probability for the last block; p0 is implicitly 1.0
    The code computes a linear-decay p_l per block (see paper Eq. (4)).
    """

    def __init__(self, block, layers: List[int], num_classes=10, p_L=0.5):
        super().__init__()
        self.inplanes = 16  # typical for CIFAR-style experiments
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # total number of residual blocks L
        self.L = sum(layers)

        # create list of survival probabilities for each block (linear decay)
        # p_0 is for the "input" and is 1; for blocks we compute p_l following Eq (4)
        self.p_list = self._compute_survival_probabilities(p_L)

        # Build layers (three stages commonly)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, start_block_idx=0)
        idx = layers[0]
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, start_block_idx=idx)
        idx += layers[1]
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, start_block_idx=idx)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # initialize weights
        self._init_weights()

    def _compute_survival_probabilities(self, p_L: float) -> List[float]:
        # p_l = 1 - (l / L) * (1 - p_L), l in [1..L]
        # We return a list of length L where index l-1 corresponds to p_l.
        L = max(1, self.L)
        p_list = [1.0 - (float(l) / L) * (1.0 - float(p_L)) for l in range(1, L+1)]
        return p_list

    def _make_layer(self, block, planes, blocks, stride=1, start_block_idx=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample using 1x1 conv (common for CIFAR variants they use different trick; this is standard)
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        for i in range(blocks):
            # survival prob for this block (global index = start_block_idx + i)
            p = self.p_list[start_block_idx + i]
            if i == 0:
                layers.append(block(self.inplanes, planes, stride, downsample, p_survive=p))
            else:
                layers.append(block(self.inplanes, planes, 1, None, p_survive=p))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)   # CIFAR style first conv
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# -------------------------
# Helper to construct a CIFAR-ResNet (e.g., 110-layer)
# Example: for the ResNet-110 in the paper they use 3 groups each with 18 blocks
# (which corresponds to 54 residual blocks). Here layers=[18,18,18]
# -------------------------
def stochastic_resnet_cifar(layers=[18, 18, 18], num_classes=10, p_L=0.5):
    return StochasticResNet(StochasticBasicBlock, layers, num_classes=num_classes, p_L=p_L)


# -------------------------
# Example usage (training skeleton)
# -------------------------
if __name__ == "__main__":
    # quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = stochastic_resnet_cifar(layers=[3,3,3], num_classes=10, p_L=0.5).to(device)
    # small network: 3 blocks per stage (for quick runs)
    x = torch.randn(8, 3, 32, 32).to(device)
    model.train()
    out = model(x)
    print("train out shape:", out.shape)
    model.eval()
    out_eval = model(x)
    print("eval out shape:", out_eval.shape)

    # Training skeleton (very short)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss()
    # for epoch in range(epochs):
    #     model.train()
    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         logits = model(images)
    #         loss = criterion(logits, labels)
    #         loss.backward()
    #         optimizer.step()
    #     # validation
    #     model.eval()
    #     # run validation loop...
