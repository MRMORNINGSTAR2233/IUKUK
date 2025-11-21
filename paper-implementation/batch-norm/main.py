import torch
import torch.nn as nn

class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        num_features: number of channels (input feature dimension)
        eps: small value to avoid division by zero
        momentum: used for running mean/var during inference
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # Learnable scale (gamma) & shift (beta)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running stats for inference
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Compute mean/variance across batch
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Normalize
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            # Use stored statistics during inference
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # Scale and shift
        return self.gamma * x_hat + self.beta


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # Learnable params
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # x shape: (N, C, H, W)
        if self.training:
            # Compute per-channel stats across batch + spatial dims
            batch_mean = x.mean(dim=[0, 2, 3])
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)

            # Normalize: need unsqueeze to broadcast to NCHW
            x_hat = (x - batch_mean[None, :, None, None]) / torch.sqrt(
                batch_var[None, :, None, None] + self.eps
            )

            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

        else:
            x_hat = (x - self.running_mean[None, :, None, None]) / torch.sqrt(
                self.running_var[None, :, None, None] + self.eps
            )

        # Scale + shift
        return self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]


#Custom BatchNorm layers can now be used in neural network models.
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = BatchNorm2d(64)
        self.fc = nn.Linear(64*8*8, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)
