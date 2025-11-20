#!/usr/bin/env python3
"""
rbm_nrelu_visual_all.py (macOS SAFE VERSION)

RBM with Gaussian visible & NReLU hidden + full visualizations:
 - 2D filters
 - Reconstructions
 - 3D pre-activation surface + ReLU boundary
 - Rotating 3D GIF (macOS compatible)
 - t-SNE 2D & 3D
 - Feature evolution GIF across epochs

Dependencies:
    pip install torch torchvision scikit-learn imageio matplotlib
"""
import os
import math
import argparse
import random
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs_rbm", type=int, default=20)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--lr_rbm", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--save_dir", type=str, default="./rbm_visual_all")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--tsne_samples", type=int, default=3000)
parser.add_argument("--pca_subset", type=int, default=5000)
parser.add_argument("--gif_fps", type=int, default=12)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, "visuals"), exist_ok=True)
os.makedirs(os.path.join(args.save_dir, "animations"), exist_ok=True)

# -----------------------------------------------------------
# Seeds
# -----------------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device(args.device)

# -----------------------------------------------------------
# Load MNIST
# -----------------------------------------------------------
mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

def dataset_to_tensors(dataset):
    xs, ys = [], []
    for img, lab in dataset:
        xs.append(img.view(-1))
        ys.append(lab)
    return torch.stack(xs).float(), torch.tensor(ys, dtype=torch.long)

x_train, y_train = dataset_to_tensors(mnist_train)
x_test, y_test = dataset_to_tensors(mnist_test)

# normalize
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / (std + 1e-8)
x_test  = (x_test - mean)  / (std + 1e-8)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size)

INPUT_DIM = 784
HIDDEN = args.hidden_size

# -----------------------------------------------------------
# RBM CLASS (Gaussian visible + NReLU hidden)
# -----------------------------------------------------------
class RBM:
    def __init__(self, n_visible, n_hidden):
        self.W = torch.randn(n_visible, n_hidden) * 0.01
        self.b_v = torch.zeros(n_visible)
        self.b_h = torch.zeros(n_hidden)

        self.W_m = torch.zeros_like(self.W)
        self.bv_m = torch.zeros_like(self.b_v)
        self.bh_m = torch.zeros_like(self.b_h)

        self.W = self.W.to(device)
        self.b_v = self.b_v.to(device)
        self.b_h = self.b_h.to(device)

    def sample_hidden_nrelu(self, v):
        x = v @ self.W + self.b_h
        var = torch.sigmoid(x).clamp(min=1e-8)
        noise = torch.randn_like(x) * torch.sqrt(var)
        h = F.relu(x + noise)
        return h, x

    def hidden_det(self, v):
        x = v @ self.W + self.b_h
        return F.relu(x), x

    def reconstruct_visible(self, h):
        return h @ self.W.t() + self.b_v

    def cd1_update(self, v0, lr, momentum):
        h0, _ = self.sample_hidden_nrelu(v0)
        v1 = self.reconstruct_visible(h0)
        h1, _ = self.sample_hidden_nrelu(v1)

        B = v0.shape[0]
        dW = (v0.t() @ h0 - v1.t() @ h1) / B
        dbv = (v0 - v1).mean(0)
        dbh = (h0 - h1).mean(0)

        self.W_m = momentum * self.W_m + lr * dW
        self.bv_m = momentum * self.bv_m + lr * dbv
        self.bh_m = momentum * self.bh_m + lr * dbh

        self.W += self.W_m
        self.b_v += self.bv_m
        self.b_h += self.bh_m

        return ((v0 - v1)**2).mean().item()

# -----------------------------------------------------------
# VISUALIZATION HELPERS
# -----------------------------------------------------------
def save_filters_grid(W, filename, n_cols=8, max_filters=64):
    W = W.detach().cpu().numpy()
    F = min(W.shape[1], max_filters)
    rows = math.ceil(F / n_cols)

    fig, axes = plt.subplots(rows, n_cols, figsize=(n_cols, rows))
    axes = axes.flatten()

    for i in range(rows*n_cols):
        ax = axes[i]
        ax.axis("off")
        if i < F:
            img = W[:,i].reshape(28,28)
            ax.imshow(img, cmap='gray')

    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def save_reconstruction_example(rbm, x, filename):
    with torch.no_grad():
        h,_ = rbm.sample_hidden_nrelu(x.unsqueeze(0).to(device))
        recon = rbm.reconstruct_visible(h).squeeze(0).cpu()

    fig,axs = plt.subplots(1,2,figsize=(5,3))
    axs[0].imshow(x.view(28,28), cmap='gray'); axs[0].axis('off')
    axs[1].imshow(recon.view(28,28), cmap='gray'); axs[1].axis('off')
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

# -----------------------------------------------------------
# 3D PREACTIVATION SURFACE + ROTATING GIF (MAC-SAFE)
# -----------------------------------------------------------
def plot_preact_surface_on_pca(rbm, pca, filename_prefix, neuron_idx=0, grid_n=60):

    xs = np.linspace(-3,3,grid_n)
    ys = np.linspace(-3,3,grid_n)
    Xg, Yg = np.meshgrid(xs,ys)

    XY = np.stack([Xg.ravel(),Yg.ravel()], axis=1)

    coords = np.zeros((XY.shape[0], pca.n_components_))
    coords[:,:2] = XY

    V = pca.inverse_transform(coords)
    Vt = torch.tensor(V, dtype=torch.float32, device=device)

    with torch.no_grad():
        Z = (Vt @ rbm.W[:,neuron_idx] + rbm.b_h[neuron_idx]).cpu().numpy().reshape(grid_n, grid_n)

    # static surface
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xg, Yg, Z, cmap='viridis')
    ax.contour(Xg, Yg, Z, levels=[0], colors='red')
    fig.savefig(f"{filename_prefix}_surface.png")
    plt.close(fig)

    # rotating GIF (macOS safe: use fig.canvas.buffer_rgba())
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xg, Yg, Z, cmap='viridis')
    ax.contour(Xg, Yg, Z, levels=[0], colors='red')

    frames = []
    out_gif = f"{filename_prefix}_rotating.gif"

    for angle in range(0, 360, 360 // args.gif_fps):
        ax.view_init(30, angle)
        fig.canvas.draw()

        img = np.asarray(fig.canvas.buffer_rgba())
        frames.append(img)

    imageio.mimsave(out_gif, frames, fps=args.gif_fps)
    plt.close(fig)

    return out_gif

# -----------------------------------------------------------
# t-SNE VISUALIZATION
# -----------------------------------------------------------
def tsne_and_plot(features, labels, prefix):
    idx = np.random.choice(len(features), min(args.tsne_samples, len(features)), replace=False)
    feats = features[idx]
    labs = labels[idx]

    tsne2 = TSNE(n_components=2, max_iter=800, perplexity=30)
    Y2 = tsne2.fit_transform(feats)
    fig,ax = plt.subplots(figsize=(7,6))
    sc=ax.scatter(Y2[:,0],Y2[:,1],c=labs,cmap='tab10',s=6)
    fig.savefig(prefix+"_tsne2.png"); plt.close(fig)

    tsne3 = TSNE(n_components=3, max_iter=800, perplexity=30)
    Y3 = tsne3.fit_transform(feats)
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(Y3[:,0],Y3[:,1],Y3[:,2],c=labs,cmap='tab10',s=6)
    fig.savefig(prefix+"_tsne3.png"); plt.close(fig)

# -----------------------------------------------------------
# FEATURE EVOLUTION ANIMATION (EPOCHS)
# -----------------------------------------------------------
def animate_feature_evolution(history, labels, out_path):

    fig,ax = plt.subplots(figsize=(6,6))
    scat = ax.scatter(history[0][:,0], history[0][:,1], c=labels, cmap='tab10', s=6)

    def update(i):
        scat.set_offsets(history[i])
        ax.set_title(f"Epoch {i}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=500)
    ani.save(out_path, writer='pillow', fps=6)
    plt.close(fig)


# -----------------------------------------------------------
# TRAIN RBM + RECORD FEATURE EVOLUTION
# -----------------------------------------------------------
rbm = RBM(INPUT_DIM, HIDDEN)
subset_idx = np.random.choice(len(x_train), args.pca_subset, replace=False)
x_subset = x_train[subset_idx].to(device)
labels_subset = y_train[subset_idx].numpy()

feature_history = []

print("Beginning RBM pretraining...")
for ep in range(args.epochs_rbm):
    loss_sum=0; nb=0
    for xb,_ in train_loader:
        xb=xb.to(device)
        loss_sum+=rbm.cd1_update(xb,args.lr_rbm,args.momentum); nb+=1
    print(f"Epoch {ep+1}/{args.epochs_rbm} recon_mse={loss_sum/nb:.6f}")

    with torch.no_grad():
        feats,_ = rbm.hidden_det(x_subset)
        feats = feats.cpu().numpy()

    if ep==0:
        pca_evo = PCA(n_components=2)
        pca_evo.fit(feats)

    feature_history.append(pca_evo.transform(feats))

# -----------------------------------------------------------
# FINAL VISUALIZATIONS
# -----------------------------------------------------------
print("Generating final visualizations...")

save_filters_grid(rbm.W, f"{args.save_dir}/visuals/filters.png")
save_reconstruction_example(rbm, x_test[0], f"{args.save_dir}/visuals/recon.png")

# PCA for ReLU surface
pca_input = PCA(n_components=10)
pca_input.fit(x_train[:10000].numpy())

gif_path = plot_preact_surface_on_pca(
    rbm, pca_input, 
    f"{args.save_dir}/visuals/neuron0",
    neuron_idx=0
)
print("3D rotating GIF saved:", gif_path)

# t-SNE
with torch.no_grad():
    h_tsne,_ = rbm.hidden_det(x_train.to(device))
    tsne_and_plot(h_tsne.cpu().numpy(), y_train.numpy(), f"{args.save_dir}/visuals/rbm")

# feature evolution animation
evo_path = f"{args.save_dir}/animations/feature_evolution.gif"
animate_feature_evolution(feature_history, labels_subset, evo_path)
print("Feature evolution GIF saved:", evo_path)

print("DONE! All results saved in:", args.save_dir)
