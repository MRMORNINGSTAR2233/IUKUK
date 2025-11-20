#!/usr/bin/env python3
"""
rbm_nrelu_cifar10.py  (MAC-SAFE VERSION)

CIFAR-10 RBM + NReLU training and full visualization:
 - CIFAR-10 (32x32x3 = 3072 dims)
 - Gaussian visible RBM
 - Noisy ReLU hidden units (NReLU)
 - Filter visualization (RGB)
 - Reconstructions
 - PCA-based ReLU boundary surface
 - Rotating 3D GIF (buffer_rgba for macOS)
 - t-SNE 2D & 3D
 - Feature evolution GIF

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
# CLI SETUP
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs_rbm", type=int, default=25)
parser.add_argument("--hidden_size", type=int, default=500)
parser.add_argument("--lr_rbm", type=float, default=1e-4)
parser.add_argument("--momentum_rbm", type=float, default=0.9)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--save_dir", type=str, default="./rbm_cifar10_visuals")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--tsne_samples", type=int, default=4000)
parser.add_argument("--pca_subset", type=int, default=6000)
parser.add_argument("--gif_fps", type=int, default=12)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, "visuals"), exist_ok=True)
os.makedirs(os.path.join(args.save_dir, "animations"), exist_ok=True)

# -----------------------------------------------------------
# SEED
# -----------------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device(args.device)

# -----------------------------------------------------------
# LOAD CIFAR-10
# -----------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
testset  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

def flatten_dataset(ds):
    X = []
    Y = []
    for img, y in ds:
        X.append(img.view(-1))
        Y.append(y)
    return torch.stack(X).float(), torch.tensor(Y, dtype=torch.long)

x_train, y_train = flatten_dataset(trainset)
x_test , y_test  = flatten_dataset(testset)

# normalize globally
mean = x_train.mean()
std  = x_train.std()
x_train = (x_train - mean) / (std + 1e-8)
x_test  = (x_test - mean) / (std + 1e-8)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

INPUT_DIM = 32*32*3
HIDDEN = args.hidden_size

# -----------------------------------------------------------
# RBM (Gaussian visible + NReLU)
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

    def cd1(self, v0, lr, momentum):
        h0,_ = self.sample_hidden_nrelu(v0)
        v1 = self.reconstruct_visible(h0)
        h1,_ = self.sample_hidden_nrelu(v1)

        B = v0.shape[0]
        dW  = (v0.t() @ h0 - v1.t() @ h1) / B
        dbv = (v0 - v1).mean(0)
        dbh = (h0 - h1).mean(0)

        # Clip gradients to prevent explosion
        dW = torch.clamp(dW, -1.0, 1.0)
        dbv = torch.clamp(dbv, -1.0, 1.0)
        dbh = torch.clamp(dbh, -1.0, 1.0)

        self.W_m  = momentum*self.W_m  + lr*dW
        self.bv_m = momentum*self.bv_m + lr*dbv
        self.bh_m = momentum*self.bh_m + lr*dbh

        self.W  += self.W_m
        self.b_v += self.bv_m
        self.b_h += self.bh_m

        return ((v0 - v1)**2).mean().item()

# -----------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -----------------------------------------------------------
def save_cifar_filters(W, filename, max_filters=36):
    """
    CIFAR filters: reshape to (3,32,32)
    """
    W = W.detach().cpu().numpy()
    F = min(W.shape[1], max_filters)
    rows = int(math.sqrt(F))
    cols = rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis("off")
            img = W[:, idx].reshape(3,32,32)
            img = np.transpose(img, (1,2,0))
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img)
            idx += 1
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def save_reconstruction(rbm, x, filename):
    with torch.no_grad():
        h,_ = rbm.sample_hidden_nrelu(x.unsqueeze(0).to(device))
        v = rbm.reconstruct_visible(h).squeeze(0).cpu()

    orig = x.view(3,32,32).permute(1,2,0)
    rec  = v.view(3,32,32).permute(1,2,0)

    orig = (orig - orig.min())/(orig.max()-orig.min()+1e-8)
    rec  = (rec  - rec .min())/(rec .max()-rec .min()+1e-8)

    fig,ax=plt.subplots(1,2,figsize=(6,3))
    ax[0].imshow(orig); ax[0].axis("off")
    ax[1].imshow(rec ); ax[1].axis("off")
    fig.savefig(filename)
    plt.close(fig)

# -----------------------------------------------------------
# PCA SURFACE + ROTATING GIF (MAC SAFE)
# -----------------------------------------------------------
def pca_surface_gif(rbm, pca, prefix, neuron_idx, grid_n=40):
    xs = np.linspace(-3,3,grid_n)
    ys = np.linspace(-3,3,grid_n)
    Xg,Yg = np.meshgrid(xs,ys)

    XY = np.stack([Xg.ravel(),Yg.ravel()], axis=1)
    coords = np.zeros((XY.shape[0], pca.n_components_))
    coords[:,:2] = XY

    V = pca.inverse_transform(coords)
    Vt = torch.tensor(V, dtype=torch.float32, device=device)

    with torch.no_grad():
        Z = (Vt @ rbm.W[:,neuron_idx] + rbm.b_h[neuron_idx]).cpu().numpy()
    Z = Z.reshape(grid_n,grid_n)

    # static surface
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(Xg,Yg,Z,cmap='viridis')
    ax.contour(Xg,Yg,Z,levels=[0],colors='red')
    fig.savefig(f"{prefix}_surface.png")
    plt.close(fig)

    # rotating gif
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(Xg,Yg,Z,cmap='viridis')
    ax.contour(Xg,Yg,Z,levels=[0],colors='red')

    frames=[]
    out_gif = f"{prefix}_rotating.gif"

    for a in range(0,360,360//args.gif_fps):
        ax.view_init(30,a)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        frames.append(img)

    imageio.mimsave(out_gif,frames,fps=args.gif_fps)
    plt.close(fig)

# -----------------------------------------------------------
# t-SNE VIS
# -----------------------------------------------------------
def tsne_visual(features, labels, prefix):
    idx = np.random.choice(len(features), min(args.tsne_samples, len(features)), replace=False)
    feats = features[idx]
    labs  = labels[idx]

    print("Running t-SNE 2D...")
    tsne2 = TSNE(n_components=2, perplexity=40, max_iter=1000)
    Y2 = tsne2.fit_transform(feats)
    plt.figure(figsize=(7,6))
    plt.scatter(Y2[:,0],Y2[:,1],c=labs,cmap='tab10',s=5)
    plt.savefig(prefix+"_tsne2.png")
    plt.close()

    print("Running t-SNE 3D...")
    tsne3 = TSNE(n_components=3, perplexity=40, max_iter=1000)
    Y3 = tsne3.fit_transform(feats)
    fig=plt.figure(figsize=(7,6))
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(Y3[:,0],Y3[:,1],Y3[:,2],c=labs,s=5,cmap='tab10')
    fig.savefig(prefix+"_tsne3.png")
    plt.close(fig)

# -----------------------------------------------------------
# Feature evolution animation
# -----------------------------------------------------------
def feature_evolution(history, labels, outpath):
    fig,ax = plt.subplots(figsize=(6,6))
    scat = ax.scatter(history[0][:,0], history[0][:,1], c=labels, cmap='tab10', s=5)

    def update(i):
        scat.set_offsets(history[i])
        ax.set_title(f"Epoch {i}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=450)
    ani.save(outpath,writer='pillow',fps=6)
    plt.close(fig)

# -----------------------------------------------------------
# TRAIN RBM
# -----------------------------------------------------------
rbm = RBM(INPUT_DIM, HIDDEN)
subset_idx = np.random.choice(len(x_train), args.pca_subset, replace=False)
x_subset = x_train[subset_idx].to(device)
labels_subset = y_train[subset_idx].numpy()

feature_hist = []

print("Training RBM on CIFAR-10...")

for ep in range(args.epochs_rbm):
    total=0; c=0
    for xb,_ in train_loader:
        xb = xb.to(device)
        total+=rbm.cd1(xb, args.lr_rbm, args.momentum_rbm)
        c+=1
    print(f"Epoch {ep+1}/{args.epochs_rbm}: recon={total/c:.4f}")

    with torch.no_grad():
        h,_ = rbm.hidden_det(x_subset)
        h=h.cpu().numpy()

    if ep==0:
        pca_evo = PCA(n_components=2)
        pca_evo.fit(h)

    feature_hist.append(pca_evo.transform(h))

# -----------------------------------------------------------
# FINAL VISUALIZATIONS
# -----------------------------------------------------------
save_cifar_filters(rbm.W, f"{args.save_dir}/visuals/filters.png")
save_reconstruction(rbm, x_test[0], f"{args.save_dir}/visuals/recon.png")

# PCA for surface visual
pca_input = PCA(n_components=10)
pca_input.fit(x_train[:8000].numpy())

pca_surface_gif(
    rbm, pca_input,
    f"{args.save_dir}/visuals/neuron0",
    neuron_idx=0
)

with torch.no_grad():
    feats,_ = rbm.hidden_det(x_train.to(device))
    tsne_visual(feats.cpu().numpy(), y_train.numpy(), f"{args.save_dir}/visuals/rbm")

feature_evolution(feature_hist, labels_subset, f"{args.save_dir}/animations/feature_evolution.gif")

print("\nALL RESULTS SAVED IN:", args.save_dir)
