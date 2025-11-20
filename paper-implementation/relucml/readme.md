📘 README — RBM + NReLU Visualization Framework

Implementation & Visualization of “Rectified Linear Units Improve Restricted Boltzmann Machines” (Nair & Hinton, ICML 2010)

🧠 1. Introduction

This repository provides a complete PyTorch implementation of the RBM architecture proposed in the ICML 2010 paper by Nair & Hinton:

**Rectified Linear Units Improve Restricted Boltzmann Machines**  
by Vinod Nair & Geoffrey Hinton.

This project extends the original ideas by including full scientific visualizations, including:

- 2D filter visualization
- 3D ReLU boundary plots
- 3D rotating surface GIFs
- t-SNE embeddings (2D & 3D)
- Feature evolution across epochs (animated video)
- Reconstructions of test images

The implementation is built around MNIST for simplicity but can be adapted to any dataset.

⚡ 2. Core Theory

2.1 What is an RBM?

An RBM (Restricted Boltzmann Machine) is an energy-based probabilistic model with:

- Visible units \( v \in \mathbb{R}^D \)
- Hidden units \( h \in \mathbb{R}^H \)

The energy function:

\[ E(v, h) = -v^\top W h - b_v^\top v - b_h^\top h \]

Probability of a configuration:

\[ p(v, h) = \frac{e^{-E(v, h)}}{Z} \]

where \( Z \) is the partition function (intractable, estimated indirectly).

Training uses Contrastive Divergence (CD):

\[ \Delta W = \eta (\langle vh^\top \rangle_{data} - \langle vh^\top \rangle_{model}) \]

2.2 Why ReLUs Improve RBMs (Nair & Hinton 2010)

Traditional RBMs used binary hidden units:

\[ h_j \in \{0, 1\} \]

But this places strong limits on model expressiveness.

Nair & Hinton’s key idea:

A ReLU hidden unit behaves like an infinite collection of tied binary units, each with shifted bias:

\[ h = \sum_{k=1}^\infty \sigma(x - k) \]

They show this infinite sum is well approximated by:

\[ h \approx \log(1 + e^x) \]

Which behaves similarly to:

\[ h = \max(0, x) \]

Final approximation used in training (Noisy ReLU = NReLU):

\[ h = \max(0, x + \epsilon), \quad \epsilon \sim \mathcal{N}(0, \sigma(x)) \]

Where \( x = v^\top W + b_h \).

This stochasticity allows ReLUs to act like a probabilistic version of standard neural ReLUs.

2.3 Why ReLUs are Better

✔ (1) Intensity-equivariance

If you scale an input by any positive constant:

\[ v' = \alpha v \]

Then:

\[ h' = \max(0, W(\alpha v)) = \alpha h \]

This means the network is invariant to lighting changes, making it excellent for:

- Face recognition
- Image classification
- Robotics
- Medical imaging (contrast changes)

✔ (2) Exponential mixture of linear models

Each ReLU is a switch that selects between:

- Off region (0)
- Linear region (x)

A hidden layer of \( H \) ReLUs creates up to:

\[ 2^H \]

linear regimes → extremely expressive models.

✔ (3) Better gradient flow

ReLUs avoid the saturation of sigmoids and allow deeper RBMs to train.

🎯 3. Real-Life Examples of ReLU-RBM Usage

🧬 A. Medical Imaging – CT/MRI

Contrast varies widely between scans. ReLUs maintain relative intensity, improving robustness of:

- Tumor detection
- Tissue classification
- Organ segmentation

🛂 B. Face Verification (LFW dataset)

The original paper showed huge improvements using ReLUs for:

- Face identity verification
- Robustness against brightness
- Cosine similarity–based matching

🚗 C. Autonomous Driving

Light conditions change constantly; ReLU RBMs retain feature structure:

- At night
- In tunnels
- During bright daylight
- Under shadows

🤖 D. Robotics + Industrial Automation

Robots analyzing objects under varying illumination benefit from intensity-invariant features.

🛠 4. Features of This Implementation

This repository includes:

✔ Full PyTorch RBM implementation
- Gaussian visible units
- Noisy ReLU hidden units
- CD-1 training
- Momentum + weight decay

✔ Fully interactive visualizations
- Filter grids
- Reconstructions
- PCA-based ReLU region visualization
- 3D pre-activation surface plots
- Animated rotating 3D GIFs
- Feature evolution GIF across epochs
- t-SNE 2D + 3D

📦 5. Installation

1. Clone the repo:
	```
	git clone <your-repo-url>
	cd <repo-folder>
	```

2. Install dependencies:
	```
	pip install torch torchvision scikit-learn imageio matplotlib
	```

	Optional (for .mp4 animations):
	```
	sudo apt install ffmpeg
	```

▶️ 6. Running the Code

- Basic run
  ```
  python rbm_nrelu_visual_all.py
  ```

- Fast mode
  ```
  python rbm_nrelu_visual_all.py --epochs_rbm 5 --tsne_samples 500
  ```

- High-quality mode
  ```
  python rbm_nrelu_visual_all.py --epochs_rbm 40 --hidden_size 500 --tsne_samples 6000
  ```

- Force GPU
  ```
  python rbm_nrelu_visual_all.py --device cuda
  ```

🗂 7. Output Directory Structure

After running the script, you will get:

```
rbm_visual_all/
│
├── visuals/
│   ├── filters_final.png
│   ├── recon_final.png
│   ├── rbm_features_tsne2.png
│   ├── rbm_features_tsne3.png
│   ├── neuron0_surface.png
│   ├── neuron0_rotating.gif
│   └── ...
│
└── animations/
	 ├── feature_evolution.gif
	 └── ...
```

🎨 8. Visualization Examples Explained

8.1 Filter Grid (2D)

Shows RBM weight vectors reshaped into \( 28 \times 28 \) images.

8.2 Reconstruction Visualization

Shows how well the RBM reconstructs test images.

8.3 3D ReLU Boundary Visualization

We sample a 2D PCA subspace → lift it to input space → compute preactivations:

\[ x = v^\top W + b_h \]

Plot surface:

- Surface height = \( x \)
- Red contour = ReLU boundary where \( x = 0 \)

8.4 3D Rotating GIF

An animated spin-around of the surface.

8.5 t-SNE Embeddings

We compute deterministic hidden activations:

\[ h = \max(0, x) \]

And apply:

- t-SNE 2D
- t-SNE 3D

Colors represent MNIST labels.

8.6 Feature Evolution (GIF)

At each RBM epoch, features are extracted and projected through PCA:

\[ h_{epoch} \rightarrow \text{PCA} \rightarrow \mathbb{R}^2 \]

The animation shows how the representation stabilizes.

🔧 9. Extending the Project

You can easily add:

- CIFAR-10 training
- FashionMNIST
- Deep belief network (stacked RBMs)
- Contrastive Divergence k > 1
- Persistent CD (PCD)

Just ask and I can generate the code.

❤️ 10. Credits

Original idea from:

Vinod Nair & Geoffrey Hinton  
“Rectified Linear Units Improve Restricted Boltzmann Machines,” ICML 2010.

All visualizations and extended infrastructure created for this project.
