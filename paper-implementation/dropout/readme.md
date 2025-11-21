📘 Dropout Experiment — Reproducing Srivastava et al. (2014) with MNIST & CIFAR-10

This project implements convolutional neural networks (CNNs) using Dropout, following the classic paper:
“Dropout: A Simple Way to Prevent Neural Networks from Overfitting” — Srivastava et al., 2014
Local paper reference: /mnt/data/srivastava14a.pdf

The goal is to understand dropout through:

A faithful reproduction of the paper’s MNIST architecture

Modern PyTorch implementations

Visual explanations (2D, interactive 3D plots)

Training metrics and comparisons

Optional CIFAR-10 experiments with dropout

🚀 What is Dropout? (Simple Explanation)

Dropout is a regularization technique that prevents overfitting by randomly turning off neurons during training.
This forces the network to learn robust, independent features instead of depending on specific neurons.

Real-life analogy

If you study with different friends absent each day, you become stronger in every subject because you can’t depend on just one expert.
That’s exactly what dropout does for neural networks.

📂 What This Project Includes
✔️ 1. MNIST CNN (as in the original paper)

Architecture:

Conv5x5 → ReLU → MaxPool
Conv5x5 → ReLU → MaxPool
Fully Connected 1024
Dropout (p = 0.5)
Output Layer (10 classes)


Hyperparameters follow the paper:

Dropout probability: 0.5 on hidden layers

SGD: lr = 0.1, momentum = 0.95

Max-norm constraint on incoming weights

Batch size: 100–256

✔️ 2. CIFAR-10 Experiment (optional)

Includes a deeper CNN suitable for CIFAR-10 with dropout applied before fully connected layers.

✔️ 3. Training Metrics & Visualization

The notebook records and plots:

Training & test loss curves

Training & test accuracy curves

Plots help visualize the effect of dropout on generalization.

✔️ 4. Image Visualizations

2D matplotlib visualizations of MNIST & CIFAR images

Interactive 3D surface plots (Plotly) showing pixel intensities in 3D
Great for building intuition about image data & model inputs.

✔️ 5. GPU Support

Automatically uses GPU if available:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

📁 Files Included

mnist_dropout_full.ipynb — main notebook

mnist_dropout_cnn.ipynb — simple version

srivastava14a.pdf — uploaded paper

Checkpoints saved during training (optional)

🧠 How to Run
1. Install dependencies
pip install torch torchvision matplotlib plotly tqdm

2. Open the notebook
jupyter notebook mnist_dropout_full.ipynb

3. Run all cells

The notebook will:

Download MNIST

Train the CNN with dropout

Plot accuracy & loss curves

Display 2D & 3D visualizations

(Optional) Train on CIFAR-10

📊 Example Results

Typical MNIST results with dropout:

Test accuracy: ~99%

Shows less overfitting compared to networks without dropout

Typical CIFAR-10 results:

Test accuracy: 75–82% (small convnet)

🔍 Why This Matters

The Dropout paper was a major milestone in deep learning.
It showed that a simple trick — randomly dropping neurons — acts like training many subnetworks and averaging them, dramatically improving generalization.

This repository helps you see and feel that effect using clean, reproducible experiments.

📝 Reference

Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014)
“Dropout: A Simple Way to Prevent Neural Networks from Overfitting”
JMLR 15(1):1929-1958
Included locally at: /mnt/data/srivastava14a.pdf