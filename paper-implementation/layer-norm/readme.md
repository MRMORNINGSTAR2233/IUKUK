# 📘 Layer Normalization Deep Dive — README

## 🔍 A Hands-On Exploration of Layer Normalization

This repository provides a complete practical guide to Layer Normalization (LN) — a powerful technique in deep learning that stabilizes and accelerates training, especially in RNNs, Transformers, and small-batch learning environments.

The notebook included here demonstrates LN using:

- 🔥 Heatmaps of activation correlations
- 🌐 Plotly 3D interactive visualizations
- 📌 Animated gradient flow through training
- 🧪 Training comparisons: No Norm vs BatchNorm vs LayerNorm
- ⚙️ Implementations in MLPs, RNNs, CNNs, Transformer blocks
- 💡 Effects of LN initialization (γ, β)

Based on the foundational research paper:

**Layer Normalization**  
Jimmy Lei Ba, Jamie Ryan Kiros & Geoffrey E. Hinton  
University of Toronto & Google  
arXiv:1607.06450 (2016)  
https://arxiv.org/abs/1607.06450

## 🧠 Why Layer Normalization?

Batch Normalization normalizes activations across a batch which causes issues:

| Issue              | BatchNorm Problem          |
|--------------------|----------------------------|
| Small batch sizes  | Unstable statistics        |
| Sequence training  | Needs separate stats per timestep |
| Online / RL settings | Batch size = 1 → fails    |
| Inference          | Stats differ from training |

LayerNorm solves this by normalizing across the hidden units within each sample, enabling:

- ✔ Stable & faster RNN training
- ✔ Same behavior in training & inference
- ✔ Strong performance even with tiny batches
- ✔ Robust gradient flow

## 📂 Repository Structure

```
📁 LayerNorm-DeepDive/
│
├── layernorm_visualization.ipynb   # 🔥 Main interactive notebook
├── README.md                       # 📘 This documentation
└── assets/                         # (Optional) Generated figures / animations
```

## 🚀 Features Demonstrated

### 🔹 1️⃣ Custom LayerNorm Implementation (from Paper Eq. 15–16)

- Learnable γ (gain) and β (bias) per feature
- Stable statistics per-training case

### 🔹 2️⃣ Visualizations Include

| Visualization              | What It Shows                          |
|----------------------------|----------------------------------------|
| Heatmaps                   | LN reduces hidden unit correlation     |
| 3D Plotly                  | Activations become normalized & spherical |
| Training Loss Animated     | LN speeds convergence                  |
| Gradient Flow Animation    | LN prevents exploding/vanishing gradients |

### 🔹 3️⃣ Architecture Comparisons

| Model              | Normalization Tested          |
|--------------------|-------------------------------|
| MLP                | Baseline vs LN                |
| RNN (LSTM)         | BatchNorm vs LayerNorm        |
| CNN                | BatchNorm vs LayerNorm        |
| Transformer Block  | BatchNorm vs LayerNorm        |

## 🔧 Installation

You can run locally or in Google Colab.

### Requirements

- python >= 3.8
- torch >= 2.0
- numpy
- matplotlib
- seaborn
- scikit-learn
- plotly

Install via pip:

```bash
pip install torch numpy matplotlib seaborn scikit-learn plotly
```

## ▶️ Running the Notebook

Launch Jupyter:

```bash
jupyter notebook layernorm_visualization.ipynb
```

Or upload to Google Colab:

👉 Open Colab: https://colab.research.google.com/

📌 Upload the notebook file

## 📊 Key Takeaways

- ✔ LN keeps mean ≈ 0 and variance ≈ 1 per sample
- ✔ LN makes networks robust to input scale and weight initialization
- ✔ LN accelerates convergence and improves final accuracy
- ✔ Gradients flow smoother across layers & time steps
- ✔ LN is now standard in Transformers (BERT, GPT, T5…)

## 📑 Scientific Notes

- LN is invariant to rescaling a layer’s weights
- LN stabilizes hidden-to-hidden dynamics in recurrent networks
- LN outperforms BN in long sequence tasks & online learning
- BN remains better for CNNs with large batch sizes

Reference Table from paper:

| Model Type                  | Best Normalization |
|-----------------------------|---------------------|
| RNNs / NLP                  | ⭐ LayerNorm       |
| Transformers                | ⭐ LayerNorm       |
| CNNs (batch ≥ 32)           | 👍 BatchNorm       |
| Online / Reinforcement Learning | ⭐ LayerNorm    |

## 🙌 Acknowledgements

This work builds on:

- 📄 Ba, Kiros & Hinton — Layer Normalization (2016)
- 🔍 Ioffe & Szegedy — Batch Normalization (2015)
