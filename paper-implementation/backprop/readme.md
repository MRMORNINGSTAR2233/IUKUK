# Efficient BackProp — PyTorch Implementation & Visualizations

This project is a complete, practical implementation of the classic paper “Efficient BackProp” — LeCun, Bottou, Orr, Müller (1998) featuring:

- PyTorch implementation of all recommended optimization tricks
- 2D & 3D loss-surface visualizations
- PCA weight-space trajectory plots
- Hessian diagnostics (H·v products + power method eigenvalue estimation)
- Experiments demonstrating each principle from the paper
- Clear explanations to help you master the underlying concepts

This repo is designed for both learning and research experimentation.

## 🚀 What You Will Learn

This project gives you practical intuition and mastery over:

- ✅ Backpropagation and the core update rule: $$\Delta w_i = -\eta x_i \delta$$
- ✅ Why input normalization accelerates convergence
- ✅ Why LeCun’s recommended activation outperforms sigmoids
- ✅ Why weight initialization must follow: $$\sigma_w = \frac{1}{\sqrt{\text{fan-in}}}$$
- ✅ Why SGD often outperforms batch gradient descent
- ✅ How momentum reduces zig-zag in steep valleys
- ✅ What the Hessian is and how curvature affects learning
- ✅ How to compute Hessian–vector products (H·v) in PyTorch
- ✅ How to estimate the top Hessian eigenvalue via the power method
- ✅ How to visualize high-dimensional optimization with PCA
- ✅ How to create 2D & 3D loss landscapes of a neural network

## 📁 Project Structure

```
project/
│
├── efficient_backprop_pytorch.py      # Core implementation and visualization utilities
├── README.md                          # You are here
├── data/                              # Synthetic datasets (2D Gaussians)
├── experiments/                       # Optional: saved figures & logs
└── notebooks/
    ├── EfficientBackProp_Full.ipynb   # Full tutorial notebook (all experiments)
    └── Hessian_DeepDive.ipynb         # Notebook focused on Hessian & curvature
```

## 🧠 Techniques Implemented (Directly from the Paper)

1. **Input Standardization & Whitening**
   - Shift inputs to zero mean
   - Scale to unit variance
   - Optional PCA whitening
   - Improves Hessian conditioning and speeds up convergence.

2. **LeCun’s Recommended Activation**
   - $$1.7159 \times \tanh\left(\frac{2}{3} \times x\right)$$
   - Chosen to avoid saturation and maintain stable gradient flow.

3. **LeCun Weight Initialization**
   - $$\text{std} = \frac{1}{\sqrt{\text{fan-in}}}$$
   - Keeps activations within a useful range at the start of training.

4. **SGD / Mini-batch Training with Momentum**
   - Faster convergence
   - Better minima
   - Less zig-zag behavior
   - Exactly as recommended in Efficient BackProp.

5. **Hessian Diagnostics**
   - Compute Hessian–vector products (H·v)
   - Approximate largest eigenvalue
   - Track curvature during training

6. **2D & 3D Loss-Surface Visualization**
   - Fix all parameters except two
   - Plot contour & 3D surface
   - Overlay training trajectory

7. **PCA Visualization of Weight Trajectories**
   - Compress entire training path into 2D/3D for intuitive understanding.

## 📊 Visualizations Included

- ✅ Training loss (log-scale)
- ✅ PCA trajectory (2D & 3D)
- ✅ Loss-surface contour + trajectory overlay
- ✅ 3D loss surface
- ✅ Hessian top eigenvalue over training
- ✅ Comparison plots for different initializations, activations & preprocessing

## ▶️ How to Run

1. Install dependencies
   ```
   pip install torch numpy matplotlib scikit-learn
   ```

2. Run the main script
   ```
   python efficient_backprop_pytorch.py
   ```

3. Or launch the full notebook
   ```
   jupyter notebook notebooks/EfficientBackProp_Full.ipynb
   ```

## 🧪 Experiments You Can Reproduce

Each experiment is directly tied to a specific section of the paper.

1. **Baseline (bad preprocessing)**
   - Logistic sigmoid
   - Poor initialization
   - No normalization
   - Batch gradient
   - Expect: slow convergence, zig-zag trajectories, large eigenvalue spread.

2. **Add Input Normalization**
   - Expect: faster convergence, improved conditioning, smoother trajectories.

3. **Use Recommended Activation (LeCunTanh)**
   - Expect: less saturation, better gradient flow.

4. **Use Proper LeCun Initialization**
   - Expect: stable activations, faster initial descent.

5. **SGD vs Batch Gradient**
   - SGD tends to:
     - Converge faster
     - Escape shallow minima
     - Reach better solutions

6. **Momentum**
   - Expect: reduced oscillation in steep directions.

7. **PCA Weight Trajectories**
   - Visualize how optimization moves through weight space.

8. **Hessian Eigenvalue Analysis**
   - Track top eigenvalue across epochs to see conditioning change.

## 🎯 Goal of this Project

By completing the experiments and visualizations in this repository, you will develop:

- Intuition for why deep learning optimization is hard
- Understanding of conditioning, curvature, and the Hessian
- Practical mastery of initialization & preprocessing techniques
- Geometric insights into gradient descent dynamics

All grounded in one of the most important optimization papers ever written.

## 📘 References

LeCun, Yann; Bottou, Léon; Orr, Genevieve; Müller, Klaus.  
*Efficient BackProp* (1998).
