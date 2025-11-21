# 📘 README — Adam Optimizer: Theory, Intuition, Formulas & Applications

This document is a complete, teacher-style guide explaining the Adam optimizer, based on the paper “Adam: A Method for Stochastic Optimization” by Kingma & Ba (2015). It covers theory, math, visual intuition, convergence, and real-world use cases.

## ⭐ 1. Overview

Adam (Adaptive Moment Estimation) is a first-order gradient-based optimization algorithm designed for:

- Large-scale machine learning
- Noisy or stochastic gradients
- High-dimensional parameter spaces
- Sparse gradients
- Non-stationary objectives

Adam combines the strengths of momentum and adaptive learning rates, giving it excellent performance across deep learning tasks.

— Abstract & Introduction 

[1412.6980v9](https://arxiv.org/abs/1412.6980v9)

## 🔢 2. Notation

| Symbol | Meaning |
|--------|---------|
| \(\theta\) | Parameters |
| \(f(\theta)\) | Stochastic objective |
| \(f_t(\theta)\) | Stochastic sample at step \(t\) |
| \(g_t\) | Gradient at step \(t\): \(g_t = \nabla f_t(\theta_{t-1})\) |
| \(m_t\) | 1st moment (mean) |
| \(v_t\) | 2nd moment (variance estimate) |
| \(\alpha\) | Learning rate |
| \(\beta_1, \beta_2\) | Decay rates |
| \(\epsilon\) | Numerical stabilizer |

## ⚙️ 3. The Adam Algorithm

From Algorithm 1 in the paper. 

[1412.6980v9](https://arxiv.org/abs/1412.6980v9)

### Step-by-step

1. **Compute gradient**  
    \(g_t = \nabla_\theta f_t(\theta_{t-1})\)

2. **Update biased moment estimates**  
    **Momentum:**  
    \(m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\)  
    **Variance estimate:**  
    \(v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\)

3. **Bias correction**  
    \(\hat{m}_t = \frac{m_t}{1 - \beta_1^t}\)  
    \(\hat{v}_t = \frac{v_t}{1 - \beta_2^t}\)

4. **Parameter update**  
    \(\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\)

## ⚙️ 4. Hyperparameters

Defaults recommended in the paper (page 2). 

[1412.6980v9](https://arxiv.org/abs/1412.6980v9)

| Parameter | Default |
|-----------|---------|
| Learning rate \(\alpha\) | 0.001 |
| \(\beta_1\) | 0.9 |
| \(\beta_2\) | 0.999 |
| \(\epsilon\) | 1e-8 |

## 🧠 5. Intuition Behind Adam

- **Momentum (from SGD + Momentum)**  
  Adam accumulates a moving average of gradients → smooths noisy updates.

- **Adaptive learning rates (from RMSProp/AdaGrad)**  
  Each parameter gets its own learning rate based on gradient variance.

- **Scale-invariant updates**  
  Scaling gradients by \(c\) leaves updates unchanged:  
  \(\frac{c \hat{m}_t}{\sqrt{c^2 \hat{v}_t}} = \frac{\hat{m}_t}{\sqrt{\hat{v}_t}}\)

- **Automatic annealing**  
  Near minima, variance shrinks → step sizes shrink → stable convergence.

- **Critical bias correction**  
  Without correction, early steps become too small or too large — illustrated in Figure 4 of the paper. 

[1412.6980v9](https://arxiv.org/abs/1412.6980v9)

## 📉 6. Convergence Theory (Convex Case)

Adam is analyzed through online convex optimization using regret:

\[ R(T) = \sum_{t=1}^T [f_t(\theta_t) - f_t(\theta^*)] \]

### Main result (Theorem 4.1)

Adam achieves:

\[ R(T) = O(\sqrt{T}) \]

This matches the best known results for first-order methods.

— Section 4.1 

[1412.6980v9](https://arxiv.org/abs/1412.6980v9)

### Corollary

\[ \frac{R(T)}{T} = O(T^{-1/2}) \]

→ Average regret goes to zero.

## 🧪 7. Real-World Applications (From Experiments in the Paper)

1. **Logistic Regression**  
    Performed strongly on MNIST  
    Handled sparse 10,000-word IMDB features exceptionally  
    — Figure 1 

    [1412.6980v9](https://arxiv.org/abs/1412.6980v9)

2. **Fully Connected Neural Networks**  
    Adam outperformed SFO, RMSProp, SGD, AdaDelta.  
    — Figure 2 

    [1412.6980v9](https://arxiv.org/abs/1412.6980v9)

3. **Convolutional Networks (CNNs)**  
    On CIFAR-10, Adam matched or outperformed other adaptive methods.  
    — Figure 3 

    [1412.6980v9](https://arxiv.org/abs/1412.6980v9)

4. **VAEs**  
    Bias correction crucial.  
    — Figure 4 

    [1412.6980v9](https://arxiv.org/abs/1412.6980v9)

### Additional examples

Used across:

- Transformers
- NLP models
- Recommender systems
- Reinforcement learning
- Speech models
- Variational inference

## 📐 8. Worked Example

Let:

- \(g_1 = 0.1\)
- \(\beta_1 = 0.9\), \(\beta_2 = 0.999\)
- \(\alpha = 0.001\)

Compute moments:  
\(m_1 = 0.01\)  
\(v_1 = 10^{-5}\)

Bias corrections:  
\(\hat{m}_1 = 0.1\)  
\(\hat{v}_1 = 0.01\)

Update:  
\(\theta_1 = \theta_0 - 0.001\)

## 📚 9. AdaMax — Adam’s L∞ Variant

Described in Section 7.1 and Algorithm 2. 

[1412.6980v9](https://arxiv.org/abs/1412.6980v9)

Replace second moment with:  
\(u_t = \max(\beta_2 u_{t-1}, |g_t|)\)

Update:  
\(\theta_t = \theta_{t-1} - \frac{\alpha}{1 - \beta_1^t} \frac{m_t}{u_t}\)

Benefits:

- No bias correction needed for \(u_t\)
- Simpler bound: \(|\Delta_t| \leq \alpha\)
- More stable for large gradients

## 🔄 10. Temporal Averaging (for Better Generalization)

Section 7.2. 

[1412.6980v9](https://arxiv.org/abs/1412.6980v9)

Exponential moving average of parameters:  
\(\bar{\theta}_t = \beta_2 \bar{\theta}_{t-1} + (1 - \beta_2) \theta_t\)

Bias-corrected:  
\(\tilde{\theta}_t = \frac{\bar{\theta}_t}{1 - \beta_2^t}\)

Improves performance similarly to Polyak-Ruppert averaging.

## 🧾 11. Summary

| Feature | Benefit |
|---------|---------|
| Momentum | Smooths gradients |
| Adaptive LRs | Per-parameter learning rates |
| Bias correction | Stable early updates |
| Sparse-gradient efficiency | Great for NLP, Recommender systems |
| Scale invariance | Stable across architectures |
| Bound on update | Convergent & predictable |

Adam is widely used because:

- Works well out-of-the-box
- Requires little tuning
- Performs consistently across domains
- Handles noisy, sparse, and complex gradients
