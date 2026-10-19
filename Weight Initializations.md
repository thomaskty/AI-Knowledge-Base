# Comprehensive Guide to Weight Initialization Techniques

Weight initialization is a critical hyperparameter step in deep learning. Its goal is to prevent activations from vanishing or exploding during the forward pass and to prevent gradients from vanishing or exploding during backpropagation.

---

## 1. Core Mechanics & Motivation

The central objective behind modern initialization schemes (like **Xavier/Glorot** and **He/Kaiming**) is to keep the expected **variance of activations and gradients approximately constant** from one layer to the next.

$$\text{Var}(h^{(l)}) \approx \text{Var}(h^{(l-1)}) \quad \text{and} \quad \text{Var}\left(\frac{\partial L}{\partial h^{(l-1)}}\right) \approx \text{Var}\left(\frac{\partial L}{\partial h^{(l)}}\right)$$

### 1.1. Why Zero Initialization Fails (The Symmetry Problem)

If all weights in a neural network layer are initialized to zero ($W = \mathbf{0}$):

1. **Forward Pass:** For any input vector $\mathbf{x}$, the pre-activation $z_i = \sum_j w_{ij} x_j + b_i = 0$. All neurons in a layer output the exact same value:

   $$a_i = g(z_i) = g(0)$$

2. **Backward Pass:** The gradient of the loss with respect to weight $w_{ij}$ is given by:

   $$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial z_i} \cdot x_j = \delta_i \cdot x_j$$

   Since every neuron in layer $l$ receives identical activations from layer $l-1$ and produces identical outputs, the backpropagated error $\delta_i$ is identical for all neurons $i$.

3. **Symmetry Result:** Every weight in layer $l$ receives the exact same update during gradient descent ($\Delta w_{ij} = \Delta w_{kj}$). The network collapses into a single-neuron model copied $m$ times, destroying its capacity to learn distinct feature representations.

---

### 1.2. Random Initialization & Variance Scaling

To break symmetry, weights are sampled from random distributions (e.g., Gaussian or Uniform). However, the **scale (variance)** of these random values is crucial:

* **Weights too small ($\text{Var}(W) \to 0$):** In deep networks, multiplying by small weight matrices across many layers shrinks signal variance toward $0$. Activations vanish in the forward pass, and gradients vanish during backpropagation.
* **Weights too large ($\text{Var}(W) \to \infty$):** Signals expand exponentially across layers, causing forward activation values and backward gradient updates to explode, or forcing saturating activations (like Sigmoid or Tanh) into flat, zero-gradient regions.

---

## 2. Mathematical Foundations

Consider a linear layer without bias:

$$y = Wx$$

Where:
* $x \in \mathbb{R}^{n_{\text{in}}}$ is the input vector.
* $W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$ is the weight matrix.
* $y \in \mathbb{R}^{n_{\text{out}}}$ is the output vector.

For a single output neuron $y_i$:

$$y_i = \sum_{j=1}^{n_{\text{in}}} w_{ij} x_j$$

### Assumptions for Variance Analysis

1. Inputs $x_j$ are independent and identically distributed (i.i.d.) with zero mean: $\mathbb{E}[x_j] = 0$ and $\text{Var}(x_j) = \text{Var}(x)$.
2. Weights $w_{ij}$ are i.i.d. with zero mean: $\mathbb{E}[w_{ij}] = 0$ and $\text{Var}(w_{ij}) = \text{Var}(w)$.
3. Inputs $x_j$ and weights $w_{ij}$ are mutually independent.

### Variance Expansion of a Product

For independent random variables $A$ and $B$:

$$\text{Var}(AB) = \mathbb{E}[A]^2 \text{Var}(B) + \mathbb{E}[B]^2 \text{Var}(A) + \text{Var}(A)\text{Var}(B)$$

Since $\mathbb{E}[w_{ij}] = 0$ and $\mathbb{E}[x_j] = 0$:

$$\text{Var}(w_{ij} x_j) = \text{Var}(w_{ij}) \text{Var}(x_j) = \text{Var}(w) \text{Var}(x)$$

Summing across all $n_{\text{in}}$ independent inputs:

$$\text{Var}(y_i) = \text{Var}\left(\sum_{j=1}^{n_{\text{in}}} w_{ij} x_j\right) = \sum_{j=1}^{n_{\text{in}}} \text{Var}(w_{ij} x_j) = n_{\text{in}} \text{Var}(w) \text{Var}(x)$$

---

## 3. Xavier (Glorot) Initialization

Xavier initialization was designed for symmetric, linear-like activation functions around zero (e.g., Sigmoid, Tanh, or Identity).

---

### 3.1. Forward Pass Constraint

To prevent activations from expanding or shrinking as they pass forward through layers, we enforce:

$$\text{Var}(y_i) = \text{Var}(x)$$

Substituting our variance relation:

$$n_{\text{in}} \text{Var}(w) \text{Var}(x) = \text{Var}(x) \implies n_{\text{in}} \text{Var}(w) = 1$$

$$\text{Var}(w) = \frac{1}{n_{\text{in}}} \quad \text{(Forward Constraint)}$$

---

### 3.2. Backward Pass Constraint

Now consider backpropagation. Let $L$ be the loss scalar. Define the gradient error term for output neuron $i$:

$$\delta_i^{(\text{curr})} = \frac{\partial L}{\partial y_i}$$

By the chain rule, the error propagated backward to input $x_j$ (the previous layer's output) is:

$$\delta_j^{(\text{prev})} = \frac{\partial L}{\partial x_j} = \sum_{i=1}^{n_{\text{out}}} \frac{\partial L}{\partial y_i} \frac{\partial y_i}{\partial x_j} = \sum_{i=1}^{n_{\text{out}}} \delta_i^{(\text{curr})} w_{ij}$$

Notice that in the backward pass, error signals flow through the **transposed weight matrix** $W^T$ across $n_{\text{out}}$ channels.

Applying the variance rule to the backpropagated error (assuming $\mathbb{E}[\delta_i^{(\text{curr})}] = 0$):

$$\text{Var}\left(\delta_j^{(\text{prev})}\right) = n_{\text{out}} \text{Var}(w) \text{Var}\left(\delta^{(\text{curr})}\right)$$

To preserve gradient variance during backpropagation ($\text{Var}(\delta^{(\text{prev})}) = \text{Var}(\delta^{(\text{curr})})$), we enforce:

$$n_{\text{out}} \text{Var}(w) = 1$$

$$\text{Var}(w) = \frac{1}{n_{\text{out}}} \quad \text{(Backward Constraint)}$$

---

### 3.3. Reconciling Both Constraints

In general, layers have different numbers of inputs and outputs ($n_{\text{in}} \neq n_{\text{out}}$). We cannot satisfy both constraints simultaneously.

Xavier & Glorot proposed taking the **harmonic mean** of both constraints:

$$\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

---

### 3.4. Concrete Distributions for Xavier Initialization

#### 1. Xavier Normal Distribution
Weights are drawn from a normal distribution with zero mean:

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

#### 2. Xavier Uniform Distribution
For a uniform distribution $U(-a, a)$, its variance is $\text{Var}(U) = \frac{(a - (-a))^2}{12} = \frac{4a^2}{12} = \frac{a^2}{3}$.

Setting $\frac{a^2}{3} = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ gives $a = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$:

$$W_{ij} \sim U\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

---

## 4. He (Kaiming) Initialization

Xavier initialization assumes activations operate linearly near zero. However, non-linear functions like **ReLU** break this assumption.

$$\text{ReLU}(z) = \max(0, z)$$

---

### 4.1. The ReLU Variance Reduction Problem

For a zero-mean symmetric input distribution $z \sim \mathcal{N}(0, \sigma^2)$, ReLU sets approximately $50\%$ of inputs to zero:

* For $z > 0$: $\text{ReLU}(z) = z$
* For $z \le 0$: $\text{ReLU}(z) = 0$

Because half of the variance density is clipped to zero, the variance of the output after ReLU is halved:

$$\mathbb{E}[\text{ReLU}(z)^2] = \frac{1}{2} \text{Var}(z)$$

If Xavier initialization ($\text{Var}(w) = \frac{1}{n_{\text{in}}}$) is paired with ReLU, the activation variance shrinks by a factor of $0.5$ at every single layer:

$$\text{Var}(a^{(l)}) = \frac{1}{2} \cdot \text{Var}(a^{(l-1)})$$

In a 30-layer deep network, activation scale drops to $0.5^{30} \approx 9.3 \times 10^{-10}$, leading to vanishing activations.

---

### 4.2. He Mathematical Derivation

To compensate for the $50\%$ signal drop caused by ReLU, He initialization doubles the variance of the initialized weights:

$$n_{\text{in}} \cdot \left( \frac{1}{2} \text{Var}(w) \right) \cdot \text{Var}(x) = \text{Var}(x)$$

$$\frac{1}{2} n_{\text{in}} \text{Var}(w) = 1$$

$$\text{Var}(w) = \frac{2}{n_{\text{in}}}$$

---

### 4.3. Concrete Distributions for He Initialization

#### 1. He Normal Distribution

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

#### 2. He Uniform Distribution
Using $U(-a, a)$ where $\frac{a^2}{3} = \frac{2}{n_{\text{in}}}$ gives $a = \sqrt{\frac{6}{n_{\text{in}}}}$:

$$W_{ij} \sim U\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \, \sqrt{\frac{6}{n_{\text{in}}}}\right)$$

---

## 5. Summary & Comparison Matrix

| Initialization Scheme | Target Activation Functions | Target Variance ($\text{Var}(w)$) | Normal Scale ($\sigma$) | Uniform Bounds ($[-a, a]$) |
| :--- | :--- | :--- | :--- | :--- |
| **Zero** | Never for weights (Causes symmetry lock) | $0$ | $0$ | $[0, 0]$ |
| **Small Random** | Shallow Networks / Linear | Custom (e.g., $0.01$) | $0.01$ | $[-0.01, 0.01]$ |
| **Xavier (Glorot)** | Sigmoid, Tanh, Linear, Softmax | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ | $\sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}$ | $\pm \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$ |
| **He (Kaiming)** | ReLU, Leaky ReLU, PReLU, ELU | $\frac{2}{n_{\text{in}}}$ | $\sqrt{\frac{2}{n_{\text{in}}}}$ | $\pm \sqrt{\frac{6}{n_{\text{in}}}}$ |

---

## 6. Implementation References (PyTorch & TensorFlow)

### PyTorch

```python
import torch.nn as nn

# Xavier / Glorot Initialization (for Tanh / Sigmoid)
nn.init.xor_normal_(layer.weight)
nn.init.xor_uniform_(layer.weight)

# He / Kaiming Initialization (for ReLU / Leaky ReLU)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
```