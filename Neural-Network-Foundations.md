A perceptron is a simple model of a neuron and one of the earliest building blocks of neural networks. The word is also used for the simplest trainable classifier: a single artificial unit that combines input values and makes a decision.

## 1. Biological neuron: the inspiration

A biological neuron receives signals through its dendrites. The cell body integrates those signals, and if the combined input is strong enough, the neuron produces an electrical spike that travels along its axon. Synapses determine how strongly one neuron influences another, and their strengths can change with learning.

This is an analogy, not a literal simulation. Artificial neurons use numbers and mathematical operations; biological neurons use complex electrochemical processes and timing.

## 2. The classical perceptron

For input features $x_1,\ldots,x_n$, the perceptron computes a weighted sum plus a bias:

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

Here, $w_i$ says how strongly feature $x_i$ affects the decision, and $b$ shifts the decision boundary. A classic perceptron applies a hard threshold:

$$
\hat{y} =
\begin{cases}
1 & \text{if } z \ge 0 \\
0 & \text{if } z < 0
\end{cases}
$$

For example, if $x=(2,1)$, $w=(0.5,-1)$, and $b=0.2$, then $z=(0.5)(2)+(-1)(1)+0.2=0.2$, so the perceptron predicts 1.

With one perceptron, the boundary between the two classes is a straight line in two dimensions, a plane in three dimensions, or a hyperplane in higher dimensions. A single perceptron can learn only linearly separable classifications. It cannot solve XOR by itself.

## 3. The modern artificial neuron

A modern artificial neuron keeps the weighted sum but usually applies a smooth or piecewise-smooth activation function:

$$
a = f(z) = f\left(\sum_i w_i x_i+b\right)
$$

The output $a$ can be a score, a hidden-layer feature, or a probability-like value. Common activations include:

- **Sigmoid:** $\sigma(z)=1/(1+e^{-z})$, which maps values into $(0,1)$.
- **Tanh:** maps values into $(-1,1)$.
- **ReLU:** $\max(0,z)$, which keeps positive values and zeros negative ones.

The classical step function makes a hard yes/no decision and has zero derivative almost everywhere, so it is not convenient for gradient-based learning. Modern networks typically use differentiable activations in their hidden layers, then choose an output and loss suited to the task.

An artificial neuron is often described as a simplified model of a biological neuron: inputs correspond loosely to signals, weights to connection strengths, the weighted sum to integration, and the activation to a response.

## 4. Multi-layer perceptron (MLP)

An MLP connects artificial neurons in layers:

1. **Input layer:** receives the feature values.
2. **Hidden layers:** transform the representation using weighted sums and nonlinear activations.
3. **Output layer:** produces a prediction, such as a class score or a number.

A layer can be written compactly as:

$$
\mathbf{a}^{(l)} = f\left(W^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}\right)
$$

where $W^{(l)}$ contains the layer's weights and $\mathbf{b}^{(l)}$ its biases. Stacking layers lets an MLP build increasingly useful combinations of features. Nonlinear activations matter: without them, multiple affine layers collapse into one affine transformation, so depth would not add expressive power.

Hidden layers allow MLPs to represent nonlinear decision boundaries, including XOR, given a suitable architecture and training.

## 5. How learning works

Training adjusts weights and biases so the network's predictions better match target values.

**The classical perceptron learning rule.** For labels $y\in\{-1,+1\}$, predict $\hat{y}=\operatorname{sign}(\mathbf{w}\cdot\mathbf{x}+b)$. If an example is misclassified, update:

$$
\mathbf{w} \leftarrow \mathbf{w}+\eta y\mathbf{x},
\qquad
b \leftarrow b+\eta y
$$

Here $\eta>0$ is the learning rate. Repeating this over examples converges to a separating boundary when the training data are linearly separable. It does not solve the general nonlinear case.

**Gradient-based learning for modern networks.** A typical training step is:

1. Run a **forward pass** to compute predictions.
2. Calculate a **loss**, a number measuring prediction error.
3. Use **backpropagation** (the chain rule) to compute how the loss changes with each parameter.
4. Update parameters, for example with gradient descent: $\theta\leftarrow\theta-\eta\nabla_\theta L$.

For classification, common losses include cross-entropy; for numeric prediction, mean squared error is common. Optimizers such as SGD and Adam use the gradients to update parameters. Autograd libraries build the computation graph during the forward pass and differentiate it automatically during backpropagation.

## 6. Numeric autograd example

Consider one sigmoid neuron with input $x=2$, target $y=1$, weight $w=0.5$, and bias $b=0$. Use binary cross-entropy loss.

**Forward pass**

$$
z=wx+b=(0.5)(2)+0=1
$$

$$
p=\sigma(z)\approx0.7311
$$

$$
L=-\ln(p)\approx0.3133
$$

For a sigmoid output with binary cross-entropy, the derivative of the loss with respect to the logit is $p-y$:

$$
\frac{\partial L}{\partial z}=p-y\approx-0.2689
$$

Since $z=wx+b$:

$$
\frac{\partial L}{\partial w}=(p-y)x\approx-0.5379,
\qquad
\frac{\partial L}{\partial b}=p-y\approx-0.2689
$$

With learning rate $\eta=0.1$, gradient descent gives:

$$
w_{\text{new}}=0.5-0.1(-0.5379)\approx0.5538
$$

$$
b_{\text{new}}=0-0.1(-0.2689)\approx0.0269
$$

The negative gradients mean increasing these parameters slightly will reduce the loss for this example.

```python
import torch
import torch.nn.functional as F

x = torch.tensor(2.0)
target = torch.tensor(1.0)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

logit = w * x + b
loss = F.binary_cross_entropy_with_logits(logit, target)
loss.backward()  # Autograd fills in w.grad and b.grad

print(torch.sigmoid(logit).item())  # about 0.7311
print(loss.item())                  # about 0.3133
print(w.grad.item(), b.grad.item()) # about -0.5379, -0.2689

with torch.no_grad():
    w -= 0.1 * w.grad
    b -= 0.1 * b.grad
```

`binary_cross_entropy_with_logits` combines sigmoid and cross-entropy in a numerically stable way. `backward()` computes the gradients; the final block performs the parameter update. In a larger model, autograd applies the chain rule through every operation in the network.

## 7. Vanishing and exploding gradients

Backpropagation applies the chain rule across layers. A gradient reaching an early layer contains a product of many local derivatives. If these factors are repeatedly small, the product shrinks toward zero; if they are repeatedly large, it can grow dramatically.

- **Vanishing gradients:** gradients become tiny as they travel backward. Early layers then learn very slowly or appear to stop learning. Saturated sigmoid or tanh activations are common contributors because their derivatives are small in saturated regions.
- **Exploding gradients:** gradients grow excessively. Updates can become unstable, causing large parameter jumps, erratic loss, or numerical overflow such as infinities and NaNs.

Common ways to reduce these problems include:

- Use suitable initialization (for example, Xavier/Glorot or He initialization).
- Use activations such as ReLU-family functions where appropriate.
- Use normalization and residual/skip connections in deep networks.
- Use gated architectures such as LSTMs or GRUs for long sequence dependencies.
- Clip gradient norms to limit unusually large updates, especially for recurrent models.
- Check the learning rate and monitor gradient norms and loss during training.

These techniques help, but the right choice depends on the architecture, data, and training objective.