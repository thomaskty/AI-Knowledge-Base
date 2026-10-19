# Biological Neuron: The Inspiration

A biological neuron receives signals through its dendrites. The cell body integrates those signals, and if the combined input is strong enough, the neuron produces an electrical spike that travels along its axon. Synapses determine how strongly one neuron influences another, and their strengths can change with learning.

This is an analogy, not a literal simulation. Artificial neurons use numbers and mathematical operations; biological neurons use complex electrochemical processes and timing.

## The Classical Perceptron

For input features $x_1,\ldots,x_n$, a perceptron computes a weighted sum plus a bias:

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

Using column vectors $x,w \in \mathbb{R}^{n}$, the same computation can be written as:

$$
z = w^T x + b
$$

Here, $w_i$ indicates how strongly feature $x_i$ affects the decision, and $b$ shifts the decision boundary. The weighted sum $z$ is called the **pre-activation**. In some contexts, it is also called a **logit**.

A classic perceptron applies a hard threshold:

$$
\hat{y} =
\begin{cases}
1 & \text{if } z \geq 0, \\
0 & \text{if } z < 0.
\end{cases}
$$

For example, if $x=(2,1)^T$, $w=(0.5,-1)^T$, and $b=0.2$, then:

$$
z = (0.5)(2) + (-1)(1) + 0.2 = 0.2
$$

Since $z \geq 0$, the perceptron predicts $\hat{y}=1$.

With one perceptron, the decision boundary between two classes is a straight line in two dimensions, a plane in three dimensions, or a hyperplane in higher dimensions. A single perceptron can learn only linearly separable classifications.

**Perceptron definition:** A perceptron is a mathematical model of a single decision-making unit that learns a linear decision boundary to separate two classes.

Historically, the term *perceptron* referred to the complete learning model proposed by Frank Rosenblatt: the computational unit and the algorithm used to update its weights.

In modern deep learning, the word *neuron* usually refers to a computational unit that applies a weighted sum followed by an activation function.

The original perceptron uses the **perceptron learning rule**, not backpropagation. One common form of the update is:

$$
w_{\text{new}} = w_{\text{old}} + \eta (y-\hat{y})x
$$

$$
b_{\text{new}} = b_{\text{old}} + \eta (y-\hat{y})
$$

Here, $y$ is the true label, $\hat{y}$ is the predicted label, and $\eta$ is the learning rate. The weight update above assumes $x$ is a column vector.

## The Modern Artificial Neuron

### Why do modern neural networks rarely use the original perceptron?

- Gradient-based optimization cannot effectively train a hard step activation using ordinary derivatives: its derivative is zero almost everywhere and undefined at the threshold.
- Modern neural networks typically use differentiable or piecewise-differentiable activation functions, allowing gradient-based learning.

**Modern artificial neuron:** A computational unit that transforms input features into an output by applying a weighted linear transformation followed by an activation function.

A modern artificial neuron keeps the weighted sum but usually applies a smooth or piecewise-smooth activation function:

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

$$
a = f(z)
$$

The output $a$ can be a score, a hidden-layer feature, or a probability-like value. Common activation functions include:

- **Sigmoid:**

  $$
  \sigma(z) = \frac{1}{1+e^{-z}}
  $$

  It maps real-valued inputs into the interval $(0,1)$.
- **Tanh:**

  $$
  \tanh(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}}
  $$

  It maps real-valued inputs into $(-1,1)$.
- **ReLU:**

  $$
  \operatorname{ReLU}(z) = \max(0,z)
  $$

  It outputs zero for negative inputs and preserves positive inputs.

During training, the model's trainable parameters—typically weights and biases—are updated. The activation function is usually fixed rather than learned, although some architectures use learnable activation parameters.

Training is the process of finding parameter values that minimize a chosen loss function.

## Multi-Layer Perceptron (MLP)

Instead of asking one neuron to solve an entire problem, an MLP combines many neurons that can learn different features or patterns.

- Each neuron has its own weights and bias.
- Neurons in a layer can receive the same input values but apply different weights, allowing them to learn different representations.
- In a fully connected or **dense layer**, each neuron connects to every neuron in the previous layer. Each connection has its own weight.

An MLP connects artificial neurons in layers:

1. **Input layer:** Receives the feature values.
2. **Hidden layers:** Transform representations using weighted sums and nonlinear activation functions.
3. **Output layer:** Produces a prediction, such as a class score, probability, or numeric value.

For a layer with input vector $a^{(l-1)}$, weights $W^{(l)}$, and bias $b^{(l)}$, the forward computation is:

$$
z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = f^{(l)}\left(z^{(l)}\right)
$$

The exact dimensions depend on whether activations are represented as row or column vectors. Here, activations are column vectors and each row of $W^{(l)}$ corresponds to one output neuron.

## Perceptron, Neuron, MLP, FFN, and NN

- A **Multi-Layer Perceptron (MLP)** is a feedforward neural network that uses fully connected layers.
- Despite its historical name, a modern MLP typically does not use the original perceptron's binary step activation. It uses activation functions such as ReLU, sigmoid, or tanh.
- Differentiable or piecewise-differentiable activations enable gradient-based optimization with backpropagation.
- **Feedforward neural network (FNN)** describes a network in which information moves from input toward output without recurrent connections. An MLP is one type of FNN.
- **Dense network** is often used informally to refer to a network built from fully connected layers.
- **Neural network (NN)** is the broader category and includes many architectures beyond MLPs, such as convolutional, recurrent, and transformer networks.

The terms MLP, FNN, and dense network are related, but they are not always exact synonyms.

## Autograd and Backpropagation (Matrix Example)

Consider a batch of two examples with three input features each. Let $X$ have shape $(2,3)$, $W_1$ have shape $(3,4)$, and $W_2$ have shape $(4,1)$.

For this example, we use two linear transformations without biases or activation functions:

$$
M = XW_1
$$

$$
\hat{Y} = MW_2
$$

The resulting shapes are:

- $X \in \mathbb{R}^{2\times3}$
- $W_1 \in \mathbb{R}^{3\times4}$
- $M \in \mathbb{R}^{2\times4}$
- $W_2 \in \mathbb{R}^{4\times1}$
- $\hat{Y} \in \mathbb{R}^{2\times1}$

Let $Y$ be the target vector. Define the **sum of squared errors** over the two examples as:

$$
L = \sum_{j=1}^{2}(Y_j-\hat{Y}_j)^2
$$

There is no factor of $1/2$ in this loss, so the derivative includes a factor of 2.

### Gradients by the chain rule

The gradient with respect to the predictions is:

$$
\frac{\partial L}{\partial \hat{Y}} = 2(\hat{Y}-Y)
$$

The gradient with respect to the second weight matrix is:

$$
\frac{\partial L}{\partial W_2}
= M^T\frac{\partial L}{\partial \hat{Y}}
= 2M^T(\hat{Y}-Y)
$$

The gradient with respect to the intermediate matrix is:

$$
\frac{\partial L}{\partial M}
= \frac{\partial L}{\partial \hat{Y}}W_2^T
= 2(\hat{Y}-Y)W_2^T
$$

The gradient with respect to the first weight matrix is:

$$
\frac{\partial L}{\partial W_1}
= X^T\frac{\partial L}{\partial M}
= 2X^T(\hat{Y}-Y)W_2^T
$$

### Numerical example

Use the following input matrix:

$$
X =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$


$$
W_1 =
\begin{bmatrix}
1 & 0 & 2 & 1 \\
0 & 1 & 1 & 0 \\
1 & 2 & 0 & 1
\end{bmatrix}
$$

$$
W_2 =
\begin{bmatrix}
1 \\
2 \\
1 \\
0
\end{bmatrix}
$$

The target vector is:

$$
Y =
\begin{bmatrix}
10 \\
20
\end{bmatrix}
$$

First, calculate \(M=XW_1\):

$$
M =
\begin{bmatrix}
4 & 8 & 4 & 4 \\
10 & 17 & 13 & 10
\end{bmatrix}
$$

Then calculate the predictions:

$$
\hat{Y}=MW_2
=
\begin{bmatrix}
4+16+4+0 \\
10+34+13+0
\end{bmatrix}
=
\begin{bmatrix}
24 \\
57
\end{bmatrix}
$$

The prediction error is:

$$
\hat{Y}-Y =
\begin{bmatrix}
14 \\
37
\end{bmatrix}
$$

The loss is:

$$
L = 14^2+37^2=196+1369=1565
$$

The gradient with respect to the predictions is:

$$
\frac{\partial L}{\partial \hat{Y}}
=2(\hat{Y}-Y)
=
\begin{bmatrix}
28 \\
74
\end{bmatrix}
$$

The gradient with respect to $W_2$ is:

$$
\begin{aligned}
\frac{\partial L}{\partial W_2}
&=M^T
\begin{bmatrix}
28 \\
74
\end{bmatrix} \\
&=
\begin{bmatrix}
4 & 10 \\
8 & 17 \\
4 & 13 \\
4 & 10
\end{bmatrix}
\begin{bmatrix}
28 \\
74
\end{bmatrix} \\
&=
\begin{bmatrix}
852 \\
1482 \\
1074 \\
852
\end{bmatrix}
\end{aligned}
$$

The gradient with respect to M is:

$$
\begin{aligned}
\frac{\partial L}{\partial M}
&=
\begin{bmatrix}
28 \\
74
\end{bmatrix}
\begin{bmatrix}
1 & 2 & 1 & 0
\end{bmatrix} \\
&=
\begin{bmatrix}
28 & 56 & 28 & 0 \\
74 & 148 & 74 & 0
\end{bmatrix}
\end{aligned}
$$

Finally, the gradient with respect to $W_1$ is:

$$
\begin{aligned}
\frac{\partial L}{\partial W_1}
&=X^T\frac{\partial L}{\partial M} \\
&=
\begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}
\begin{bmatrix}
28 & 56 & 28 & 0 \\
74 & 148 & 74 & 0
\end{bmatrix} \\
&=
\begin{bmatrix}
324 & 648 & 324 & 0 \\
426 & 852 & 426 & 0 \\
528 & 1056 & 528 & 0
\end{bmatrix}
\end{aligned}
$$

All gradient shapes match their corresponding parameter or intermediate matrix:

- $\partial L/\partial \hat{Y}$: $(2,1)$
- $\partial L/\partial W_2$: $(4,1)$
- $\partial L/\partial M$: $(2,4)$
- $\partial L/\partial W_1$: $(3,4)$

**Important:** This is a simplified linear network used to demonstrate the chain rule. Real neural networks commonly include biases and nonlinear activations, whose derivatives must also be included during backpropagation.

Autograd systems calculate these derivatives by tracking tensor operations and applying the chain rule. Backpropagation is the algorithmic process of propagating gradients backward through the computation graph; automatic differentiation is a way to calculate those gradients.

## Vanishing and Exploding Gradients

Suppose a network has many layers. During backpropagation, the gradient is repeatedly multiplied by derivatives of activation functions and by weight matrices. As a result, the gradient can shrink or grow substantially as it travels toward earlier layers.

- **Vanishing gradients:** If the relevant derivatives and weight transformations repeatedly shrink gradient magnitudes, gradients reaching early layers can become extremely small. Those layers then learn very slowly.
- **Exploding gradients:** If the transformations repeatedly amplify gradient magnitudes, gradients can become extremely large, leading to unstable parameter updates.

### Why sigmoid can contribute to vanishing gradients

For the sigmoid activation function,

$$
\sigma'(x)=\sigma(x)\left(1-\sigma(x)\right)
$$

Its derivative is at most 0.25, attained when $\sigma(x)=0.5$. When sigmoid units operate in saturated regions, their derivatives are much smaller than this maximum.

Repeated multiplication by small derivatives can contribute to vanishing gradients in deep networks. The actual gradient also depends on the weight matrices and the other operations in the network.

### ReLU and the dying ReLU problem

For ReLU,

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

Its derivative is:

$$
\frac{d}{dx}\operatorname{ReLU}(x)=
\begin{cases}
1 & \text{if } x>0, \\
0 & \text{if } x<0.
\end{cases}
$$

At \(x=0\), ReLU is not differentiable; implementations choose a subgradient, commonly zero.

For positive inputs, ReLU's derivative is 1, so it does not shrink the gradient through the activation itself. This can help mitigate one source of vanishing gradients. However, weight matrices can still shrink or amplify gradients.

For negative inputs, ReLU's derivative is zero. If a unit remains inactive for the inputs it receives, it may stop receiving useful gradient updates. This is known as the **dying ReLU problem**.

ReLU therefore does not eliminate vanishing gradients; it changes some of the conditions that cause them.

### The goal: stable gradient flow

Ideally, gradients should neither vanish nor explode as they propagate through a deep network. Informally, we want gradient magnitudes to remain at a useful scale across layers. One simplified way to express this goal is:

$$
\left\|
\frac{\partial L}{\partial h^{(l)}}
\right\|
\approx
\left\|
\frac{\partial L}{\partial h^{(l+1)}}
\right\|
$$

This is an intuition rather than a strict requirement for every pair of layers: gradient magnitudes can legitimately vary by layer and task.

There is no rule that every individual derivative must be approximately 1. The goal is for the overall gradient flow to remain numerically manageable and useful for learning.

A gradient tells the optimizer how the loss changes with respect to a parameter:

- **Direction:** The sign indicates which direction locally increases or decreases the loss.
- **Magnitude:** The size indicates how sensitive the loss is to a parameter change, relative to the chosen parameterization and scale.

The optimizer uses gradients to update parameters. Stable gradient flow helps the network learn effectively throughout its depth.
