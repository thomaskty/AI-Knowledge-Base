> # Normalization Methods

## Covariate Drift

Covariate Shift is a phenomenon in machine learning where the distribution of the input features (covariates) changes between the training and testing phases, while the relationship between the inputs and the outputs remains the same.

$$
P_{\text{train}}(X) \neq P_{\text{test}}(X),
\qquad
P_{\text{train}}(Y|X) = P_{\text{test}}(Y|X).
$$

## Internal Covariate Shift - ICS

Covariate shift to the intermediate layers of a neural network.
Instead of the input data distribution changing between training and testing, the distribution of the activations (outputs) produced by one layer changes during training, causing the next layer to receive inputs from a constantly changing distribution.
This was the primary motivations behind the development of Batch Normalization. The idea was that if the activations could be normalized to maintain a consistent distribution throughout training, each layer would learn under more stable conditions, leading to faster and more reliable convergence.
Reducing ICS is not the primary reason batch normalization improves performance. The current understanding is that Batch Normalization mainly helps by smoothing the optimization landscape, making gradient based optimization more stable and allowing larger learning rates.

## Batch Normalization

Used to stabilize and accelerate the training of deep neural networks
Normalize the activations of a layer using the statistics (mean and variance) computed from the current mini batch.
BatchNorm is usually applied before the activation function because its goal is to stabilize the distribution of the values entering the nonlinearity. If normalization were applied after ReLu, it would operate on already-clipped activations, potentially reintroducing negative values and reducing the intended effect of the activation.
By normalizing the pre-activations, the activation function consistently receives inputs with a controlled mean and variance, leading to more stable and efficient training.

Consider an input matrix :

$$
X=
\begin{bmatrix}
2 & 5 & 8 & 10\\
4 & 7 & 6 & 14\\
6 & 9 & 10 & 18
\end{bmatrix}
$$

Here, rows represents the each sample in the batch, columns represents the features. We can take the mean and variance of each feature ;

$$
\mu=
[4,\;7,\;8,\;14] \\
\sigma
=
[1.63,\;1.63,\;1.63,\;3.27]
$$

Now each value is normalized using the statistics of its own feature.

$$
X_{\text{norm}}=
\begin{bmatrix}
-1.22 & -1.22 & \;\;0.00 & -1.22\\
\;\;0.00 & \;\;0.00 & -1.22 & \;\;0.00\\
\;\;1.22 & \;\;1.22 & \;\;1.22 & \;\;1.22
\end{bmatrix}
$$

While this makes training more stable, it also restricts the representation learned by the network. Some neurons may actually benefit from producing activations with a different mean or variance. If we always force the outputs to have mean 0 and variance 1, the network loses this flexibility.

To overcome this, Batch Normalization introduces two learnable parameters for every feature (or neuron):
γ (Gamma): scales the normalized activations.
β (Beta): shifts the normalized activations.

$$
Y=\gamma X_{\text{norm}}+\beta \\ 

X_{\text{norm}}=
\begin{bmatrix}
-1.22 & -1.22 & 0.00 & -1.22\\
0.00 & 0.00 & -1.22 & 0.00\\
1.22 & 1.22 & 1.22 & 1.22
\end{bmatrix}

\\ 
\gamma=[2,\;1,\;0.5,\;3] \quad \beta=[1,\;0,\;-2,\;0]
\\
Y=
\begin{bmatrix}
-1.44 & -1.22 & -2.00 & -3.66\\
1.00 & 0.00 & -2.61 & 0.00\\
3.44 & 1.22 & -1.39 & 3.66
\end{bmatrix}
$$

Notice that the outputs are no longer constrained to have mean 0 and variance 1. Instead, the network learns the most suitable distribution for each neuron while still benefiting from the stability provided by normalization.
`<u><b>`Running Mean and Running Variance Computation in Batch Normalization (BN)`</u></b>`
During training, Batchnorm normalizes using the current mini-batch statistics. During inference, there is no mini-batch ( or we don’t want the output to depend on other test samples), so it uses statistics accumulated during training.

$$
W_1=
\begin{bmatrix}
1&0&0\\
0&1&0\\
0&0&1
\end{bmatrix}
\quad
W_2=
\begin{bmatrix}
1&0&0&1\\
0&1&1&0\\
1&1&0&0
\end{bmatrix}
\quad
X^{(1)}=
\begin{bmatrix}
1&2&3\\
5&4&7
\end{bmatrix}
$$

First linear layer computation;

$$
Z_1=
XW_1=
\begin{bmatrix}
1&2&3\\
5&4&7
\end{bmatrix}
$$

Computing the mean vector and variance vector we get;

$$
\mu_1=
\begin{bmatrix}
3&3&5
\end{bmatrix}
\quad
\sigma_1^2=
\begin{bmatrix}
4&1&4
\end{bmatrix}
$$

Normalized output becomes;

$$
\hat Z_1=
\begin{bmatrix}
-1&-1&-1\\
1&1&1
\end{bmatrix}
$$

Now we update the running statistics for the BN(1);

$$
\text{Running Mean(1)} = [0,0,0]
\\ 
\text{Running Variance(1)} = [1,1,1]
\\ 
\text{New Running Mean} = 0.9\cdot[0,0,0] + 0.1 \cdot [3,3,5] = [0.3, 0.3,0.5]
\\
\text{New Running Variance} = 0.9\cdot[1,1,1]+0.1\cdot[4,1,4] = [1.3,1.0,1.3]
$$

Second Linear Layer Computation & BN(2) Running Statistics Update

After Batch Normalization (and assuming an identity activation function for simplicity), $\hat{Z}_1$ passes through the second weight matrix $W_2$:

$$
W_2 = 
\begin{bmatrix}
1 & 0 & 0 & 1 \\
0 & 1 & 1 & 0 \\
1 & 1 & 0 & 0
\end{bmatrix}
\quad
\hat{Z}_1 = 
\begin{bmatrix}
-1 & -1 & -1 \\
1 & 1 & 1
\end{bmatrix}
$$

$$
Z_2 = \hat{Z}_1 W_2 = 
\begin{bmatrix}
-1 & -1 & -1 \\
1 & 1 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 & 1 \\
0 & 1 & 1 & 0 \\
1 & 1 & 0 & 0
\end{bmatrix}
=
\begin{bmatrix}
-2 & -2 & -1 & -1 \\
2 & 2 & 1 & 1
\end{bmatrix}
$$

Computing the column-wise mean ($\mu_2$) and variance ($\sigma_2^2$) across the $N=2$ samples in the mini-batch for each of the $4$ feature channels:

$$
\mu_2 = \begin{bmatrix} 0 & 0 & 0 & 0 \end{bmatrix}
$$

$$
\sigma_2^2 = \begin{bmatrix} 4 & 4 & 1 & 1 \end{bmatrix}
$$

$$
\hat{Z}_2 = \frac{Z_2 - \mu_2}{\sqrt{\sigma_2^2 + \epsilon}} = 
\begin{bmatrix}
\frac{-2-0}{\sqrt{4}} & \frac{-2-0}{\sqrt{4}} & \frac{-1-0}{\sqrt{1}} & \frac{-1-0}{\sqrt{1}} \\
\frac{2-0}{\sqrt{4}} & \frac{2-0}{\sqrt{4}} & \frac{1-0}{\sqrt{1}} & \frac{1-0}{\sqrt{1}}
\end{bmatrix}
=
\begin{bmatrix}
-1 & -1 & -1 & -1 \\
1 & 1 & 1 & 1
\end{bmatrix}
$$

Updating Running Statistics for BN(2)

Assuming initial running statistics for $BN_2$ are set to default values ($\text{Running Mean} = [0, 0, 0, 0]$, $\text{Running Variance} = [1, 1, 1, 1]$) and using momentum coefficient $\text{momentum} = 0.1$:

$$
\text{Running Mean(2)} = [0, 0, 0, 0]
\\
\text{Running Variance(2)} = [1, 1, 1, 1]
$$

$$
\begin{aligned}
\text{New Running Mean} &= (1 - 0.1) \cdot [0, 0, 0, 0] + 0.1 \cdot [0, 0, 0, 0] \\
&= [0.0, 0.0, 0.0, 0.0]
\end{aligned}
$$

$$
\begin{aligned}
\text{New Running Variance} &= (1 - 0.1) \cdot [1, 1, 1, 1] + 0.1 \cdot [4, 4, 1, 1] \\
&= [0.9, 0.9, 0.9, 0.9] + [0.4, 0.4, 0.1, 0.1] \\
&= [1.3, 1.3, 1.0, 1.0]
\end{aligned}
$$

## Layer Normalization

Layer Normalization (LayerNorm) normalizes the activations across the features of each individual sample, rather than across the samples in a mini-batch.
Independent of batch size
Same behaviour during training and inference
There is no running statistics

Unlike Batch Normalization (which normalizes across the batch dimension for each feature channel), **Layer Normalization** normalizes the activations **across the features for each individual sample** independently. This makes LayerNorm independent of batch size, which is why it is standard practice in Transformers and RNNs.

Given the input matrix $X$ (where rows are samples and columns are features):

$$
X = 
\begin{bmatrix}
1 & 2 & 3 \\
5 & 4 & 7
\end{bmatrix}
$$

Assume learnable affine parameters: **scale $\gamma = [1, 1, 1]$** and **shift $\beta = [0, 0, 0]$**, with a small constant $\epsilon = 0$ for simplicity.

$$
\mu_1 = \frac{1 + 2 + 3}{3} = \frac{6}{3} = 2
$$

$$
\sigma_1^2 = \frac{(1 - 2)^2 + (2 - 2)^2 + (3 - 2)^2}{3} = \frac{1 + 0 + 1}{3} = \frac{2}{3} \approx 0.6667
$$

$$
\hat{x}^{(1)} = \frac{x^{(1)} - \mu_1}{\sqrt{\sigma_1^2 + \epsilon}} = 
\begin{bmatrix}
\frac{1 - 2}{\sqrt{2/3}} & \frac{2 - 2}{\sqrt{2/3}} & \frac{3 - 2}{\sqrt{2/3}}
\end{bmatrix}
=
\begin{bmatrix}
-\sqrt{1.5} & 0 & \sqrt{1.5}
\end{bmatrix}
\approx
\begin{bmatrix}
-1.2247 & 0 & 1.2247
\end{bmatrix}
$$

$$
\mu_2 = \frac{5 + 4 + 7}{3} = \frac{16}{3} \approx 5.3333
$$

$$
\sigma_2^2 = \frac{\left(5 - \frac{16}{3}\right)^2 + \left(4 - \frac{16}{3}\right)^2 + \left(7 - \frac{16}{3}\right)^2}{3} = \frac{\frac{1}{9} + \frac{16}{9} + \frac{25}{9}}{3} = \frac{42/9}{3} = \frac{14}{9} \approx 1.5556
$$

$$
\hat{x}^{(2)} = \frac{x^{(2)} - \mu_2}{\sqrt{\sigma_2^2 + \epsilon}} = 
\begin{bmatrix}
\frac{5 - 16/3}{\sqrt{14/9}} & \frac{4 - 16/3}{\sqrt{14/9}} & \frac{7 - 16/3}{\sqrt{14/9}}
\end{bmatrix}
=
\begin{bmatrix}
-\frac{1}{\sqrt{14}} & -\frac{4}{\sqrt{14}} & \frac{5}{\sqrt{14}}
\end{bmatrix}
\approx
\begin{bmatrix}
-0.2673 & -1.0690 & 1.3363
\end{bmatrix}
$$

Since $\gamma = [1, 1, 1]$ and $\beta = [0, 0, 0]$, scaling and shifting leave the normalized values unchanged:

$$
Y = 
\begin{bmatrix}
-1.2247 & 0 & 1.2247 \\
-0.2673 & -1.0690 & 1.3363
\end{bmatrix}
$$

## RMS Normalization

RMS Norm is almost Layer Norm, except it removes one operation.

$$
\hat{x}
=\frac{x}
{\sqrt{\text{mean}(x^2)+\epsilon}} \quad , \quad  y=\gamma\hat x
$$

$$
x=
\begin{bmatrix}
1&2&3
\end{bmatrix}
$$

Taking the Mean sqauared

$$
\frac{1+4+9}{3}
=
\frac{14}{3}
=\sqrt{4.67} = 2.16
$$

$$
\hat x
=
\begin{bmatrix}
0.46&0.93&1.39
\end{bmatrix}
$$

Researchers found that removing the mean isn’t actually necessary. Simply controlling the magnitude of the activations is often enough for stable training.

## Instance Norm & Group Norm

**Instance Normalization (InstanceNorm)** normalizes each channel of each sample independently by computing the mean and variance over the spatial dimensions (height and width). It does not depend on other samples in the batch.
**Group Normalization (GroupNorm)** divides the feature channels into several groups and normalizes each group independently. It computes statistics within each group for every sample.

---

BatchNorm → Compare this feature across different samples.
LayerNorm → Compare different features within the same sample.
RMSNorm → Scale the sample by its overall magnitude.
InstanceNorm → Normalize each image channel independently.
GroupNorm → Normalize a few channels together as a group.

---
