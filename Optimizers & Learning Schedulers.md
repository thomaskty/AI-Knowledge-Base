# Optimizers and Learning Rate Schedulers in Neural Networks

## What an optimizer does

A neural network defines a parameterized prediction function $f_\theta(x)$, where $\theta$ contains trainable weights and biases. Training minimizes a loss over a dataset.

For a dataset of $N$ examples, the empirical objective is

$$
J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell(f_\theta(x_i),y_i).
$$

An optimizer uses gradients of this objective, or estimates of them, to update the parameters. A learning rate scheduler changes the learning rate during training.

The optimizer and scheduler solve different parts of the training problem:

- **Optimizer:** determines how gradient information updates parameters.
- **Learning rate:** controls the scale of an update.
- **Scheduler:** changes the learning rate according to a rule or validation signal.

## Notation used in the equations

| Symbol | Meaning |
|---|---|
| $\theta_t$ | Parameter vector at update step $t$ |
| $J(\theta)$ | Training objective |
| $g_t$ | Gradient estimate at step $t$ |
| $\eta_t$ | Learning rate at step $t$ |
| $B$ | Mini-batch size |
| $\beta,\beta_1,\beta_2$ | Exponential averaging coefficients |
| $\epsilon$ | Small numerical-stability constant |
| $m_t$ | First-moment estimate in adaptive optimizers |
| $v_t$ | Second-moment estimate in adaptive optimizers |

Unless specified otherwise, gradients are evaluated at the current parameters. Optimizer equations are written in a common convention; frameworks may differ in details such as bias correction, epsilon placement, weight-decay semantics, and reduction defaults.

## Gradient descent

Gradient descent updates parameters in the direction opposite to the gradient because the gradient points toward the steepest local increase of the objective.

$$
\theta_{t+1}=\theta_t-\eta_t\nabla_\theta J(\theta_t).
$$

The learning rate $\eta_t$ determines the step size. If it is too large, updates can overshoot useful regions or diverge. If it is too small, training may progress very slowly.

In practice, neural-network training often uses a gradient estimate computed from a subset of the training data rather than evaluating the full objective at every update.

## Batch gradient descent

Batch gradient descent computes the gradient using all $N$ training examples for every update:

$$
g_t=\frac{1}{N}\sum_{i=1}^{N}
\nabla_\theta \ell_i(\theta_t),
\qquad
\theta_{t+1}=\theta_t-\eta_t g_t.
$$

### Characteristics

- Each update uses the complete training set.
- The gradient is deterministic for fixed parameters and data, assuming deterministic computation.
- A single update can be expensive for a large dataset.
- There are fewer parameter updates per epoch than with mini-batch training.
- The full gradient can still be expensive to compute and does not guarantee a globally optimal solution for a non-convex neural-network objective.

**When it may be useful:** small datasets or optimization problems where the full gradient is cheap to compute.

## Stochastic gradient descent (SGD)

In the strict sense, stochastic gradient descent estimates the gradient using one randomly selected training example:

$$
g_t=\nabla_\theta \ell_{i_t}(\theta_t),
\qquad
\theta_{t+1}=\theta_t-\eta_t g_t.
$$

The sampled gradient is an estimate of the full-data gradient. Under uniform sampling, its expectation is the full-data gradient at the current parameters.

### Characteristics

- One example contributes to each update.
- Updates are inexpensive but noisy.
- Gradient noise can make the loss fluctuate from step to step.
- Many frequent updates can make progress quickly, but hardware utilization may be poor compared with vectorized mini-batches.

In everyday deep-learning usage, “SGD” often refers to mini-batch SGD rather than one-example updates. The batch size and implementation should therefore be stated explicitly.

## Mini-batch gradient descent

Mini-batch gradient descent computes the average gradient over a batch of $B$ examples:

$$
g_t=\frac{1}{B}\sum_{i\in\mathcal{B}_t}
\nabla_\theta \ell_i(\theta_t),
\qquad
\theta_{t+1}=\theta_t-\eta_t g_t.
$$

Here $\mathcal{B}_t$ is the mini-batch sampled at update $t$.

### Why mini-batches are the standard

- Matrix operations over a batch are efficient on GPUs and other accelerators.
- The gradient is usually less noisy than a one-example estimate.
- The update is much cheaper than computing the full-dataset gradient.
- Batch size affects gradient noise, memory usage, throughput, and the number of updates per epoch.

### Engineering considerations

- An **epoch** usually means one pass through the training dataset. With dataset size $N$ and batch size $B$, there are approximately $\lceil N/B\rceil$ updates per epoch when the final partial batch is retained.
- If the final partial batch is dropped, the number of updates can differ.
- Changing batch size changes the number of optimizer updates per epoch. Scheduler units (steps versus epochs) therefore matter.
- The gradient convention is usually a batch mean, but some objectives or frameworks use sums. This changes gradient scale and interacts with the learning rate.
- Large batches do not automatically improve generalization; their effects depend on the optimizer, learning-rate schedule, data, and training setup.

## Momentum optimization

Momentum uses an exponential moving average of gradients to smooth noisy updates and accumulate movement across steps.

A common convention is

$$
v_t=\beta v_{t-1}+g_t,
$$

$$
\theta_{t+1}=\theta_t-\eta_t v_t,
$$

where $v_0=0$ and $\beta\in[0,1)$ controls how much of the previous accumulated direction is retained.

Some implementations use the alternative convention $v_t=\beta v_{t-1}+(1-\beta)g_t$. This rescales the velocity and changes the exact interpretation of the learning rate, so use the convention documented by the framework.

### What the momentum coefficient does

- If $\beta=0$, the update reduces to ordinary gradient descent under the first convention.
- A larger $\beta$ retains more history and smooths short-term gradient fluctuations.
- A smaller $\beta$ makes the update respond more quickly to the current gradient.
- Values such as $0.9$ are common starting points, not universal optima.

Expanding the recurrence shows the exponentially weighted history:

$$
v_t=g_t+\beta g_{t-1}+\beta^2g_{t-2}+\cdots+\beta^t g_0.
$$

Older gradients receive exponentially smaller weights. Momentum can accelerate movement in directions where gradients consistently agree and reduce oscillation in directions where gradients alternate.

### Limitations

- The accumulated direction can lag when the gradient changes rapidly.
- A learning rate that is too large can still cause instability.
- Momentum does not adapt the learning rate independently for each parameter.

## Nesterov accelerated gradient (NAG)

Nesterov momentum evaluates the gradient at a look-ahead parameter position rather than only at the current position. This gives the update a chance to account for where the momentum term is about to move the parameters.

One common form is

$$
v_t=\beta v_{t-1}
+\nabla_\theta J(\theta_t-\eta_t\beta v_{t-1}),
$$

$$
\theta_{t+1}=\theta_t-\eta_t v_t.
$$

The point $\theta_t-\eta_t\beta v_{t-1}$ is the look-ahead position under this velocity convention.

Another equivalent-looking formulation is often written with a different definition of velocity or parameter update order. These forms should not be mixed term by term without reconciling their conventions.

### Intuition

- **Momentum:** uses the gradient at the current parameters and combines it with previous gradients.
- **Nesterov momentum:** estimates the gradient at the anticipated look-ahead position, which can provide a corrective signal before taking the full momentum step.

### Engineering notes

NAG is useful when momentum-based SGD is a good fit and the look-ahead correction helps. It still requires tuning the learning rate and momentum coefficient, and it does not eliminate sensitivity to learning-rate choice.

## Adaptive learning-rate optimizers

SGD and momentum use one global learning rate for all parameters. Adaptive optimizers maintain per-parameter statistics so that update magnitudes depend on recent gradient behavior.

This can be useful when different parameters have gradients with very different scales or frequencies. The exact behavior depends on the algorithm's moving averages and numerical-stability constant.

## AdaGrad

AdaGrad accumulates the squared gradients elementwise and scales each parameter's update by the accumulated magnitude.

$$
r_t=r_{t-1}+g_t\odot g_t,
$$

$$
\theta_{t+1}
=
\theta_t-\frac{\eta}{\sqrt{r_t}+\epsilon}\odot g_t.
$$

Here $\odot$ is elementwise multiplication, and the square root and division are elementwise. The accumulator is typically initialized as $r_0=0$.

### What the accumulator does

For parameter $j$,

$$
r_{t,j}=\sum_{k=1}^{t}g_{k,j}^2.
$$

If a parameter has repeatedly received large gradients, its denominator grows and its effective step size shrinks. Parameters that receive infrequent or small gradients may retain relatively larger updates.

### Strengths

- Adapts the effective learning rate independently for each parameter.
- Can work well for sparse features and infrequent updates.
- Requires no manually chosen exponential averaging coefficient.

### Limitations

- The squared-gradient accumulator only grows.
- Effective learning rates can become very small after many updates.
- This monotonically decreasing step scale can limit progress in long deep-learning runs.

### Engineering note

AdaGrad's effective learning rate is parameter-specific:

$$
\eta_{t,j}^{\mathrm{eff}}
=
\frac{\eta}{\sqrt{\sum_{k=1}^{t}g_{k,j}^2}+\epsilon}.
$$

The global $\eta$ is only the starting scale; the accumulated history controls each parameter's actual scale.

## RMSProp

RMSProp addresses AdaGrad's continually growing accumulator by using an exponential moving average of squared gradients.

$$
s_t=\rho s_{t-1}+(1-\rho)(g_t\odot g_t),
$$

$$
\theta_{t+1}
=
\theta_t-\frac{\eta_t}{\sqrt{s_t}+\epsilon}\odot g_t.
$$

Here $\rho\in[0,1)$ is the decay coefficient for the squared-gradient average. A common starting value is around $0.9$, but the appropriate value depends on the task and implementation.

### What the decay factor means

Expanding the recurrence gives

$$
s_t=(1-\rho)\sum_{k=1}^{t}\rho^{t-k}(g_k\odot g_k)
$$

when $s_0=0$. Recent squared gradients receive larger weights; older squared gradients decay geometrically by powers of $\rho$.

- **Higher $\rho$:** longer memory and slower adaptation to changes in gradient scale.
- **Lower $\rho$:** shorter memory and faster adaptation, but noisier estimates.
- Unlike AdaGrad, old squared gradients gradually lose influence.

The effective per-parameter learning rate is

$$
\eta_{t,j}^{\mathrm{eff}}
=
\frac{\eta_t}{\sqrt{s_{t,j}}+\epsilon}.
$$

### Strengths and limitations

- Adapts to changes in gradient magnitude.
- Avoids AdaGrad's permanently accumulating squared-gradient history.
- Does not use a first-moment estimate by itself in the standard form above.
- Still needs a base learning rate and can be sensitive to hyperparameters and implementation details.

Some frameworks include momentum or other variations around RMSProp. Check the exact optimizer definition when reproducing experiments.

## Adam (Adaptive Moment Estimation)

Adam combines an exponential moving average of gradients with an exponential moving average of squared gradients. It uses these estimates to adapt each parameter's update.

### First and second moments

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t\odot g_t).
$$

Here:

- $m_t$ estimates the first moment (mean) of recent gradients.
- $v_t$ estimates the second raw moment (mean squared gradient), not the variance alone.
- $\beta_1$ controls the decay of the first-moment average.
- $\beta_2$ controls the decay of the second-moment average.

Common defaults are $\beta_1=0.9$ and $\beta_2=0.999$, though these are not guaranteed to be optimal for every problem.

### Bias correction

Both moving averages are initialized at zero, so early estimates are biased toward zero. Adam corrects this initialization bias:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
$$

The update is

$$
\theta_{t+1}
=
\theta_t-\eta_t
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

The square root and division are elementwise.

### What each part contributes

- **First moment $m_t$:** smooths the gradient direction, similar to momentum.
- **Second moment $v_t$:** estimates the scale of recent squared gradients.
- **Bias correction:** compensates for zero initialization, especially early in training.
- **$\epsilon$:** prevents division by zero and helps numerical stability.
- **$\eta_t$:** controls the overall update scale.

### Strengths

- Often converges quickly in early training.
- Handles parameters with different gradient scales.
- Works well across many deep-learning tasks and is a common default optimizer.

### Limitations

- Still requires learning-rate and hyperparameter tuning.
- Adaptive updates do not guarantee better validation performance than SGD with momentum.
- Generalization behavior depends on the model, data, schedule, and training budget.
- Weight decay must be handled deliberately; ordinary L2 regularization inside Adam is not generally equivalent to AdamW.

### Adam's effective coordinate-wise scale

Ignoring the numerator's direction and focusing on the denominator, Adam scales each coordinate approximately according to

$$
\frac{\eta_t}{\sqrt{\hat v_{t,j}}+\epsilon}.
$$

This is not the entire update: the numerator $\hat m_{t,j}$ also matters, and the moment estimates are history-dependent.

## AdamW: Adam with decoupled weight decay

AdamW separates weight decay from the adaptive gradient update. This distinction matters because adding an L2 penalty to the loss changes the gradient that Adam processes, and Adam's adaptive scaling then also acts on that penalty gradient.

A common AdamW update is

$$
\theta_{t+1}
=
\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta_t\lambda\theta_t.
$$

Equivalently, under this convention,

$$
\theta_{t+1}
=
(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

The term $-\eta_t\lambda\theta_t$ is the decoupled weight-decay contribution. Some implementations exclude selected parameters, such as biases and normalization parameters, from weight decay.

### What the decay factor means

For the decay-only part of the update,

$$
\theta_{t+1}=(1-\eta_t\lambda)\theta_t.
$$

Thus, each update multiplies the decayed parameter component by $1-\eta_t\lambda$. If $\eta_t$ or $\lambda$ changes over time, the shrinkage factor changes too.

For example, with $\eta_t=0.001$ and $\lambda=0.01$,

$$
1-\eta_t\lambda=1-(0.001)(0.01)=0.99999.
$$

This is a small shrinkage per update; the cumulative effect depends on the number of updates and the learning-rate schedule. This calculation describes the decay-only component, not the full AdamW update, because the adaptive gradient term is also present.

### L2 regularization versus decoupled weight decay

With a loss penalty $\lambda\lVert\theta\rVert_2^2$, the gradient includes $2\lambda\theta$ under that exact convention. An adaptive optimizer scales this combined gradient coordinate by coordinate. AdamW instead applies weight decay separately from the adaptive gradient step.

For plain SGD, L2 regularization and weight decay can be algebraically equivalent when coefficients and conventions are matched. For adaptive methods such as Adam, the two are generally not equivalent.

### Engineering considerations

- AdamW is a common default for Transformer and other deep-network training.
- Tune weight decay separately from the learning rate.
- Check which parameters are decayed; many recipes exclude biases and normalization parameters.
- Weight decay and learning rate interact because the per-step shrinkage includes $\eta_t\lambda$.
- A scheduler that lowers the learning rate also lowers the per-step AdamW decay contribution when $\lambda$ is fixed.

## Optimizer comparison

| Optimizer | Main mechanism | Main advantage | Main limitation |
|---|---|---|---|
| Batch gradient descent | Full-dataset gradient | Stable, deterministic full gradient | Expensive update |
| SGD / mini-batch SGD | Sampled gradient | Simple; can generalize well | Sensitive to learning rate; noisy updates |
| Momentum | Exponential average of gradients | Smooths updates and builds consistent direction | Can lag when gradients change |
| Nesterov momentum | Gradient at a look-ahead point | Adds a corrective look-ahead signal | More convention-sensitive; still needs tuning |
| AdaGrad | Cumulative squared gradients | Useful for sparse or infrequent gradients | Effective learning rates can shrink too far |
| RMSProp | EMA of squared gradients | Adapts to recent gradient scales | Hyperparameter and learning-rate sensitivity |
| Adam | EMA of gradients and squared gradients | Fast, adaptive per-parameter updates | Not always best for generalization; weight decay requires care |
| AdamW | Adam plus decoupled weight decay | Separates adaptive update from weight shrinkage | Requires tuning decay and exclusions |

## Learning-rate schedulers

A learning-rate scheduler changes $\eta_t$ during training. The optimizer still computes the update direction and any moment estimates; the scheduler controls the global learning-rate scale or, in some systems, a set of parameter-group learning rates.

Schedulers may be defined by epochs, optimizer steps, processed examples, or validation metrics. This distinction matters when batch size or dataset size changes.

### Step decay

Step decay reduces the learning rate by a fixed multiplicative factor after a specified number of epochs or updates.

A common form is

$$
\eta_t=\eta_0\gamma^{\left\lfloor t/s\right\rfloor},
$$

where:

- $\eta_0$ is the initial learning rate.
- $\gamma$ is the decay factor, usually between 0 and 1.
- $s$ is the number of scheduler steps between reductions.
- $t$ is the scheduler's time index, measured in the same units as $s$.

If scheduling by epoch, $t$ and $s$ are measured in epochs. If scheduling by optimizer step, they are measured in update steps.

#### Numerical example

Suppose $\eta_0=0.1$, $\gamma=0.5$, and the learning rate is reduced every 10 epochs. Then:

| Epoch range | Learning rate |
|---|---:|
| 0–9 | $0.1$ |
| 10–19 | $0.05$ |
| 20–29 | $0.025$ |
| 30–39 | $0.0125$ |

Each decay event multiplies the current learning rate by $0.5$. A factor $\gamma=0.1$ would reduce it tenfold at each event; a factor closer to 1 would reduce it more gently.

#### When to use it

Step decay is useful when training is organized into phases and a few planned learning-rate drops are easy to tune. The abrupt changes are simple to reason about, but the schedule does not react to validation behavior unless combined with a metric-driven policy.

### Exponential decay

Exponential decay reduces the learning rate continuously or at every scheduler step by a constant multiplicative factor:

$$
\eta_t=\eta_0\gamma^t,
\qquad 0<\gamma<1.
$$

Here $\gamma$ is the per-step decay multiplier. At each step,

$$
\eta_{t+1}=\gamma\eta_t.
$$

The learning rate therefore becomes $\gamma$ times its previous value at every scheduler step. Unlike step decay, there are no planned plateaus between discrete drops.

#### Numerical example

Let $\eta_0=0.1$ and $\gamma=0.9$. Then:

| Scheduler step $t$ | Learning rate $\eta_t$ |
|---:|---:|
| 0 | $0.1$ |
| 1 | $0.09$ |
| 2 | $0.081$ |
| 3 | $0.0729$ |
| 4 | $0.06561$ |

After $t$ steps, the learning rate is $\eta_0\gamma^t$. If $\gamma$ is very close to 1, decay is slow; if it is appreciably smaller, decay is faster.

Some frameworks parameterize exponential decay by a continuous decay rate $k$:

$$
\eta(t)=\eta_0e^{-kt}.
$$

This is equivalent to a multiplicative decay form when $\gamma=e^{-k}$ for a unit step. Do not confuse the per-step multiplier $\gamma$ with the continuous rate $k$.

### Step decay versus exponential decay

| Property | Step decay | Exponential decay |
|---|---|---|
| Shape | Piecewise constant, with abrupt drops | Smooth multiplicative decrease per step |
| Main parameters | Initial rate, factor $\gamma$, interval $s$ | Initial rate, multiplier $\gamma$ or continuous rate $k$ |
| Learning-rate change | Only at scheduled boundaries | At every scheduler step |
| Useful when | Distinct training phases are desired | Gradual reduction is desired |
| Main caution | Boundary drops may be abrupt | Decay may become too small or too slow depending on $\gamma$ and duration |

### Scheduler time units are important

A scheduler can advance once per epoch or once per optimizer update. For exponential decay, changing from epoch-based to step-based scheduling without changing the factor can radically change the total decay.

For example, a multiplier of $0.99$ applied once per epoch is very different from $0.99$ applied hundreds of times per epoch. When changing batch size, gradient accumulation, or dataset size, verify the intended scheduler time unit.

## Other scheduler patterns worth recognizing

### Warmup

Warmup starts with a small learning rate and increases it over an initial number of steps. It is often used with Adam/AdamW in Transformer training to reduce instability at the beginning of training.

A simple linear warmup over $T_w$ steps is

$$
\eta_t=\eta_{\max}\frac{t}{T_w},
\qquad 0\le t\le T_w.
$$

Implementations may start at step zero or use slightly different endpoint conventions.

### Cosine decay

Cosine decay smoothly decreases the learning rate from a maximum to a minimum over a chosen horizon:

$$
\eta_t=\eta_{\min}
+\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\left(\pi\frac{t}{T}\right)\right),
\qquad 0\le t\le T.
$$

Here $T$ is the planned decay duration. At $t=0$, the learning rate is $\eta_{\max}$; at $t=T$, it reaches $\eta_{\min}$. Warmup may be combined with cosine decay.

### Reduce on plateau

A metric-driven scheduler reduces the learning rate when a monitored validation metric fails to improve for a configured number of checks. This differs from step and exponential decay because the timing depends on observed validation behavior.

When using it, specify the monitored metric, whether lower or higher is better, patience, minimum improvement threshold, reduction factor, and minimum learning rate.

## Practical training workflow

- Choose a baseline optimizer suitable for the model and task. AdamW is a common starting point for many modern deep networks; SGD with momentum remains useful in many settings.
- Select a sensible initial learning rate. Learning-rate choice often has a larger effect than small changes to optimizer hyperparameters.
- Decide whether the scheduler advances by epoch, optimizer step, or validation event.
- Record batch size, gradient accumulation, total optimizer updates, warmup duration, and scheduler horizon.
- Track training loss and validation metrics. A falling training loss with worsening validation performance can indicate overfitting, but may also reflect a metric mismatch or distribution shift.
- If using AdamW, tune weight decay separately and verify which parameters are excluded.
- If training is unstable, check learning rate, gradient scale, data preprocessing, loss reduction, mixed-precision settings, and gradient clipping where appropriate.
- Save the best validation checkpoint and evaluate it on a held-out test set.

## Common implementation mistakes

- **Confusing batch and mini-batch SGD:** state the batch size and gradient-reduction convention.
- **Advancing a scheduler at the wrong frequency:** per-step and per-epoch schedules are not interchangeable.
- **Using a decay factor without defining its time unit:** a factor applied every update differs substantially from one applied every epoch.
- **Assuming Adam and AdamW weight decay are equivalent:** they generally are not.
- **Ignoring bias correction in Adam:** early moment estimates differ from the bias-corrected values.
- **Changing batch size without revisiting the schedule:** this changes updates per epoch and may change total optimizer steps.
- **Using an excessively large learning rate:** adaptive optimizers do not eliminate divergence or unstable updates.
- **Comparing optimizer runs with different training budgets:** compare optimizer steps, processed examples, wall-clock time, and validation performance as appropriate.
- **Assuming a scheduler improves results automatically:** its schedule must fit the optimizer, dataset, and training duration.

## Engineering checklist

Before running a training experiment, record:

- Optimizer name and exact variant.
- Initial learning rate and parameter groups.
- Momentum or moment coefficients.
- Epsilon value and implementation convention where relevant.
- Weight-decay coefficient and parameters excluded from decay.
- Batch size, gradient accumulation, and number of optimizer updates.
- Scheduler type, decay factor, interval or horizon, and time unit.
- Warmup configuration and minimum learning rate, if applicable.
- Gradient clipping and mixed-precision configuration, if used.
- Validation metric, checkpoint-selection rule, and stopping criterion.

## Key takeaways

- Gradient descent differs by how much data is used to estimate each update: full batch, one example, or a mini-batch.
- Momentum smooths gradients over time; Nesterov momentum evaluates a gradient at a look-ahead point.
- AdaGrad uses cumulative squared gradients, while RMSProp uses an exponential moving average of squared gradients.
- Adam combines first- and second-moment estimates and applies bias correction.
- AdamW decouples weight decay from Adam's adaptive gradient update.
- Step decay multiplies the learning rate by a factor at scheduled boundaries; exponential decay multiplies it at every scheduler step.
- A decay factor has meaning only together with its time unit and schedule horizon.
- Optimizer choice, learning rate, scheduler, batch size, and weight decay should be treated as a coordinated training configuration and evaluated on validation data.
