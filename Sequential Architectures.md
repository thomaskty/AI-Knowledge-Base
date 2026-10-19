
# Sequential Models in Neural Networks

Sequential models learn from ordered observations where the meaning of an input depends partly on its position and surrounding context. Examples include natural language, speech, sensor streams, financial time series, event logs, biological sequences, and user interaction histories.

This note covers recurrent neural networks (RNNs), bidirectional recurrence, encoder–decoder architectures, attention, long short-term memory (LSTM), and gated recurrent units (GRU). The focus is on mathematical foundations, gradient behavior, tensor shapes, architectural trade-offs, and practical engineering considerations.

## Sequential Data and Task Patterns

A sequence of length $T$ is represented as

$$
X = (x_1, x_2, \ldots, x_T), \qquad x_t \in \mathbb{R}^{d_x}
$$

where $x_t$ is the feature vector at time step $t$ and $d_x$ is the input dimension.

A minibatch is commonly represented as a tensor of shape `[batch, time, features]`, although some libraries and recurrent kernels use `[time, batch, features]`.

Unlike independent observations, sequential observations exhibit dependencies across positions. These dependencies may be local, long-range, periodic, hierarchical, or conditional on a latent state.

Common sequence-learning task patterns include:

- **Sequence-to-one:** Map an input sequence to a single prediction. Examples include sentiment classification, document classification, and forecasting a target from a historical window.
- **One-to-sequence:** Generate a sequence conditioned on a fixed input representation. Image captioning is a classical example.
- **Sequence-to-sequence, aligned:** Produce one output per input time step. Examples include token classification, part-of-speech tagging, and frame-level labeling.
- **Sequence-to-sequence, unaligned:** Map an input sequence to an output sequence of potentially different length. Examples include machine translation and speech recognition.
- **Autoregressive prediction:** Predict the next token or observation conditioned on preceding context.
- **Sequence labeling:** Predict a label at every position, potentially using both past and future context when the task permits it.

Autoregressive sequence modeling commonly factorizes a joint distribution as

$$
p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{<t})
$$

where $x_{<t}$ denotes all observations preceding position $t$.

For conditional generation, the corresponding factorization is

$$
p(y_{1:T_y} \mid X)
=
\prod_{t=1}^{T_y} p(y_t \mid y_{<t}, X)
$$

The factorization makes the prediction dependency explicit: the current output depends on previous outputs and, for conditional generation, the input sequence.

## Core Challenges in Sequence Modeling

### Variable-Length Sequences

Sequences frequently have different lengths. Batching therefore requires padding, packed sequences, masking, or other variable-length processing strategies.

Padding must not affect the loss or be treated as genuine observations by the recurrent state transition.

### Long-Range Dependencies

A prediction may depend on evidence many time steps in the past. Repeated nonlinear transformations can make it difficult for the network to preserve information or propagate useful gradients across long distances.

### Temporal Credit Assignment

A loss at a later time step must assign responsibility to earlier inputs and hidden states. Backpropagation through time (BPTT) computes these gradients through the unrolled recurrent computation graph.

### Exposure Bias

During autoregressive training, a decoder often receives the ground-truth previous token. At inference, it receives its own generated token.

Consequently, inference can expose the model to contexts that differ from those encountered during teacher-forced training.

### Causality and Information Leakage

Bidirectional recurrence and unrestricted attention can use future positions. This is valid for offline sequence labeling but can introduce leakage in online prediction or forecasting.

For a causal forecasting task, every feature and transformation must be available at the prediction timestamp.

### Irregular Sampling and Missing Observations

A conventional recurrent network processes observations in sequence order but does not inherently understand the elapsed time between observations.

For irregularly sampled data, useful approaches include supplying time-delta features, timestamp encodings, missingness indicators, or using architectures designed for irregular temporal observations.

## Recurrent Neural Networks

A recurrent neural network maintains a hidden state that summarizes information from the sequence observed so far.

The simple RNN recurrence is

$$
h_t = \phi(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

where:

- $x_t \in \mathbb{R}^{d_x}$ is the current input.
- $h_{t-1} \in \mathbb{R}^{d_h}$ is the previous hidden state.
- $W_{xh} \in \mathbb{R}^{d_h \times d_x}$ is the input-to-hidden weight matrix.
- $W_{hh} \in \mathbb{R}^{d_h \times d_h}$ is the recurrent weight matrix.
- $b_h \in \mathbb{R}^{d_h}$ is the bias vector.
- $\phi$ is a nonlinear activation, commonly `tanh`.

A prediction head may transform the hidden state into output logits:

$$
o_t = W_{hy}h_t + b_y
$$

For multiclass classification, the probabilities are

$$
\hat{y}_t = \operatorname{softmax}(o_t)
$$

For sequence-to-one classification, a prediction head can operate on the final valid hidden state. For sequence labeling, the head commonly operates on each time-step representation.

The correct readout depends on the task, sequence lengths, and padding convention.

### Parameter Sharing and Temporal Unrolling

The same recurrent parameters are reused at every time step:

$$
h_t = f_\theta(x_t, h_{t-1})
$$

The sequence computation can be unrolled as

$$
h_1 = f_\theta(x_1,h_0)
$$

$$
h_2 = f_\theta(x_2,h_1)
$$

$$
\vdots
$$

$$
h_T = f_\theta(x_T,h_{T-1})
$$

Parameter sharing allows the model to process sequences of varying lengths without introducing a new transition matrix for each position.

The initial state $h_0$ is commonly zero, learned, or provided by another network.

Although the recurrence reuses the same parameters, the unrolled computational graph has depth proportional to the sequence length. This temporal depth is a major source of optimization difficulty.

### Backpropagation Through Time

BPTT applies the chain rule through the unrolled recurrent computation graph.

For a sequence-level objective, the total loss may be expressed as

$$
\mathcal{L} = \sum_{t=1}^{T} \ell_t
$$

where $\ell_t$ is the loss associated with time step $t$.

A recurrent parameter contributes to the loss at multiple time steps. Its gradient therefore accumulates contributions from every position where it is used.

The influence of an earlier hidden state $h_k$ on a later state $h_t$ is governed by a product of stepwise Jacobians:

$$
\frac{\partial h_t}{\partial h_k}
=
\frac{\partial h_t}{\partial h_{t-1}}
\frac{\partial h_{t-1}}{\partial h_{t-2}}
\cdots
\frac{\partial h_{k+1}}{\partial h_k}
$$

The expression illustrates how gradient propagation depends on repeated matrix multiplications.

If these transformations repeatedly contract relevant gradient directions, gradients vanish. If they repeatedly amplify relevant directions, gradients explode.

The behavior depends on the recurrent weights, activation derivatives, input trajectory, and the directions in which the Jacobians act.

### Vanishing Gradients

Vanishing gradients occur when gradients become increasingly small as they propagate backward through time.

For a `tanh` activation,

$$
\frac{d}{dz}\tanh(z) = 1-\tanh^2(z)
$$

The derivative approaches zero when the activation saturates near $-1$ or $1$.

Repeated multiplication by small derivatives and contracting recurrent transformations can make gradients from distant time steps negligible.

Consequences include:

- Poor learning of long-range dependencies.
- Stronger influence from recent observations than distant observations.
- Slow optimization of parameters associated with early sequence positions.
- Difficulty retaining relevant information over long sequences.

Gated architectures, appropriate initialization, attention mechanisms, and alternative sequence architectures can mitigate these problems.

### Exploding Gradients

Exploding gradients occur when repeated transformations amplify gradient magnitudes.

They may produce unstable parameter updates, sudden loss spikes, and non-finite model parameters.

Global-norm gradient clipping is a common mitigation:

$$
g \leftarrow g \cdot
\min\left(1,\frac{\tau}{\lVert g\rVert_2}\right)
$$

where $g$ is the concatenated gradient vector and $\tau$ is the clipping threshold.

When $\lVert g\rVert_2 > \tau$, the gradient direction is preserved while its magnitude is reduced.

Gradient clipping stabilizes optimization but does not resolve every underlying conditioning problem.

### Truncated Backpropagation Through Time

Full BPTT propagates gradients through the entire unrolled sequence. For long streams, this can require substantial memory and computation.

Truncated BPTT limits gradient propagation to windows of length $K$.

Advantages include:

- Reduced memory requirements.
- Lower training cost for long streams.
- Practical stateful training on continuous sequences.

The trade-off is that gradient paths crossing truncation boundaries are removed. Hidden state can still be carried forward between windows, so information may cross a boundary during the forward pass even though its gradient does not.

When implementing truncated BPTT, carry state only across contiguous chunks from the same stream and detach the carried state when the intended gradient boundary is reached.

Reset state at genuine sequence boundaries.

## Bidirectional Recurrent Neural Networks

A bidirectional RNN processes a sequence in both forward and backward directions.

The forward recurrence is

$$
\overrightarrow{h_t}
=
f_{\rightarrow}(x_t,\overrightarrow{h}_{t-1})
$$

The backward recurrence is

$$
\overleftarrow{h_t}
=
f_{\leftarrow}(x_t,\overleftarrow{h}_{t+1})
$$

The representations are commonly concatenated:

$$
h_t =
[\overrightarrow{h_t};\overleftarrow{h_t}]
$$

The resulting representation incorporates information from both preceding and following positions.

If each direction has hidden size $d_h$, concatenation produces a representation of dimension $2d_h$.

### Advantages

- Access to both left and right context.
- Strong representations for offline sequence labeling.
- Useful contextual information for named-entity recognition, document tagging, and full-utterance speech processing.

### Limitations

- Future context is unavailable in strictly causal streaming.
- Processing generally requires access to the complete sequence.
- Bidirectional representations can introduce information leakage when used incorrectly in forecasting.
- The recurrent computation is more expensive than a single-direction model.

A bidirectional RNN is appropriate when the complete input sequence is available before the prediction must be made.

## Encoder–Decoder Architecture

An encoder–decoder separates input representation learning from output sequence generation.

The encoder processes the source sequence:

$$
h_t^{enc} = f_{enc}(x_t,h_{t-1}^{enc})
$$

In a basic encoder–decoder, the final encoder state becomes a fixed context vector:

$$
c = h_{T_x}^{enc}
$$

The decoder generates the target sequence conditioned on this representation and previously generated tokens:

$$
s_t = f_{dec}(y_{t-1},s_{t-1},c)
$$

The next-token distribution is

$$
p(y_t \mid y_{<t},X)
=
\operatorname{softmax}(W_o s_t+b_o)
$$

where:

- $X$ is the source sequence.
- $T_x$ is the source length.
- $s_t$ is the decoder hidden state.
- $y_{t-1}$ is the previous target token.
- $c$ is the source context.

Depending on the architecture, the context may initialize the decoder state, be concatenated with decoder inputs, or be supplied through attention.

### Teacher Forcing

During training, the decoder commonly receives the ground-truth previous token:

$$
s_t = f_{dec}(y_{t-1}^{*},s_{t-1},c)
$$

where $y_{t-1}^{*}$ denotes the reference token.

The standard token-level negative log-likelihood objective is

$$
\mathcal{L}
=
-\sum_{t=1}^{T_y}
\log p(y_t^{*}\mid y_{<t}^{*},X)
$$

where $T_y$ is the target sequence length.

Padding positions must be excluded from the loss.

Teacher forcing makes training efficient because the ground-truth target sequence is available. However, it creates a difference between training and inference conditions.

### Autoregressive Inference

At inference, the decoder consumes its own previous output.

Common decoding strategies include:

- **Greedy decoding:** Select the highest-probability token at each step.
- **Beam search:** Maintain several candidate partial sequences and expand the most promising hypotheses.
- **Sampling:** Draw tokens from the predicted distribution, optionally after applying temperature or other sampling transformations.

Greedy decoding is computationally efficient but makes locally optimal decisions. Beam search explores more candidate sequences but does not guarantee the globally optimal output unless the search is exhaustive under the relevant scoring objective.

Generation usually terminates when an end-of-sequence token is produced or a maximum length is reached.

### The Fixed-Context Bottleneck

A basic encoder–decoder compresses the entire source sequence into a single fixed-size context vector.

As source length and semantic complexity increase, this representation can become an information bottleneck. The decoder may struggle to retrieve specific source details needed at later output positions.

Attention addresses this limitation by allowing the decoder to access encoder states dynamically.

## Attention Mechanism

Attention allows a model to retrieve relevant information from a collection of representations using content-dependent weights.

For an encoder–decoder with attention, let the encoder produce

$$
H = (h_1,h_2,\ldots,h_{T_x})
$$

At decoder time step $t$, the decoder state $s_t$ is compared with each encoder state $h_i$.

### Attention Scores

A compatibility function computes an unnormalized score:

$$
e_{t,i} = \operatorname{score}(s_t,h_i)
$$

The scores are normalized over source positions using softmax:

$$
\alpha_{t,i}
=
\frac{\exp(e_{t,i})}
{\sum_{j=1}^{T_x}\exp(e_{t,j})}
$$

For an unmasked source sequence,

$$
\sum_{i=1}^{T_x}\alpha_{t,i}=1
$$

The attention weights describe how the model distributes attention across source positions at the current decoder step.

### Context Vector

The context vector is the weighted sum of encoder representations:

$$
c_t = \sum_{i=1}^{T_x}\alpha_{t,i}h_i
$$

Unlike a fixed encoder summary, $c_t$ changes with the decoder state and can emphasize different source positions at different output steps.

The decoder can use the context to update its state or calculate output logits.

### Additive Attention

Additive attention, commonly associated with Bahdanau attention, computes scores using a learned feed-forward transformation:

$$
e_{t,i}
=
v^\top
\tanh(W_s s_t + W_h h_i + b)
$$

where $W_s$, $W_h$, $v$, and $b$ are learned parameters.

This formulation can accommodate query and key representations with different original dimensions because they are projected into a common scoring space.

### Dot-Product Attention

Dot-product attention computes compatibility using an inner product:

$$
e_{t,i} = s_t^\top h_i
$$

The query and key dimensions must match.

Scaled dot-product attention uses

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V
$$

where:

- $Q$ contains query vectors.
- $K$ contains key vectors.
- $V$ contains value vectors.
- $d_k$ is the key dimension.
- $M$ is an additive attention mask.

The scaling factor $\sqrt{d_k}$ helps control the magnitude of dot-product scores as the key dimension increases.

A mask typically assigns zero to allowed positions and a sufficiently large negative value to disallowed positions before softmax.

### Attention Complexity and Interpretation

For query length $T_q$ and key length $T_k$, full attention computes pairwise query–key interactions with complexity proportional to

$$
O(T_qT_kd_k)
$$

for the score computation, excluding other projection and value-aggregation costs.

When query and key lengths are both $T$, the pairwise interaction component is quadratic in sequence length.

Attention improves access to individual sequence positions, but attention weights should not automatically be treated as faithful causal explanations of model behavior.

Attention is a mechanism, not a complete architecture. It can be integrated with recurrent encoders and decoders or used as the foundation of non-recurrent architectures such as Transformers.

## Long Short-Term Memory Networks

Long short-term memory networks introduce a cell state and gating mechanisms to regulate information flow through time.

A standard LSTM maintains two states:

- $h_t$: hidden state exposed to subsequent layers or prediction heads.
- $c_t$: cell state used for controlled information retention.

The cell state provides an additive update path that can support more effective gradient propagation than a simple RNN.

### LSTM Equations

The input gate is

$$
i_t =
\sigma(W_i x_t+U_i h_{t-1}+b_i)
$$

The forget gate is

$$
f_t =
\sigma(W_f x_t+U_f h_{t-1}+b_f)
$$

The output gate is

$$
o_t =
\sigma(W_o x_t+U_o h_{t-1}+b_o)
$$

The candidate cell content is

$$
\tilde{c}_t =
\tanh(W_c x_t+U_c h_{t-1}+b_c)
$$

The cell state is updated as

$$
c_t =
f_t\odot c_{t-1}
+
i_t\odot\tilde{c}_t
$$

The hidden state is

$$
h_t =
o_t\odot\tanh(c_t)
$$

Here, $\sigma$ denotes the sigmoid function and $\odot$ denotes element-wise multiplication.

The gate values lie between zero and one. Each hidden dimension can learn its own gating behavior.

These equations describe a common LSTM variant without peephole connections.

### Understanding the Gates

**Input gate $i_t$**

Controls how much candidate information is written into the cell state.

**Forget gate $f_t$**

Controls how much of the previous cell state is retained.

**Candidate state $\tilde{c}_t$**

Proposes new content derived from the current input and previous hidden state.

**Output gate $o_t$**

Controls how much of the cell state is exposed through the hidden state.

The cell-state update combines retained information and newly written information:

$$
c_t =
\underbrace{f_t\odot c_{t-1}}_{\text{retained memory}}
+
\underbrace{i_t\odot\tilde{c}_t}_{\text{new information}}
$$

When the forget gate remains near one and the input gate is small, the cell state can preserve information across multiple time steps.

The additive update provides a comparatively direct gradient path. In a simplified view, the direct contribution of the previous cell state is modulated by $f_t$. If the forget gates remain near one, this path can preserve gradient magnitude more effectively than repeatedly applying a strongly contractive nonlinear transition.

However, the complete gradient also includes dependencies through the gates and hidden states. LSTMs do not eliminate vanishing or exploding gradients in every circumstance.

### LSTM Parameter Count

For input dimension $d_x$ and hidden dimension $d_h$, a standard single-layer LSTM has approximately

$$
N_{\text{LSTM}}
=
4(d_xd_h+d_h^2+d_h)
$$

parameters when one bias vector is counted per gate transformation.

The factor of four corresponds to the input gate, forget gate, output gate, and candidate cell transformation.

Actual framework parameter counts may differ because of separate input and recurrent biases, peephole connections, projection layers, and other implementation choices.

### LSTM Engineering Considerations

- A positive forget-gate bias is sometimes used to encourage retention early in training.
- Both hidden state and cell state must be carried between chunks in stateful training.
- States should be reset between unrelated sequences.
- Recurrent state must be aligned with the correct stream when processing continuous data in chunks.
- Gate computations increase parameter count and computation relative to a simple RNN.
- Fused recurrent kernels can improve runtime substantially, depending on the framework and hardware.

LSTMs are useful when gated memory is beneficial, but their additional complexity does not guarantee better performance than a GRU or a simpler architecture.

## Gated Recurrent Units

A gated recurrent unit uses gates to regulate the hidden-state update without maintaining a separate cell state.

A common GRU convention uses an update gate, a reset gate, and a candidate hidden state.

### GRU Equations

The update gate is

$$
z_t =
\sigma(W_z x_t+U_z h_{t-1}+b_z)
$$

The reset gate is

$$
r_t =
\sigma(W_r x_t+U_r h_{t-1}+b_r)
$$

The candidate hidden state is

$$
\tilde{h}_t =
\tanh
\left(
W_h x_t
+
U_h(r_t\odot h_{t-1})
+
b_h
\right)
$$

The hidden-state update is

$$
h_t =
(1-z_t)\odot h_{t-1}
+
z_t\odot\tilde{h}_t
$$

In this convention:

- $z_t$ is the update gate.
- $r_t$ is the reset gate.
- $\tilde{h}_t$ is the candidate hidden state.

The reset gate controls how much of the previous hidden state contributes to the candidate. The update gate interpolates between the previous state and the candidate.

### Understanding the GRU

The update equation combines old and candidate information:

$$
h_t =
\underbrace{(1-z_t)\odot h_{t-1}}_{\text{retained state}}
+
\underbrace{z_t\odot\tilde{h}_t}_{\text{candidate contribution}}
$$

When $z_t$ is close to zero, the previous hidden state is retained strongly.

When $z_t$ is close to one, the candidate state contributes strongly.

The reset gate affects candidate construction rather than directly interpolating the final hidden state.

**Implementation convention warning:** Some references and framework implementations define the update gate using the complementary convention:

$$
h_t =
z_t\odot h_{t-1}
+
(1-z_t)\odot\tilde{h}_t
$$

This is a convention difference, not necessarily an architectural disagreement. Verify the gate definition and reset-gate placement when comparing equations, porting weights, or reproducing results.

Framework implementations may also differ in where the reset gate is applied relative to the recurrent matrix multiplication.

### GRU Parameter Count

For input dimension $d_x$ and hidden dimension $d_h$, a standard GRU has approximately

$$
N_{\text{GRU}}
=
3(d_xd_h+d_h^2+d_h)
$$

parameters when one bias vector is counted per transformation.

The factor of three corresponds to the update gate, reset gate, and candidate transformation.

Actual parameter counts depend on framework-specific bias layouts and implementation details.

### GRU Engineering Considerations

- GRUs generally have fewer parameters than comparable standard LSTMs.
- They maintain one recurrent state instead of separate hidden and cell states.
- Their reduced gate complexity can make them computationally attractive.
- They can perform comparably to LSTMs, depending on the dataset and task.
- Their relative advantage should be evaluated under the target sequence lengths, latency requirements, and training budget.

A GRU is a strong baseline when gated recurrence is appropriate and a relatively compact model is desirable.

## Comparing Sequential Architectures

| Architecture | State or information mechanism | Strength | Limitation | Typical application |
|---|---|---|---|---|
| Simple RNN | Single hidden state | Low architectural complexity | Difficult long-range optimization | Short dependencies and baselines |
| Bidirectional RNN | Forward and backward states | Uses left and right context | Requires future context | Offline sequence labeling |
| LSTM | Hidden state, cell state, and gates | Controlled memory retention | More parameters and computation | Recurrent tasks with long dependencies |
| GRU | Single gated hidden state | Compact gated recurrence | No separate cell state | Efficient recurrent baselines |
| Encoder–decoder | Encoder representation and autoregressive decoder | Variable-length input/output mapping | Fixed-context bottleneck without attention | Translation and sequence generation |
| Encoder–decoder with attention | Dynamic context over encoder states | Selective access to source information | Additional compute and memory | Conditional sequence generation |
| Transformer | Attention-based contextual representations | Parallelizable sequence processing during training | Full attention has quadratic pairwise complexity | Language and general sequence modeling |

These architectures are not mutually exclusive. An encoder–decoder may use LSTM or GRU cells, and attention may be integrated into either recurrent or non-recurrent models.

The appropriate architecture depends on sequence length, causal constraints, data volume, computational budget, latency, and the evaluation objective.

## Tensor Shapes and Implementation Details

Assume a batch-first input tensor with shape

`[B, T, D_in]`

where:

- `B` is batch size.
- `T` is sequence length.
- `D_in` is input feature dimension.
- `H` is hidden size.

Typical output shapes are:

| Component | Typical shape |
|---|---|
| Input sequence | `[B, T, D_in]` |
| Unidirectional RNN outputs | `[B, T, H]` |
| Bidirectional RNN outputs | `[B, T, 2H]` |
| Unidirectional final hidden state, one layer | `[1, B, H]` in common PyTorch conventions |
| Bidirectional final hidden state, one layer | `[2, B, H]` in common PyTorch conventions |
| LSTM cell state | Same leading dimensions as its hidden state |

Exact shapes depend on the framework, number of layers, directionality, batch layout, and projection configuration.

In PyTorch, for example, `batch_first=True` changes the input and output layout but does not change the conventional layer/direction-first layout of the returned final states.

### Common Shape Errors

**Confusing time and batch dimensions**

An input with shape `[B, T, D_in]` is not interchangeable with `[T, B, D_in]` unless the recurrent layer is configured for the corresponding layout.

**Selecting the last padded output**

Using `outputs[:, -1, :]` can select a padded position instead of the final valid observation.

For variable-length sequences, use the correct final valid position, packed-sequence support, or the recurrent layer's final-state output as appropriate.

**Ignoring bidirectional dimensions**

Concatenated bidirectional outputs usually have dimension $2H$, not $H$.

**Mixing hidden and cell states**

An LSTM requires both $h_t$ and $c_t$. A standard GRU requires only $h_t$.

**Mismatched loss shapes**

For token-level prediction, align logits, targets, and masks carefully. Exclude padded target positions from the objective.

### Dimensional Consistency

For the simple RNN equation,

$$
h_t = \phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
$$

the input transformation has dimensions

$$
(d_h \times d_x)(d_x) \rightarrow (d_h)
$$

and the recurrent transformation has dimensions

$$
(d_h \times d_h)(d_h) \rightarrow (d_h)
$$

Both terms and the bias must have the same hidden dimension.

For batched computation, matrix multiplications and broadcasting must preserve the batch dimension consistently.

## Training and Optimization Checklist

### Data Preparation

- Define the meaning of one time step explicitly.
- Preserve chronological order when the task depends on time.
- Use chronological train, validation, and test splits for forecasting when random splitting would introduce future-data leakage.
- Fit preprocessing transformations only on training data.
- Distinguish missing observations from padding.
- Apply masks consistently to recurrent computation and loss calculation.
- Handle empty, truncated, and unusually long sequences.
- Verify that all features are available at the intended prediction timestamp.

### Optimization

- Establish a simple baseline before increasing model complexity.
- Monitor training loss, validation loss, and task-specific metrics.
- Track gradient norms when diagnosing unstable training.
- Apply global-norm clipping when exploding gradients are a concern.
- Tune learning rate, hidden size, layer count, batch size, and sequence truncation length.
- Use early stopping based on a validation metric aligned with deployment requirements.
- Evaluate multiple random seeds when training variance is material.
- Inspect non-finite losses, gradients, and parameters.

### Regularization

- Tune dropout based on the model and dataset.
- Distinguish recurrent dropout from dropout applied between stacked recurrent layers.
- Verify the framework's dropout semantics; some implementations apply dropout only between recurrent layers.
- Use weight decay where appropriate.
- Avoid assuming that additional layers or larger hidden states will improve generalization.

### Stateful Sequence Processing

- Reset hidden states between independent sequences unless the task explicitly requires state carryover.
- Preserve state only across contiguous chunks from the same stream.
- Carry both hidden and cell states for LSTM models.
- Detach carried states when implementing truncated BPTT.
- Ensure that shuffling and batching do not accidentally associate a state with the wrong sequence.
- Validate that training-time and inference-time state handling are consistent.

### Evaluation

- Use task-appropriate metrics rather than relying only on loss.
- Report performance by sequence length where long-range behavior matters.
- Evaluate rare patterns and relevant subgroups.
- For forecasting, compare against naïve and domain-specific baselines.
- Evaluate generation quality beyond teacher-forced next-token likelihood.
- Test robustness to missing observations, noise, truncation, and distribution shifts.
- Measure end-to-end latency, including preprocessing, batching, and decoding.

## Common Misconceptions

**RNNs cannot learn long-range dependencies.**

RNNs can learn such dependencies, but gradient propagation through many recurrent transitions can make learning them difficult.

**LSTMs eliminate vanishing gradients.**

LSTM cell states can provide a more favorable gradient path, but vanishing and exploding gradients can still occur.

**Attention is itself a recurrent architecture.**

Attention is a mechanism for content-based information retrieval. It can be used within recurrent architectures or in architectures without recurrence.

**Bidirectional models are always better.**

Bidirectional models use future context and are unsuitable when that information is unavailable at prediction time.

**The final hidden state is always the best sequence representation.**

A single state can become an information bottleneck. Pooling, attention, or task-specific representations may perform better.

**More gates guarantee better performance.**

Performance depends on the data, optimization, architecture, regularization, and deployment constraints. LSTM and GRU should be compared empirically.

**Teacher-forced evaluation fully represents generation quality.**

Teacher-forced likelihood measures predictions under reference prefixes. Free-running generation can behave differently because previous predictions are model-generated.

**Attention weights are complete explanations.**

Attention weights indicate how attention is distributed, but they do not automatically establish causal importance or provide a complete explanation of the prediction.

## Architecture Selection Guide

Choose a **simple RNN** when dependencies are short, computational simplicity is important, or a baseline is needed.

Choose an **LSTM** when gated memory and a separate cell state are useful candidates for modeling the task.

Choose a **GRU** when gated recurrence is appropriate and a compact recurrent architecture is desirable.

Choose a **bidirectional RNN** when the full input sequence is available before the prediction must be made.

Choose an **encoder–decoder with attention** when an output sequence must access specific positions in a variable-length input sequence.

Compare against **Transformer-based architectures** for large-scale language tasks and many general sequence-learning problems. Recurrence is not automatically the strongest choice merely because the data is sequential.

The final decision should be empirical. Keep the evaluation protocol fixed, compare predictive quality and parameter count, and measure latency and memory under realistic sequence lengths.

## Compact Equation Reference

### Simple RNN

$$
h_t = \phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
$$

### Bidirectional RNN

$$
h_t =
[\overrightarrow{h_t};\overleftarrow{h_t}]
$$

### LSTM

$$
i_t=\sigma(W_i x_t+U_i h_{t-1}+b_i)
$$

$$
f_t=\sigma(W_f x_t+U_f h_{t-1}+b_f)
$$

$$
o_t=\sigma(W_o x_t+U_o h_{t-1}+b_o)
$$

$$
\tilde{c}_t=\tanh(W_c x_t+U_c h_{t-1}+b_c)
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t
$$

$$
h_t=o_t\odot\tanh(c_t)
$$

### GRU

$$
z_t=\sigma(W_z x_t+U_z h_{t-1}+b_z)
$$

$$
r_t=\sigma(W_r x_t+U_r h_{t-1}+b_r)
$$

$$
\tilde{h}_t=
\tanh(W_h x_t+U_h(r_t\odot h_{t-1})+b_h)
$$

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde{h}_t
$$

### Attention

$$
\alpha_{t,i}
=
\frac{\exp(e_{t,i})}
{\sum_j\exp(e_{t,j})}
$$

$$
c_t=\sum_i\alpha_{t,i}h_i
$$

### Scaled Dot-Product Attention

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V
$$

### Autoregressive Factorization

$$
p(y_{1:T}\mid X)
=
\prod_{t=1}^{T}p(y_t\mid y_{<t},X)
$$

### Gradient Clipping

$$
g \leftarrow
g\cdot
\min\left(1,\frac{\tau}{\lVert g\rVert_2}\right)
$$

### Parameter Count Estimates

For a standard single-layer RNN with one bias vector:

$$
N_{\text{RNN}}
=
d_xd_h+d_h^2+d_h
$$

For a standard single-layer LSTM with one bias vector per gate transformation:

$$
N_{\text{LSTM}}
=
4(d_xd_h+d_h^2+d_h)
$$

For a standard single-layer GRU with one bias vector per transformation:

$$
N_{\text{GRU}}
=
3(d_xd_h+d_h^2+d_h)
$$

These are reference estimates. Framework-specific bias layouts, projections, peephole connections, and other implementation details can change the actual parameter count.

## Final Engineering Perspective

Sequential modeling is fundamentally about learning useful representations of ordered information while managing temporal dependencies, gradient propagation, and inference constraints.

The central distinction is how each architecture preserves and accesses information:

- A simple RNN repeatedly transforms one hidden state.
- An LSTM regulates a separate cell state through input, forget, and output gates.
- A GRU uses update and reset gates to control a single recurrent state.
- A bidirectional RNN combines representations from past and future context.
- An encoder–decoder maps an input sequence into an output sequence.
- Attention provides dynamic access to relevant representations.
- Transformers replace recurrent state transitions with attention-based contextual computation.

For engineering work, understand the mathematical mechanism, verify the tensor and masking semantics, prevent temporal leakage, and select the architecture through controlled experiments under the actual deployment constraints.
