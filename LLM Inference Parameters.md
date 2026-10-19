# Master Guide to LLM Inference Parameters, Decoding Strategies & Mechanics

Inference parameters and execution settings dictate how a Large Language Model (LLM) processes input context, manages memory, applies decoding algorithms, and samples tokens. Fine-tuning these parameters allows developers to control output structure, creativity, determinism, latency, and resource utilization.

---

## 1. Core Mechanics: Token Probability Generation

Before adjusting parameters, it helps to understand how token selection works under the hood:

1. **Logit Generation:** The final layer of an LLM produces raw, unnormalized confidence scores called **logits** for every token in its vocabulary (often 32,000 to 128,000+ tokens).
2. **Logit Manipulation:** Parameters like `Frequency Penalty`, `Presence Penalty`, `Repetition Penalty`, and `Logit Bias` modify these raw scores.
3. **Softmax Transformation:** The modified logits pass through the Softmax function, converting them into a normalized probability distribution where all values sum to $1.0$ ($100\%$).
4. **Sampling Filters:** Parameters like `Temperature`, `Top-K`, `Top-P`, and `Min-P` filter or scale this probability distribution.
5. **Decoding & Execution:** The engine selects the next token according to the specified strategy (e.g., Greedy, Stochastic, or Beam Search) and appends it to the sequence for the next generation step.

---

## 2. Token Sampling & Selection Parameters

These parameters alter logit values or trim the probability distribution prior to selecting the next token.

### 2.1. Temperature
* **What it does:** Controls randomness and entropy by scaling raw logits before Softmax normalization.
* **Mathematical Formula:**

  $$
  \text{Softmax}(z_i) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}
  $$

  *(where $z_i$ is the logit of token $i$, and $T$ is the Temperature)*

* **How it works:**
  * **$T \to 0$ (e.g., $0.0$):** The math causes the highest-probability token to dominate near 100% of the distribution, resulting in deterministic (greedy) output.
  * **$0.1 \le T \le 0.7$:** High-probability tokens remain favored, but secondary choices retain a small probability of selection.
  * **$T > 1.0$:** Flattens the distribution, giving lower-probability (rare) tokens a much higher chance of being selected, which often leads to creative or erratic text.

### 2.2. Top-P (Nucleus Sampling)
* **What it does:** Dynamically cuts off the candidate token pool based on a cumulative probability threshold.
* **How it works:**
  1. The engine orders all vocabulary tokens by probability from highest to lowest.
  2. It computes the running cumulative probability sum.
  3. The pool is truncated as soon as the cumulative sum reaches threshold $P$ (e.g., $0.90$ or $90\%$).
  4. Tokens outside this "nucleus" are zeroed out, and remaining tokens are re-normalized.
* **Dynamic Nature:** When the model is confident in a choice (e.g., 95% probability on a single token), the pool shrinks to 1 token. When uncertain, the pool expands to include multiple candidates.

### 2.3. Top-K Sampling
* **What it does:** Restricts the candidate token pool to strictly the top $K$ most likely tokens.
* **How it works:**
  1. Orders all tokens by probability.
  2. Keeps the top $K$ candidates (e.g., $K = 40$) and discards all others.
  3. Re-normalizes the probabilities of the retained tokens to sum to $1.0$.

### 2.4. Min-P (Minimum Probability Truncation)
* **What it does:** Truncates candidate tokens based on a dynamic cutoff relative to the **top-ranked token's probability**.
* **Mathematical Rule:**

  $$
  \text{Threshold} = \text{Probability of Top Token} \times \text{Min-P}
  $$

* **How it works:**
  * If `Min-P = 0.05` ($5\%$) and the top token has a probability of $0.80$, the threshold is $0.04$ ($4\%$). Tokens below $4\%$ probability are removed.
  * If the top token has only a $0.20$ probability, the threshold adjusts down to $0.01$ ($1\%$). This provides better output coherence in tail-end distributions than static `Top-K` or `Top-P`.

### 2.5. Frequency Penalty
* **What it does:** Penalizes tokens based on **how many times they have already appeared** in the generated output sequence.
* **Mechanism:** Subtractions are applied to logit scores proportional to token occurrence counts. This actively stops the model from repeating identical words or phrases in loops.

### 2.6. Presence Penalty
* **What it does:** Applies a one-time flat penalty to any token that has **appeared at least once** in the output sequence.
* **Mechanism:** Unlike Frequency Penalty, the deduction does not scale with repetition count. It encourages the model to introduce new topics and vocabulary.

### 2.7. Repetition Penalty
* **What it does:** A multiplicative penalty applied to previously generated tokens (commonly used in backends like vLLM, TGI, and Ollama).
* **Mechanism:** Logits of past tokens are divided by the penalty factor (if logit > 0) or multiplied (if logit < 0). A setting of $1.0$ means no penalty, while values above $1.0$ (e.g., $1.15$) reduce repetitive text generation.

### 2.8. Logit Bias
* **What it does:** Manually modifies specific token scores prior to Softmax processing.
* **Mechanism:** Accepts a map of Token IDs to integer bias values (typically `-100` to `+100`).
  * A bias of `-100` effectively bans a specific token from being generated.
  * A bias of `+100` forces the engine to select that token.

---

## 3. Decoding Strategies & Search Algorithms

Decoding strategies determine how the model searches and selects tokens across single or multiple candidate paths.

### 3.1. Greedy Decoding
* **How it works:** Selects the single token with the absolute highest probability ($T = 0$) at each step.
* **Trade-offs:** Fast and deterministic, but can fall into repetitive output cycles or miss better overall sentences because it never evaluates secondary token paths.

### 3.2. Random / Stochastic Sampling
* **How it works:** Samples probabilistic-wise from the filtered distribution.
* **Trade-offs:** Generates human-like, varied text, but requires fixed seeds for reproducible runs.

### 3.3. Beam Search
* **What it is:** A search algorithm that tracks a set of $B$ (beam width) candidate sequences in parallel rather than selecting one token at a time.
* **How it works:**
  1. Keeps the top $B$ partial candidate sequences at step $t$.
  2. Calculates cumulative sequence log-probabilities:

     $$
     \text{Score}(Y) = \sum_{t=1}^{N} \log P(y_t \mid y_1, \dots, y_{t-1})
     $$

  3. Prunes low-scoring paths, preserving only the top $B$ overall trajectories.
* **Length Penalty ($\alpha$):** Raw cumulative log-probabilities favor shorter sequences. A length penalty normalizes path scores by output size:

  $$
  \text{Normalized Score} = \frac{\text{Score}(Y)}{\left(\frac{5 + \vert{}Y\vert{}}{6}\right)^\alpha}
  $$

* **Use cases:** Translation and summarization where global sequence structure matters more than real-time generation speed.

### 3.4. Speculative Decoding
* **What it is:** An optimization technique where a smaller "draft model" rapidly generates $K$ tokens, which are then validated by the larger "target model" in a single parallel forward pass.
* **Impact:** Reduces output latency without sacrificing model accuracy or changing mathematical outputs.

### 3.5. Mirostat Sampling
* **What it is:** An adaptive algorithm designed to maintain a target **entropy level** throughout generation.
* **Mechanism:** It measures generation surprise step-by-step and dynamically adjusts truncation bounds, preventing long outputs from degrading into repetitive or incoherent text.

### 3.6. Contrastive Decoding
* **What it is:** An algorithm that weighs probabilities generated by a large target model against a smaller amateur model.
* **Mechanism:** Subtracts amateur logit probabilities from target logit probabilities ($P_{\text{target}} - P_{\text{amateur}}$), suppressing generic phrases and hallucinated responses.

---

## 4. Runtime Parameters, Formats & Memory Mechanics

These settings dictate context handling, structural constraints, stream delivery, and hardware execution.

### 4.1. Context Window
* **What it is:** The absolute maximum token capacity (prompt context + generated output) supported by the model's positional embeddings (e.g., 32k, 128k, or 1M+ tokens).
* **Impact:** Prompts exceeding this boundary will fail, truncate, or require context management techniques like sliding-window attention.

### 4.2. Max Tokens / Max Completion Tokens
* **What it is:** A hard cap set by developers limiting the maximum number of tokens an LLM can emit during a single generation request.
* **Impact:** Halts generation when reached, even if the model has not emitted an end-of-sequence token.

### 4.3. Stop Sequences
* **What it is:** Custom string literals or token IDs that instantly halt the inference engine upon generation (e.g., `["\nUser:", "```", "<|im_end|>"]`).

### 4.4. Response Format (Structured Outputs)
* **What it is:** Enforces structural formatting rules on outputs (such as requiring valid `JSON` or adherence to a Pydantic schema).
* **Mechanism (Grammar-Constrained Decoding):** Inference engines build a state machine (Context-Free Grammar) from the provided schema. At each step, tokens violating structural syntax receive a probability of $0$, ensuring $100\%$ schema compliance without requiring retries.

### 4.5. Streaming (`stream: true / false`)
* **What it is:** A execution setting that determines how output tokens are delivered to the client application.
* **Non-Streaming (`stream: false`):** The server waits for generation to finish completely, returning the full text block in a single HTTP response.
* **Streaming (`stream: true`):** Emits generated tokens immediately over Server-Sent Events (SSE) as chunked deltas, lowering **Time to First Token (TTFT)** for user interfaces.

### 4.6. Seed
* **What it is:** An integer value used to initialize the pseudo-random number generator (PRNG).
* **Impact:** Setting a static seed alongside identical parameters ($T, P, K$) ensures deterministic, repeatable token generation across runs.

### 4.7. Key-Value (KV) Cache
* **What it is:** A VRAM memory optimization that stores intermediate Key and Value matrix activations from self-attention layers for prompt and generated tokens.
* **Why it matters:**
  * Without KV Cache, calculating token $N$ requires recomputing attention across all prior $N-1$ tokens, resulting in $O(N^2)$ computational complexity.
  * With KV Cache, past activations are retrieved directly from memory, reducing step generation to $O(N)$ complexity.
* **Related Runtime Concepts:**
  * **PagedAttention (vLLM):** Allocates KV cache memory in non-contiguous pages (like virtual memory in operating systems) to reduce VRAM fragmentation.
  * **KV Cache Quantization (FP8 / INT8 / INT4):** Compresses cached KV matrices from FP16 to lower bit-rates to free up VRAM for larger context windows.