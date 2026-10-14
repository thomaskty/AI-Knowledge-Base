# AI-Knowledge-Base
1. Unicode, Text Normalization, Preprocessing, and Tokenization Fundamentals
2. Subword Tokenization Algorithms: BPE, WordPiece, Unigram, and SentencePiece
3. Byte-Level Tokenization, Vocabulary Construction, Special Tokens, and Tokenizer Training Objectives
4. Tokenization Efficiency, Morphology, Rare Words, Multilingual Bias, and Sequence-Length Trade-offs
5. Positional Representations: Sinusoidal Encoding, Learned Embeddings, Relative Positions, RoPE, and ALiBi
6. Context-Length Extension, Position Interpolation, and Sequence-Level Computational Cost
7. Transformer Architecture: Evolution from Recurrent Networks, Encoder-Decoder Designs, and Model Variants
8. Scaled Dot-Product Attention: Mathematical Derivation, Query-Key-Value Projections, and Softmax Normalization
9. Multi-Head Attention, Self-Attention, Cross-Attention, Causal Masking, and Padding Masks
10. Transformer Blocks: Feed-Forward Networks, Residual Connections, Layer Normalization, and Pre-Norm versus Post-Norm
11. Attention Complexity and Optimization: Quadratic Costs, FlashAttention, Sparse Attention, and Linear Attention
12. Efficient Attention Architectures: MQA, GQA, MLA, and KV-Cache Optimization
13. Attention Behavior: Entropy, Concentration, Attention Sinks, Positional Effects, and Interpretability Limitations
14. Transformer Initialization, Gradient Propagation, Parameter Counting, FLOP Estimation, and Computational Bottlenecks
15. Long-Context Transformers: Attention Scaling, Memory Limitations, and Reasoning Constraints
16. Foundation Model Architectures: GPT, BERT, RoBERTa, T5, and Encoder-Decoder Trade-offs
17. Language Model Pretraining: Autoregressive Prediction, Masked Language Modeling, and Text-to-Text Objectives
18. Foundation Model Scaling: Training Data, Parameter Counts, Compute-Optimal Scaling, and Chinchilla Laws
19. Pretraining Data Engineering: Corpus Construction, Deduplication, Quality Filtering, Data Mixtures, and Curriculum Learning
20. Emergent Capabilities, In-Context Learning, Zero-Shot Generalization, Few-Shot Learning, and Memorization
21. Domain Adaptation, Continual Pretraining, Parametric Memory, Knowledge Retention, and Catastrophic Forgetting
22. LLM Optimization: Cross-Entropy Loss, Batch Design, Gradient Accumulation, AdamW, Learning-Rate Schedules, Gradient Clipping, and Mixed-Precision Training
23. Distributed Model Training: Data, Tensor, Pipeline, Sequence, Context, and Expert Parallelism
24. Instruction Tuning: Supervised Fine-Tuning, Instruction Dataset Design, and Training Quality
25. Parameter-Efficient Fine-Tuning: LoRA, QLoRA, Prefix Tuning, Prompt Tuning, and Full Fine-Tuning Trade-offs
26. Preference Learning and Alignment: Reward Models, RLHF, PPO, DPO, IPO, KTO, and ORPO
27. Alignment Data Generation: Rejection Sampling, Best-of-N, Synthetic Instructions, Self-Improvement, and Distillation
28. Alignment Reliability: Constitutional AI, Critique-Revision Training, Reward Hacking, Alignment Tax, and Capability Trade-offs
29. Mixture-of-Experts Training: Routing, Expert Specialization, Load Balancing, Sparse Computation, and Expert Collapse
30. Model Merging, Continual Fine-Tuning, Catastrophic Forgetting, Data Contamination, Memorization, and Privacy Risks
31. Autoregressive Inference: Next-Token Distributions, Greedy Decoding, Beam Search, and Stopping Criteria
32. Sampling Strategies: Temperature, Top-k, Top-p, Min-p, Typical Sampling, and Probability Distribution Control
33. Generation Quality Controls: Repetition Penalties, Frequency and Presence Penalties, Logit Bias, and Length Normalization
34. Advanced Decoding: Contrastive Search, Self-Consistency, Best-of-N, Reranking, and Verifier-Guided Generation
35. Reasoning-Time Compute: Test-Time Search, Multi-Token Prediction, Speculative Decoding, and Inference-Time Scaling
36. Structured Generation: Grammar-Constrained Decoding, Schema Enforcement, and Constrained Token Selection
37. LLM Serving Performance: Prefill, Decode, KV Caching, Prefix Caching, and Continuous Batching
38. Inference Performance Engineering: TTFT, Inter-Token Latency, Throughput, Memory Constraints, and Latency-Cost Trade-offs
39. LLM Quantization: INT8, INT4, Weight-Only, Activation, and KV-Cache Quantization
40. Prompt Engineering Fundamentals: Instruction Hierarchies, Conditional Inference, Zero-Shot, Few-Shot, and Example Selection
41. Reasoning Prompting: Chain-of-Thought Evaluation, Problem Decomposition, Least-to-Most, and Plan-and-Solve
42. Search-Based Reasoning: Tree-of-Thought, Graph-of-Thought, Multiple Reasoning Paths, and Self-Consistency
43. Agentic Prompting: ReAct, Planning, Tool Use, Reflection, Critique, and Iterative Self-Refinement
44. Context Engineering: Context Selection, Ordering, Position Sensitivity, Compression, Information Preservation, and Prompt Caching
45. Prompt Optimization and Evaluation: Automatic Search, Continuous Prompts, Ablation Studies, Robustness, and Failure Analysis
46. Instruction Security: Prompt Injection, Instruction Conflicts, Ambiguity, and Structured Output Reliability
47. Knowledge Graphs: Graph Representation, Entity Resolution, Relationship Modeling, Graph Traversal, and Graph-Based Retrieval
48. Reinforcement Learning Fundamentals: Markov Decision Processes, Policies, Value Functions, Rewards, and Bellman Equations
49. Reinforcement Learning Algorithms: Value-Based Methods, Policy Gradients, Actor-Critic Methods, and Proximal Policy Optimization
50. Bandits and Sequential Decision-Making: Exploration-Exploitation, Online Experimentation, and Adaptive Policies
51. Causal Inference: Structural Causal Models, Directed Acyclic Graphs, Confounding, Selection Bias, and Causal Identification
52. Interventions and Counterfactuals: The Do-Operator, Potential Outcomes, Treatment Effects, and Randomized Experiments
53. Decision Theory: Bayesian Decisions, Expected Utility, Risk Sensitivity, Cost-Sensitive Thresholds, and Value of Information
54. Causal Evaluation and Policy Assessment: Distribution Shift, LLM Explanations, Pipeline Experiments, and Autonomous Agent Decisions
55. Distributed Systems Engineering for AI: Gradient Synchronization, All-Reduce, Parameter Sharding, Optimizer Partitioning, and Hybrid Parallelism
56. Distributed Reliability and Performance: Checkpointing, Fault Tolerance, Stragglers, Load Imbalance, and Pipeline Scheduling
57. LLM Serving Infrastructure: Request Scheduling, Queueing Theory, Cache Allocation, Tail Latency, and Scaling Efficiency
58. Model Compression: Quantization-Aware Training, Post-Training Quantization, Calibration, and Quantization Granularity
59. Structural Compression: Low-Rank Approximation, SVD, Pruning, Sparsity, Weight Sharing, and Parameter Tying
60. Efficient Model Computation: Activation Checkpointing, Recomputation, Memory-Compute Trade-offs, and Energy Efficiency
61. Knowledge Distillation: Teacher-Student Architectures, Distillation Objectives, Temperature Scaling, and Quality Preservation
62. Efficiency Evaluation: Compression-Induced Degradation, Performance per Watt, and Quality-Latency-Memory Trade-offs
63. Generative AI Architecture Selection: Prompting, RAG, Fine-Tuning, Pretraining, Single versus Multi-Model Systems, and Agent versus Workflow Designs
64. Retrieval, Memory, and Computation Trade-offs: Dense, Sparse, and Hybrid Retrieval; Parametric Knowledge; Context Expansion; Caching; and Online versus Offline Processing
65. AI System Design and Optimization: Quality, Cost, Latency, Calibration, Reliability, Observability, Fault Isolation, Experimental Design, Bottleneck Analysis, Graceful Degradation, and Multi-Objective Optimization
66. Neural Network Internals: Token Embeddings, Weight Tying, Logit Projection, Activation Functions, and Normalization Variants
67. Representation Learning: Embedding Spaces, Contrastive Learning, Metric Learning, and Representation Geometry
68. Embedding Models: Bi-Encoders, Cross-Encoders, Embedding Pooling, Similarity Metrics, and Embedding Evaluation
69. Retrieval-Augmented Generation: Query Transformation, Hybrid Retrieval, Reranking, Context Assembly, Grounded Generation, and Citation Verification
70. Advanced Retrieval Systems: Multi-Query Retrieval, Parent-Child Retrieval, Hypothetical Document Embeddings, Graph RAG, and Iterative Retrieval
71. RAG Optimization: Chunking Strategies, Retrieval Recall, Context Precision, Faithfulness, Attribution, and End-to-End Evaluation
72. Multimodal Foundation Models: Vision-Language Architectures, Image Encoders, Visual Tokenization, Cross-Modal Alignment, and Image Grounding
73. Generative Modeling Beyond Autoregressive LLMs: Variational Autoencoders, GANs, Normalizing Flows, Diffusion Models, and Flow Matching
74. Diffusion and Flow-Based Generation: Denoising Objectives, Noise Schedules, Score Matching, Guidance, and Sampling Efficiency
75. LLM Evaluation Science: Perplexity, Benchmark Design, Statistical Significance, Human Evaluation, Data Contamination, and LLM-as-a-Judge Reliability
76. RAG and Agent Evaluation: Retrieval Metrics, Answer Correctness, Faithfulness, Tool-Use Accuracy, Task Completion, and Trajectory Evaluation
77. AI Safety and Alignment: Hallucination Mitigation, Calibration, Uncertainty Estimation, Abstention, Refusal Behavior, and Adversarial Robustness
78. LLM Security and Privacy: Data Poisoning, Model Extraction, Membership Inference, Jailbreaks, Indirect Prompt Injection, and Sensitive Data Leakage
79. Agent Architecture: Tool Calling, Planning, State Management, Short-Term and Long-Term Memory, Workflow Orchestration, and Multi-Agent Coordination
80. Agent Reliability: Tool Failure Recovery, Idempotency, Checkpointing, Human Approval, Sandboxing, and Bounded Autonomous Execution
81. LLM Application Engineering: Structured Extraction, Function Calling, API Integration, Schema Validation, Output Parsing, and Error Recovery
82. AI Application Observability: Tracing, Token and Cost Accounting, Quality Monitoring, Feedback Collection, Regression Testing, and Production Drift Detection
83. AI Deployment and MLOps: Model Serving, Versioning, CI/CD, Evaluation Gates, Canary Releases, Rollbacks, and Infrastructure Reproducibility
84. AI Governance and Responsible Deployment: Data Lineage, Licensing, Consent, Model Cards, Auditability, Regulatory Compliance, and Risk Management
85. Synthetic Data Engineering: Data Generation, Quality Filtering, Diversity Assessment, Privacy Risks, Distribution Matching, and Synthetic-to-Real Evaluation
86. Advanced Post-Training: GRPO, Reinforcement Learning with Verifiable Rewards (RLVR), Process Supervision, and Outcome-Based Supervision
87. LLM Architecture Innovations: State-Space Models, Selective State Spaces, Hybrid Attention Architectures, Recurrent Language Models, and Transformer Alternatives
88. Memory-Augmented Language Models: External Memory, Persistent Memory, Retrieval-Based Memory, Memory Compression, and Memory Update Policies
89. AI System Benchmarking: Workload Characterization, Capacity Planning, Load Testing, Cost Modeling, Reliability Testing, and Reproducible Performance Comparisons
90. Generative AI Research Methodology: Paper Reproduction, Hypothesis Testing, Ablation Studies, Baseline Selection, Error Analysis, and Research-to-Production Translation
