# Transformer Research Book
## Complete Table of Contents with Chapter Topics and Subtopics

> A comprehensive technical book covering the Transformer architecture from first principles through
> state-of-the-art research as of 2025. Spans language, vision, audio, multimodal, science, efficiency,
> and post-transformer architectures. Every part includes dedicated training and evaluation chapters.

---

## Book Overview

| Part | Title | Chapters | Focus |
|------|-------|----------|-------|
| I | Foundations | 1–6 | Architecture fundamentals, math, tokenization, pre-training basics |
| II | Language Models | 7–17 | BERT, GPT, T5, scaling, alignment, retrieval, RAG |
| III | Vision Transformers | 18–27 | ViT, Swin, detection, video understanding, video generation |
| IV | Audio, Speech and Music | 28–34 | ASR, TTS, audio generation, unified audio-visual models |
| V | Multimodal Transformers | 35–42 | CLIP, VLMs, omni models, code, 3D, robotics |
| VI | Architecture Innovations and Efficiency | 43–55 | Flash Attention, KV cache, MoE, PEFT, quantization, distributed training |
| VII | Transformers in Science | 56–64 | Proteins, genomics, drug discovery, climate, medical imaging |
| VIII | Beyond Transformers and Frontiers | 65–74 | SSMs, Mamba, RWKV, hybrids, reasoning, open problems |

**Total: 74 chapters | 17 training chapters | 17 evaluation chapters | ~950+ estimated pages**

---

## Part I: Foundations

*Goal: Build complete mathematical and architectural understanding of the vanilla Transformer from scratch,
with no assumed prior knowledge of the architecture.*

---

### Chapter 1: Why Transformers? A Historical Arc

**The problem with sequential models**
- Recurrent Neural Networks (RNNs): hidden state, BPTT, vanishing and exploding gradients
- Long Short-Term Memory (LSTM): cell state, forget/input/output gates, gradient flow analysis
- Gated Recurrent Units (GRU): simplified gating, comparison with LSTM
- The fundamental bottleneck: sequential computation prevents parallelism during training
- CNNs for sequences: temporal convolutions, WaveNet, dilated convolutions, fixed receptive field limitations

**Attention as a primitive**
- Bahdanau attention (2014): the first soft attention mechanism for seq2seq
- Luong attention: dot-product and general forms, comparison with Bahdanau
- The attention equation before transformers: context vectors, alignment scores
- Why attention alone is not sufficient: positional blindness, quadratic memory

**Transformers arrive**
- "Attention Is All You Need" (Vaswani et al., 2017): the paradigm shift
- Why the transformer won: parallelism, global receptive field, scalability
- The 2018-2019 explosion: BERT, GPT, XLNet, the pre-training era begins
- Timeline of major transformer milestones: 2017 to 2025

---

### Chapter 2: Mathematical Prerequisites

**Linear algebra review for attention**
- Matrix multiplication as the core operation
- Vectors as token representations, matrices as transformations
- The role of dimensionality: embedding dimension d_model, head dimension d_k
- Outer products, dot products, and their geometric interpretation

**The softmax function**
- Definition and properties: probability normalization, differentiability
- Numerical stability: subtracting the maximum before exponentiation
- Temperature scaling: sharpening vs flattening distributions
- Softmax saturation and gradient vanishing in extreme logit regimes

**Layer normalization**
- Batch norm vs layer norm: why batch norm fails for variable-length sequences
- Layer norm formula: mean and variance over the feature dimension
- RMS Norm (Root Mean Square Normalization): simplified variant used in LLaMA, Gemma
- Pre-norm vs post-norm placement and training stability implications

**Scaled dot-product attention derivation**
- Why scale by sqrt(d_k): preventing dot product growth in high dimensions
- The full attention formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
- Complexity analysis: O(n^2 d) time, O(n^2) memory for the attention matrix
- Gradient flow through the softmax: why certain initializations matter

**Feed-forward network mathematics**
- Two-layer MLP structure: W_1, activation, W_2
- The expansion ratio: why 4x is standard, what it controls
- Activation functions: ReLU, GELU, SwiGLU, GLU variants
- Universal approximation: FFN as a key-value memory store (Geva et al., 2021)

---

### Chapter 3: Attention Is All You Need — The Original Paper

**Paper overview and context**
- The machine translation setting: WMT English-German, English-French
- Previous state-of-the-art: ConvS2S, Wu et al. (2016) GNMT
- Core thesis: self-attention can replace recurrence and convolution entirely

**Encoder architecture**
- Input embedding and positional encoding
- Stack of N=6 identical layers
- Each layer: multi-head self-attention + position-wise FFN
- Residual connections and layer normalization placement
- Encoder output: contextual representations for every input token

**Decoder architecture**
- Autoregressive generation: masked self-attention for causal modeling
- Cross-attention: decoder attends to encoder output
- The encoder-decoder attention mechanism: keys and values from encoder, queries from decoder
- Output linear layer and softmax over vocabulary
- Why the decoder is causal: preventing future token leakage

**Multi-head attention**
- Motivation: allowing the model to attend to different representation subspaces simultaneously
- Splitting into h heads: projecting Q, K, V to d_k = d_model / h dimensions
- Concatenation and final linear projection
- Practical implementation: reshaping and parallel computation
- What different heads learn: positional, syntactic, semantic specialization

**Positional encodings (original)**
- Why position information is needed: the permutation-equivariance problem
- Sinusoidal encoding: sin/cos at different frequencies
- Why sinusoidal: relative position representable as a linear function, extrapolation to longer sequences
- Learned positional embeddings: alternative approach, comparison

**Training details in the original paper**
- Adam optimizer with warm-up and decay schedule
- Label smoothing: epsilon = 0.1
- Dropout: residual, attention, embedding
- Multi-GPU training: 8 NVIDIA P100s, base model in 12 hours
- BLEU scores and comparison with prior work

---

### Chapter 4: Inside the Transformer Block

**The complete block anatomy**
- Pre-norm transformer block (modern standard): LN before attention
- Post-norm transformer block (original paper): LN after residual
- Why pre-norm is now dominant: training stability, gradient flow at initialization
- The DeepNorm variant: scaled residuals for extremely deep transformers

**Self-attention in depth**
- Query, Key, Value projections: W_Q, W_K, W_V parameter matrices
- The QK similarity as a content-addressable memory
- Attention patterns: local vs global, diagonal, vertical stripes
- Causal masking: lower-triangular mask implementation, -inf for masked positions
- Multi-head attention: implementation with einsum, batched matrix multiplication

**Feed-forward sublayer in depth**
- The two-layer MLP: W_1 (expansion), W_2 (projection)
- SwiGLU activation: Swish(xW_1) * (xW_2), used in LLaMA, PaLM, Gemma
- Why SwiGLU works: gating mechanism, smoother gradients
- The FFN as a key-value memory: neurons as memory slots (Geva et al.)
- FFN parameter count vs attention parameter count: FFN dominates at typical ratios

**Residual connections**
- Original motivation: avoiding vanishing gradients in deep networks (He et al., 2016)
- Residual stream interpretation: information superhighway through the network
- Residual stream as superposition of features (mechanistic interpretability framing)
- Effect on learning dynamics: identity initialization, gradient propagation

**Normalization variants**
- Layer Norm: formula, placement options, parameter count
- RMS Norm: removes mean centering, faster, equivalent empirically in practice
- Pre-norm transformers: training stability without warm-up, enables deeper models
- Deep Norm: scaled initialization + scaled residuals, 1000-layer transformers

**Activation functions survey**
- ReLU: sparse activations, dead neuron problem
- GELU: smooth approximation to ReLU, used in BERT, GPT-2
- SiLU / Swish: self-gated, smooth, slightly better than GELU
- SwiGLU: gated linear unit variant, dominant in modern LLMs
- GeGLU: GELU-gated linear unit

---

### Chapter 5: Tokenization and Embeddings

**The tokenization problem**
- Why character-level is too fine-grained: long sequences, weak statistical patterns
- Why word-level is too coarse: OOV tokens, morphology, code and math
- The Goldilocks zone: subword tokenization

**Byte Pair Encoding (BPE)**
- Algorithm: start from characters, iteratively merge the most frequent pair
- Vocabulary construction: merge rules as a learned artifact
- GPT-2 BPE: 50,257 tokens, byte-level to handle all Unicode
- Tiktoken (GPT-3/4): cl100k_base and o200k_base vocabularies
- Fertility: average tokens per word across languages, multilingual mismatch

**WordPiece and SentencePiece**
- WordPiece: likelihood-maximizing variant, used in BERT
- The ## prefix convention in BERT tokenization
- SentencePiece: language-agnostic, treats whitespace as a token
- Unigram language model tokenizer: alternative to BPE, probabilistic
- T5, LLaMA, Gemma all use SentencePiece variants

**Byte-level tokenization**
- Byte-level BPE (GPT-2): no unknown tokens, full Unicode coverage
- BBPE vocabulary efficiency vs compression rate trade-off
- Byte fallback in SentencePiece: handling rare scripts

**Token embeddings**
- The embedding table: shape [vocab_size, d_model]
- Tied input/output embeddings: weight sharing, Press and Wolf (2017)
- Embedding initialization: small normal distribution, scaling
- The embedding dimension choice: d_model in {512, 768, 1024, 2048, 4096, 8192}
- Token frequency and embedding quality: rare tokens and their representations

**Special tokens**
- [CLS], [SEP], [MASK] in BERT-style encoders
- \<BOS\>, \<EOS\>, \<PAD\> in decoder models
- System prompt tokens, tool call delimiters, image placeholders in modern models

---

### Chapter 6: Pre-training Fundamentals

**Pre-training objectives**
- Masked Language Modeling (MLM): BERT-style 15% masking, mask/random/unchanged split
- Causal Language Modeling (CLM): GPT-style next-token prediction
- Span corruption: T5-style, mask contiguous spans of tokens
- Prefix LM: part causal, part bidirectional (UniLM, T5 variants)
- Denoising objectives: BART-style text infilling, sentence permutation, deletion

**Data collection and curation**
- Web crawl data: Common Crawl, C4, FineWeb, RedPajama
- Quality filtering: language detection, perplexity filtering, deduplication
- Deduplication methods: MinHash LSH, exact substring matching, Bloom filters
- Data mixture: books, code, academic, web -- ratio decisions and their effects
- Data contamination: preventing benchmark leakage into pre-training
- Tokenization and packing: efficient batching, no wasted padding

**Optimization for pre-training**
- AdamW: Adam with decoupled weight decay (Loshchilov and Hutter, 2019)
- Learning rate schedules: linear warm-up, cosine decay, constant then decay
- Gradient clipping: global norm clipping at 1.0
- Mixed precision training: FP16 vs BF16, loss scaling, master weights in FP32
- Weight decay, dropout rates during pre-training

**Training stability**
- Loss spikes: causes (data quality, learning rate) and recovery strategies
- Gradient accumulation: simulating large batch sizes across many steps
- Batch size scaling: linear scaling rule, warm-up for large batches
- Checkpoint saving strategy: frequency, rolling vs permanent checkpoints
- Resuming from failure: deterministic data ordering, RNG state saving

---

## Part II: Language Models

*Goal: Cover the full spectrum of language model architectures, pre-training regimes, alignment techniques,
retrieval systems, and distributed training, with rigorous benchmarking methodology.*

---

### Chapter 7: Encoder-Only Models — BERT and Beyond

**BERT (Devlin et al., 2018)**
- Architecture: 12-layer base, 24-layer large, bidirectional self-attention
- Pre-training tasks: MLM + Next Sentence Prediction (NSP)
- Fine-tuning paradigm: add a task-specific head, fine-tune the full model
- Contextualized representations: how BERT differs from static word2vec embeddings
- BERT's impact: near-simultaneous improvement across 11 NLP tasks

**BERT variants and improvements**
- RoBERTa: removes NSP, dynamic masking, longer training, more data -- significant gains from better recipes
- ALBERT: parameter sharing across layers, factorized embedding, inter-sentence coherence objective
- DeBERTa: disentangled attention (separate content and position), enhanced mask decoder
- DistilBERT: knowledge distillation, 40% smaller, 60% faster, 97% of BERT performance
- SpanBERT: span masking, span boundary objective, strong on coreference and QA

**Multilingual and domain-specific encoders**
- mBERT: multilingual BERT, 104 languages, cross-lingual transfer
- XLM-R: trained on 2.5TB of CommonCrawl in 100 languages, outperforms mBERT
- BioBERT, SciBERT, LegalBERT, FinBERT: domain-specific pre-training effects
- ClinicalBERT: EHR text, clinical NLP downstream tasks

**Use cases for encoder-only models**
- Text classification: sentiment, intent, toxicity
- Named entity recognition: token-level classification
- Question answering: span extraction
- Semantic textual similarity: [CLS] embedding or pooled output
- When to choose encoder-only over generative: efficiency, discriminative tasks

---

### Chapter 8: Decoder-Only Models — The GPT Lineage

**GPT-1 (Radford et al., 2018)**
- 12-layer transformer decoder, causal masking
- Unsupervised pre-training + supervised fine-tuning
- Task-specific input transformations: delimiters for classification, NLI, QA

**GPT-2 (Radford et al., 2019)**
- Scaling to 1.5B parameters, byte-level BPE
- Zero-shot task performance: the first glimpse of emergent capability
- WebText dataset, 40GB high-quality web text
- Language models are (unsupervised) multitask learners

**GPT-3 (Brown et al., 2020) and in-context learning**
- 175B parameters, the first frontier-scale model
- Few-shot learning: task demonstration in the prompt, no gradient update
- Prompt engineering as a skill: task framing, exemplar selection
- Limitations: context length, compute cost, hallucination

**The open-source lineage**
- OPT (Meta): reproducible baseline at GPT-3 scale, released with training logs
- BLOOM: multilingual, 176B, BigScience collaborative effort
- Falcon: RefinedWeb dataset, GQA, efficient architecture
- LLaMA (Touvron et al., 2023): 7B-65B, trained on public data only -- shifts open-source landscape
- LLaMA 2: improved data mix, RLHF-trained chat models, extended context
- LLaMA 3 / 3.1 / 3.3: 8B to 405B, GQA, 128K context, strong multilingual
- Mistral 7B: sliding window attention, GQA, strong performance at small scale
- Mixtral 8x7B and 8x22B: sparse MoE with Mistral blocks
- DeepSeek V2/V3: multi-head latent attention (MLA), MoE, Chinese-developed frontier open model

**GPT-4 and the closed frontier**
- Multimodal input (vision): GPT-4V
- RLHF and system prompts at scale
- Long context (128K): what changes when context is large
- GPT-4o: native multimodal, unified architecture for text + vision + audio

**Architectural refinements in modern decoders**
- GQA (Grouped Query Attention): fewer key-value heads, faster inference, used in LLaMA 2/3
- MQA (Multi-Query Attention): single key-value head, aggressive memory reduction
- Sliding window attention in Mistral: local windows + global every N layers
- Multi-head Latent Attention (MLA) in DeepSeek: low-rank KV compression

---

### Chapter 9: Encoder-Decoder Models

**T5 (Raffel et al., 2020) — Text-to-Text Transfer Transformer**
- The text-to-text unification: all tasks formatted as string-to-string
- C4 dataset: Colossal Clean Crawled Corpus, 750GB
- Span corruption objective: mask spans of average length 3, replace with sentinel tokens
- Architecture choices: relative positional bias, no absolute PE
- Model sizes: T5-Small through T5-11B
- Flan-T5: instruction fine-tuned variant, strong zero-shot performance

**BART (Lewis et al., 2020)**
- Denoising autoencoder: corrupt text, train encoder-decoder to reconstruct
- Corruption strategies: token masking, deletion, text infilling, sentence permutation, rotation
- Strong on summarization, question generation, data-to-text
- mBART: multilingual BART for cross-lingual generation

**Other encoder-decoder models**
- UL2: Mixture of Denoisers, unifying MLM and span corruption and causal objectives
- mT5: multilingual T5, 101 languages, mC4 dataset
- PEGASUS: gap sentence generation for abstractive summarization
- ProphetNet: future n-gram prediction for better long-form generation

**When to use encoder-decoder**
- Conditional generation: the task has a clearly defined input and target
- Summarization, translation, question generation, data-to-text
- Encoder-decoder vs decoder-only for generation: quality vs versatility trade-off

---

### Chapter 10: Scaling Laws and Emergent Abilities

**The Kaplan et al. (2020) scaling laws**
- Power law relationship between loss and model size (N), data (D), compute (C)
- Key finding: optimal allocation is compute-proportional, N and D should scale together
- IsoFLOP curves: for a fixed compute budget, what is the optimal model size?
- The training-compute frontier: not worth training smaller models longer

**Chinchilla scaling laws (Hoffmann et al., 2022)**
- Chinchilla finding: Kaplan models were undertrained -- equal scaling of N and D
- Chinchilla-optimal compute allocation: ~20 tokens per parameter
- Impact on model design: GPT-3 would have been better trained at 70B with more data
- Limitations: inference cost not considered, downstream task performance vs loss

**Beyond Chinchilla: the inference-optimal regime**
- Llama's insight: train smaller models on more tokens for better inference efficiency
- The Llama 3 approach: 8B model trained on 15T tokens -- far past Chinchilla
- The compute-inference Pareto frontier: deploy smaller, better-trained models
- Data-constrained scaling: when you run out of quality data

**Emergent abilities**
- Definition: abilities not predictably extrapolated from smaller models
- Examples: chain-of-thought reasoning, few-shot arithmetic, code generation at scale
- Wei et al. (2022): classification of emergent tasks across model families
- Controversy: Schaeffer et al. (2023) argue emergence is a metric artifact
- Phase transitions in capabilities: what causes discontinuous jumps

**Scaling in non-language domains**
- Scaling laws for vision transformers: Zhai et al. (2022)
- Scaling laws for code: how model scale correlates with HumanEval pass rate
- Multimodal scaling: image-text data size vs text-only size effects
- Scientific models: do biology/chemistry LMs follow the same scaling laws?

---

### Chapter 11: Instruction Tuning, RLHF, and Alignment

**Supervised Fine-Tuning (SFT)**
- Instruction-following data collection: FLAN, Self-Instruct, Alpaca, ShareGPT
- Data quality vs quantity: 1000 high-quality examples vs 1M noisy ones
- Prompt formatting: system prompts, user/assistant turn structure
- SFT training: low learning rate, few epochs, avoiding catastrophic forgetting

**Reinforcement Learning from Human Feedback (RLHF)**
- The InstructGPT pipeline (Ouyang et al., 2022): SFT → reward model → PPO
- Reward model training: Bradley-Terry model on preference pairs
- PPO (Proximal Policy Optimization): KL penalty to prevent reward hacking
- Reward hacking and overoptimization: Goodhart's Law in practice
- RLHF at scale: OpenAI, Anthropic, DeepMind approaches compared

**Direct Preference Optimization (DPO)**
- The DPO insight: implicit reward model, no separate RL loop
- DPO loss: cross-entropy on preferred vs rejected, with reference model KL
- Why DPO is simpler and often competitive with PPO
- Variants: IPO, KTO, SimPO, ORPO -- stability vs performance trade-offs
- When PPO still wins: complex instruction following, math, coding

**Constitutional AI and RLAIF**
- Constitutional AI (Anthropic, 2022): rule-guided self-critique and revision
- RLAIF: using an LLM as the preference labeler instead of humans
- Self-play fine-tuning (SPIN): iterative self-improvement via preference learning
- Scalable oversight: weak-to-strong generalization, debate, amplification

**Alignment tax and capability trade-offs**
- Alignment tax: does RLHF reduce raw capability? Evidence and counter-evidence
- Safety vs helpfulness: Pareto frontier in alignment
- System prompts and meta-prompting as alignment tools

---

### Chapter 12: Long-Context and Memory Augmentation

**The context window scaling problem**
- Quadratic attention cost: 128K context = 16x more expensive than 32K
- KV cache memory: linear growth with sequence length
- Positional encoding extrapolation: models often fail past training length
- Long-context benchmarks: RULER, NIAH (Needle in a Haystack), ScrollS

**Extending context windows**
- RoPE-based interpolation: YaRN, LongRoPE, NTK-aware scaling
- Fine-tuning for long context: continued training on long documents
- Sliding window + global attention hybrid (Longformer, Mistral)
- LongLoRA: efficient fine-tuning for long context with shifted sparse attention
- Positional interpolation (PI): scaling down position indices, simple and effective

**Retrieval-Augmented Generation (RAG) as memory**
- RAG pipeline: chunking, embedding, retrieval, reranking, generation
- Chunk size and overlap: trade-offs for retrieval precision
- Late interaction vs dense retrieval (more in Chapter 13)
- RAG vs long context: cost, freshness, faithfulness trade-offs
- GraphRAG: knowledge graph augmented retrieval

**External memory mechanisms**
- MemGPT: OS-inspired memory management, virtual context windows
- KNN-LM: k-nearest neighbor interpolation at inference
- RETRO (Borgeaud et al., 2022): retrieval-enhanced transformer, DB integrated into training
- Memorizing Transformers: attention over cached past activations
- Titans: neural long-term memory as a learnable module (Meta, 2024)

---

### Chapter 13: Retrieval and Ranking with Transformers

**The information retrieval landscape**
- Classical retrieval: TF-IDF, BM25 -- lexical matching, inverted indexes
- The semantic gap: "car" vs "automobile", paraphrase, abbreviation
- Dense retrieval: embed query and document, cosine similarity in vector space
- The bi-encoder vs cross-encoder distinction: efficiency vs quality

**Dense Passage Retrieval (DPR)**
- DPR architecture: two BERT encoders (question, passage), in-batch negatives
- Training: contrastive loss with hard negatives from BM25
- FAISS: approximate nearest neighbor search for billion-scale retrieval
- DPR vs BM25: when dense retrieval wins (semantic queries) and when it loses (rare terms)

**ColBERT: Late Interaction**
- MaxSim: query token attends to best-matching passage token
- ColBERT v2: distillation + residual compression, 100x storage reduction
- Late interaction vs full cross-attention: the efficiency-quality curve
- PLAID: efficient ColBERT serving, progressive candidate pruning

**Sparse Learned Retrieval**
- SPLADE: sparse lexical expansion via MLM output, combines BM25 and DPR advantages
- uniCOIL: term weighting with BERT, compatible with BM25 indexes
- Hybrid retrieval: BM25 + dense vector, reciprocal rank fusion (RRF)

**Cross-Encoders for Reranking**
- Cross-encoder architecture: concatenate query and passage, single forward pass
- MonoBERT, MiniLM rerankers: small fast cross-encoders for second-stage ranking
- Listwise vs pointwise vs pairwise reranking objectives
- Cascade retrieval: retrieve → first-stage rank → cross-encoder rerank → generate

**RAG pipeline architectures**
- Naive RAG: retrieve then generate
- Advanced RAG: query rewriting, hypothetical document embeddings (HyDE), re-ranking
- Modular RAG: routing, fusion, adaptive retrieval
- Agentic RAG: iterative retrieval, multi-hop reasoning
- Embedding model choices: BGE, E5, Nomic Embed, text-embedding-3 comparison

**Multimodal and cross-modal retrieval**
- Image-text retrieval with CLIP embeddings
- Video retrieval: temporal aggregation strategies
- Code retrieval: CodeBERT, UniXcoder, function-level embeddings

---

### Chapter 14: Training Language Models at Scale

**Data parallelism**
- Naive data parallelism: replicate model across GPUs, gradient all-reduce
- DDP (Distributed Data Parallel): overlapping backward and communication
- Gradient accumulation: local accumulation before all-reduce, communication reduction
- Batch size scaling: linear rule, square root rule, warm-up requirement

**Model parallelism**
- Tensor parallelism (Megatron-LM): split attention heads and FFN columns across GPUs
- Megatron-LM specifics: column-parallel and row-parallel linear layers, AllReduce placement
- Pipeline parallelism: split layers across GPUs, micro-batch interleaving (GPipe, PipeDream)
- 1F1B schedule: reducing pipeline bubble to near-zero

**ZeRO: Zero Redundancy Optimizer**
- ZeRO Stage 1: partition optimizer states across ranks
- ZeRO Stage 2: partition gradients + optimizer states
- ZeRO Stage 3: partition parameters + gradients + optimizer states
- ZeRO-Infinity: offload to CPU/NVMe, enabling trillion-parameter models
- FSDP (Fully Sharded Data Parallel): PyTorch-native ZeRO Stage 3 equivalent

**3D parallelism**
- Combining data + tensor + pipeline parallelism
- Deciding the parallelism degrees: cluster topology, interconnect bandwidth, memory limits
- Megatron-DeepSpeed 3D parallelism: used for GPT-NeoX, BLOOM, LLaMA pre-training
- Communication volume analysis: when tensor vs pipeline vs data parallelism is the bottleneck

**Training infrastructure**
- GPU cluster topology: NVLink within node, InfiniBand/RoCE across nodes
- Compute vs memory vs bandwidth bound operations in transformers
- Activation checkpointing (gradient checkpointing): trade compute for memory
- FlashAttention role in distributed training (see Chapter 43)
- Monitoring: loss curves, gradient norms, activation statistics, MFU (model FLOP utilization)

**Chinchilla-optimal data planning**
- Token budget per model size
- When to stop training: optimal stopping vs continued training for inference efficiency
- Data epochs: whether to repeat data, effects of second epoch quality
- Curriculum learning: easy-to-hard, domain-proportional scheduling

**Training failures and recovery**
- Loss spikes: detection, causes, rollback strategy
- Hardware failures: checkpoint frequency, redundant checkpointing
- Silent data corruption (SDC) on large clusters: bitflip detection
- Straggler mitigation in pipeline parallelism

---

### Chapter 15: Evaluating Language Models

**Intrinsic metrics**
- Perplexity: definition, limitations as a proxy for downstream task performance
- Bits-per-character, bits-per-byte: cross-domain perplexity comparison
- Why low perplexity does not guarantee good generation

**Knowledge and reasoning benchmarks**
- The saturation cycle: GLUE → SuperGLUE → MMLU, each saturated within 2-4 years
- MMLU (57 subjects, zero/few-shot): now saturated at 93%+ for frontier models
- MMLU-Pro (TIGER-AI-Lab, NeurIPS 2024): 12K graduate-level questions, 10 answer choices, 16-33% harder
- GPQA-Diamond: PhD-level "Google-proof" questions in biology, physics, chemistry
- Humanity's Last Exam (HLE, Nature 2025): 2,500 questions from 1,000 experts, best model ~45%
- SimpleQA (OpenAI, 2024): factuality and calibration measurement, targeting hallucination
- BIG-Bench Hard: 23 hardest tasks from original 200+ task suite
- HellaSwag, ARC, WinoGrande: commonsense reasoning suite (saturated at frontier)

**Mathematics and formal reasoning benchmarks**
- MATH and MATH-500: competition math, now saturated at frontier (>90%)
- AIME 2025: 30 olympiad-level problems, best model at 100%
- FrontierMath (Epoch AI, 2024): 350 research-level problems, Fields Medalists involved, best ~50%
- ARC-AGI-2 (2025): abstract visual reasoning, top score 24%
- ARC-AGI-3 (2026): interactive reasoning, planning, memory -- frontier AI <1%, humans 100%

**Code and software engineering benchmarks**
- HumanEval and MBPP: historical baselines, now contaminated
- SWE-bench Verified (Princeton NLP): real GitHub issue resolution, top ~81%
- LiveCodeBench (ICLR 2025): temporally segmented competitive programming, contamination-resistant
- MLE-bench (OpenAI, ICLR 2025): 75 Kaggle ML competitions, testing ML engineering skills
- The pass@k metric: unbiased estimator for code generation evaluation

**Open-ended generation and human preference**
- MT-Bench: 80 multi-turn questions, LLM-as-judge scoring
- AlpacaEval 2.0: length-controlled win rates for instruction following
- Chatbot Arena / LMArena (rebranded Jan 2026): Elo ratings from crowdsourced pairwise comparison
- Domain-specific arenas: SciArena (Allen AI), BioMedArena (NIH/DataTecnica)
- WildBench (Allen AI, ICLR 2025): 1,024 real user tasks, 0.98 correlation with Arena
- GDPval (OpenAI, 2025): 1,320 professional tasks across 44 occupations
- IFEval: verifiable instruction following with 25 constraint types, multilingual extensions
- HELM: holistic multi-metric evaluation framework

**LLM-as-judge**
- Core methodology: single-answer grading, pairwise comparison, reference-based
- Known biases: verbosity, self-preference, position bias
- Chain-of-thought judging: improved correlation with human judgments (0.51 → 0.66 Spearman ρ)
- Multi-agent debate (MAJ-Eval): group deliberation for higher agreement with humans
- Crowd Comparative Reasoning, Mixture of Prompts (MoPs)

**Agentic evaluation**
- Why agentic eval differs: multi-step, tool use, environment interaction, trajectory quality
- GAIA / GAIA-2: multi-step reasoning with web browsing and dynamic environments
- WebArena (CMU): autonomous web task completion, ~60% success rate
- Tau-bench / Tau-2 (Sierra Research, ICLR 2025): customer service agents, database state checking
- BFCL V4 (Berkeley): function calling / tool use leaderboard across languages

**Evaluating reasoning models**
- The "thinking" model paradigm: o1/o3, DeepSeek R1, Claude extended thinking
- Why standard benchmarks undercount reasoning ability: variable compute budgets
- Compute-optimal evaluation: accuracy per dollar/token, same-cost comparisons
- Chain-of-thought monitorability (OpenAI 2025): monitoring hidden reasoning for safety
- TTT-Bench (EMNLP 2025): reasoning-specific evaluation

**Safety and alignment evaluation**
- TruthfulQA: measuring hallucination on adversarial questions
- BBQ: bias benchmark for question answering across 9 social dimensions
- Toxicity: Perspective API, ToxiGen
- HarmBench: standardized red-teaming with Attack Success Rate (ASR) classifiers
- ALERT (Babelscape, 2024): 45K instructions, 6 macro / 32 micro safety categories
- Red-teaming protocols: RedBench (2026), domain-specific (FINBench, SGToxicGuard)
- Constitutional AI self-critique scoring: self-evaluation against written principles

**Contamination and eval validity**
- Training data contamination: when benchmarks appear in pre-training data
- Contamination detection: n-gram overlap, perplexity analysis, temporal segmentation
- Case studies: HumanEval contamination, MMLU in training corpora
- Dynamic benchmarks: LiveBench (monthly updates), LiveCodeBench, EvoEval (transformed variants)
- Long-context evaluation: HELMET (ICLR 2025) -- synthetic tasks ≠ downstream performance
- Reproducibility: temperature, system prompt, decoding strategy standardization

---

### Chapter 16: Evaluating Retrieval and RAG Systems

**Information retrieval metrics**
- Precision@k, Recall@k: basic hit-based metrics
- Mean Reciprocal Rank (MRR): how high is the first relevant result?
- NDCG@k (Normalized Discounted Cumulative Gain): graded relevance, position-weighted
- MAP (Mean Average Precision): area under the precision-recall curve

**BEIR benchmark**
- 18 heterogeneous retrieval datasets: question answering, biomedical, Twitter, news, finance
- Why BEIR is hard: zero-shot transfer across very different domains
- BEIR results for dense retrieval vs BM25 vs hybrid: where dense retrieval fails
- MTEB (Massive Text Embedding Benchmark): retrieval + classification + clustering + STS

**RAG-specific evaluation**
- Faithfulness: does the generated answer stay grounded in the retrieved context?
- Answer relevance: is the answer actually responsive to the question?
- Context relevance: did retrieval return relevant documents?
- RAGAS: automated RAG evaluation framework using LLM-as-judge for all three dimensions
- ARES: reference-free automated evaluation for RAG

**End-to-end RAG evaluation**
- Closed-domain vs open-domain: known corpus vs web-scale
- Multi-hop QA benchmarks: HotpotQA, MuSiQue, 2WikiMultiHopQA
- Citation accuracy: does the model correctly attribute claims to sources?
- Hallucination tracing: which step -- retrieval or generation -- is responsible?

---

## Part III: Vision Transformers

*Goal: Cover the full landscape of transformer-based computer vision, from image classification
through dense prediction, video understanding, and video generation.*

---

### Chapter 17: Vision Transformer (ViT)

**The ViT paper (Dosovitskiy et al., 2020)**
- Core insight: treat image patches as tokens, apply standard transformer
- Patch embedding: 16x16 patches, linear projection to d_model
- Positional embeddings for 2D: 1D learned positions, 2D sinusoidal, interpolation at new resolutions
- [CLS] token: global representation for classification, vs global average pooling
- ViT-B (86M), ViT-L (307M), ViT-H (632M): scaling configurations

**Why ViT works at scale**
- Lack of inductive biases: no translation equivariance, no locality -- hurts on small data
- Advantage on large data: global attention can learn any spatial relationship
- JFT-300M pre-training: ViT-H/14 beats ResNet152x4 on ImageNet
- ViT vs CNN comparison across data regimes: when each architecture wins
- ViT-22B (Google, 2023): scaling to 22B parameters, QK normalization for stability
- The 2025 consensus: ViTs dominate at scale, CNNs for edge/small data, hybrids (ConvNeXt, CoAtNet) in practice

**DeiT: Data-efficient image transformers**
- Training ViT without large datasets: knowledge distillation from CNN teacher
- Distillation token: a separate CLS-like token targeting the teacher's soft logits
- Strong augmentation: RandAugment, MixUp, CutMix, random erasing, stochastic depth
- DeiT-III: refined 3-augmentation strategy, LayerScale, 87.7% ImageNet without external data

**Self-supervised ViT pre-training**
- DINO (Caron et al., 2021): self-distillation with no labels, student-teacher with EMA
- DINO features: emergent foreground segmentation in attention maps, excellent k-NN classification
- iBOT: combining DINO self-distillation with masked image modeling for patch-level features
- MAE (He et al., 2022): masked autoencoder, 75% masking, asymmetric encoder-decoder
- Why MAE works: high masking ratio forces semantic understanding, not texture; 4x training speedup
- I-JEPA (Assran et al., 2023): predict masked representations in latent space, not pixel space
- V-JEPA 2 (Meta, 2025): extending representation-space prediction to video at massive scale

**DINOv2**
- Curated dataset: LVD-142M, data curation pipeline removing near-duplicates and memes
- Combined objectives: DINO self-distillation loss + iBOT masked prediction loss
- Universal visual features: one encoder for depth, segmentation, classification without fine-tuning
- Register tokens (Darcet et al., ICLR 2024): fixing high-norm artifact tokens in large ViTs
  - Problem: background patches hijacked for internal bookkeeping in ViT-L+
  - Solution: append learnable register tokens to absorb global computations

**DINOv3 and beyond**
- DINOv3 (Meta, Aug 2025): Gram Anchoring loss, 7B teacher, 1.7B images, +6 mIoU over DINOv2
  - Gram Anchoring: regularizes second-order feature correlations to prevent dense feature degradation
  - 88.4% ImageNet fine-tuned, 91.1% ImageNet-R (outstanding OOD robustness)
- AIMv2 (Apple, CVPR 2025): autoregressive patch prediction (not masked), 89.5% ImageNet frozen at 3B
  - Multimodal decoder generating both patches and text tokens
- SigLIP 2 (Google DeepMind, Feb 2025): unified contrastive + captioning + self-distillation + masked prediction
  - Default vision encoder for PaliGemma 2 and production VLMs

---

### Chapter 18: Hierarchical Vision Transformers

**The limitation of ViT for dense prediction**
- Single-scale features: ViT produces one resolution of features, problematic for detection
- High resolution is expensive: 14x14 patches at high resolution = O(n^2) attention cost
- Multi-scale feature maps: why FPN-style hierarchies are needed for detection/segmentation
- The design space: how to build hierarchical multi-scale features into a transformer

**Swin Transformer (Liu et al., 2021)**
- Shifted window attention: restrict attention to local windows, shift between layers
- Hierarchical stages: 4x, 8x, 16x, 32x downsampled feature maps (like ResNet stages)
- Patch merging: spatial pooling by concatenation + projection
- Relative positional bias: learned 2D relative bias table
- Swin-T, -S, -B, -L sizes: comparable to ResNet-50 through ResNet-200 compute

**Swin Transformer V2**
- Scaling to 3B parameters: post-norm, scaled cosine attention, log-spaced continuous PE
- Res-Post-Norm: combined post-norm stability with residual connections
- Training on 192x192 then fine-tuning on 1536x1536: window size adaptation
- Flash Window Attention (Zhang, 2025): IO-aware kernel for window attention, 3x speedup over FlashAttention for short-sequence windowed workloads

**PVT (Pyramid Vision Transformer)**
- Spatial-reduction attention: downsample K and V in deeper stages
- Progressive shrinking: overlap patch embedding for richer spatial context
- PVT v2: linear complexity attention via pooling

**BEiT: Bidirectional Encoder representation from Image Transformers**
- Masked image modeling: predict discrete visual tokens (dVAE from DALL-E)
- Self-supervised ViT training analogous to BERT
- BEiT v2: vector-quantized visual tokenizer, richer targets
- BEiT-3: image as a foreign language — unified multimodal pretraining (no BEiT-4; line plateaued)

**Hiera: Simplicity wins (Ryali et al., Meta ICML 2023)**
- Strip all bells and whistles: no relative position bias, no shifted windows
- Plain ViT blocks with local attention (early stages) and global attention (later stages)
- MAE pretraining recovers accuracy lost from architectural simplification
- 2.4x faster than MViTv2 on images, 5.1x faster on video
- Adopted as vision encoder in SAM 2: real-world validation at massive scale

**FasterViT and efficient hierarchical designs (NVIDIA, ICLR 2024)**
- Hierarchical Attention (HAT): carrier tokens for global context within local windows
- Each window has dedicated tokens that participate in both local and global attention
- Near-linear complexity with image resolution; SOTA Pareto-front (accuracy vs throughput)
- EfficientViT (MIT Han Lab): multi-scale linear attention, 8.8x latency reduction over SegFormer
- FastViT (Apple): structural reparameterization (RepMixer) for mobile deployment
- TransNeXt (CVPR 2024): biomimetic aggregated attention simulating foveal vision

**Linear attention revival for hierarchical ViTs**
- L2ViT (Zheng et al., 2025): local concentration module fixes linear attention's distribution collapse
- Alternating enhanced linear global attention with local window attention
- 84.4% ImageNet-1K Top-1 (87.0% with ImageNet-22K pretraining at 384 resolution)
- The question: can linear attention replace softmax in hierarchical backbones?

**Attention-free hierarchical backbones**
- FocalNet (Yang et al., NeurIPS 2022): focal modulation replaces self-attention entirely
- Depth-wise convolutions + gated aggregation + element-wise modulation; outperforms Swin
- Vision KAN (2026): Kolmogorov-Arnold Networks in a 4-stage hierarchical backbone
- The pattern: hierarchical multi-stage design transcends the attention mechanism itself

**Mamba-Transformer hybrids: the SSM challenge (2024-2025)**
- VMamba (NeurIPS 2024): 2D Selective Scan (SS2D) with four-way scanning for vision
- MambaVision (NVIDIA, CVPR 2025): Mamba early stages + attention late stages; SOTA Pareto-front
- MSVMamba (NeurIPS 2024): multi-scale scanning — hierarchy within the scan itself
- MAP (CVPR 2025): masked autoregressive pretraining for hybrid Mamba-Transformer backbones
- Brief treatment here; full SSM coverage in Part VIII (Post-Transformer Architectures)

**Self-supervised pretraining reshapes hierarchical backbones**
- DINOv3 (Meta, 2025): 7B parameters, 1.7B images, Gram anchoring for dense feature stability
- Frozen DINOv3 backbone outperforms fine-tuned hierarchical models on dense prediction
- The emerging insight: pretraining recipe matters as much as architectural complexity
- InternViT-6B (OpenGVLab): hierarchical ViT scaled to 6B for multimodal LLMs via dynamic tiling
- SigLIP 2 (Google, 2025): sigmoid contrastive + captioning + self-distillation + masked prediction

**The landscape in 2026: convergence and open questions**
- Swin/PVT lines plateaued (no v3 releases); ideas absorbed into newer architectures
- Three competing paradigms: window attention (Swin-family), simplified plain ViT (Hiera), SSM hybrids (MambaVision)
- The pretraining-vs-architecture tension: DINOv3 shows frozen simple backbones rival complex ones
- Open question: will hierarchical structure be designed into the architecture or emerge from pretraining?

---

### Chapter 19: Transformers for Detection and Segmentation

**DETR: detection as set prediction (Carion et al., 2020)**
- The insight: no anchors, no NMS, no hand-designed components
- Object queries: N learned query vectors, each predicts one object or "no object"
- Bipartite matching loss: Hungarian algorithm assigns predictions to ground truth
- DETR limitations: slow convergence (500 epochs), poor small-object detection

**Deformable DETR: fixing attention cost**
- Deformable attention: each query attends to K learned reference points, not all pixels
- Multi-scale deformable attention: attend across FPN levels with O(HWK) cost
- Convergence: 10x faster than DETR, 50 epochs matches DETR at 500

**The DETR improvement arc**
- DN-DETR: denoising training — add noise to GT boxes, train decoder to reconstruct
- Co-DETR: collaborative hybrid assignment — one-to-one + one-to-many auxiliary heads
- DINO (detection): contrastive denoising + mixed query selection, state-of-the-art
- RT-DETR: real-time detection transformer, efficient hybrid encoder, no NMS
- RF-DETR (ICLR 2026): DINOv2 backbone + NAS, >60 mAP on COCO at ≤40ms latency

**Open-vocabulary and unified detection**
- Grounding DINO: open-vocabulary detection conditioned on text prompts
- DINO-X: unified detection + segmentation + grounding + counting + pose + captioning
- Florence-2: sequence-to-sequence formulation for all vision tasks in 0.2B-0.7B params

**Segmentation transformers**
- SegFormer: hierarchical transformer encoder + lightweight MLP decoder
- Mask2Former: universal architecture (semantic, instance, panoptic) via masked attention
- SegGPT: in-context segmentation, coloring as a proxy task

**Segment Anything: promptable segmentation**
- SAM: promptable segmentation at any granularity (points, boxes, masks, text)
- SAM 2: video segmentation with streaming memory, real-time at 44 FPS
- Efficient-SAM2 (ICLR 2026): sparse window routing + sparse memory retrieval, 1.68x speedup
- SAM 3 (ICLR 2026): concept-prompted segmentation, dual encoder-decoder, SA-Co benchmark
- Grounded SAM 2: detection (Grounding DINO / DINO-X) + SAM 2 for open-set tracking

**DINO and DINOv2 for dense prediction**
- Emergent segmentation from DINO attention maps
- DINOv2 + linear probes for depth estimation, semantic segmentation
- Depth Pro (Apple): zero-shot metric depth estimation using DINOv2 features

---

### Chapter 20: Image Generation with Transformers

**The image generation revolution**
- From GANs to diffusion: why diffusion models won
- Latent diffusion: compress to latent space, denoise there, decode back
- The role of the autoencoder: VQ-VAE, KL-VAE, and the trade-off between compression and fidelity

**Autoregressive image generation**
- DALL-E (2021): dVAE tokenization + GPT-style autoregressive transformer
- Parti (Google, 2022): scaling autoregressive image generation to 20B parameters
- The limitation: autoregressive models struggle with global coherence and spatial relationships

**Diffusion Transformers (DiT)**
- DiT (Peebles and Xie, 2022): replacing UNet with a vision transformer in latent diffusion
- AdaLN-Zero: conditioning transformer layers on timestep and text via layer norm modulation
- Scaling laws for DiT: doubling parameters consistently improves FID

**MMDiT and rectified flow**
- MMDiT (SD3/FLUX): dual-stream multimodal diffusion transformer with joint self-attention
- Rectified flow: straight-line paths from noise to data, fewer sampling steps
- Text encoding: T5-XXL, CLIP, and dual text encoder strategies

**The frontier of image generation (2025-2026)**
- FLUX 2 (Black Forest Labs): 32B rectified flow transformer, 4K photorealism, readable text
- Stable Diffusion 3.5: MMDiT-X architecture, 8B params, 1MP resolution
- GPT Image 1.5 (OpenAI): native multimodal, replaces DALL-E 3
- SANA (NVIDIA, ICLR 2025): 0.6B linear DiT, 20× smaller than FLUX, 100× faster, DC-AE with 32× compression
- Midjourney V8 (Mar 2026): rewritten engine, native 2K, 5× faster

**Controllability and editing**
- ControlNet: injecting spatial conditioning (depth, edge, pose) via zero-initialized parallel branch
- IP-Adapter: image prompt conditioning through decoupled cross-attention
- InstructPix2Pix: instruction-based image editing
- Inpainting and outpainting with diffusion transformers

---

### Chapter 21: Video Transformers

**The video understanding challenge**
- Temporal dimension: motion, action, causal structure
- Computation: naive ViT on video = O(T*H*W)^2 attention
- Data: large labeled video datasets (Kinetics, Something-Something, HowTo100M)

**TimeSformer (Bertasius et al., 2021)**
- Factorized space-time attention: spatial attention then temporal attention per patch
- Joint space-time attention: full attention over all patches and frames (expensive)
- Divided space-time attention: best trade-off of accuracy and efficiency
- "Is space-time attention all you need for video?": surprisingly, yes for classification

**ViViT (Arnab et al., 2021)**
- Four model variants: factorized encoder, factorized self-attention, factorised dot-product attention, full attention
- Tubelet embedding: 3D patch tokenization across time and space
- Pre-training on large image datasets, fine-tuning on video

**Video Swin Transformer and hierarchical video backbones**
- 3D shifted windows: temporal + spatial locality combined
- Local 3D windows: t*h*w window size, cycle-shifted along all three dimensions
- Strong performance on Kinetics, SomethingSomething at manageable cost
- Hiera for video: mask unit attention pooling across space and time, 87.3% on K400
- The simplicity thesis revisited: pretraining beats architectural complexity for video

**Self-supervised video pre-training**
- VideoMAE: masked autoencoder for video, 90% masking ratio, tube masking
- VideoMAE V2: dual masking for billion-scale pretraining (encoder subset + decoder subset)
- V-JEPA (Meta, 2024): joint embedding predictive architecture, predict in feature space not pixel space
- V-JEPA 2 (Meta, 2025): world model on 1M+ hours, two-stage training, 77.3% SSv2, zero-shot robot planning
- From pixel prediction to latent prediction: why feature-space targets outperform pixel reconstruction

**Video foundation models**
- InternVideo2 (ECCV 2024): progressive training unifying masked modeling + contrastive + next-token prediction, 6B, SOTA on 60+ tasks
- InternVideo-Next (Dec 2025): Encoder-Predictor-Decoder framework, latent world model, first to beat video-text pretrained models without video-text supervision
- Video-LLMs: VideoLLaMA 2 (spatial-temporal convolution connector), LongVILA (2048 frames, 1M+ tokens)
- The convergence: from task-specific video models to unified video understanding

**Token efficiency for video**
- The token explosion problem: why video transformers need aggressive compression
- PruneVid: pruning 80%+ tokens while maintaining performance
- Run-length tokenization: 40% faster training via temporal redundancy
- Spatiotemporal token compression (CVPR 2026): scaling to longer durations

---

### Chapter 22: Video Generation with Transformers

**The video generation problem**
- Challenges: temporal consistency, physical plausibility, high resolution, long duration
- Data: WebVid, Panda-70M, OpenVid, curated film datasets
- Evaluation difficulty: quality, motion, physics, prompt fidelity

**Early transformer-based video generation**
- VideoGPT (2021): VQ-VAE for compression, transformer autoregressive in latent space
- CogVideo (2022): pre-trained image generation + temporal attention modules
- Phenaki: variable-length video generation from text prompts using transformer

**Diffusion transformers for video**
- DiT (Peebles and Xie, 2022): replace UNet with transformer in latent diffusion
- MMDiT: dual-stream multimodal diffusion transformer (FLUX/SD3 foundation)
- Video DiT: extend DiT with temporal attention, spatiotemporal patches
- CogVideoX (2024): 3D full attention on video tokens, expert transformer

**Sora and the frontier of video generation**
- Sora (OpenAI, 2024): video as spacetime patches, scaling DiT to minutes of high-res video
- Spacetime patch tokenization: flexible resolution and duration
- World model framing: Sora as simulation of physical world dynamics
- Sora 2 (Sep 2025): multimodal MM-DiT, synchronized audio, 10-25s generation

**The open-source video generation ecosystem**
- Open-Sora 2.0 (Mar 2025): MMDiT dual-stream architecture, matches Sora for $200K
- HunyuanVideo 1.5 (Tencent, Nov 2025): 8.3B, 3D causal VAE (16× spatial, 4× temporal)
- Wan 2.2 (Alibaba, Jul 2025): first MoE video gen, two-expert noise-level routing, 27B/14B active
- Mochi 1 (Genmo): 10B AsymmDiT, 128× compression VAE, 3D RoPE
- LTX-Video / LTX-2 (Lightricks): real-time DiT, 5s video in 2s, simultaneous audio+video

**Architectural components for video generation**
- 3D causal VAE: joint spatial-temporal compression, group causal convolution (CVPR 2025)
- Full spatiotemporal attention vs factorized: quality vs efficiency trade-off
- 3D RoPE: extending rotary embeddings to time dimension
- Asymmetric architectures: AsymmDiT (4× visual vs text params), AsymmVAE
- MoE routing by noise level: high-noise expert (layout) + low-noise expert (detail)
- Classifier-free guidance in video: motion guidance, camera control

**Controllable generation and video editing**
- Video ControlNet: depth, pose, edge conditioning across frames
- EPiC: camera control with <1% extra params via anchor-ControlNet
- Motion prompting (CVPR 2025): controlling generation with motion trajectories
- Stable Video Diffusion: image-to-video with temporal layers
- VideoPainter: dual-stream any-length video inpainting and editing
- Tokenflow: propagating edits temporally through feature-space attention

---

### Chapter 23a: Training Vision Transformers — Classification and Transfer Learning

**The ViT training landscape**
- Why ViT training is harder than CNN training: no inductive biases, data hunger, optimization fragility
- Three eras: supervised pre-training (2020), self-supervised revolution (2021-2023), modern recipes (2024-2026)

**The modern supervised training recipe**
- ViT-5 (Feb 2026): QKNorm, RoPE+APE dual encoding, registers, SwiGLU, RMSNorm — 84.2% ImageNet
- DeiT III recipe: only 3 augmentations, LayerScale, stochastic depth, RegNetY teacher distillation
- Data regime comparison: ImageNet-1K vs 21K vs JFT-300M/3B — when each matters
- Hands-on: full ViT-B supervised training loop on ImageNet with HuggingFace datasets

**Data augmentation for vision transformers**
- The augmentation suite: RandAugment, AutoAugment, TrivialAugment
- MixUp and CutMix: interpolation and patch-swapping augmentation with full PyTorch code
- Regularization as augmentation: stochastic depth (DropPath), patch dropout, LayerScale
- Synthetic data augmentation: diffusion-generated training data (CVPR 2026), 20% gains from realistic synthetic data

**Self-supervised pre-training strategies**
- DINO/DINOv2 training: teacher-student, centering, sharpening, EMA update, curated LVD-142M
- Hands-on: complete DINO training loop with multi-crop augmentation using ImageNet from HF datasets
- MAE training: 75% masking, asymmetric encoder-decoder, DailyMAE 5.8× speedup
- Hands-on: complete MAE pre-training loop with masking and reconstruction code
- FastDINOv2: spectral-domain curriculum, 62% training time with matched accuracy
- iBOT / BEiT: discrete tokenizer, masked prediction in token space
- Contrastive objectives: MoCo v3, SigLIP 2 unified recipe

**Fine-tuning vision transformers**
- Layer-wise learning rate decay (LLRD): exponential LR decay per layer group
- Hands-on: full fine-tuning pipeline on COCO detection with LLRD and HuggingFace datasets
- Resolution adaptation: interpolate 2D positional embeddings, Swin window size adaptation
- PEFT for vision (2025-2026): Image-LoRA, head-selection via influence scores, visual prompt tuning
- Make LoRA Great Again: adaptive singular values + MoE alignment

**Detection and segmentation training recipes**
- DETR training: bipartite matching loss, DN-DETR denoising, Co-DETR hybrid assignment
- SAM training at scale: SA-1B data engine, promptable training loop
- DINOv2 + linear probes: zero-shot dense prediction without fine-tuning

**Numerical precision and training efficiency**
- Mixed precision: BF16 preferred for ViT stability, FP16 loss scaling strategies
- FP8 training (TWEO, 2025): full FP8 pre-training with 36% throughput gain, no architecture changes
- Distributed training: tensor parallelism for ViT-G, FSDP for DINOv2, activation checkpointing

**Complete classification training recipes**
- Recipe 1: ViT-Base ImageNet classification (DeiT III) — full hyperparameter table
- Recipe 2: DINOv2-style self-supervised pre-training — full hyperparameter table
- Recipe 3: Fine-tuning for COCO detection with LLRD — full hyperparameter table

---

### Chapter 23b: Training Image Generation Transformers

**Latent diffusion fundamentals for transformers**
- From pixel-space to latent-space: why VAE encoding makes transformer-based diffusion tractable
- The SD VAE: encoder, decoder, KL regularization, f=8 spatial compression, 4-channel latent
- Hands-on: encoding images to latent space and reconstructing with a pretrained VAE

**DiT and MMDiT training**
- DiT architecture recap: patchify latents, AdaLN-Zero conditioning, final linear head
- Rectified flow: linear interpolation between data and noise, V-prediction loss
- Classifier-free guidance training: random conditioning dropout at 10-20%
- Hands-on: complete DiT training loop on ImageNet with HuggingFace datasets
- MMDiT: dual-stream text-image processing, FLUX architecture, joint attention blocks

**Training text-to-image models**
- Text encoder training: frozen T5-XXL + CLIP, dual text conditioning
- Noise schedule design: logit-normal sampling for rectified flow, resolution-dependent schedules
- Progressive resolution training: start at 256×256, scale to 1024×1024
- Hands-on: text-conditioned DiT training step with dual text encoders

**Consistency distillation for few-step generation**
- Consistency models: map any point on the denoising trajectory to the final clean output
- SANA-Sprint (ICCV 2025): continuous-time consistency distillation + latent adversarial distillation
- DMD2: distribution matching distillation without regression loss, 1.28 FID on ImageNet-64
- Adversarial Diffusion Distillation (ADD): discriminator at each timestep, 1-4 step generation
- SenseFlow (ICLR 2026): distribution matching for flow-based models (SD3.5, FLUX)
- Hands-on: consistency distillation training step

**ControlNet and adapter training**
- ControlNet: zero-initialized copies of encoder blocks, condition on depth/edge/pose
- IP-Adapter: image prompt adapter with decoupled cross-attention
- Training recipes and hyperparameters for conditional generation

**Complete image generation training recipes**
- Recipe 1: DiT-L/2 class-conditional ImageNet 256×256 — full hyperparameter table
- Recipe 2: Text-to-image rectified flow at 512×512 — full hyperparameter table
- Recipe 3: Consistency distillation from a trained teacher — full hyperparameter table

---

### Chapter 23c: Training Video Generation Transformers

**3D VAE training for video compression**
- From 2D VAE to 3D: temporal compression with causal 3D convolutions
- Group causal convolution (CVPR 2025): fixing quality inconsistency across frames
- Open-Sora 2.0 Video DC-AE: EfficientViT blocks, 8×8×8 compression, pixel-shuffle decoder
- Hands-on: 3D VAE encoder-decoder forward pass and latent space visualization

**Video diffusion transformer training**
- CogVideoX expert transformer: expert adaptive LayerNorm, 3D full attention
- Full 3D attention vs factorized spatial-temporal: quality vs cost tradeoff
- 3D RoPE: rotary position embeddings extended to spatiotemporal coordinates
- Hands-on: complete video diffusion training step with 3D attention

**Mixed image-video training**
- Domain-specific normalization: separate batch norm statistics for images and videos
- Why mixed training matters: preventing image quality forgetting
- Open-Sora 2.0: zero-padding images to match video input format
- Mixed video-length training: randomized durations with zero-padding

**Progressive training pipeline**
- Open-Sora 2.0 three-stage pipeline: text-to-image → image-to-video → high-res video
- Checkpoint transfer between stages: what to keep, what to reinitialize
- Resolution and duration scaling: $200K total training cost for commercial quality
- Hands-on: progressive training stage transitions

**Pyramidal flow matching for efficient training**
- Temporal pyramids: recent frames at full resolution, older frames compressed
- Token reduction: 119K → 15K for 10-second 241-frame video (8× reduction)
- Training the pyramid: multi-resolution denoising within a single forward pass
- Hands-on: pyramidal flow matching training step

**Data curation and filtering for video training**
- Video data pipeline: filtering, captioning, deduplication, quality scoring
- Caption generation: using VLMs to generate dense video descriptions
- Aesthetic and motion filtering: removing static or low-quality clips

**Complete video generation training recipes**
- Recipe 1: CogVideoX-style text-to-video at 480p — full hyperparameter table
- Recipe 2: Open-Sora 2.0 progressive pipeline — full hyperparameter table
- Recipe 3: Pyramidal flow matching for long video — full hyperparameter table

---

### Chapter 24: Evaluating Vision Transformers

**Image classification benchmarks**
- ImageNet-1K: top-1 and top-5 accuracy, 1000 classes, saturation at >92% for foundation models
- ImageNet-21K: fine-grained, 21K classes, transfer learning evaluation
- ImageNet variants for robustness: ImageNet-C (corruptions), ImageNet-R (renditions), ImageNet-A (adversarial)
- ObjectNet: distribution-shifted test set designed to remove spurious correlations
- Geographic fairness: 7-20% accuracy gaps between regions, FHIBE (81+ countries), GeoBS information-theoretic framework, Dollar Street socioeconomic diversity

**Detection and segmentation benchmarks**
- COCO detection: AP, AP_50, AP_75 across box and mask; RF-DETR >60 AP (first real-time, ICLR 2026)
- COCONut (CVPR 2024): 383K images, 5.18M panoptic masks; COCO-ReM (ECCV 2024): refined masks via SAM
- LVIS: long-tail detection, 1203 categories, frequency-stratified evaluation
- ADE20K semantic segmentation: mean IoU across 150 classes
- Cityscapes: autonomous driving scenes, panoptic segmentation

**Video understanding benchmarks**
- Kinetics-400/600/700: action recognition, frame bias problem
- Something-Something V2: motion-centric, temporal reasoning required
- AVA: spatiotemporal action localization; YOLO-Act +28.18 mAP gain
- EgoSchema: 5000+ questions, models <33% vs humans 76%; EgoPlan-Bench2: planning across 4 domains
- EASG-Bench (2025): egocentric action scene graphs for video QA
- ActivityNet QA, MSRVTT: video question answering benchmarks

**Image generation metrics**
- FID: Frechet Inception Distance, Gaussian assumption limitations, sample size sensitivity
- Beyond FID: CMMD (robust distribution metric), GenEval (compositional), DreamSim (perceptual)
- DiT achieving FID 1.73 on ImageNet-256

**Video generation evaluation**
- FVD (Frechet Video Distance): 3D convolution feature distribution distance, poor temporal sensitivity
- FID per frame: per-frame image quality
- CLIP-SIM: text-video alignment score
- JEDi: JEPA-based alternative to FVD, 34% better human alignment, 16% sample requirement
- VBench and VBench-2.0 (2025): 18 capabilities across Human Fidelity, Creativity, Controllability, Physics, Commonsense
- WorldScore (ICCV 2025): unified world generation benchmark; WCS for object permanence and causal compliance
- Sora evaluation: ad-hoc expert assessment, physical plausibility, prompt fidelity

**Efficiency evaluation**
- FLOPs ≠ real latency: ViT with 5.00 GFLOPs = 1.75x latency of CNN with 4.95 GFLOPs
- EER (Efficient Error Rate): parameters, bits, FLOPs, model size
- Throughput, peak memory, and hardware-specific benchmarks

**VLM-as-Judge for vision evaluation**
- Prometheus-Vision: first open-source VLM evaluator with custom rubrics
- UNIVERSE: VLM-based evaluator for video world model rollouts
- Biases and calibration challenges for VLM judges

---

## Part IV: Audio, Speech and Music

*Goal: Cover transformer-based audio processing across speech recognition, synthesis, music generation,
and unified audio-visual models, with evaluation grounded in domain-standard metrics.*

---

### Chapter 24: Speech Recognition Transformers

**wav2vec 2.0 (Baevski et al., 2020)**
- Raw waveform input: CNN feature extractor + transformer encoder
- Contrastive pre-training: predict masked latent speech representations
- Codebook: discrete speech units via quantization module
- Fine-tuning on labeled data: CTC decoder on top of transformer
- Low-resource ASR: 10 minutes of labeled audio with wav2vec 2.0 pre-training

**HuBERT (Hsu et al., 2021)**
- BERT-style self-supervised: predict offline cluster assignments for masked frames
- K-means clustering on MFCC features, then on HuBERT features (iterative refinement)
- HuBERT Large pre-trained on 60K hours Libri-Light: state-of-the-art low-resource ASR

**Whisper (Radford et al., 2022)**
- Supervised at scale: 680K hours of labeled multilingual audio from the internet
- Log-mel spectrogram input: 80 mel bins, 25ms windows, 10ms hop
- Sequence-to-sequence transformer: encoder + decoder, joint multilingual + task modeling
- Multitask: transcription, translation, language ID, timestamp prediction
- Whisper large v3: improved multilingual, timestamp accuracy, reduced hallucination

**Conformer architecture**
- Combining convolution and attention: local patterns (conv) + global context (attention)
- Conformer block: feed-forward → multi-head attention → convolution → feed-forward (Macaron)
- Conformer vs Transformer on LibriSpeech: consistent improvement from conv module
- Use in streaming ASR: causal convolution and causal attention variants

**CTC decoding**
- Connectionist Temporal Classification: blank token, collapse rule, forward-backward algorithm
- CTC vs attention decoder: CTC for streaming, attention for quality
- Hybrid CTC/attention: joint training, beam search with CTC prefix score

---

### Chapter 25: Text-to-Speech and Voice Synthesis

**Transformer TTS baselines**
- FastSpeech 2: non-autoregressive, duration predictor, pitch and energy prediction
- Tacotron 2 + WaveNet: autoregressive mel generation then neural vocoder

**Neural codec language models**
- EnCodec: residual vector quantization (RVQ) audio codec, multi-scale codebook
- Codec tokens as the speech vocabulary: discrete audio tokens for LM training

**VALL-E (Wang et al., 2023)**
- In-context TTS: 3-second enrollment audio as acoustic prompt
- Autoregressive codec language model: AR model for first quantizer, NAR for residuals
- Zero-shot voice cloning: match speaker timbre from 3-second reference
- VALL-E 2: improved consistency, naturalness, prosody transfer

**VoiceBox (Meta, 2023)**
- Flow matching: non-autoregressive, fast generation via ODE
- Text-conditioned audio infilling: in-context speech synthesis
- Multilingual: 6 languages, style transfer, noise removal

**Modern zero-shot TTS frontier**
- CosyVoice: supervised semantic tokens + flow matching, multilingual
- F5-TTS: flow matching with DiT, simple and strong zero-shot
- Seed-TTS (ByteDance): ultra-realistic, speaker cloning, emotion control

---

### Chapter 26: Audio Understanding and Music Generation

**Self-supervised audio pre-training**
- Audio Spectrogram Transformer (AST): patch-based ViT applied to mel spectrograms
- AudioMAE: masked autoencoder for audio, 80% masking, pre-trained on AudioSet
- SSAST: joint discriminative + generative pre-training

**Audio language models**
- AudioLDM 2: latent diffusion with AudioMAE encoder as conditioning
- AudioPaLM: unify speech and text in a single PaLM decoder
- Any-to-Any: audio generation from any input modality

**Music generation**
- Jukebox (OpenAI): VQ-VAE + sparse transformer, raw audio, 1.2M parameters
- MusicGen (Meta, 2023): single-stage decoder, EnCodec tokens, text/melody conditioning
- MusicLM (Google, 2023): hierarchical audio LM, MuLan for text-audio alignment
- Stable Audio: latent diffusion with long-form temporal autoencoder

**Sound event detection and classification**
- AudioSet: weakly-labeled, 527 classes, 2M clips
- Transformer-based sound event detection: patch embedding of spectrograms
- Zero-shot audio classification with CLAP: CLIP-like contrastive audio-text pre-training

---

### Chapter 27: Unified Audio-Visual-Text Models

**ImageBind (Meta, 2023)**
- Binding six modalities: image, text, audio, depth, thermal, IMU
- Image as hub: pair each modality with image, transitive alignment
- Emergent zero-shot cross-modal retrieval: audio → 3D, text → IMU without paired data

**AudioCLIP**
- Extending CLIP to audio: three-way contrastive (audio, image, text)
- Audio-visual sound localization: which pixels correspond to which sounds?

**Moshi (Kyutai, 2024)**
- Real-time dialogue: simultaneous speech generation and understanding
- Inner monologue: text backbone with speech token streaming
- Full-duplex conversation: model can speak and listen simultaneously

**Qwen-Audio and modern audio LLMs**
- Qwen-Audio: Whisper encoder + Qwen LLM, multi-task audio understanding
- Gemini audio: native audio tokens in Gemini architecture
- GPT-4o audio: native audio in/out without ASR-as-preprocessing

---

### Chapter 28: Training Audio Transformers

**Input representations and their training implications**
- Log-mel spectrogram: frequency decomposition, standard for classification and ASR
- Raw waveform: end-to-end, avoids handcrafted features, higher memory cost
- Codec tokens: discrete audio units, enable language model training on audio
- Multi-resolution inputs: coarse + fine resolution for efficiency

**Self-supervised audio pre-training objectives**
- Masked spectrogram prediction (AudioMAE): reconstruct masked time-frequency patches
- Contrastive pre-training (wav2vec 2.0): match masked frames to quantized targets
- CLAP: contrastive audio-language pre-training, AudioSet + free text descriptions
- Online clustering (HuBERT): offline targets, iterative refinement

**Data augmentation for audio**
- SpecAugment: time masking and frequency masking on mel spectrograms
- Mixup for audio: weighted combination of two spectrograms and labels
- RIR convolution: apply room impulse responses to simulate different acoustics
- Background noise addition: simulate real-world degraded speech

**TTS training specifics**
- Duration modeling: forced alignment vs learned duration predictor
- Flow matching training: conditional flow matching on mel spectrogram or codec tokens
- Multi-speaker training: speaker embedding conditioning (d-vector, x-vector)
- Zero-shot voice cloning: acoustic prompt encoding, ensuring speaker consistency

**Distributed training for large audio models**
- Audio sequence length: 10 seconds at 16kHz = 160K samples, spectrogram = ~1000 frames
- Chunked processing: divide long audio into overlapping windows, stitch with cross-attention
- Streaming inference: causal models, chunk-wise processing for real-time ASR

---

### Chapter 29: Evaluating Audio and Speech Models

**ASR evaluation**
- Word Error Rate (WER): the standard metric, insensitive to capitalization/punctuation
- Character Error Rate (CER): preferred for non-segmenting languages (Chinese, Japanese)
- LibriSpeech: clean and other test sets, academic ASR benchmark
- Common Voice: multilingual, community-contributed, accent variation
- Challenging benchmarks: FLEURS (multilingual), CHiME-6 (overlapping speech), AMI (meeting)

**TTS evaluation**
- MOS (Mean Opinion Score): 5-point naturalness scale via human listening tests
- UTMOS: automated MOS prediction, strong correlation with human MOS
- Word Error Rate via ASR: intelligibility proxy
- Speaker similarity: ECAPA-TDNN or WavLM cosine similarity to reference speaker
- MUSHRA: multiple stimuli with hidden reference, broadband audio quality

**Audio classification and sound event detection**
- AudioSet mAP: mean average precision over 527 classes
- DCASE Challenge: yearly benchmark, scene classification + sound event detection + SELD
- SUPERB: benchmark for speech processing models, 10 tasks, single pre-trained model evaluated

**Music generation evaluation**
- FAD (Frechet Audio Distance): feature distribution distance using VGGish or music encoder
- KL divergence on audio feature distributions
- Human evaluation: musicality, creativity, prompt fidelity, harmony

---

## Part V: Multimodal Transformers

*Goal: Cover the architecture, training, and evaluation of models that jointly process multiple modalities,
including vision-language, omni models, document understanding, 3D, and robotics.*

---

### Chapter 30: Contrastive Multimodal Learning

**CLIP (Radford et al., 2021)**
- Architecture: image encoder (ViT or ResNet) + text encoder (Transformer), separate towers
- Contrastive objective: maximize similarity of matched pairs, minimize unmatched
- Training data: 400M (image, text) pairs from the internet (WIT dataset)
- Zero-shot classification: embed class names as text, cosine similarity to image embedding
- CLIP for downstream tasks: frozen features, fine-tuning, linear probing comparison

**ALIGN (Jia et al., 2021)**
- 1.8B noisy image-text pairs from alt-text: scale compensates for noise
- EfficientNet image encoder + BERT text encoder
- Demonstrates that scale + noise > small clean datasets for contrastive pre-training

**Florence and Florence-2**
- Unified representation: CLIP-like foundation with grounding, detection, segmentation heads
- Florence-2: sequence-to-sequence architecture for all vision tasks
- Prompt engineering for visual grounding: region description, OCR, referring expressions

**SigLIP**
- Sigmoid loss instead of softmax: no global batch normalization, more stable
- SigLIP-SO400M: trained on 4B samples, outperforms CLIP with better training efficiency
- Implications for large-batch contrastive training

**CLIP limitations and extensions**
- Compositional reasoning: CLIP fails on spatial and relational ("a cat to the left of a dog")
- SugarCrepe and ARO benchmarks: measuring compositional understanding
- NegCLIP: adding hard negative text pairs to address compositionality
- Long CLIP: extending CLIP to longer text captions (77 → 248 tokens)

---

### Chapter 31: Vision-Language Models (VLMs)

**BLIP (Li et al., 2022)**
- Multi-task pre-training: ITC (image-text contrastive) + ITM (image-text matching) + LM
- CapFilt: generate captions with BLIP, filter with ITM score to clean noisy web data
- Three model variants for different downstream tasks

**BLIP-2 and the Q-Former**
- Q-Former: lightweight module bridging frozen image encoder and frozen LLM
- Two-stage training: align image features to Q-Former, then to LLM
- 32 learned query vectors: compress variable image patches to fixed-length representation
- Bootstrapping: reuse frozen pre-trained components, only Q-Former parameters learned

**Flamingo (Alayrac et al., 2022)**
- Interleaved image-text sequences: few-shot multimodal learning from examples
- Perceiver Resampler: cross-attention over image features, fixed output tokens
- Gated cross-attention dense layers inserted into frozen Chinchilla LLM
- 16-shot vs 32-shot performance on VQA, COCO captioning, visual dialogue

**LLaVA and instruction-tuned VLMs**
- LLaVA: simple linear projection connecting CLIP ViT to LLaMA
- Instruction tuning data: GPT-4 generated visual conversations
- LLaVA 1.5: MLP connector, CLIP ViT-L/336px, SFT on mixture of datasets
- LLaVA-Next (LLaVA-1.6): dynamic high resolution, 4x4 grid of patches

**GPT-4V and Gemini Vision**
- GPT-4V: multimodal GPT-4, vision encoding details not published
- Gemini 1.0 Ultra: native multimodal training, image+audio+video+text
- Gemini 1.5 Pro: 1M context window, long video understanding

**InternVL and the open-source frontier (2024-2025)**
- InternVL 1.5 / 2.0: dynamic high-resolution image tiling, competitive with GPT-4V
- Qwen-VL 2.5: strong OCR, document understanding, multilingual vision
- LLaVA-Onevision: unified architecture for single-image, multi-image, video

---

### Chapter 32: Omni Models — Any-to-Any Generation

**GPT-4o architecture and capabilities**
- Native multimodal: single model processing text, image, audio simultaneously
- No pipeline bottleneck: end-to-end audio without ASR intermediate step
- Real-time conversation: 232ms median latency for audio responses
- Emotional intelligence: tone, nuance, and prosody in voice responses

**Gemini 1.5 / 2.0**
- Gemini 1.5 Pro: 1M context window, 10 hours of audio or 1 hour of video in context
- Mixture-of-experts architecture: efficient large context processing
- Gemini 2.0: multimodal output (image + audio generation), tool use, agent capabilities

**Unified-IO and Unified-IO 2**
- Tokenize everything: image pixels, audio spectrograms, actions, 3D, text in one vocabulary
- Single sequence-to-sequence model for 120+ tasks
- Architecture: 2D and 1D VQ-VAE tokenizers + T5-style encoder-decoder

**Qwen-Omni**
- Any-to-any: text, image, audio, video inputs and outputs
- Separate codec for each modality, unified transformer backbone
- State-of-the-art on both language and multimodal tasks simultaneously

**Design principles for omni models**
- Native vs adapter-based multimodality: quality vs development speed trade-off
- Unified vocabulary: merging token spaces across modalities
- Modality-specific encoders vs tokenizers: when to use each
- Cross-modal interference: negative transfer between modalities during training

---

### Chapter 33: Document and Code Understanding

**Document transformers**
- LayoutLM: 2D positional embeddings from bounding boxes, text + layout pre-training
- LayoutLMv2/v3: vision-integrated, image-text joint pre-training, masked image modeling
- DocFormer: multi-modal attention with spatial, text, visual features
- Document QA: DocVQA dataset, InfoVQA, ChartQA

**Code transformers**
- Codex (Chen et al., 2021): GPT-based, fine-tuned on GitHub code, powers Copilot
- Code pre-training data: GitHub, CodeSearchNet, The Stack, The Stack v2
- CodeT5 and CodeT5+: encoder-decoder for code understanding and generation
- StarCoder 1/2: 15.5B, 80+ programming languages, Fill-in-the-Middle objective
- DeepSeek-Coder: competitive open-source code model, strong on competitive programming

**AST-aware and program-structure-aware models**
- TreeSitter parsing: structured representation of source code
- GraphCodeBERT: data flow graph as additional input modality
- Code as trees: attention over AST nodes, hierarchical code representation

**Table and structured data understanding**
- TAPAS: table pre-training with relative cell position embeddings
- OmniTab: joint natural language and tabular data pre-training
- TableFormer: structure-aware table parsing

---

### Chapter 34: Transformers for 3D, Point Clouds and Robotics

**3D point cloud transformers**
- Point Transformer (Zhao et al., 2021): local attention in 3D space, k-NN graph
- Point Transformer V2 and V3: hierarchical, state-of-the-art 3D segmentation and detection
- PCT (Point Cloud Transformer): global attention with offset attention
- 3D object detection: VoxelTransformer, SST (Sparse Transformer for LiDAR)

**Transformer-based robotics**
- Gato (Reed et al., 2022): multi-embodiment transformer, text + image + action tokens
- RT-1: efficient transformer for robot action prediction, real-world manipulation
- RT-2 (Brohan et al., 2023): VLM fine-tuned for robot actions, semantic generalization
- Octo: open-source robot foundation model, flexible action head, 800K robot trajectories

**World models and planning**
- DreamerV3: latent world model with transformer, planning in imagination
- GAIA-1: generative world model for autonomous driving, video prediction
- UniSim: universal simulator of interactive 3D worlds, action-conditioned video generation

---

### Chapter 35: Training Multimodal Transformers

**Modality alignment pre-training**
- Stage 1: align visual encoder to language model with large-scale image-text data
- Stage 2: instruction tuning on high-quality multimodal conversations
- Stage 3 (optional): RLHF / DPO for preference alignment of multimodal outputs

**Data for multimodal pre-training**
- Image-text pairs: CC3M, CC12M, LAION-400M, LAION-5B, DataComp-1B
- Interleaved documents: MMC4 (from Common Crawl), OBELICS
- High-quality instruction data: LLaVA-Instruct, ShareGPT4V, SVIT
- Video-text: WebVid-10M, HowTo100M, Panda-70M

**Contrastive pre-training specifics**
- In-batch negatives: why they work and when they fail (false negatives)
- Temperature annealing in CLIP training: start high, anneal low
- Large batch requirements for CLIP: 32K+ batch size for strong negatives
- Hard negative mining: constructing difficult pairs from near-duplicates

**Joint training challenges**
- Modality imbalance: text data is orders of magnitude larger than labeled image data
- Catastrophic forgetting: fine-tuning on visual tasks can hurt language performance
- Modal dropout: randomly drop modalities to improve unimodal performance
- Loss weighting: balancing multiple objectives (ITC + ITM + LM) during training

**Distributed training for multimodal models**
- Separate image encoder and LLM: potential for independent parallelism strategies
- Communication overhead: broadcasting image features to all ranks in DDP
- Gradient checkpointing for image encoder: save memory for high-resolution inputs
- Mixed precision for vision: FP16 for image encoder, BF16 for language model

---

### Chapter 36: Evaluating Multimodal Models

**Vision-language understanding**
- VQAv2: open-ended visual question answering, balanced yes/no
- GQA: compositional question answering with structured semantics
- TextVQA: text reading in images, OCR-dependent QA
- OK-VQA: external knowledge required, not fully answerable from image alone

**Image captioning**
- COCO captioning: CIDEr, BLEU-4, METEOR, ROUGE-L
- NoCaps: novel object captioning, out-of-domain objects from Open Images
- Limitations of n-gram metrics: CIDER rewards fluency not factual accuracy

**Holistic multimodal benchmarks**
- MMBench: multiple-choice questions across 20 perception and reasoning dimensions
- MMMU: massive multi-discipline multimodal understanding, college-level questions
- MM-Vet: complex multimodal tasks requiring integration of recognition and knowledge
- OpenCompass multimodal leaderboard: aggregated ranking across 20+ benchmarks

**Multimodal hallucination evaluation**
- POPE: object hallucination, binary questions about object presence
- HallusionBench: visual and language hallucination, adversarial image-text pairs
- AMBER: ambiguous multimodal references and object attribution

**Video QA**
- PerceptionTest: spatio-temporal reasoning, causality, fluid dynamics
- EgoSchema: 5K egocentric video QA, long-form video understanding
- Video-MME: video multi-modal evaluation, short/medium/long video categories

---

## Part VI: Architecture Innovations and Efficiency

*Goal: Deep technical coverage of every major architectural innovation enabling faster training,
better inference, longer context, sparser computation, and smaller deployable models.*

---

### Chapter 37: FlashAttention — IO-Aware Exact Attention

**The memory bottleneck in standard attention**
- Attention matrix size: N^2 floats for sequence length N -- 100K context = 40GB
- Memory bandwidth as the bottleneck: GPU compute is underutilized waiting for HBM reads
- The roofline model: attention is memory-bandwidth-bound, not compute-bound

**GPU memory hierarchy**
- SRAM (shared memory, L1): ~20MB on A100, extremely fast (~19TB/s bandwidth)
- HBM (High Bandwidth Memory, DRAM): 40-80GB on A100, slower (~2TB/s)
- The key insight: avoid materializing the full N×N attention matrix in HBM

**FlashAttention (Dao et al., 2022)**
- Tiling: compute attention in blocks that fit in SRAM
- Online softmax: numerically stable incremental softmax update
- No materialization: never write the full attention matrix to HBM
- Exact computation: identical output to standard attention, not an approximation
- Speedup: 2-4x end-to-end training speedup, 5-20x memory reduction for attention

**FlashAttention-2 (Dao, 2023)**
- Better parallelism: parallelize over sequence length dimension
- Work partitioning: minimize non-matmul FLOPs, better utilization of tensor cores
- 2x faster than FlashAttention-1, 9x faster than PyTorch standard attention
- Support for GQA and MQA natively

**FlashAttention-3 (Shah et al., 2024)**
- Hopper GPU (H100) specific optimizations: WGMMA and TMA instructions
- Asynchrony: overlap softmax computation with next GEMM
- FP8 support: mixed precision for further speedup
- Flash-Decoding: parallelism over sequence length for long-context autoregressive generation

**Impact and adoption**
- FlashAttention is now the default in HuggingFace, PyTorch, all major training frameworks
- Enables 128K+ context training that was infeasible before
- Flash-Decoding: parallel decoding across KV cache positions for long-context inference
- Ring Attention: distribute FlashAttention over multiple GPUs for million-token contexts

---

### Chapter 38: The KV Cache — Architecture, Optimization and Management

**What is the KV cache?**
- Autoregressive generation: at each step, recompute all previous K and V tensors -- expensive
- KV cache: store and reuse previous K, V tensors, avoid redundant computation
- Memory cost: 2 * layers * heads * head_dim * seq_len * bytes_per_element
- Example: LLaMA 3 8B, 128K context, BF16 = 32 layers × 32 heads × 128 dim × 128K × 2 bytes = ~100GB

**KV cache architecture considerations**
- Per-layer KV cache: every transformer layer has its own K, V cache
- Multi-Head Attention (MHA): heads × seq_len × head_dim per layer
- Multi-Query Attention (MQA): 1 KV head, k/v shared across query heads -- 8-32x smaller KV cache
- Grouped-Query Attention (GQA): G KV heads (G < H), used in LLaMA 2/3, Mistral
- Multi-Head Latent Attention (MLA): DeepSeek V2/V3, compress KV to low-rank latent vector

**Paged attention (vLLM)**
- The fragmentation problem: variable-length sequences waste contiguous memory reservations
- Pages: divide KV cache into fixed-size pages, allocate on demand
- Block table: virtual-to-physical page mapping, like OS virtual memory
- Copy-on-write for beam search and parallel sampling: share pages until they diverge

**Continuous batching**
- Static batching: pad all sequences to max length, waste tokens
- Continuous batching: add new requests to batch as old ones finish, iteration-level scheduling
- vLLM and TGI: continuous batching + paged attention = foundation of LLM serving

**KV cache compression**
- StreamingLLM: discard middle tokens, keep attention sinks (first tokens) + recent window
- H2O (Heavy Hitter Oracle): evict KV cache entries with low attention score
- KV cache quantization: INT8/INT4 KV cache, small quality loss, 2-4x memory reduction
- Speculative decoding: use small draft model to fill cache, verify with large model

**Prefix caching**
- System prompt reuse: identical system prompt → cache KV and reuse across requests
- RadixAttention (SGLang): trie-based prefix cache, automatic reuse of common prefixes
- Deployment efficiency: prefix cache can reduce TTFT (Time To First Token) dramatically

---

### Chapter 39: Positional Encodings — A Complete Taxonomy

**Fixed absolute positional encodings**
- Sinusoidal PE (Vaswani et al.): frequencies, phase, no learned parameters, extrapolation possible
- Why sinusoidal does not extrapolate well in practice: attention patterns degrade

**Learned absolute positional encodings**
- Lookup table: one learned vector per position up to max_len
- GPT-2, BERT: learned absolute PE, simple and effective within training length
- Cannot extrapolate: positions outside training range have no learned representation

**Relative positional encodings**
- Relative position bias (Shaw et al., 2018): add learned bias to attention logits based on relative distance
- T5 relative bias: shared learnable scalar per relative position bucket
- ALiBi (Press et al., 2022): linear bias penalty proportional to distance, no added parameters
- ALiBi extrapolation: near-perfect length generalization, model trains at 1024, infers at 2048+

**Rotary Position Embeddings (RoPE)**
- Core idea: rotate query and key vectors by position-dependent angle in 2D subspaces
- Properties: relative position naturally emerges from rotation subtraction (Rd^T Rq = Rq-d)
- Implementation: multiply Q/K by complex exponentials, efficient with einops
- Adoption: LLaMA 1/2/3, Mistral, Qwen, Falcon, Gemma, DeepSeek -- de facto standard
- Long-term decay: dot product naturally decays with distance for most frequencies

**Extending RoPE for long context**
- Positional interpolation (PI): scale position indices down to fit training range
- NTK-aware interpolation: scale high-frequency dimensions less, avoid aliasing
- YaRN: non-uniform frequency scaling + attention temperature correction
- LongRoPE: per-dimension evolutionary search for optimal interpolation factors, 2M context

**2D and N-dimensional positional encodings**
- 2D sinusoidal PE for images: row and column frequencies concatenated
- RoPE for ViT: applying rotation to 2D patch coordinates
- LieRE: Lie algebra generalization of RoPE to N-dimensional spaces
- Video PE: extending RoPE to temporal dimension (T-RoPE)

---

### Chapter 40: Sparse and Linear Attention

**The quadratic attention problem**
- Self-attention cost: O(n^2) time and memory -- hard limit for very long sequences
- When quadratic cost matters: sequences > 8K tokens, DNA, audio, video, documents
- Approximate vs sparse vs linear: three strategies to break the O(n^2) barrier

**Sparse attention patterns**
- Local window attention: each token attends to ±w neighbors
- Strided sparse attention: attend to every s-th token for long-range coverage
- Random attention: random sparse connections for global information flow
- Combined: Longformer, BigBird -- local + global + random

**Longformer**
- Sliding window attention: each token attends to local window of size 2w
- Global attention for special tokens: [CLS], question tokens, memory tokens
- Efficient CUDA implementation: custom CUDA kernels for sliding window
- Applications: long document classification, QA over book-length documents

**BigBird**
- Combines: random attention + local window + global tokens
- Theoretical guarantee: BigBird is a universal approximator of full attention
- Applications: genomic sequence modeling, long document tasks

**Linear attention**
- Kernel trick: approximate softmax attention as a linear kernel product
- Performer (Choromanski et al.): FAVOR+ random feature map for softmax approximation
- Linformer: project K and V to fixed dimension r << n, O(nr) attention
- Limitations: loss of expressiveness, poor in-context learning compared to softmax attention

**Reformer**
- Locality Sensitive Hashing (LSH): hash Q and K, attend only within the same bucket
- Reversible residual connections: avoid storing all activations during backward
- O(n log n) attention, O(1) memory for residuals

---

### Chapter 41: Mixture of Experts (MoE)

**The MoE concept**
- Conditional computation: activate only a subset of parameters per input
- Expert: a feed-forward network with its own weights
- Router: a learned linear layer outputting expert weights per token
- Sparsely gated MoE: top-k routing, only k of N experts activated

**Switch Transformer (Fedus et al., 2022)**
- k=1 routing: simplest and most stable, each token goes to exactly one expert
- Load balancing loss: auxiliary loss to prevent all tokens routing to one expert
- Expert capacity factor: set max tokens per expert, drop excess tokens
- Scaling: Switch-C at 1.6T parameters, near-linear scaling efficiency

**Expert routing in practice**
- Top-2 routing: two experts per token, weighted combination
- Token dropping: when expert capacity exceeded, tokens are skipped -- quality impact
- Expert parallelism: each GPU holds a subset of experts, route tokens via all-to-all
- DeepSeek MoE innovation: fine-grained experts (smaller, more numerous) + shared experts

**Mixtral and GPT-style MoE**
- Mixtral 8x7B: 8 experts, 2 active, 46.7B total / 12.9B active parameters
- Mixtral 8x22B: 141B total / 39B active, near Llama 3 70B quality at 39B compute
- Deployment: only active parameters need to be in GPU memory for inference

**MoE training challenges**
- Expert collapse: one or few experts dominate training, others undertrained
- Load balancing loss: auxiliary loss weight tuning, z-loss for router stability
- Gradient sparsity: most experts receive no gradient for a given batch
- Communication overhead: all-to-all routing is expensive in distributed training

**MoE for multimodal and vision**
- MoE ViT: sparse experts in vision transformers for efficient large-scale training
- V-MoE: outperforms dense ViT at same compute, used in scaling studies
- LIMoE: language-image mixture of experts, unified multimodal routing

---

### Chapter 42: Parameter-Efficient Fine-Tuning

**Why PEFT?**
- Full fine-tuning cost: update all 7-70B+ parameters for each task
- Multi-tenant deployment: different LoRA adapters on same base model
- Catastrophic forgetting: full fine-tuning on small data destroys pre-training

**LoRA (Hu et al., 2022)**
- Low-rank decomposition: ΔW = BA, B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
- Applied to: W_Q, W_K, W_V, W_O (attention) and W_up, W_down (FFN)
- Rank r=8 to r=64: 10,000x fewer parameters than full fine-tuning
- Merging: A×B can be merged back into W at inference for zero overhead

**QLoRA (Dettmers et al., 2023)**
- 4-bit NF4 quantization: Normal Float 4, designed for normally distributed weights
- Double quantization: quantize the quantization constants themselves
- Paged optimizers: offload optimizer states to CPU when GPU memory overflows
- Democratization: fine-tune 65B model on single 48GB GPU

**LoRA variants**
- LoRA+: asymmetric learning rates for A and B matrices
- DoRA: weight decomposition into magnitude + direction, fine-tune direction with LoRA
- LoRA-FA: freeze A matrix, only learn B -- further reduced memory
- MoE-LoRA: multiple LoRA adapters as experts, route per token

**Adapter layers**
- Bottleneck adapters (Houlsby et al., 2019): small FFN inserted in transformer block
- Parallel adapters: added in parallel to sublayers, better gradient flow
- Adapter fusion: combine task-specific adapters without catastrophic forgetting

**Prefix and prompt tuning**
- Prefix tuning (Li and Liang, 2021): prepend learnable tokens to keys and values
- Prompt tuning (Lester et al., 2021): soft prompt in input embedding space only
- P-tuning v2: deep prompt tuning across all layers, competitive with fine-tuning

---

### Chapter 43: Quantization, Pruning and Distillation

**Post-Training Quantization (PTQ)**
- Weight-only quantization: quantize weights, dequantize on the fly for matmul
- GPTQ (Frantar et al., 2022): second-order weight update, layer-by-layer quantization
- AWQ (Lin et al., 2023): activation-aware, protect salient weights, scale-friendly
- SmoothQuant: smooth activation outliers into weights, enable W8A8 quantization

**Quantization formats**
- INT8: 8-bit signed integer, near lossless for most models
- INT4: 4-bit, aggressive, requires careful calibration
- NF4 (QLoRA): 4-bit normal float, optimal for normally distributed weights
- FP8: 8-bit float with exponent, hardware support on H100, emerging standard
- GGUF format: community quantization for CPU inference (llama.cpp)

**Structured pruning**
- Head pruning: remove attention heads with low importance scores
- FFN neuron pruning: identify and remove low-magnitude neurons
- Layer pruning: remove entire transformer blocks (ShortGPT, LLM-Pruner)
- Magnitude-based vs gradient-based importance scoring

**Knowledge distillation**
- Task-agnostic distillation: DistilBERT -- 6-layer student, BERT teacher, 97% performance
- Sequence-level KD: student matches teacher output distribution
- Layer mapping: student layer learns from corresponding teacher layer
- TinyLLaMA: 1.1B student from LLaMA 2 teacher, trained on 3T tokens

**Speculative decoding**
- Small draft model proposes k tokens, large model verifies in parallel
- Acceptance rate depends on draft-target model alignment
- Medusa: multiple parallel decoding heads on the same model
- Eagle: draft in feature space, higher acceptance rate

---

### Chapter 44: Distributed Training — Full Reference

*This chapter provides the definitive cross-domain reference for distributed training of transformers at scale.*

**Theoretical foundations**
- Parallelism strategies taxonomy: data, tensor, pipeline, sequence, expert
- Communication primitives: AllReduce, AllGather, ReduceScatter, AllToAll
- Bandwidth and latency: NVLink (600 GB/s) vs PCIe (64 GB/s) vs InfiniBand (800 Gb/s)
- Roofline model: when are you compute-bound vs memory-bound vs communication-bound?

**Data parallelism at scale**
- DDP (PyTorch): gradient all-reduce after backward, bucketed communication
- FSDP: shard parameters, gradients, optimizer states; gather only for forward/backward
- ZeRO-1/2/3: communication volume analysis for each stage
- ZeRO-Infinity: CPU and NVMe offloading for trillion-parameter models

**Tensor parallelism (Megatron-LM)**
- Column-parallel linear: split output features across GPUs
- Row-parallel linear: split input features across GPUs
- Attention head parallelism: split heads across tensor-parallel ranks
- All-reduce placement: minimal communication in critical path

**Pipeline parallelism**
- GPipe: F-then-B schedule, high memory (all micro-batch activations)
- PipeDream 1F1B: interleave forward and backward, memory-efficient
- Virtual pipeline stages: increase number of stages, reduce bubble fraction
- Pipeline bubble: fraction of time wasted at startup and shutdown

**Sequence parallelism**
- Ring attention: distribute FlashAttention over sequence dimension across GPUs
- Ulysses (DeepSpeed): all-to-all to parallelize attention heads × sequence chunks
- Combined: 1M+ token training feasible with sequence × tensor × data parallelism

**Expert parallelism (for MoE)**
- Each GPU holds a subset of expert weights
- All-to-all communication: route tokens to correct GPU for expert computation
- Token dropping vs overflow buffers: quality vs efficiency trade-off

**Optimization in distributed settings**
- Gradient accumulation over micro-batches: amortize communication
- Mixed precision: BF16 parameters, FP32 optimizer states
- Activation checkpointing + recomputation scheduling
- Optimizer sharding: ZeRO Stage 1 with Adam -- partition first/second moments

**Domain-specific distributed training notes**
- Vision (Part III): simpler -- large ViTs primarily need data + tensor parallelism
- Audio (Part IV): short sequences, memory less critical; batch size matters for contrastive
- Multimodal (Part V): separate encoder and LLM parallelism strategies can be combined
- Science (Part VII): small batch sizes common, limited GPU access -- ZeRO Stage 3 critical
- Post-Transformer (Part VIII): SSM scan parallelism is different from attention parallelism

---

### Chapter 45: Training Efficient and Sparse Models

**MoE-specific training**
- Expert initialization: importance of identical initialization to avoid early collapse
- Z-loss: regularization on router logits for stability (PaLM-style)
- Load balancing loss weight: too high → route instability, too low → expert collapse
- Expert gradient sparsity: most experts receive zero gradient per batch, implications for Adam

**PEFT training protocols**
- LoRA rank selection: practical guidance for different tasks and model sizes
- Learning rate sensitivity: LoRA typically needs 10x higher LR than full fine-tuning
- Training stability with QLoRA: gradient clipping, warm-up, NF4 outlier behavior
- Multi-adapter composition: LoRAHub, TIES-merging, model merging for multi-task

**Hardware-aware training kernels**
- Triton: writing custom GPU kernels in Python for non-standard operations
- FlashAttention integration: plug into standard torch.nn.MultiheadAttention
- SSM scan kernels (Mamba): parallel prefix scan implementation, CUDA-specific
- Low-bit matmul: INT8 GEMM with CUTLASS, FP8 with cuBLAS on H100

**Quantization-Aware Training (QAT)**
- Straight-through estimator: gradient through rounding during backward
- QAT vs PTQ trade-off: QAT recovers 1-2% quality but requires full training run
- LLM.int8(): mixed-precision matmul with 8-bit outlier handling
- FP8 pre-training (Nvidia): train from scratch in FP8, no quality loss vs BF16

---

### Chapter 46: Benchmarking Efficiency

**Training efficiency metrics**
- MFU (Model FLOP Utilization): actual FLOP/s / theoretical peak FLOP/s
- GPU utilization vs MFU: utilization can be high while MFU is low (memory-bound)
- Throughput: tokens/sec/GPU during training
- Memory per token: important for batch size and sequence length planning

**Inference efficiency metrics**
- TTFT (Time to First Token): prefill latency, scales with input length
- TPOT (Time per Output Token): decoding latency, depends on KV cache and batch size
- Throughput: tokens/sec for a given batch size
- Cost per million tokens: deployment cost comparison across model families

**The quality-efficiency Pareto frontier**
- Model size vs benchmark score: larger is generally better but diminishing returns
- Inference compute vs benchmark score: training-efficient models may be inference-inefficient
- Edge deployment benchmarks: latency on mobile, Raspberry Pi, laptop NPU
- MLPerf Inference: standardized benchmark across hardware and model configurations

**Long Range Arena (LRA)**
- 6 tasks: ListOps, byte-level text classification, retrieval, image classification, pathfinder, spatial tasks
- Designed to stress-test long-sequence modeling: sequences up to 16K tokens
- Standard evaluation for efficient attention methods and SSM alternatives
- Critique: LRA does not predict real NLP downstream task performance well

**Efficiency on specific hardware**
- A100 SXM vs A100 PCIe vs H100 vs RTX 4090: different compute/memory trade-offs
- Apple M-series (ANE, unified memory): different optimization target
- Qualcomm NPU (on-device): INT4 models only, strict latency constraints
- AMD MI300X: HBM3, competitive for inference

---

## Part VII: Transformers in Science

*Goal: Show how transformer architectures have penetrated biology, chemistry, climate science,
and medicine, with honest assessment of where they work and where they fail.*

---

### Chapter 47: Protein Structure Prediction

**The protein folding problem**
- Proteins are the machinery of life: enzymes, receptors, antibodies, motors
- Structure determines function: knowing the 3D shape predicts how a protein acts
- Historical methods: X-ray crystallography, cryo-EM, NMR -- expensive, slow
- The CASP challenge: biennial competition, measure AI against experimentally determined structures

**AlphaFold 2 (Jumper et al., 2021)**
- Multiple Sequence Alignment (MSA) as input: evolutionary information from related sequences
- Evoformer: alternating attention over MSA and pairwise residue distance matrix
- Structure module: predict 3D backbone + sidechain torsion angles from representations
- CASP14 results: TM-score > 0.9 for most targets, effectively solved for globular proteins
- Nobel Prize in Chemistry 2024: Hassabis and Jumper recognized

**AlphaFold 3 (Abramson et al., 2024)**
- Beyond single proteins: predict structure of proteins + DNA + RNA + ligands + ions jointly
- Diffusion module: replace structure module with diffusion network (DDPM-style)
- Evoformer retained: co-evolution of sequences still central to the architecture
- Accuracy: surpasses all existing docking tools on protein-ligand interaction
- AlphaFold Server: free academic access, 200M+ predicted structures in database

**ESMFold and language model approaches**
- ESM (Evolutionary Scale Modeling): protein language model pre-trained on sequences alone
- ESMFold: single forward pass from sequence to structure, no MSA required
- Trade-off: 60x faster than AlphaFold 2, slightly less accurate
- ESM-2: 15B parameter protein LM, emergent structural information

**RoseTTAFold All-Atom (RFAA)**
- All-atom: predict any biomolecule including small molecules, metal ions, modified residues
- Complementary to AlphaFold 3, fully open-source
- Applications: covalent binding prediction, post-translational modifications

**Applications to drug discovery**
- Virtual screening: evaluate protein-ligand binding with AlphaFold 3 scoring
- De novo drug design: use structure predictions to guide generative molecule design
- Isomorphic Labs: AlphaFold 3 integrated into real drug discovery campaigns
- Antibody design: predict antibody-antigen interfaces, design binders

---

### Chapter 48: Genomics and Molecular Biology

**DNA and RNA language models**
- DNABERT: BERT pre-trained on human genome with k-mer tokenization
- DNABERT-2: byte pair encoding for DNA, multi-species pre-training
- Nucleotide Transformer: 2.5B parameters, 850 species, benchmark across 18 downstream tasks
- HyenaDNA: sub-quadratic model for 1M base-pair context

**Evo: Genomic Foundation Model**
- 7B parameter model, 131K token context, trained on 2.7M prokaryotic sequences
- Single nucleotide tokenization: predict regulatory elements, functional sequences
- Multi-scale applications: molecular to genomic scale understanding
- Zero-shot fitness prediction: predict effect of mutations without task-specific labels

**RNA biology**
- RNA secondary structure prediction: transformer-based contact prediction
- RNA-FM: foundation model for RNA sequences, capture functional elements
- SpliceBERT: pre-training on pre-mRNA sequences, predict splicing sites

**Single-cell transcriptomics**
- scGPT: pre-trained on 33M human cells, cell type annotation, perturbation prediction
- Geneformer: 30M cells, transfer learning for gene network inference
- Cell as a sentence: genes as tokens, expression levels as a continuous signal

---

### Chapter 49: Drug Discovery and Chemistry

**Molecular transformers**
- SMILES representation: linearize molecular graph as a string, apply BERT pre-training
- MolBERT, ChemBERTa: SMILES-based pre-training on PubChem, downstream property prediction
- SELFIES: self-referencing embedded strings, always valid molecular representation

**Graph-based molecular models**
- Molecular graphs: atoms as nodes, bonds as edges
- Graphormer: transformer on molecular graphs, global and local attention, 2D/3D encodings
- SE(3) equivariant attention: 3D-structure-aware attention respecting rotational symmetry

**Generative drug design**
- ChemGPT: autoregressive generation of SMILES strings
- Transformer-based VAE: encode molecule to latent, decode to property-optimized structure
- AlphaFold 3 conditioned generation: generate ligands that fit a target pocket
- PCMol: multi-target transformer using AlphaFold2 protein embeddings as conditioning

**Reaction prediction and retrosynthesis**
- Molecular Transformer (Schwaller et al., 2019): seq2seq for predicting reaction products
- Retrosynthesis: given product, predict starting materials -- crucial for synthesis planning
- IBM RXN for Chemistry: cloud-based reaction prediction with transformer backend

---

### Chapter 50: Physics, Climate and Scientific Simulation

**Weather and climate forecasting**
- FourCastNet (Pathak et al., 2022): Fourier Neural Operator + attention, global weather at 25km
- Pangu-Weather (Bi et al., 2023): 3D Earth Transformer, hierarchical spatiotemporal attention
- GraphCast (Lam et al., 2023): graph neural network + attention, 10-day forecast in <1 minute
- Performance vs ECMWF: GraphCast and Pangu-Weather now match operational forecast center on many metrics

**Neural PDE solvers**
- Transformer as a PDE solver: token = discretization point, attention = influence kernel
- FNO (Fourier Neural Operator): spectral mixing via FFT, universal approximator for PDEs
- Neural operator theory: operator learning vs function learning
- Applications: fluid simulation, electromagnetics, material mechanics

**Physics-informed transformers**
- PINN with transformer backbone: penalize physics equation residuals during training
- Symmetry-equivariant transformers: build physical symmetries (rotation, translation) into architecture
- Applications: particle physics, astronomy, molecular dynamics

**Scientific simulation and AI for science**
- Cosmological simulations: transformer models for dark matter halo finding
- Materials science: crystal structure prediction, transformer on atomic positions
- Tokamak plasma control: transformer for real-time control of fusion reactors (DeepMind, 2022)

---

### Chapter 51: Medical Imaging and Clinical AI

**Medical vision transformers**
- TransFuse: parallel transformer + CNN branches, fusion for medical image segmentation
- SwinUNet: pure Swin Transformer encoder-decoder for 2D medical segmentation
- nnFormer: 3D volumetric segmentation with interleaved local and global attention

**Foundation models for medical imaging**
- SAM applied to medical images: zero-shot segmentation with point/box prompts
- MedSAM: fine-tuned SAM on 1.5M medical image-mask pairs, 10 imaging modalities
- SAM 2 for medical video: surgical video segmentation, tracking
- CONCH: pathology foundation model, slide-level reasoning, survival prediction

**Radiology report generation**
- CheXpert and MIMIC-CXR: chest X-ray datasets with expert-written reports
- Vision-language models for radiology: CheXagent, Med-PaLM M
- Temporal change detection: attend to prior study for interval comparison
- Structured report generation: findings, impressions, recommendations

**Electronic Health Records (EHR) transformers**
- Clinical BERT: pre-trained on MIMIC-III clinical notes
- BEHRT: patient timeline as a sequence, disease trajectory prediction
- Transformer for survival analysis: time-varying clinical covariates as tokens
- Privacy considerations: differential privacy in medical transformer training

**Drug-disease interaction**
- DRKG: drug repurposing knowledge graph with transformer embeddings
- BioMedBERT: PubMed pre-training, biomedical NLP
- Clinical trial outcome prediction: transformer on inclusion/exclusion criteria + trial metadata

---

### Chapter 52: Training Scientific Foundation Models

**Domain-specific tokenization**
- Molecular SMILES tokenization: atom-level, k-mer BPE, SELFIES variants
- DNA tokenization: single nucleotide, k-mer (3, 6, 8), byte pair encoding
- Protein tokenization: amino acid vocabulary of 20 + special tokens
- Spectrogram tokenization for scientific signals: EEG, seismic, spectroscopy

**Pre-training on scientific data**
- MSA as input: encoding evolutionary information, how to handle gaps and insertions
- Multi-species pre-training: species as a conditioning variable or as separate vocabularies
- Small dataset challenge: genomics has millions not billions of samples vs NLP trillions
- Data augmentation for biology: reverse complement for DNA, rotational augmentation for molecules

**Transfer learning for scientific tasks**
- Frozen vs fine-tuned backbone: limited labeled data often favors frozen features
- Low-resource fine-tuning: LoRA and adapters for scientific domain adaptation
- Multi-task pre-training: predict 50+ molecular properties simultaneously
- Active learning loops: use model uncertainty to select next wet-lab experiments

**Regulatory and reproducibility considerations**
- FDA guidance on AI in drug development: model documentation requirements
- Reproducibility: molecular benchmark data leakage, scaffold splits vs random splits
- Wet-lab validation gap: in silico accuracy does not guarantee experimental success

---

### Chapter 53: Evaluating Scientific Transformers

**Protein structure evaluation**
- TM-score: template modeling score, 0-1, values >0.5 indicate same fold
- GDT-TS: global distance test, fraction of residues within 1/2/4/8 Angstrom
- RMSD: root mean square deviation, sensitive to outliers
- CASP metrics: official CASP15 GDT-HA and TM-score distributions across targets
- Confidence calibration: pLDDT score accuracy, when to trust AlphaFold predictions

**Molecular property prediction benchmarks**
- MoleculeNet: 17 datasets across physicochemical, biophysical, biological, physiological
- Scaffold split vs random split: scaffold split tests generalization to novel chemical scaffolds
- OGB (Open Graph Benchmark): HIV, PCBA, MUV datasets with standardized splits

**Genomics benchmarks**
- Genomics Long Range Benchmark (GenomicsBench): downstream tasks at various scales
- Nucleotide Transformer benchmark: 18 tasks including histone modification, promoter prediction
- CADD score correlation: compare to established variant effect predictors

**Weather forecast evaluation**
- RMSE and ACC (Anomaly Correlation Coefficient) on pressure levels
- Tropical cyclone track error
- Precipitation categorical scores: CSI, ETS
- Comparison to ECMWF IFS and other operational NWP models

**Clinical AI evaluation**
- AUC-ROC for disease prediction tasks
- Clinical NLP benchmarks: MedQA (USMLE), PubMedQA, MedMCQA
- External validation: performance on held-out hospitals, different demographics
- Clinical utility studies: does the model actually help clinicians? Prospective trial design

---

## Part VIII: Beyond Transformers and Frontiers

*Goal: Cover post-transformer architectures (SSMs, RWKV, hybrids) and the frontier research areas
of reasoning, agents, mechanistic interpretability, and theoretical foundations.*

---

### Chapter 54: Structured State Space Models (SSMs)

**State space models from control theory**
- Linear time-invariant (LTI) system: dx/dt = Ax + Bu, y = Cx + Du
- Discretization: bilinear (Tustin) method, converting continuous-time to discrete-time
- SSM as a convolution: SSMs are equivalent to long convolutions, enable parallel training
- SSM as a recurrence: efficient O(1) inference per token via hidden state

**S4 (Gu et al., 2022)**
- HIPPO matrices: high-order polynomial projection operators, designed for long-range memory
- Diagonalization: make A diagonal + low-rank for efficient computation
- S4 = single long convolution: compute via FFT in O(n log n)
- State-of-the-art on Long Range Arena (pathfinder, ListOps, sequential CIFAR)

**S4 variants**
- DSS (Diagonal State Spaces): fully diagonal A, simpler and fast
- S5: multi-input multi-output SSM with parallel scan
- H3: SSM layer designed to match attention on language modeling

**Hyena**
- Sub-quadratic attention replacement: long convolution + element-wise multiplication
- Hyena hierarchy: stacking sub-quadratic operators
- Strong language modeling perplexity with no attention

**The parallel scan algorithm**
- Blelloch scan: O(log n) depth parallel prefix computation
- Key to SSM training efficiency: converts sequential recurrence to O(n log n) parallel
- Hardware implementation: requires custom CUDA kernel, Mamba relies on this

---

### Chapter 55: Mamba and Selective SSMs

**Limitation of fixed SSMs**
- S4 uses fixed A, B, C matrices: same transition for every input, no input-dependent selection
- Selective copying task: S4 fails because it cannot focus on relevant inputs
- Induction heads: transformers handle this naturally via content-based attention

**Mamba (Gu and Dao, 2023)**
- Selective SSM: make B, C, and Δ (timescale) functions of the input
- Δ controls how much to update the hidden state vs maintain it
- Selective filtering: Mamba can choose to ignore or focus on specific tokens
- Hardware-aware algorithm: recompute SSM state in SRAM during backward, no HBM materialization
- Results: outperforms Transformer++ at 3B parameters on language modeling

**Mamba architecture**
- Mamba block: linear projection → SSM + conv → output gate → linear projection
- No attention: fully recurrent at inference, fully parallel at training
- H dimension splits into two paths: one through SSM, one through skip gate
- Comparison: similar structure to gated MLP, replaces FFN in transformer

**Mamba-2 (Dao and Gu, 2024)**
- State Space Duality: show SSD (Structured State Space Duality) unifies attention and SSM
- Mamba-2 = attention with head dimension 1, special masking, and state compression
- SSD can be computed as matrix multiplication, leverages tensor cores efficiently
- Mamba-2 is 2-8x faster than Mamba-1 on training

**Mamba applications beyond language**
- Vision Mamba: ViM, applying Mamba to image patches with bidirectional scan
- MedMamba, VideoMamba: domain-specific applications
- Genomics: Mamba for very long DNA sequences (chromosome scale)
- Audio: Mamba for raw waveform, strong on long audio classification

**Theoretical analysis**
- Expressivity: Mamba in TC^0 complexity class, same as constant-depth transformers
- Copy task limitations: Mamba cannot perfectly copy arbitrary long sequences
- In-context learning: Mamba can learn linear functions in-context, weaker than Transformer

---

### Chapter 56: RWKV and RetNet — Recurrent Rivals

**RWKV (Peng et al., 2023)**
- Motivation: combine RNN efficiency at inference with Transformer parallelism at training
- Architecture: Time Mixing (token mixing) + Channel Mixing (feed-forward analogue)
- Time Mixing: WKV operator: attention-like but with exponential decay, parallel training formulation
- Channel Mixing: gated FFN with sigmoid-based routing
- Inference: O(1) per token, O(d) state size -- true recurrence at inference

**RWKV training formulation**
- Token shift: one-step delay provides prior-token information without attention
- WKV kernel: computed in parallel via cumulative sum, O(n) total
- No positional encoding needed: temporal decay encodes position implicitly

**RWKV versions**
- RWKV v4 (2023): first language-competitive recurrent model released publicly
- RWKV v5 Eagle: improved training recipe, beats Mamba on many benchmarks
- RWKV v6 Finch / GoldFinch: data-dependent decay, improved expressivity
- RWKV v7: attention-like data-dependent state transition, strongest RWKV yet
- Deployment: 1.5B Windows Copilot devices run RWKV v5 via Microsoft

**RetNet (Sun et al., 2023)**
- Retention mechanism: like attention but with exponential decay applied to relative positions
- Three computation modes: parallel (training), recurrent (inference), chunk-wise (long context)
- Retention = attention with a special relative position decay matrix D
- Theoretical connection: retention is a special case of linear attention with decay

**xLSTM (Beck et al., 2024)**
- Extended LSTM: sLSTM (scalar memory, new gating) + mLSTM (matrix memory cell)
- mLSTM: matrix-valued memory for higher storage capacity, covariance update rule
- Parallelizable: unlike classical LSTM, xLSTM supports efficient GPU training
- xLSTM-7B: 3.5x faster training throughput than comparable Transformer

---

### Chapter 57: Hybrid Architectures

**Why hybrids?**
- Pure SSMs: great efficiency, weaker in-context learning than Transformers
- Pure Transformers: strong everywhere but quadratic cost
- Hybrids: use attention where it helps most (few layers), SSM everywhere else

**Jamba (Lenz et al., 2024)**
- Mamba + Transformer + MoE layers: striped hybrid
- Pattern: 1 attention layer per 7 Mamba layers
- Competitive with Llama 2 70B on reasoning, much faster on long contexts
- Jamba 1.5: production model, 256K context, 52B active / 398B total

**Griffin and RecurrentGemma (De et al., 2024)**
- RG-LRU: real-gated linear recurrent unit, efficient recurrence
- Griffin block: 2 recurrent layers + 1 local attention layer (Hawk blocks)
- RecurrentGemma: production-ready Griffin variant from Google DeepMind
- Griffin-14B: competitive with Llama 2 despite using less training data

**Samba (Ren et al., 2025)**
- Sliding window attention + Mamba: combines local context with selective state
- Outperforms pure Mamba and pure sliding window attention
- Strong at 1.3B-3.8B scale on language modeling and downstream tasks

**Titans (Behrouz et al., 2024)**
- Three memory types: short-term (attention), long-term (neural memory), persistent (parameters)
- Meta in-context learning: store surprising information in neural memory at test time
- Combines all three in one forward pass via gating

**Theoretical expressivity of hybrids**
- Mixing attention and recurrence: complement weaknesses of each
- Striped vs fusion patterns: attention every N layers vs interspersed within layers
- Edge model results: Samba and RWKV7 outperform Llama 3.2 and Qwen2.5 at 1.5B
- Frontier conclusion: full attention still dominates at 70B+, hybrids competitive below 14B

---

### Chapter 58: Reasoning, Agents and Test-Time Compute

**Chain-of-thought reasoning**
- Wei et al. (2022): chain-of-thought prompting with few-shot exemplars
- Zero-shot CoT: "Let's think step by step" -- emergent at ~100B parameters
- Least-to-most prompting: decompose problem, solve sub-problems sequentially
- Tree of Thoughts: explore multiple reasoning branches, backtrack, evaluate

**Reasoning models: OpenAI o1/o3 and DeepSeek-R1**
- O1 (OpenAI, 2024): extended "thinking" before answering, trained with RL
- Test-time compute scaling: spending more inference compute improves math/coding/science
- DeepSeek-R1: open-weight reasoning model, RL from scratch with verifiable rewards
- GRPO (Group Relative Policy Optimization): DeepSeek's RL algorithm, group-based reward
- Process Reward Models (PRM): reward correct intermediate steps, not just final answer

**Tool use and agentic transformers**
- ReAct: interleave reasoning (Thought) and acting (Action/Observation) in context
- Toolformer: self-supervised API call insertion during generation
- Function calling in GPT-4: structured JSON tool invocation
- Code interpreter: Python execution as a tool, arithmetic and data analysis

**Multi-agent transformer systems**
- AutoGPT, BabyAGI: early autonomous agent loops
- Multi-agent debate: two transformer instances debate to improve factual accuracy
- Mixture of Agents (MoA): aggregate outputs of multiple LLMs as a "virtual large model"
- Society of Mind: specialized agents with memory + orchestrator transformer

**Memory for agents**
- Episodic memory: store past experiences, retrieve relevant ones
- Semantic memory: knowledge base, RAG-augmented fact retrieval
- Procedural memory: fine-tuned weights encoding repeated behaviors
- Mem0 and memory infrastructure: production memory management for AI agents

---

### Chapter 59: Training Post-Transformer Architectures

**SSM initialization**
- HIPPO initialization for A matrix: critical for long-range memory in S4
- Mamba A initialization: diagonal negative real values for stable decay
- Why standard random initialization fails for SSMs: state matrix must encode memory structure

**Mamba-specific training considerations**
- Selective scan: custom CUDA kernel required for efficient training
- Gradient through scan: parallel prefix sum backward, no BPTT-style truncation
- Hardware: Mamba-1 CUDA kernel, Mamba-2 tensor-core-friendly GEMM formulation
- Memory: SSM state is O(d_state × d_model) per layer, much less than KV cache

**Converting Transformers to RWKV/Mamba**
- RWKV from Transformer: retain weights, replace attention with RWKV operators
- MambaFormer conversion: train adapter to bridge attention and SSM representations
- Practical motivation: reuse vast pre-trained Transformer knowledge for recurrent inference

**Hybrid training strategy**
- Which layers get attention, which get SSM: empirical vs principled choices
- Layer-wise learning rates for hybrid: attention layers may need different LR from SSM layers
- Mixing ratio ablation: 1:7 attention:Mamba studied in Jamba, domain-dependent optimum

**Reasoning model training**
- Verifiable reward signals: math problems have ground truth, code has test suites
- GRPO: group sampling, normalize reward within group, no value network needed
- Cold-start vs warm-start RL: start from SFT model vs start from base model directly
- Format rewards: encouraging proper thinking format during early RL training

---

### Chapter 60: Benchmarking Transformers vs Alternatives

**Language modeling perplexity comparisons**
- Standard: Pile, WikiText-103, LAMBADA
- At equal parameter count: Mamba matches Transformer at 1-3B
- At equal compute: SSMs often win on byte-level modeling (no tokenizer advantage)
- Frontier scale (70B+): full attention still leads all alternatives as of 2025

**In-context learning stress tests**
- Recall task (MQAR): multi-query associative recall, requires exact lookup
- Mamba ICL weakness: struggles with tasks requiring arbitrary associative memory
- RWKV ICL: weaker than Transformer on few-shot tasks, especially multi-hop
- Transformer advantage: attention heads implement induction via direct QK interaction

**Long-context benchmarks**
- RULER: synthetic tasks at 4K to 128K context
- NIAH (Needle in a Haystack): recall specific fact buried in long context
- SSM state compression limits: fixed hidden state cannot recall all facts from 128K context
- Mamba + attention hybrid: retrieval head attention layers restore long-context recall

**Copy and CoT tasks**
- Copy task: repeat arbitrary sequence -- Mamba fails beyond state dimension
- Selective copying: Mamba handles (its design target), Transformer also handles
- Chain-of-thought reasoning: Transformer > Mamba because scratchpad = external memory
- Mamba reasoning limitation: single forward pass, no token-by-token scratchpad

**Edge and on-device benchmarks**
- Latency at batch size 1: SSMs win significantly (constant state, no KV growth)
- RWKV-5 on Windows Copilot: 1.5B tokens/sec on Qualcomm NPU
- Mamba vs Transformer inference throughput: 5x at 128K context length
- Practical deployment: hybrids require both CUDA attention and SSM kernels -- deployment complexity

---

### Chapter 61: Open Problems and Research Frontiers

**Mechanistic interpretability**
- Circuits: identify specific computational circuits implementing tasks (indirect object identification)
- Superposition: how transformers store more features than dimensions
- Sparse autoencoders (SAEs): decompose transformer activations into interpretable features
- Anthropic's dictionary learning: finding monosemantic neurons via SAE
- Open question: can we fully reverse-engineer a frontier model?

**World models and physical reasoning**
- Does GPT-4 have a world model? Board state tracking experiments (Othello-GPT)
- Physical intuition: transformers fail on novel physics scenarios requiring simulation
- V-JEPA: learn predictive representations of video without explicit reconstruction
- GAIA-1 and world model for driving: simulation-based planning from transformer predictions
- Open question: is autoregressive prediction sufficient to learn a world model?

**Continual learning**
- Catastrophic forgetting: fine-tuning on new tasks erases old knowledge
- Replay methods, EWC, PackNet: mitigating forgetting in transformers
- Lifelong learning benchmarks: SEALQA, CLiMB
- In-context continual learning: using context window as short-term adaptation mechanism

**Sample efficiency**
- LLMs require 10^12 tokens; humans learn from 10^9 words
- Symbol binding: transformers may learn statistical correlations not causal structure
- Grounded language learning: embodied agents, language from perception
- Open question: is scale the right axis, or do we need architectural inductive biases?

**Theoretical foundations**
- Circuit complexity: Transformers are in TC^0 -- cannot express all polynomial-time algorithms
- Looping and CoT: chain-of-thought expands effective depth, bypasses TC^0 limitations
- In-context learning theory: Transformers implement gradient descent in their forward pass (Akyürek et al.)
- Length generalization: why transformers fail to generalize to longer sequences, ICLR 2024 findings

**Sustainability and compute efficiency**
- Training carbon footprint: GPT-3 ≈ 500 tons CO2, GPT-4 estimated 10-100x more
- The efficiency imperative: do more with less, Chinchilla-optimal training, PEFT
- Open models vs closed: open science vs safety considerations
- Small language models (SLMs): Phi-2, Phi-3, Gemma 2 2B -- strong quality at 2-4B
- The future of architecture search: will transformers be replaced or will they continue to dominate?

---

## Appendices

### Appendix A: Mathematical Reference
- Matrix calculus for attention: Jacobians, the chain rule for QKV
- Einsum notation: a unified language for tensor operations
- Complexity cheatsheet: time and memory for every operation discussed in the book

### Appendix B: Key Papers Timeline
- 2017-2019: The foundation years (Transformer, BERT, GPT-1/2)
- 2020-2021: The scale era (GPT-3, T5, ViT, DALL-E, AlphaFold 2)
- 2022: The efficiency and multimodal year (FlashAttention, CLIP-based VLMs, Whisper, RLHF)
- 2023: Instruction tuning and open source (LLaMA, Mistral, LLaVA, DPO, Mamba)
- 2024: Reasoning and omni models (AlphaFold 3, GPT-4o, Sora, RWKV v6, Jamba, DeepSeek V2)
- 2025: Agents and post-transformer consolidation (DeepSeek-R1, Mamba-2, RWKV v7, hybrid frontier)

### Appendix C: Benchmark Reference Table
- Full list of every benchmark mentioned in the book with modality, metric, and citation

### Appendix D: Notation Reference
- Consistent notation used throughout: d_model, d_k, h, N, T, vocab_size, batch conventions

### Appendix E: Claude Code Project Structure
- Recommended directory layout for the HTML book
- Slash command reference: /new-chapter, /add-diagram, /build-toc
- Diagram taxonomy: which chapter uses which diagram type

---

*End of Table of Contents*

> Total chapters: 61 core chapters + 5 appendices
> Training chapters: Part I Ch.6, Part II Ch.14, Part III Ch.22, Part IV Ch.28, Part V Ch.35,
>   Part VI Ch.44+45, Part VII Ch.52, Part VIII Ch.59 — 10 dedicated training chapters
> Evaluation chapters: Part II Ch.15+16, Part III Ch.23, Part IV Ch.29, Part V Ch.36,
>   Part VI Ch.46, Part VII Ch.53, Part VIII Ch.60 — 9 dedicated evaluation chapters
> New vs v1: Video generation (Ch.21), FlashAttention dedicated chapter (Ch.37),
>   KV Cache dedicated chapter (Ch.38), Distributed Training full reference (Ch.44),
>   Retrieval and Ranking (Ch.13), Retrieval Evaluation (Ch.16)
