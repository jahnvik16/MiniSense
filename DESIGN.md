# MiniSense — Architecture & Design Document

## 1. Architectural Overview

MiniSense implements a two-level multi-agent architecture designed to answer nuanced business questions over customer feedback data:

```
                  +-----------------------------------+
                  |        Business Question          |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |        Orchestrator / Planner     |
                  +-----------------------------------+
                     |              |              |
           TaskSpec  |     TaskSpec |     TaskSpec |
                     v              v              v
           +-------------+  +-------------+  +-------------------+
           |  DataAgent  |  |  RAGAgent   |  |  ComparisonAgent  |
           +-------------+  +-------------+  +-------------------+
                  |                 |                  |
           DataAgentOutput    RAGAgentOutput    ComparisonOutput
                  \                 |                 /
                   \                |                /
                    v               v               v
                  +-----------------------------------+
                  |         Final Synthesis           |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |   Coherent Business Explanation   |
                  +-----------------------------------+
```

### Core Design Principles
1. **Separation of Reasoning and Computation**: Large Language Models handle semantic comprehension, query planning, and final synthesis. All arithmetic (CSAT, NPS, percentage deltas, cohort counts) is delegated to deterministic Python functions.
2. **Strict Typed Contracts**: All communication between agents is enforced using Pydantic models (`TaskSpec`, `DataAgentOutput`, `RAGAgentOutput`, `ComparisonAgentOutput`). Sub-agents never emit unstructured free-form text.
3. **Document-Grounded RAG**: Explanations for customer complaints or feedback are strictly grounded in retrieved product policies and FAQs, preventing hallucinations.

---

## 2. Agent Roles & Contracts

- **Orchestrator / Planner**:
  - Parses incoming natural language questions.
  - Decomposes high-level inquiries into one or more `TaskSpec`s containing target agent enum, instruction, and deterministic filter arguments.
- **DataAgent**:
  - Executes analytical calculations against survey records.
  - Invokes pure functions (`compute_csat`, `compute_nps`, `compute_sentiment_breakdown`, `filter_surveys`).
- **RAGAgent**:
  - Executes semantic search against indexed FAQ documentation.
  - Returns ranked `DocumentChunk` items with source tags and relevance scores.
- **ComparisonAgent**:
  - Coordinates multi-cohort metric queries via `DataAgent`.
  - Determines statistical deltas and identifies leadership segments.

---

## 3. Fine-Tuning Design Proposal: Domain Adaptation for 10,000 Surveys/Day

### 3.1 Data Strategy & Curation
To fine-tune an efficient model capable of handling 10,000 survey responses/day, we establish a curated dataset balancing human expert verification with teacher-model distillation:
- **High-Quality Labels**: Annotations combine domain expert labels (20% sample audited by senior analysts) and frontier-model teacher labels (80% generated with GPT-4o / Claude 3.5 Sonnet using few-shot rubric verification).
- **Class Balance**: Stratified sampling ensures balanced representation across all 8 feedback themes (Wait Time, Food Quality, Pricing, Staff, Cleanliness, Membership, Facilities, App Experience) and sentiment polarities (positive, neutral, negative/complaint).
- **Train/Validation/Test Separation**: The dataset is partitioned into strict 70% train / 15% validation / 15% test splits, stratified by theme, rating score (1–5), and collection period to eliminate temporal and topic leakage.

### 3.2 Dataset Sizing
- **Target Size**: **6,000 to 8,000 high-quality labeled examples** (~750–1,000 per theme).
- At 10,000 daily incoming surveys, this sample size captures long-tail complaint vernacular, slang, multi-theme nuances, and strict JSON output schemas while avoiding overfitting and keeping annotation budgets manageable.

### 3.3 Base Model Selection & Rationale
- **Selected Model**: **Llama-3.1-8B-Instruct** (alternative: **Mistral-7B-Instruct-v0.3**).
- **Rationale**: 8B parameter models offer state-of-the-art reasoning, robust structured JSON adherence, and open Apache-2.0 / community licenses. Crucially, 8B models fit entirely within single-GPU VRAM (16GB–24GB on NVIDIA L4 or A10G), enabling high-throughput inference at sub-second latencies.

### 3.4 Fine-Tuning Technique: LoRA vs. QLoRA vs. Full Fine-Tuning
- **Selected Technique**: **16-bit LoRA (Low-Rank Adaptation)** with rank $r=16$, $\alpha=32$, and dropout $0.05$.
- **Target Modules**: Applied to all linear attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Comparison & Trade-offs**:
  - *Full Fine-Tuning*: Rejected due to prohibitive compute requirements ($>8\times$ GPU hours), risk of catastrophic forgetting of general reasoning, and inability to dynamically swap task-specific adapter weights.
  - *QLoRA (4-bit NF4)*: Excellent for low-memory training environments, but introduces dequantization latency overhead and subtle accuracy degradation during low-latency batch serving.
  - *16-bit LoRA*: Delivers the optimal equilibrium of fast training convergence, full fp16/bf16 serving performance on single GPUs, and modular adapter portability (~65MB artifact size).

### 3.5 Tooling & Infrastructure
- **Frameworks**: Hugging Face **TRL (`SFTTrainer`)**, **PEFT**, and **PyTorch**.
- **Acceleration**: FlashAttention-2 and DeepSpeed ZeRO-2 for efficient gradient distribution.
- **Experiment Tracking**: MLflow / Weights & Biases for hyperparameter logging, loss curves, and artifact versioning.

### 3.6 Training Pipeline
1. **Schema Validation**: Raw survey pairs are validated against Pydantic schema contracts.
2. **Loss Masking**: User prompt and context tokens are masked (`label = -100`); cross-entropy loss is computed exclusively on target response tokens (structured JSON + synthesized rationale).
3. **Training Execution**: 3 epochs, cosine learning rate decay with initial LR $2 \times 10^{-4}$, warmup ratio $0.03$, and gradient clipping at $1.0$.
4. **Validation & Checkpointing**: Automated checkpoint evaluation on validation splits every 250 steps.
5. **Artifact Registration**: Export and sign versioned LoRA adapters to an internal model registry.

### 3.7 Evaluation Metrics & Production Readiness Criteria
Before promoting any adapter to staging or production, it must satisfy clear gating thresholds:
- **Classification Performance**: Macro-F1 $\ge 0.90$ and per-class F1 $\ge 0.85$ across all 8 feedback themes and sentiment classes.
- **Frontier Model Parity**: Win/tie rate $\ge 92\%$ against frontier models (GPT-4o) in blinded LLM-as-a-judge evaluations assessing coherence and accuracy.
- **Zero Math Hallucination**: 100% adherence to numerical values supplied by deterministic data tools (strict string/numerical matching against input tool payloads).
- **Schema Reliability**: 100% valid Pydantic JSON parsing over 2,000 held-out test scenarios.
- **Latency & Cost SLA**: p99 inference latency $< 350\text{ms}$ on an NVIDIA L4 GPU; serving cost $<\$1.20$ per 10,000 responses (compared to $\approx \$25.00$–$\$40.00$ with frontier APIs).

### 3.8 Serving Architecture, Canary Rollout & Dual-Model Operation
- **Runtime Engine**: **vLLM** with multi-LoRA serving enabled, hosting the Llama-3.1-8B base model alongside versioned adapters.
- **Shadow Deployment**: The fine-tuned adapter runs concurrently with the frontier LLM on 100% of live traffic for 7 days. Outputs are asynchronously compared for schema fidelity and semantic alignment without affecting user-facing responses.
- **Canary Rollout**: Traffic progressively migrates: 5% $\rightarrow$ 25% $\rightarrow$ 100% over two weeks.
- **Fallback Circuit Breaker**: If output schema parsing fails or adapter confidence score drops below 0.80, the request automatically falls back to the frontier model (Gemini / GPT-4o).
- **Adapter Versioning**: Semantic adapter tags (e.g., `survey-adapter:v1.2.0`) allow instant rolling deploys and zero-downtime rollbacks via routing configuration.

### 3.9 Future-Proofing Input/Output Contracts
To prevent model lock-in and decouple LLM weights from orchestrator logic:
- All agent interactions remain strictly governed by immutable Pydantic contracts (`TaskSpec`, `DataAgentResult`, `RAGAgentResult`, `FinalAnswer`).
- The fine-tuned model is trained to output JSON strictly conforming to these Pydantic schemas.
- If upstream orchestrator logic or downstream tools evolve, contract schemas use semantic versioning (`schema_version="2.0"`). New adapters are trained against the versioned contract, allowing heterogeneous agent versions to coexist safely in production.
