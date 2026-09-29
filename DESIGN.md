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
1. **Separation of Reasoning and Computation**: Foundational language models handle semantic comprehension and final narrative synthesis. All arithmetic (CSAT, percentages, response counts, cohort deltas) is delegated to deterministic Python functions.
2. **Deterministic Intent Decomposition in Planning**: Rather than making an external LLM call to decide sub-agent dispatching, the Orchestrator applies deterministic pattern-and-intent decomposition. This ensures zero planning latency overhead, eliminates routing hallucinations, guarantees 100% test reproducibility, and avoids redundant token expenditure.
3. **Strict Typed Contracts**: All communication between agents is enforced using immutable Pydantic models (`TaskSpec`, `DataAgentResult`, `RAGAgentResult`, `ComparisonAgentResult`, `FinalAnswer`). Sub-agents never emit unstructured free-form text.
4. **Document-Grounded RAG**: Explanations for customer complaints or feedback are strictly grounded in retrieved product policies and FAQs, preventing hallucinations.

---

## 2. Agent Roles & Contracts

- **Orchestrator / Planner**:
  - Parses incoming natural language questions via deterministic intent classification.
  - Decomposes high-level inquiries into one or more `TaskSpec`s containing target agent enum, instruction, and deterministic filter arguments.
- **DataAgent**:
  - Executes analytical calculations against survey records.
  - Invokes pure functions (`compute_csat`, `compute_average_rating`, `count_responses`, `get_top_themes`, `filter_by_date`).
- **RAGAgent**:
  - Executes semantic search against indexed GreenLeaf Bistro FAQ documentation.
  - Returns ranked `DocumentChunk` items with source tags and relevance scores.
- **ComparisonAgent**:
  - Coordinates temporal or cohort comparison queries.
  - Computes exact deterministic deltas with zero-denominator safety.

---

## 3. Fine-Tuning Design Proposal: Domain Adaptation for 10,000 Surveys/Day

### 3.1 Data Strategy & Curation
To adapt an efficient open-weights model to 10,000 daily surveys, we construct a curated training set balancing domain expert audits (20%) with frontier-model teacher distillation (80% generated with GPT-4o / Claude 3.5 Sonnet using few-shot rubric validation). Sampling maintains strict class balance across all 8 feedback themes and sentiment polarities. Data is partitioned into a strict 70% train / 15% validation / 15% test split, stratified by theme, rating score (1–5), and date to eliminate temporal and topic leakage.

### 3.2 Dataset Sizing
The target dataset comprises **6,000 to 8,000 high-quality labeled examples** (~750–1,000 per theme). For a daily volume of 10,000 surveys, this volume adequately covers long-tail complaint expressions and multi-theme feedback while remaining cost-effective to annotate and audit.

### 3.3 Base Model Selection & Rationale
**Llama-3.1-8B-Instruct** (or **Mistral-7B-Instruct-v0.3**). 8B parameter models offer strong instruction following, reliable structured JSON compliance, and permissive community or Apache-2.0 licenses. Critically, 8B models fit within 16GB–24GB VRAM (single NVIDIA L4 or A10G GPU), enabling cost-effective serving at low latencies.

### 3.4 Fine-Tuning Technique: LoRA vs. QLoRA vs. Full Fine-Tuning
We adopt **16-bit LoRA (Low-Rank Adaptation)** with rank $r=16$, $\alpha=32$, and dropout $0.05$ across attention and MLP linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Full fine-tuning is rejected due to excessive GPU compute costs ($>8\times$), risk of catastrophic forgetting, and inability to dynamically hot-swap adapters. QLoRA (4-bit NF4) is suitable for low-VRAM training but introduces dequantization latency penalties during low-latency serving. 16-bit LoRA provides fast training convergence, full fp16 serving throughput, and modular adapter weights (~65MB).

### 3.5 Tooling & Pipeline
Implemented with Hugging Face **TRL (`SFTTrainer`)**, **PEFT**, and **PyTorch**, accelerated by FlashAttention-2. Experiment metrics and loss curves are tracked via MLflow.
*Pipeline*: Schema validation $\rightarrow$ Tokenization with response-only loss masking (masking prompt tokens to evaluate cross-entropy exclusively on completion JSON) $\rightarrow$ 3-epoch training with cosine learning rate decay ($2 \times 10^{-4}$) $\rightarrow$ Validation evaluation every 250 steps $\rightarrow$ Signed adapter export to an internal artifact registry.

### 3.6 Evaluation Metrics & Readiness Criteria
- **Classification Performance**: Macro-F1 $\ge 0.90$ and per-class F1 $\ge 0.85$ across all 8 themes.
- **Frontier Model Parity**: Win/tie rate $\ge 90\%$ against GPT-4o in blinded LLM-as-a-judge evaluations.
- **Zero Math Hallucination**: 100% exact numerical match against input tool metrics.
- **Schema Reliability**: 100% valid Pydantic JSON parsing on 2,000 held-out test scenarios.
- **Target Performance Goals**: Illustrative target of p99 inference latency $< 500\text{ms}$ on an enterprise GPU (e.g., NVIDIA L4), aiming to lower operating costs substantially compared to recurring frontier model API calls at 10,000 requests/day.

### 3.7 Serving Architecture & Canary Rollout
Served via **vLLM** with dynamic multi-LoRA support. Rollout proceeds through a 7-day shadow phase (100% background mirroring against the frontier LLM) followed by a canary migration (5% $\rightarrow$ 25% $\rightarrow$ 100%). A circuit breaker automatically routes requests to the frontier model if output JSON parsing fails or confidence drops below $0.80$.

### 3.8 Future-Proofing Input/Output Contracts
All interactions remain strictly decoupled from model weights via semantic versioning of Pydantic contracts (`schema_version="2.0"`). New adapters are trained to conform to versioned schemas, allowing heterogeneous model versions to coexist safely in production.
