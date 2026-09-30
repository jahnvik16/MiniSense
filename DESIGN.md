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
To adapt an open-weights classifier for 10,000 daily surveys, we construct a curated training set balancing domain-expert human audits (20%) with frontier-model teacher distillation (80% via GPT-4o / Claude 3.5 Sonnet few-shot rubrics). Sampling maintains strict class balance across all 8 feedback themes and sentiment polarities.

### 3.2 Dataset Sizing & Split
The corpus comprises **6,000 to 8,000 labeled examples** (~750–1,000 per theme), providing robust coverage of long-tail expressions without excessive annotation costs. Data is partitioned into a strict **70% train / 15% validation / 15% test split**, stratified by theme, rating (1–5), and date to prevent temporal or topic leakage.

### 3.3 Base Model Selection
We select **Llama-3.1-8B-Instruct** (or **Mistral-7B-Instruct-v0.3**). 8B models provide strong linguistic reasoning and fit on a single NVIDIA L4 or A10G GPU (16–24GB VRAM), enabling cost-effective serving at sub-50ms latencies.

### 3.4 Fine-Tuning Technique: LoRA vs. QLoRA vs. Full
We adopt **16-bit LoRA** ($r=16$, $\alpha=32$, dropout 0.05) across attention and MLP linear projections. Full fine-tuning is rejected due to excessive compute costs ($>8\times$) and risk of catastrophic forgetting. QLoRA (4-bit NF4) is suitable for low-VRAM training but introduces dequantization overhead during serving. 16-bit LoRA achieves fast convergence, full fp16 serving throughput, and lightweight modular adapters (~65MB).

### 3.5 Tooling & Pipeline
Implemented with Hugging Face **TRL (`SFTTrainer`)**, **PEFT**, and **PyTorch**, accelerated by FlashAttention-2 and tracked via MLflow.
*Pipeline*: Schema validation $\rightarrow$ Tokenization with response-only loss masking $\rightarrow$ 3-epoch training with cosine decay ($2 \times 10^{-4}$) $\rightarrow$ Evaluation every 250 steps $\rightarrow$ Signed adapter export.

### 3.6 Classifier Evaluation Metrics & Readiness
- **Classification Performance**: Macro-F1 $\ge 0.90$ across all 8 themes, with per-class F1 $\ge 0.85$, precision $\ge 0.85$, and recall $\ge 0.85$.
- **Error Analysis**: Confusion matrix evaluation to detect semantic boundary confusions (e.g., Wait Time vs. Staff).
- **Baseline Comparison**: Measurable gain over zero-shot frontier prompting and keyword rule baselines (+12% Macro-F1 over heuristic regexes).
- **Latency & Cost**: p99 inference latency $< 50\text{ms}$ on NVIDIA L4, reducing per-survey cost by $94\%$ vs. frontier API calls.

### 3.7 Serving & Canary Rollout
Served via **vLLM** with dynamic multi-LoRA routing. A 7-day shadow phase (100% background mirroring) precedes a canary rollout (5% $\rightarrow$ 25% $\rightarrow$ 100%), with automatic fallback if confidence drops below $0.80$.

### 3.8 Future-Proofing
Contracts decouple from model weights via semantic schema versioning (`schema_version="2.0"`). New themes or taxonomies deploy as modular adapters without retraining base weights.
