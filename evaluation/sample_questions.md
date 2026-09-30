# MiniSense — Evaluation Evidence & Benchmark Execution

This document records the empirical execution evidence for the three benchmark evaluation scenarios and five unseen generalization queries run directly through the MiniSense LangGraph multi-agent architecture against the 75,000 Appendix A synthetic survey dataset and local FAISS vector index.

---

## Scenario 1: Ranking / Complaints Analytics

### Question
> *"What are the top 3 complaints this month?"*

### 1. Hybrid LLM Planner & Task Decomposition
The LLM Planner parsed the natural-language question, interpreted `"this month"` as a temporal filter for the active evaluation window, and decomposed the inquiry into a structured `DataAgent` sub-task with `metric="negative_volume"` and `requested_limit=3`:

```json
[
  {
    "task_id": "task_1",
    "agent": "DataAgent",
    "task_type": "top_themes",
    "question": "What are the top 3 complaints this month?",
    "metric": "negative_volume",
    "dimensions": ["theme"],
    "filters": {"sentiment": "negative"},
    "date_range": {"start_date": "2026-05-01", "end_date": "2026-05-31"},
    "requested_limit": 3,
    "rationale": "Identify the top 3 complaint themes for the current month ranked by negative feedback volume."
  }
]
```

### 2. Deterministic Date Resolution
Natural language expression `"this month"` was deterministically converted to `start_date="2026-05-01"` and `end_date="2026-05-31"`.

### 3. Agent & Tool Execution
- **DataAgent Invoked Tools**: `filter_by_date`, `filter_surveys`, `count_responses`, `compute_csat`, `compute_average_rating`, `get_top_themes`.
- **Metrics Computed**:
  - Total Month Volume: 38,114 responses.
  - Overall Month Average Rating: 3.70 / 5.0.
  - Overall Month CSAT: 62.59%.
  - Top 3 Complaint Themes (ranked strictly by negative feedback count):
    1. **Pricing**: 2,047 negative complaints | 4,615 mentions | CSAT: 19.52% | Average Rating: 2.62
    2. **Membership**: 855 negative complaints | 5,031 mentions | CSAT: 62.93% | Average Rating: 3.68
    3. **Facilities**: 793 negative complaints | 5,683 mentions | CSAT: 66.78% | Average Rating: 3.78

### 4. RAG Grounding
- **RAGAgent**: Not invoked (pure empirical survey inquiry; no policy or FAQ guidance requested).

### 5. Final Synthesized Executive Answer
> "This month, the top three complaints identified from the survey data are related to Pricing, Membership, and Facilities. Specifically, the Pricing theme received 4,615 mentions, with an average rating of 2.62 and a customer satisfaction (CSAT) score of 19.52, indicating significant dissatisfaction. The Membership theme had 5,031 mentions, with a higher average rating of 3.68 and a CSAT of 62.93, suggesting a more favorable perception but still with some negative feedback (855 negative mentions). Lastly, the Facilities theme garnered 5,683 mentions, achieving an average rating of 3.78 and a CSAT of 66.78, reflecting a relatively positive sentiment compared to the other themes."

---

## Scenario 2: Longitudinal Period Comparison

### Question
> *"How did wait-time experience change from April to May?"*

### 1. Hybrid LLM Planner & Task Decomposition
The LLM Planner recognized a longitudinal comparison intent across two specific months (`April` to `May`) for the `Wait Time` feedback theme, delegating directly to `ComparisonAgent`:

```json
[
  {
    "task_id": "task_1",
    "agent": "ComparisonAgent",
    "task_type": "period_comparison",
    "question": "How did wait-time experience change from April to May?",
    "date_range": {"start_date": "2026-05-01", "end_date": "2026-05-31"},
    "comparison_period": {"start_date": "2026-04-01", "end_date": "2026-04-30"},
    "parameters": {"theme": "Wait Time", "current_label": "May", "previous_label": "April"}
  }
]
```

### 2. Deterministic Date Resolution
- `May` $\rightarrow$ `start_date="2026-05-01"`, `end_date="2026-05-31"`
- `April` $\rightarrow$ `comparison_start_date="2026-04-01"`, `comparison_end_date="2026-04-30"`

### 3. Agent & Tool Execution
- **ComparisonAgent Invoked Tools**: `filter_by_date`, `filter_surveys`, `compute_csat`, `compute_average_rating`, `count_responses`, `compare_period_metrics`.
- **Computed Metrics**:
  - April 2026 (Baseline): CSAT 12.51%, Average Rating 2.13, Volume 4,988 responses.
  - May 2026 (Current): CSAT 60.95%, Average Rating 3.74, Volume 5,022 responses.
  - Calculated Deltas:
    - CSAT Delta: **+48.44 percentage points** (+387.2% relative gain)
    - Average Rating Delta: **+1.61 stars** (+75.6% relative gain)
    - Volume Delta: +34 responses

### 4. RAG Grounding
- **RAGAgent**: Not invoked (comparison question did not request policy grounding).

### 5. Final Synthesized Executive Answer
> "Between April and May, there was a significant improvement in wait-time experience as evidenced by the survey metrics. In May, measured customer satisfaction (CSAT) was 60.95%, reflecting an increase of 48.44 points from April's CSAT of 12.51% (a 387.2% relative improvement). Additionally, average rating rose from 2.13 in April to 3.74 in May (+1.61 stars, a 75.6% gain). Response volume remained steady at 5,022 responses in May compared to 4,988 in April (+34 responses)."

---

## Scenario 3: Hybrid Analytics + RAG Policy Grounding

### Question
> *"How did wait-time experience change from April to May, and what does the FAQ say about expected wait times?"*

### 1. Hybrid LLM Planner & Task Decomposition
The LLM Planner recognized a dual-domain inquiry requiring both empirical longitudinal measurement and authoritative operational policy context, routing to `ComparisonAgent` and `RAGAgent` in parallel:

```json
[
  {
    "task_id": "task_1",
    "agent": "ComparisonAgent",
    "task_type": "period_comparison",
    "question": "How did wait-time experience change from April to May?",
    "date_range": {"start_date": "2026-05-01", "end_date": "2026-05-31"},
    "comparison_period": {"start_date": "2026-04-01", "end_date": "2026-04-30"},
    "parameters": {"theme": "Wait Time"}
  },
  {
    "task_id": "task_2",
    "agent": "RAGAgent",
    "task_type": "rag_lookup",
    "question": "What does the FAQ say about expected wait times?",
    "retrieval_query": "expected wait times",
    "requested_limit": 2
  }
]
```

### 2. RAG Retrieval & Similarity Evidence
- **Natural Language Query**: `"expected wait times"`
- **Top Retrieved Chunk**:
  - `chunk_id`: `faq_chunk_2`
  - `source`: `data/faq.txt`
  - `similarity_score`: `0.441` (Cosine Inner Product)
  - `retrieved_content`:
    > "## Q2: What is your average wait time?
    > We target under 10 minutes for counter orders during off-peak hours. Peak hours (12:00 PM–1:00 PM lunch and 6:00 PM–8:00 PM dinner) may see 15–20 minute waits due to high kitchen order volume. Beginning May 1st, dedicated express pickup stations were introduced across all bistro locations to separate advance mobile orders from walk-in registers. Guests ordering through our mobile app receive real-time kitchen tracking notifications with estimated preparation windows to help minimize in-store waiting."
- **Retrieval Strength**: High semantic match ($0.441 > 0.35$ threshold), marked `reliable=True`.
- **Limitation**: The FAQ documents operating targets and policies; it provides operational context rather than proving causal attribution for empirical metric changes.

### 3. Final Synthesized Executive Answer
> "Wait-time experience improved significantly between April and May, with measured CSAT rising from 12.51% to 60.95% (+48.44 percentage points) and average ratings increasing from 2.13 to 3.74 stars across 5,022 May responses versus 4,988 in April. According to GreenLeaf Bistro's operational policies in the FAQ, counter orders target under 10 minutes during off-peak hours, while peak meal periods (12–1 PM and 6–8 PM) expect 15–20 minute turnaround times. The FAQ notes that dedicated express pickup stations and mobile kitchen tracking notifications were introduced on May 1st across all bistro locations. While this operational initiative coincides with the observed satisfaction gains, it represents operational context rather than definitive causal proof."

---

## Generalization Scenarios (5 Unseen Questions)

The system was evaluated against 5 unseen natural-language queries to verify that planning and routing are fully generalized and not benchmark-specific:

### Unseen Query 1: Theme-Specific Monthly Inquiries
- **Question**: *"What are customers saying about food quality this month?"*
- **Planner Routing**: `DataAgent` (`theme="Food Quality"`, `start_date="2026-05-01"`, `end_date="2026-05-31"`)
- **Computed Metrics**: Volume 4,976 responses, Average Rating 4.09 / 5.0, CSAT 77.29%, Negative Count 337.
- **Synthesized Narrative**: Coherent executive summary citing high overall satisfaction (77.3% CSAT) with Food Quality in May.

### Unseen Query 2: Cleanliness Satisfaction in May
- **Question**: *"Why did cleanliness satisfaction drop in May?"*
- **Planner Routing**: `DataAgent` (`theme="Cleanliness"`, `start_date="2026-05-01"`, `end_date="2026-05-31"`)
- **Computed Metrics**: Volume 3,966 responses, Average Rating 3.84, CSAT 71.46%, Negative Count 574.
- **Synthesized Narrative**: Evaluates cleanliness empirically; correctly reports healthy May CSAT (71.5%) rather than hallucinating an imaginary drop.

### Unseen Query 3: Multi-Channel Cross-Segment Comparison
- **Question**: *"Compare kiosk vs mobile channel satisfaction."*
- **Planner Routing**: `ComparisonAgent` (`cohort_a="kiosk"`, `cohort_b="mobile"`)
- **Computed Metrics**: Kiosk CSAT 62.58% (19,107 responses) vs Mobile CSAT 62.48% (19,007 responses). CSAT delta: -0.10 points.
- **Synthesized Narrative**: Executive comparison detailing channel parity within 0.10 percentage points across 38,114 responses.

### Unseen Query 4: Authoritative Operational Policy RAG Lookup
- **Question**: *"What does the FAQ say about handling customer complaints and refunds?"*
- **Planner Routing**: `RAGAgent` (`retrieval_query="handling customer complaints and refunds"`)
- **Retrieved Chunk**: `faq_chunk_3` (similarity score `0.485`).
- **Synthesized Narrative**: Details 15-minute shift manager escalation, replacement policies, and complimentary beverage voucher thresholds for delays beyond 20 minutes.

### Unseen Query 5: Facility-Wide Complaint Driver Ranking
- **Question**: *"What are the main complaint themes across our facilities?"*
- **Planner Routing**: `DataAgent` (`task_type="top_themes"`, `metric="negative_volume"`)
- **Computed Metrics**: Ranked all 8 themes across 75,000 records.
- **Synthesized Narrative**: Identifies Pricing (2,920 negative mentions across 75k records), Wait Time, and Facilities as the top volume complaint categories.

---

## Additional Paraphrased Questions (3 Non-Benchmark Variants)

To demonstrate robustness against varied phrasing, 3 additional paraphrased queries were evaluated:

### Paraphrase 1 (Complaints Ranking Variant)
- **Question**: *"Which areas received the highest volume of customer complaints during the current month?"*
- **Planner Routing**: `DataAgent` (`task_type="top_themes"`, `start_date="2026-05-01"`, `end_date="2026-05-31"`, `metric="negative_volume"`)
- **Computed Metrics**: Pricing (2,047 negative complaints), Membership (855), Facilities (793), Wait Time (752), Cleanliness (574).
- **Synthesized Narrative**: Identifies Pricing, Membership, and Facilities as the primary negative feedback drivers in May 2026.

### Paraphrase 2 (Longitudinal Delta Variant)
- **Question**: *"What was the difference in customer satisfaction ratings for wait times between April 2026 and May 2026?"*
- **Planner Routing**: `ComparisonAgent` (`theme="Wait Time"`, `previous="2026-04-01 to 2026-04-30"`, `current="2026-05-01 to 2026-05-31"`)
- **Computed Metrics**: April CSAT 12.51% (4,988 responses) $\rightarrow$ May CSAT 60.95% (5,022 responses). CSAT delta: **+48.44 points**, average rating delta: **+1.61 stars**.
- **Synthesized Narrative**: Accurately computes and communicates the +48.44 point satisfaction surge from April to May.

### Paraphrase 3 (Hybrid Policy & Analytics Variant)
- **Question**: *"How long should customers anticipate waiting for their orders according to policy, and how did recent pickup satisfaction perform?"*
- **Planner Routing**: `RAGAgent` (`query="expected wait time for orders according to policy"`) + `DataAgent` (`theme="Wait Time"`, `date_range="2026-05-01 to 2026-05-31"`)
- **Retrieved Chunk**: `faq_chunk_2` (<10 min off-peak, 15–20 min peak, May 1 express stations).
- **Computed Metrics**: 5,022 May responses, CSAT 60.95%, Average Rating 3.74.
- **Synthesized Narrative**: Unifies operational SLA wait-time policy with empirical May pickup satisfaction metrics.

