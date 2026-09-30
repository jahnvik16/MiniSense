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
> "This month, the top three complaints identified from the survey metrics are related to Pricing, Membership, and Facilities. Specifically, the Pricing theme received 4,615 mentions, with an average rating of 2.62 and a customer satisfaction (CSAT) score of 19.52, indicating significant dissatisfaction. The Membership theme had 5,031 mentions, with a higher average rating of 3.68 and a CSAT of 62.93, suggesting a more favorable perception but still with some negative feedback (855 negative mentions). Lastly, the Facilities theme garnered 5,683 mentions, achieving an average rating of 3.78 and a CSAT of 66.78, reflecting a relatively positive sentiment compared to the other themes."

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
  - April 2026 (Baseline): CSAT 16.03%, Average Rating 2.21, Volume 4,884 responses.
  - May 2026 (Current): CSAT 64.47%, Average Rating 3.75, Volume 4,887 responses.
  - Calculated Deltas:
    - CSAT Delta: **+48.44 percentage points** (+302.2% relative gain)
    - Average Rating Delta: **+1.54 stars** (+69.7% relative gain)
    - Volume Delta: +3 responses

### 4. RAG Grounding
- **RAGAgent**: Not invoked (comparison question did not request policy grounding).

### 5. Final Synthesized Executive Answer
> "Wait-time satisfaction demonstrated a substantial upward shift between April and May. In April, customer satisfaction (CSAT) was measured at 16.03% with an average rating of 2.21 across 4,884 survey responses. In May, CSAT reached 64.47% with an average rating of 3.75 across 4,887 responses, reflecting a +48.44 percentage point increase in satisfaction and a +1.54 star improvement in average rating. Response volume remained consistent across both evaluation periods."

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
    "retrieval_query": "expected wait times peak off-peak policy",
    "requested_limit": 2
  }
]
```

### 2. RAG Retrieval & Similarity Evidence
- **Natural Language Query**: `"expected wait times peak off-peak policy"`
- **Top Retrieved Chunk**:
  - `chunk_id`: `faq_chunk_2`
  - `source`: `data/faq.txt`
  - `similarity_score`: `0.441` (Cosine Inner Product)
  - `retrieved_content`:
    > "## Q2: What is your average wait time?
    > We target under 10 minutes for counter orders during off-peak hours. Peak hours (12:00 PM–1:00 PM lunch and 6:00 PM–8:00 PM dinner) may see 15–20 minute waits due to high kitchen order volume. Beginning May 1st, dedicated express pickup stations were introduced across all bistro locations to separate advance mobile orders from walk-in registers. Guests ordering through our mobile app receive real-time kitchen tracking notifications with estimated preparation windows to help minimize in-store waiting."
- **Retrieval Strength**: High semantic match ($0.441 > 0.35$ threshold), marked `reliable=True`.
- **Limitation**: The FAQ documents operating targets and policies; it does not measure real-time queue lengths or establish causal proof for metric shifts.

### 3. Final Synthesized Executive Answer
> "Wait-time experience improved significantly between April and May, with measured CSAT rising from 16.03% to 64.47% (+48.44 percentage points) and average ratings increasing from 2.21 to 3.75 stars. According to GreenLeaf Bistro's operational policies, counter orders target under 10 minutes during off-peak hours, while peak meal periods (12–1 PM and 6–8 PM) expect 15–20 minute turnaround times. The FAQ notes that dedicated express pickup stations and mobile kitchen tracking notifications were introduced on May 1st. While this policy change coincides with the observed satisfaction gains, it provides operational context rather than establishing definitive causality."

---

## Generalization Scenarios (5 Unseen Questions)

The system was evaluated against 5 unseen natural-language queries to verify that planning and routing are fully generalized and not benchmark-specific:

### Unseen Query 1: Theme-Specific Monthly Inquiries
- **Question**: *"What are customers saying about food quality this month?"*
- **Planner Routing**: `DataAgent` (`theme="Food Quality"`, `date_range={"start_date": "2026-05-01", "end_date": "2026-05-31"}`)
- **Computed Metrics**: Volume 4,792 responses, Average Rating 3.99 / 5.0, CSAT 77.39%, Negative Count 446.
- **Synthesized Narrative**: Coherent executive summary citing high overall satisfaction (77.4% CSAT) with Food Quality in May.

### Unseen Query 2: Missing Feedback / Zero-Denominator Safety
- **Question**: *"Why did cleanliness satisfaction drop in May?"*
- **Planner Routing**: `DataAgent` (`theme="Cleanliness"`, `date_range={"start_date": "2026-05-01", "end_date": "2026-05-31"}`)
- **Safety Handling**: Evaluated cleanly with zero-denominator safety without raising division exceptions or inventing fake causes.

### Unseen Query 3: Multi-Channel Cross-Segment Comparison
- **Question**: *"Compare kiosk vs mobile channel satisfaction."*
- **Planner Routing**: `ComparisonAgent` (`cohort_a="kiosk"`, `cohort_b="mobile"`)
- **Computed Metrics**: Evaluated 18,750+ responses per channel with exact deterministic delta calculations.
- **Synthesized Narrative**: Executive comparison comparing kiosk vs mobile CSAT without hallucinations.

### Unseen Query 4: Authoritative Operational Policy RAG Lookup
- **Question**: *"What does the FAQ say about handling customer complaints and refunds?"*
- **Planner Routing**: `RAGAgent` (`retrieval_query="handling customer complaints and refunds FAQ"`)
- **Retrieved Chunk**: `faq_chunk_3` (similarity score `0.485`).
- **Synthesized Narrative**: Details 15-minute shift manager escalation, replacement policies, and complimentary beverage voucher thresholds for delays beyond 20 minutes.

### Unseen Query 5: Facility-Wide Complaint Driver Ranking
- **Question**: *"What are the main complaint themes across our facilities?"*
- **Planner Routing**: `DataAgent` (`task_type="top_themes"`, `metric="negative_volume"`)
- **Computed Metrics**: Ranked all 8 themes across 75,000 records.
- **Synthesized Narrative**: Ranks Wait Time, App Experience, and Pricing as the leading complaint categories across all facilities.
