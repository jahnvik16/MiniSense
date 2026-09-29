"""Structured Pydantic contracts and domain schemas for MiniSense multi-agent system."""

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


class TaskType(str, Enum):
    """Supported task types for sub-agents."""
    DATA_ANALYSIS = "data_analysis"
    TOP_THEMES = "top_themes"
    RAG_LOOKUP = "rag_lookup"
    PERIOD_COMPARISON = "period_comparison"
    COHORT_COMPARISON = "cohort_comparison"
    COMPARISON = "comparison"


class AgentType(str, Enum):
    """Available specialized sub-agents."""
    DATA_AGENT = "DataAgent"
    RAG_AGENT = "RAGAgent"
    COMPARISON_AGENT = "ComparisonAgent"


class TaskSpec(BaseModel):
    """Structured task specification emitted by the Orchestrator for sub-agents."""
    task_id: str = Field(description="Unique identifier for the task step")
    agent: str = Field(default="DataAgent", description="Target sub-agent identifier (e.g., DataAgent, RAGAgent, ComparisonAgent)")
    task_type: str = Field(default="data_analysis", description="Specific capability required (e.g., data_analysis, rag_lookup, comparison)")
    question: str = Field(description="Specific sub-question or instruction to address")
    start_date: str | None = Field(default=None, description="Primary evaluation start date (ISO-8601 or YYYY-MM-DD)")
    end_date: str | None = Field(default=None, description="Primary evaluation end date (ISO-8601 or YYYY-MM-DD)")
    comparison_start_date: str | None = Field(default=None, description="Baseline comparison start date")
    comparison_end_date: str | None = Field(default=None, description="Baseline comparison end date")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Deterministic arguments, cohort filters, or top-k settings")
    instruction: str | None = Field(default=None, description="Optional natural language guidance for backwards compatibility")

    @field_validator("task_id", "agent", "question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return v.strip()


class ThemeMetric(BaseModel):
    """Aggregated metrics for an individual survey theme."""
    theme: str
    count: int = Field(ge=0, description="Total responses in this theme")
    average_rating: float = Field(ge=0.0, le=5.0, description="Mean rating 1-5")
    csat: float = Field(ge=0.0, le=100.0, description="CSAT percentage (ratings >= 4)")
    sentiment_breakdown: dict[str, int] = Field(default_factory=dict, description="Counts of positive, neutral, negative")


class PeriodMetrics(BaseModel):
    """Summary metrics for a specific time window."""
    period_label: str = "custom"
    start_date: str | None = None
    end_date: str | None = None
    response_count: int = Field(ge=0)
    average_rating: float = Field(ge=0.0, le=5.0)
    csat: float = Field(ge=0.0, le=100.0)
    sentiment_breakdown: dict[str, int] = Field(default_factory=dict)
    top_themes: list[ThemeMetric] = Field(default_factory=list)


# Core sub-agent output contracts

class DataAgentResult(BaseModel):
    """Deterministic structured output returned by DataAgent."""
    response_count: int = Field(ge=0, description="Total responses analyzed")
    average_rating: float = Field(ge=0.0, le=5.0, description="Arithmetic mean of 1-5 ratings")
    csat: float = Field(ge=0.0, le=100.0, description="Percentage of responses with rating >= 4")
    top_themes: list[ThemeMetric] = Field(default_factory=list, description="Ranked theme performance metrics")
    supporting_metadata: dict[str, Any] = Field(default_factory=dict, description="Filters applied, cohort, date window, etc.")

    @field_validator("average_rating", "csat")
    @classmethod
    def round_floats(cls, v: float) -> float:
        return round(v, 2)

    @property
    def sample_size(self) -> int:
        return self.response_count

    @property
    def value(self) -> float:
        return self.csat

    @property
    def summary(self) -> str:
        return f"CSAT score is {self.csat}% across {self.response_count} responses."


class RAGAgentResult(BaseModel):
    """Structured knowledge retrieval output returned by RAGAgent."""
    query: str = Field(description="Search inquiry evaluated against the knowledge base")
    retrieved_chunks: list[str] = Field(default_factory=list, description="Text content of retrieved semantic chunks")
    scores: list[float] = Field(default_factory=list, description="Relevance or cosine similarity scores")
    source_metadata: list[dict[str, Any]] = Field(default_factory=list, description="Chunk identifiers, source paths, and section titles")
    reliable: bool = Field(default=True, description="Flag indicating whether retrieval confidence met the minimum threshold")

    @property
    def grounding_context(self) -> str:
        """Helper property joining all retrieved chunk contents for synthesis."""
        return "\n\n".join(self.retrieved_chunks)

    @model_validator(mode="after")
    def validate_lengths(self) -> "RAGAgentResult":
        if self.scores and len(self.scores) != len(self.retrieved_chunks):
            raise ValueError("Scores list length must match retrieved_chunks length")
        return self


class ComparisonAgentResult(BaseModel):
    """Structured delta analysis returned by ComparisonAgent."""
    current_period: PeriodMetrics = Field(description="Metrics for the primary evaluated period")
    previous_period: PeriodMetrics = Field(description="Metrics for the comparison baseline period")
    metric_changes: dict[str, float] = Field(default_factory=dict, description="Calculated deltas (e.g., csat_delta, average_rating_delta, count_delta)")
    summary: str = Field(default="", description="Concise executive summary of changes between periods")

    @property
    def cohort_a(self) -> str:
        return self.current_period.period_label

    @property
    def cohort_b(self) -> str:
        return self.previous_period.period_label

    @property
    def delta(self) -> float:
        return self.metric_changes.get("csat_delta", self.metric_changes.get("csat_diff", 0.0))


class FinalAnswer(BaseModel):
    """Final business-language response synthesized for the stakeholder."""
    answer: str = Field(description="Coherent business-language executive paragraph")
    supporting_metrics: dict[str, Any] = Field(default_factory=dict, description="Exact metrics backing the narrative")
    retrieved_sources: list[str] = Field(default_factory=list, description="Citations to grounding FAQ / policy chunks")
    assumptions: list[str] = Field(default_factory=list, description="Explicit business or statistical assumptions made")

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("FinalAnswer must contain a non-empty answer")
        return v.strip()


# Domain & Legacy Support Models

class SurveyRecord(BaseModel):
    """Schema representing an individual survey feedback entry."""
    id: str
    customer_id: str
    business_id: str = Field(default="loc_downtown", description="Identifier for location/business unit")
    business_name: str | None = None
    channel: str = Field(default="mobile_app", description="Collection channel (e.g., in_store, mobile_app, email)")
    cohort: str = Field(description="Customer cohort e.g., enterprise, self_serve, member, guest")
    product_tier: str = Field(default="Standard", description="Plan or membership tier")
    theme: str = Field(description="Survey theme (e.g., Food Quality, Wait Time, Staff, etc.)")
    category: str = Field(description="Survey category, aligned with theme")
    rating: int = Field(ge=1, le=5, description="Survey score rating 1-5")
    csat_score: int = Field(ge=1, le=5, description="CSAT score 1-5")
    nps_score: int = Field(ge=0, le=10, description="Correlated NPS rating 0-10")
    sentiment: str = Field(description="Sentiment classification: positive, neutral, or negative")
    feedback: str = Field(description="Free-text customer comment")
    timestamp: datetime = Field(description="ISO-8601 survey completion timestamp")


class DocumentChunk(BaseModel):
    """Grounding chunk retrieved from the FAQ/documentation store."""
    chunk_id: str
    content: str
    score: float
    source: str = "data/faq.txt"


class PeriodComparisonResult(BaseModel):
    """Comparative deltas between two time periods (used by data tools)."""
    period_a: PeriodMetrics
    period_b: PeriodMetrics
    count_delta: int
    average_rating_delta: float
    csat_delta: float
    summary: str


class DataAgentInput(BaseModel):
    """Input payload for DataAgent."""
    metric_name: str = Field(description="Target metric: 'csat', 'nps', 'count', or 'sentiment_breakdown'")
    cohort: str | None = Field(default=None, description="Optional cohort filter (e.g., 'enterprise', 'self_serve')")
    category: str | None = Field(default=None, description="Optional feedback category filter")


class DataAgentOutput(BaseModel):
    """Deterministic numerical results returned by DataAgent."""
    metric_name: str
    value: float | dict[str, Any]
    sample_size: int
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    summary: str


class RAGAgentInput(BaseModel):
    """Input payload for RAGAgent."""
    query: str
    top_k: int = 3


class RAGAgentOutput(BaseModel):
    """Grounded retrieval results returned by RAGAgent."""
    query: str
    retrieved_chunks: list[DocumentChunk]
    grounding_context: str


class ComparisonAgentInput(BaseModel):
    """Input payload for ComparisonAgent."""
    metric_name: str
    cohort_a: str
    cohort_b: str


class ComparisonAgentOutput(BaseModel):
    """Structured delta analysis returned by ComparisonAgent."""
    metric_name: str
    cohort_a: str
    cohort_a_value: float
    cohort_b: str
    cohort_b_value: float
    delta: float
    higher_cohort: str
    summary: str


class SynthesisOutput(BaseModel):
    """Final business synthesis combining deterministic metrics and RAG grounding."""
    question: str
    business_response: str = Field(description="Coherent business-language paragraph")
    metrics_used: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
