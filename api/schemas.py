"""Shapes of what the API accepts and returns."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from api import settings


def _check_text(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")
    if len(text) > settings.MAX_TEXT_CHARS:
        raise ValueError(f"text is longer than {settings.MAX_TEXT_CHARS} characters")
    return text


class SentimentRequest(BaseModel):
    text: str = Field(..., examples=["Arrived broken and customer service never replied."])

    @field_validator("text")
    @classmethod
    def _clean(cls, text: str) -> str:
        return _check_text(text)


class SentimentBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=settings.MAX_BATCH)

    @field_validator("texts")
    @classmethod
    def _clean_all(cls, texts: list[str]) -> list[str]:
        return [_check_text(t) for t in texts]


class SentimentResult(BaseModel):
    text: str
    label: Literal["positive", "negative", "uncertain"]
    positive_probability: float = Field(..., description="0 = surely negative, 1 = surely positive")


class SentimentBatchResponse(BaseModel):
    results: list[SentimentResult]


class ChurnSignals(BaseModel):
    days_since_last_order: int
    orders_last_365d: int
    spend_last_365d: float
    avg_days_between_orders: Optional[float] = Field(None, description="empty if the customer has ordered on only one day")


class ChurnResult(BaseModel):
    customer_id: int
    scored: bool
    as_of: str
    model: str
    churn_score: Optional[float] = Field(
        None, description="Higher = more likely to place no order in the next 90 days. A ranking score, not a calibrated probability.")
    flagged_at_risk: Optional[bool] = Field(None, description="True when the score is at or above the model's 80%-recall threshold")
    threshold: float
    signals: ChurnSignals
    reason: Optional[str] = Field(None, description="Why the customer was not scored")


class ChurnTopResponse(BaseModel):
    as_of: str
    model: str
    threshold: float
    customers_scored: int
    customers: list[ChurnResult]


class ComponentStatus(BaseModel):
    loaded: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    churn: ComponentStatus
    sentiment: ComponentStatus
