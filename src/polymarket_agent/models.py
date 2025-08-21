"""Domain models used by the Polymarket Demo Trader Agent."""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field


class OrderModel(BaseModel):
    """Represents a single simulated trade order."""

    id: str
    side: Literal["BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO"]
    usd_size: float
    price: float
    timestamp: datetime


class StateModel(BaseModel):
    """Persistent state of the trading agent between ticks."""

    balance: float
    holdings: float
    last_5_actions: List[str] = Field(default_factory=list)
    timestamp: datetime


class MarketOdds(BaseModel):
    """Snapshot of current Polymarket prices for the target market."""

    yes_price: float
    no_price: float
    timestamp: datetime


class NewsItem(BaseModel):
    """A news headline with an associated sentiment score (-1 … 1)."""

    headline: str
    sentiment_score: float
    timestamp: datetime


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result for a single headline."""
    
    headline: str
    sentiment_score: float = Field(
        description="Sentiment score from -1.0 (very negative for BTC price) to +1.0 (very positive for BTC price)",
        ge=-1.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief reasoning for the sentiment score (max 30 words)",
        max_length=150
    )


class SentimentAnalysisResponse(BaseModel):
    """Complete sentiment analysis response for multiple headlines."""
    
    analyses: List[SentimentAnalysis] = Field(
        description="Sentiment analysis for each headline"
    )
    overall_sentiment: float = Field(
        description="Overall market sentiment score (-1.0 to +1.0)",
        ge=-1.0,
        le=1.0
    )
    confidence: float = Field(
        description="Confidence in the analysis (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )


class MarketImpactAnalysis(BaseModel):
    """Analysis of how news impacts the likelihood of market outcome."""
    
    direction: Literal["INCREASES_YES", "INCREASES_NO", "NEUTRAL"] = Field(
        description="How the news affects the probability of the market outcome"
    )
    impact: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="Magnitude of the impact on market outcome"
    )
    confidence: float = Field(
        description="Confidence in the analysis (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of the analysis (1-2 sentences)",
        max_length=200
    )
    news_urls: List[str] = Field(
        description="URLs of the news sources analyzed"
    )
