"""Tool functions for the Polymarket demo agent.

These are exposed as LangChain-compatible tools that TraderGPT can call via
function-calling.

All tools are **pure Python** to keep the demo self-contained.  External API
calls are encapsulated behind small helper functions that can be mocked in
tests.
"""
from __future__ import annotations

import os
import random
import uuid
from datetime import datetime, timezone
from typing import List, Literal, cast

import requests

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

from .models import MarketOdds, OrderModel, StateModel, SentimentAnalysisResponse
from .storage import append_order, load_state, save_state

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

POLYMARKET_MARKET_ID = os.getenv("POLYMARKET_MARKET_ID", "btc-70k-2025-07-31")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # noqa: S105 — just env var name

# ---------------------------------------------------------------------------
# API helpers (very thin wrappers – keep sync for simplicity)
# ---------------------------------------------------------------------------

_POLYMARKET_ENDPOINT = "https://www.polymarket.com/api/v3/markets/{}"


def _fetch_polymarket_prices() -> MarketOdds:
    """Retrieve YES/NO prices for the configured market.

    The public Polymarket API returns prices in MILLI-DOLLAR units; we convert
    to regular USD.
    """
    url = _POLYMARKET_ENDPOINT.format(POLYMARKET_MARKET_ID)
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = cast(dict, resp.json())
        # Polymarket returns price in cents; convert to USD
        yes_price = data["yesPrice"] / 100.0
        no_price = data["noPrice"] / 100.0
    except Exception:  # noqa: BLE001 – any failure → fallback to random demo prices
        now = random.random()
        yes_price = round(0.3 + now * 0.4, 2)  # 0.3–0.7
        no_price = round(1 - yes_price, 2)
    return MarketOdds(yes_price=yes_price, no_price=no_price, timestamp=datetime.now(timezone.utc))


_NEWS_ENDPOINT = "https://newsapi.org/v2/everything"


def _fetch_news() -> List[str]:
    """Fetch top Bitcoin headlines and return them raw (no scoring).

    The function always returns **exactly** three headlines. In case of any
    network or API error a deterministic stub list is returned for demo / test
    purposes.
    """
    if not NEWS_API_KEY:
        # Demo / offline mode
        return [
            "Bitcoin shows resilience after Fed comments",
            "Analyst predicts BTC surge amid institutional inflows",
            "Market uncertainty as BTC faces regulatory pressure",
        ]

    params = {
        "q": "bitcoin",
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 3,
        "apiKey": NEWS_API_KEY,
    }
    try:
        resp = requests.get(_NEWS_ENDPOINT, params=params, timeout=5)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])[:3]
        headlines = [art.get("title") for art in articles if art.get("title")]
        if len(headlines) < 3:
            raise ValueError("Not enough headlines returned")
        return headlines
    except Exception:  # noqa: BLE001
        return [
            "Bitcoin price remains flat in low-volume session",
            "Investors eye ETF approval timeline",
            "Regulatory headwinds could impact crypto adoption",
        ]


# ---------------------------------------------------------------------------
# LLM-based sentiment analysis
# ---------------------------------------------------------------------------

def _analyze_sentiment_with_llm(headlines: List[str]) -> List[dict]:
    """Use LLM with structured output to analyze sentiment of Bitcoin headlines."""
    
    market_id = POLYMARKET_MARKET_ID
    headlines_text = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines)])
    
    prompt = f"""You are a financial sentiment analyst specialized in Bitcoin markets.

Analyze the following Bitcoin headlines for the market: {market_id}

Headlines:
{headlines_text}

For each headline, provide:
1. A sentiment score from -1.0 (very negative for BTC price) to +1.0 (very positive for BTC price)
2. Brief reasoning (max 30 words)

Consider:
- Market impact potential (institutional vs retail focus)
- Regulatory implications
- Technical vs fundamental news
- Short-term vs long-term price effects

Also provide an overall market sentiment and your confidence in the analysis.
"""

    try:
        llm = ChatOpenAI(
            model_name=os.getenv("POLY_AGENT_MODEL", "gpt-4o-mini"),
            temperature=0.1
        )
        
        # Use structured output with Pydantic model
        structured_llm = llm.with_structured_output(SentimentAnalysisResponse)
        response = structured_llm.invoke(prompt)
        
        # Convert to the expected format for backward compatibility
        return [
            {
                "headline": analysis.headline,
                "sentiment_score": analysis.sentiment_score,
                "reasoning": analysis.reasoning
            }
            for analysis in response.analyses
        ]
                
    except Exception as e:
        print(f"LLM sentiment analysis failed: {e}")
        # Fallback to simple keyword analysis
        return [
            {
                "headline": h, 
                "sentiment_score": _fallback_sentiment(h), 
                "reasoning": "LLM unavailable - keyword fallback"
            } 
            for h in headlines
        ]


def _fallback_sentiment(headline: str) -> float:
    """Fallback sentiment scoring if LLM fails."""
    positive = ["surge", "resilience", "record", "gain", "approval", "rise", "rally", "bull", "positive", "growth", "institutional", "adoption"]
    negative = ["pressure", "uncertainty", "decline", "risk", "headwinds", "fall", "crash", "bear", "negative", "regulatory", "ban", "concern"]
    
    s = 0.0
    lower = headline.lower()
    for kw in positive:
        if kw in lower:
            s += 0.15
    for kw in negative:
        if kw in lower:
            s -= 0.15
    return max(-1.0, min(1.0, s))


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def get_market_odds() -> dict:
    """Return current YES/NO prices.

    Output is serialisable so the LLM can read the structure easily.
    """
    odds = _fetch_polymarket_prices()
    return odds.model_dump()


def get_news() -> List[str]:
    """Return a list of raw Bitcoin news headlines."""
    return _fetch_news()


def analyze_sentiment(headlines: List[str]) -> List[dict]:
    """Analyze sentiment of headlines using LLM for sophisticated market analysis.

    Parameters
    ----------
    headlines : List[str]
        Raw news headlines to analyse.

    Returns
    -------
    List[dict]
        Each dict contains ``headline``, ``sentiment_score`` (-1 .. 1), and ``reasoning``.
    """
    return _analyze_sentiment_with_llm(headlines)


def trade(side: Literal["BUY", "SELL"], usd: float) -> dict:
    """Simulate a trade on the target market.

    Automatically adjusts trade size to maximum possible if requested amount 
    exceeds available balance/holdings. This reduces LLM costs by avoiding 
    failed trades and subsequent retry calls.
    """
    state = load_state() or StateModel(balance=1000.0, holdings=0.0, last_5_actions=[], timestamp=datetime.now(timezone.utc))

    odds = _fetch_polymarket_prices()
    price = odds.yes_price
    timestamp = datetime.now(timezone.utc)
    
    original_usd = usd
    adjusted = False

    if side == "BUY":
        if usd > state.balance:
            usd = state.balance  # Auto-adjust to max buyable
            adjusted = True
        contracts = usd / price
        state.balance -= usd
        state.holdings += contracts
    else:  # SELL
        max_sellable_usd = state.holdings * price
        if usd > max_sellable_usd:
            usd = max_sellable_usd  # Auto-adjust to max sellable
            adjusted = True
        contracts = usd / price
        state.balance += usd
        state.holdings -= contracts

    order = OrderModel(
        id=str(uuid.uuid4()),
        side=side,
        usd_size=usd,
        price=price,
        timestamp=timestamp,
    )
    state.last_5_actions = (state.last_5_actions + [f"{side} ${usd:.2f} @ {price:.2f}"])[-5:]
    state.timestamp = timestamp

    # Persist changes
    append_order(order)
    save_state(state)

    result = {
        "order": order.model_dump(),
        "new_state": state.model_dump(),
    }
    
    if adjusted:
        result["adjusted"] = True
        result["original_amount"] = original_usd
        result["executed_amount"] = usd
        result["note"] = f"Auto-adjusted from ${original_usd:.2f} to ${usd:.2f} (max available)"

    return result


TOOLS = [get_market_odds, get_news, analyze_sentiment, trade]
