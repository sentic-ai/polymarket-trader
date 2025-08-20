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

from langchain_core.tools import tool

from .models import MarketOdds, OrderModel, SentimentAnalysisResponse

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


def _fetch_polymarket_data() -> dict:
    """Retrieve market data including YES/NO prices and volume.

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
        volume = data.get("volume", 0)  # Volume in USD
    except Exception:  # noqa: BLE001 – any failure → fallback to time-based demo data
        # Use current time rounded to 5-minute intervals as seed for consistent data
        now = datetime.now(timezone.utc)
        # Round to 5-minute intervals: 0, 5, 10, 15, etc.
        time_slot = (now.hour * 60 + (now.minute // 5) * 5)
        
        # Generate consistent data for this 5-minute window
        random.seed(time_slot)  # Create consistent seed
        base_price = random.random()
        base_volume = random.random()
        random.seed()  # Reset seed for other random operations
        
        yes_price = round(0.3 + base_price * 0.4, 2)  # 0.3–0.7
        no_price = round(1 - yes_price, 2)
        volume = round(50000 + base_volume * 200000, 2)  # $50K-$250K demo volume
    
    return {
        "yes_price": yes_price,
        "no_price": no_price,
        "volume": volume,
        "timestamp": datetime.now(timezone.utc)
    }


_NEWS_ENDPOINT = "https://newsapi.org/v2/everything"


def _fetch_news() -> List[str]:
    """Fetch top Bitcoin headlines and return them raw (no scoring).

    The function always returns **exactly** three headlines. In case of any
    network or API error a deterministic stub list is returned for demo / test
    purposes.
    """
    if not NEWS_API_KEY:
        # Demo / offline mode with time-based rotation
        return _get_time_based_demo_news()

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
        return _get_time_based_demo_news()


def _get_time_based_demo_news() -> List[str]:
    """Generate time-based demo news that changes every 5 minutes."""
    # Use current time rounded to 5-minute intervals as seed
    now = datetime.now(timezone.utc)
    time_slot = (now.hour * 60 + (now.minute // 5) * 5)  # 5-minute intervals
    
    # Predefined news sets that rotate
    news_sets = [
        [
            "Bitcoin shows resilience after Fed comments",
            "Analyst predicts BTC surge amid institutional inflows", 
            "Market uncertainty as BTC faces regulatory pressure",
        ],
        [
            "Bitcoin price remains flat in low-volume session",
            "Investors eye ETF approval timeline",
            "Regulatory headwinds could impact crypto adoption",
        ],
        [
            "Major institutional investor adds Bitcoin to portfolio",
            "Technical analysis suggests BTC breakout imminent",
            "Mining difficulty adjustment impacts market sentiment",
        ],
        [
            "Central bank digital currencies spark Bitcoin debate",
            "Whale movements detected on blockchain analytics",
            "Options expiry creates volatility in BTC markets",
        ],
        [
            "Lightning Network adoption reaches new milestone",
            "Energy concerns resurface in crypto discussions",
            "Bitcoin correlation with traditional markets weakens",
        ],
        [
            "DeFi protocols integrate Bitcoin through wrapped tokens",
            "Geopolitical tensions drive safe haven demand for BTC",
            "Bitcoin futures open interest hits record levels",
        ]
    ]
    
    # Select news set based on time slot
    set_index = (time_slot // 5) % len(news_sets)  # Rotate every 5 minutes
    return news_sets[set_index]


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

@tool
def get_market_data() -> dict:
    """Return current market data including YES/NO prices and trading volume.

    Output is serializable so the LLM can read the structure easily.
    """
    data = _fetch_polymarket_data()
    # Convert datetime to ISO string for JSON serialization
    return {
        "yes_price": data["yes_price"],
        "no_price": data["no_price"], 
        "volume": data["volume"],
        "timestamp": data["timestamp"].isoformat()
    }


@tool
def get_news() -> List[str]:
    """Return a list of raw Bitcoin news headlines."""
    return _fetch_news()


@tool
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


@tool  
def trade(action: Literal["BUY", "SELL"], position: Literal["YES", "NO"], usd: float) -> dict:
    """Execute a trade on the target market.
    
    This tool simulates trade execution and returns the trade details.
    Validation and balance checks are performed in apply_updates.
    
    Parameters:
    - action: BUY or SELL
    - position: YES or NO (which side to trade)
    - usd: Trade amount in USD
    """
    # Basic validation: minimum and maximum trade amounts
    MIN_TRADE_USD = 1.0
    MAX_TRADE_USD = 10000.0
    
    if usd < MIN_TRADE_USD:
        return {
            "tool_name": "trade",
            "error": f"Trade amount ${usd:.2f} is below minimum of ${MIN_TRADE_USD:.2f}",
            "success": False
        }
    
    if usd > MAX_TRADE_USD:
        return {
            "tool_name": "trade", 
            "error": f"Trade amount ${usd:.2f} exceeds maximum of ${MAX_TRADE_USD:.2f}",
            "success": False
        }
    
    market_data = _fetch_polymarket_data()
    yes_price = market_data["yes_price"] 
    no_price = market_data["no_price"]
    timestamp = datetime.now(timezone.utc)
    
    # Determine which price to use based on position
    if position == "YES":
        price = yes_price
    else:  # NO
        price = no_price
    
    contracts = usd / price
    
    # Create combined side for backwards compatibility
    side = f"{action}_{position}"
    
    order = OrderModel(
        id=str(uuid.uuid4()),
        side=side,
        usd_size=usd,
        price=price,
        timestamp=timestamp,
    )

    return {
        "tool_name": "trade",
        "order": {
            "id": order.id,
            "side": order.side,
            "usd_size": order.usd_size,
            "price": order.price,
            "timestamp": order.timestamp.isoformat()
        },
        "side": side,
        "usd_amount": usd,
        "price": price,
        "contracts": contracts,
        "action_summary": f"{side} ${usd:.2f} @ {price:.2f}",
        "success": True
    }


TOOLS = [get_market_data, get_news, analyze_sentiment, trade]
