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
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Literal

import requests

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

from langchain_core.tools import tool
from langgraph.types import interrupt

from .models import OrderModel, SentimentAnalysisResponse, MarketImpactAnalysis

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

POLYMARKET_MARKET_ID = os.getenv("POLYMARKET_MARKET_ID", "516713")  # Default to USDT depeg market

# News caching configuration
MAX_CACHED_URLS = 100
MAX_USAGE_COUNT = 30
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Legacy NewsAPI (not used anymore)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # noqa: S105 — just env var name

# Global market context cache
_CURRENT_MARKET_CONTEXT = None

# ---------------------------------------------------------------------------
# Cache management helpers
# ---------------------------------------------------------------------------

def _manage_news_cache(cache: Dict[str, int]) -> Dict[str, int]:
    """Clean up news usage cache based on limits.
    
    Args:
        cache: Current cache dict {url: usage_count}
        
    Returns:
        Cleaned cache dict
    """
    if not cache:
        return cache
        
    # If ANY URL hits max usage count, clear entire cache
    if any(count > MAX_USAGE_COUNT for count in cache.values()):
        print(f"DEBUG: Max usage count ({MAX_USAGE_COUNT}) reached, clearing entire news cache")
        return {}
    
    # If cache exceeds max URLs, clear entire cache
    if len(cache) > MAX_CACHED_URLS:
        print(f"DEBUG: Max cached URLs ({MAX_CACHED_URLS}) reached, clearing entire news cache")
        return {}
    
    return cache


def _increment_url_usage(cache: Dict[str, int], url: str) -> Dict[str, int]:
    """Increment usage count for a URL in the cache.
    
    Args:
        cache: Current cache dict
        url: URL to increment
        
    Returns:
        Updated cache dict
    """
    new_cache = cache.copy()
    new_cache[url] = new_cache.get(url, 0) + 1
    return _manage_news_cache(new_cache)


def _clean_text_content(text: str) -> str:
    """Remove ANSI escape codes and clean up text content.
    
    Args:
        text: Raw text that might contain ANSI codes
        
    Returns:
        Cleaned text without escape codes
    """
    if not text:
        return text
        
    # Remove ANSI escape codes (like \x1b[0m, \x1b[31m, etc.)
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    cleaned = ansi_escape.sub('', text)
    
    # Remove other common escape sequences
    cleaned = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', cleaned)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def _get_market_context() -> Dict[str, str]:
    """Get cached market context or fetch if not available.
    
    Returns:
        Dict with 'title' and 'description' of current market
    """
    global _CURRENT_MARKET_CONTEXT
    
    if _CURRENT_MARKET_CONTEXT is None:
        try:
            market_data = _fetch_real_market_data(POLYMARKET_MARKET_ID)
            _CURRENT_MARKET_CONTEXT = {
                'title': market_data['title'],
                'description': market_data['description']
            }
            print(f"DEBUG: Cached market context for market: {market_data['title']}")
        except Exception as e:
            print(f"DEBUG: Failed to fetch market context: {e}")
            # Fallback context
            _CURRENT_MARKET_CONTEXT = {
                'title': 'Unknown Market',
                'description': 'Market context unavailable'
            }
    
    return _CURRENT_MARKET_CONTEXT


# ---------------------------------------------------------------------------
# API helpers (very thin wrappers – keep sync for simplicity)
# ---------------------------------------------------------------------------

_GAMMA_API_ENDPOINT = "https://gamma-api.polymarket.com/markets"


def _fetch_real_market_data(market_id: str) -> dict:
    """Fetch complete market data from Polymarket Gamma API.
    
    Returns market metadata including title, description, prices, volume, etc.
    """
    try:
        resp = requests.get(_GAMMA_API_ENDPOINT, params={"id": market_id}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if not data or len(data) == 0:
            raise ValueError(f"No market found with ID: {market_id}")
            
        market = data[0]  # API returns array, take first result
        
        # Parse the data we need
        # Handle outcomePrices - could be string or array
        outcome_prices = market["outcomePrices"]
        if isinstance(outcome_prices, str):
            # Parse JSON string like "[\"0.055\", \"0.945\"]"
            import json
            outcome_prices = json.loads(outcome_prices)
        
        return {
            "id": market["id"],
            "title": market["question"],
            "description": market["description"],
            "end_date": market["endDate"],
            "yes_price": float(outcome_prices[0]),  # First outcome is YES
            "no_price": float(outcome_prices[1]),   # Second outcome is NO
            "volume": float(market["volume"]),
            "liquidity": float(market["liquidity"]),
            "active": market["active"],
            "closed": market["closed"]
        }
        
    except Exception as e:
        print(f"Failed to fetch real market data for ID {market_id}: {e}")
        # Return fallback data
        return {
            "id": market_id,
            "title": "Bitcoin to reach $70,000 by July 31, 2025",
            "description": "Fallback market - API unavailable",
            "end_date": "2025-07-31T23:59:59Z",
            "yes_price": 0.5,
            "no_price": 0.5,
            "volume": 100000.0,
            "liquidity": 10000.0,
            "active": True,
            "closed": False
        }


def _fetch_polymarket_data() -> dict:
    """Retrieve market data including YES/NO prices and volume using real API."""
    try:
        # Use the real market data function
        market_data = _fetch_real_market_data(POLYMARKET_MARKET_ID)
        
        return {
            "yes_price": market_data["yes_price"],
            "no_price": market_data["no_price"],
            "volume": market_data["volume"],
            "timestamp": datetime.now(timezone.utc)
        }
        
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




def _fetch_news(search_query: str = "general", usage_cache: Dict[str, int] = None) -> List[dict]:
    """Fetch structured news data and return single best unused news item.

    The function returns exactly one news item in a list. Uses Tavily if API key
    is available, otherwise falls back to time-based demo news.
    
    Args:
        search_query: Search query or question to execute with Tavily
        usage_cache: Dict of URL -> usage_count for avoiding duplicates
    """
    if usage_cache is None:
        usage_cache = {}
        
    # Try Tavily first if API key is available
    if TAVILY_API_KEY and TavilyClient:
        try:
            tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
            response = tavily_client.search(search_query, search_depth="advanced", time_range="week", max_results=15)
            if response and 'results' in response:
                # Parse all results with scores
                all_news = []
                for result in response['results']:
                    title = result.get('title', '').strip()
                    content = result.get('content', '').strip()
                    url = result.get('url', '').strip()
                    score = result.get('score', 0.0)
                    
                    if title and len(title) > 10 and url:
                        # Clean title and content from ANSI escape codes
                        clean_title = _clean_text_content(title)
                        clean_content = _clean_text_content(content)
                        
                        # Truncate content after cleaning
                        if len(clean_content) > 500:
                            clean_content = clean_content[:500] + '...'
                        
                        all_news.append({
                            'title': clean_title,
                            'content': clean_content,
                            'url': url,
                            'score': score
                        })
                
                if all_news:
                    # Filter by relevance score >= 0.4, but ensure at least top 5
                    high_relevance = [item for item in all_news if item['score'] >= 0.4]
                    if len(high_relevance) < 5:
                        # Take top 5 by score regardless of threshold
                        filtered_news = sorted(all_news, key=lambda x: x['score'], reverse=True)[:5]
                    else:
                        filtered_news = high_relevance
                    
                    # Find best item to use (unused or lowest usage count)
                    selected_item = None
                    min_usage = float('inf')
                    
                    for item in filtered_news:
                        usage_count = usage_cache.get(item['url'], 0)
                        if usage_count == 0:  # Unused item found
                            selected_item = item
                            break
                        elif usage_count < min_usage:  # Track lowest usage
                            min_usage = usage_count
                            selected_item = item
                    
                    if selected_item:
                        # Remove score from final output
                        final_item = {k: v for k, v in selected_item.items() if k != 'score'}
                        return [final_item]
                    
        except Exception as e:
            print(f"Tavily search failed, using fallback: {e}")
    
    # Fallback to time-based demo news (return first item only)
    fallback_news = _get_time_based_demo_news()
    return [fallback_news[0]] if fallback_news else []


def _get_time_based_demo_news() -> List[dict]:
    """Generate time-based demo news that changes every 5 minutes."""
    # Use current time rounded to 5-minute intervals as seed
    now = datetime.now(timezone.utc)
    time_slot = (now.hour * 60 + (now.minute // 5) * 5)  # 5-minute intervals
    
    # Predefined news sets that rotate - now with structured data
    news_sets = [
        [
            {
                'title': "Bitcoin shows resilience after Fed comments",
                'content': "Bitcoin's price remained stable following the Federal Reserve's latest policy announcement, demonstrating the cryptocurrency's growing maturity and institutional acceptance in traditional financial markets.",
                'url': "https://example-news.com/bitcoin-fed-resilience"
            },
            {
                'title': "Analyst predicts BTC surge amid institutional inflows",
                'content': "Leading cryptocurrency analyst forecasts a significant price increase for Bitcoin as institutional investors continue to allocate funds to digital assets, citing improved regulatory clarity.",
                'url': "https://example-news.com/btc-institutional-surge"
            },
            {
                'title': "Market uncertainty as BTC faces regulatory pressure",
                'content': "Bitcoin markets show signs of volatility as regulatory authorities worldwide continue to develop frameworks for cryptocurrency oversight, creating uncertainty among traders.",
                'url': "https://example-news.com/btc-regulatory-pressure"
            }
        ],
        [
            {
                'title': "Bitcoin price remains flat in low-volume session",
                'content': "Trading volumes hit multi-week lows as Bitcoin consolidated around key support levels, with market participants awaiting clearer directional signals.",
                'url': "https://example-news.com/btc-low-volume-flat"
            },
            {
                'title': "Investors eye ETF approval timeline",
                'content': "Market attention focuses on pending Bitcoin ETF applications as regulatory decisions could significantly impact institutional adoption and price discovery.",
                'url': "https://example-news.com/btc-etf-timeline"
            },
            {
                'title': "Regulatory headwinds could impact crypto adoption",
                'content': "Industry experts warn that increasing regulatory scrutiny may slow mainstream cryptocurrency adoption, though clearer rules could ultimately benefit the sector.",
                'url': "https://example-news.com/crypto-regulatory-headwinds"
            }
        ],
        [
            {
                'title': "Major institutional investor adds Bitcoin to portfolio",
                'content': "A Fortune 500 company announced the addition of Bitcoin to its treasury reserves, signaling growing corporate acceptance of cryptocurrency as a store of value.",
                'url': "https://example-news.com/institutional-btc-portfolio"
            },
            {
                'title': "Technical analysis suggests BTC breakout imminent",
                'content': "Chart patterns indicate Bitcoin may be approaching a significant price movement, with key resistance levels being tested amid increasing trading activity.",
                'url': "https://example-news.com/btc-technical-breakout"
            },
            {
                'title': "Mining difficulty adjustment impacts market sentiment",
                'content': "Bitcoin's latest mining difficulty adjustment reflects network health and could influence market sentiment as mining economics continue to evolve.",
                'url': "https://example-news.com/btc-mining-difficulty"
            }
        ]
    ]
    
    # Add more news sets to reach 6 total (keeping it shorter for demo)
    # Select news set based on time slot
    set_index = (time_slot // 5) % len(news_sets)  # Rotate every 5 minutes
    return news_sets[set_index]


# ---------------------------------------------------------------------------
# LLM-based market impact analysis
# ---------------------------------------------------------------------------

def _analyze_market_impact_with_llm(news_items: List[dict]) -> dict:
    """Use LLM to analyze how news impacts market outcome probability."""
    
    if not news_items:
        return {
            "direction": "NEUTRAL",
            "impact": "LOW", 
            "confidence": 0.0,
            "reasoning": "No news items to analyze",
            "news_urls": []
        }
    
    # Get market context
    context = _get_market_context()
    market_title = context['title']
    market_description = context['description']
    
    # Build news text for analysis
    news_text = ""
    news_urls = []
    for i, item in enumerate(news_items):
        title = item.get('title', '')
        content = item.get('content', '')
        url = item.get('url', '')
        news_urls.append(url)
        news_text += f"{i+1}. Title: {title}\n   Content: {content}\n   Source: {url}\n\n"
    
    prompt = f"""You are a prediction market analyst. Analyze how the following news impacts the likelihood of this market outcome.

MARKET CONTEXT:
Title: {market_title}
Description: {market_description}

NEWS TO ANALYZE:
{news_text}

TASK: Determine how this news affects the probability of the market resolving to "YES".

OPTIONS:
- INCREASES_YES: News makes the outcome MORE likely
- INCREASES_NO: News makes the outcome LESS likely  
- NEUTRAL: News is irrelevant, spam, or has no clear impact on the market outcome

Consider:
- Is this news directly related to the market outcome?
- Does it provide new information that affects probability?
- How significant is this impact? (LOW/MEDIUM/HIGH)
- How confident are you in this assessment? (0.0 to 1.0)
- Provide 1-2 sentence reasoning

Focus on:
- Direct causal relationships to the market outcome
- New information vs already known facts
- Market timing and relevance
- Magnitude of potential impact

If news is unrelated, spam, or provides no meaningful signal, choose NEUTRAL with LOW impact.
"""

    try:
        llm = ChatOpenAI(
            model_name=os.getenv("POLY_AGENT_MODEL", "gpt-5-mini"),
            temperature=1
        )
        
        # Use structured output with Pydantic model
        structured_llm = llm.with_structured_output(MarketImpactAnalysis)
        response = structured_llm.invoke(prompt)
        
        # Add news URLs to response
        return {
            "direction": response.direction,
            "impact": response.impact,
            "confidence": response.confidence,
            "reasoning": response.reasoning,
            "news_urls": news_urls
        }
                
    except Exception as e:
        print(f"LLM market impact analysis failed: {e}")
        # Fallback to neutral analysis
        return {
            "direction": "NEUTRAL",
            "impact": "LOW", 
            "confidence": 0.0,
            "reasoning": "Analysis unavailable - LLM error occurred",
            "news_urls": news_urls
        }


# ---------------------------------------------------------------------------
# LLM-based sentiment analysis (deprecated)
# ---------------------------------------------------------------------------

def _analyze_sentiment_with_llm(news_items: List[dict]) -> List[dict]:
    """Use LLM with structured output to analyze sentiment of news items."""
    
    market_id = POLYMARKET_MARKET_ID
    news_text = ""
    for i, item in enumerate(news_items):
        title = item.get('title', '')
        content = item.get('content', '')
        url = item.get('url', '')
        news_text += f"{i+1}. Title: {title}\n   Content: {content}\n   Source: {url}\n\n"
    
    prompt = f"""You are a financial sentiment analyst specialized in prediction markets.

Analyze the following news items for the market: {market_id}

News Items:
{news_text}

For each news item, provide:
1. A sentiment score from -1.0 (very negative for market outcome) to +1.0 (very positive for market outcome)
2. Brief reasoning (max 30 words)

Consider:
- Market impact potential (institutional vs retail focus)
- Regulatory implications  
- Technical vs fundamental news
- Short-term vs long-term price effects
- Content depth and source reliability

Analyze both the title and content for comprehensive sentiment assessment.

Also provide an overall market sentiment and your confidence in the analysis.
"""

    try:
        llm = ChatOpenAI(
            model_name=os.getenv("POLY_AGENT_MODEL", "gpt-5-mini"),
            temperature=1
        )
        
        # Use structured output with Pydantic model
        structured_llm = llm.with_structured_output(SentimentAnalysisResponse)
        response = structured_llm.invoke(prompt)
        
        # Convert to the expected format with URLs
        results = []
        for i, analysis in enumerate(response.analyses):
            result = {
                "headline": analysis.headline,
                "sentiment_score": analysis.sentiment_score,
                "reasoning": analysis.reasoning
            }
            # Add URL if available from original news items
            if i < len(news_items):
                result["url"] = news_items[i].get("url", "")
            results.append(result)
        
        return results
                
    except Exception as e:
        print(f"LLM sentiment analysis failed: {e}")
        # Fallback to simple keyword analysis
        return [
            {
                "headline": item.get('title', ''), 
                "sentiment_score": _fallback_sentiment(item.get('title', '')), 
                "reasoning": "LLM unavailable - keyword fallback",
                "url": item.get('url', '')
            } 
            for item in news_items
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


def _extract_market_analysis(messages: List) -> dict:
    """Extract market analysis from recent analyze_market_impact tool results.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Dict with market analysis or fallback dict
    """
    if not messages:
        return {
            "direction": "UNKNOWN",
            "impact": "UNKNOWN",
            "confidence": 0.0,
            "analysis": "No market analysis available - no messages found"
        }
    
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'name') and msg.name == 'analyze_market_impact':
            try:
                import json
                if isinstance(msg.content, str):
                    result = json.loads(msg.content)
                elif isinstance(msg.content, dict):
                    result = msg.content
                else:
                    continue
                
                if isinstance(result, dict):
                    return {
                        "direction": result.get('direction', 'UNKNOWN'),
                        "impact": result.get('impact', 'UNKNOWN'),
                        "confidence": result.get('confidence', 0.0),
                        "analysis": result.get('reasoning', 'No reasoning provided')
                    }
                    
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue
    
    return {
        "direction": "UNKNOWN",
        "impact": "UNKNOWN",
        "confidence": 0.0,
        "analysis": "No market impact analysis found - trading based on market data only"
    }


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
def get_news(search_query: str) -> List[dict]:
    """Search for news articles relevant to trading decisions.
    
    Returns exactly one news item (in a list) to avoid duplicate analysis across runs.
    Uses caching to ensure fresh content each time.
    
    Parameters:
    - search_query: Specific search query or question to execute with Tavily.
      Examples: "What is causing USDT to depeg?", "Why is Tether losing its peg?", 
                "Latest news on stablecoin depegging events", "USDT regulatory concerns 2025"
    
    Returns:
    - List containing one news item with 'title', 'content', and 'url' fields
    """
    print(f"DEBUG: Agent requested news search: '{search_query}'")
    
    # Get usage cache from global state (will be set by the agent)
    usage_cache = getattr(get_news, '_usage_cache', {})
    print(f"DEBUG: Using news cache with {len(usage_cache)} cached URLs")
    
    return _fetch_news(search_query, usage_cache)


@tool
def analyze_market_impact(news_items: List[dict]) -> dict:
    """Analyze how news impacts the likelihood of the market outcome.

    IMPORTANT: This tool requires news items as input. Use the results from get_news() calls.
    
    CORRECT USAGE PATTERN:
    1. First call: get_news("your search query") 
    2. Then call: analyze_market_impact(news_items_from_step_1)
    
    Example workflow:
    - news_result = get_news("USDT stability concerns 2025")
    - impact_result = analyze_market_impact(news_result)

    Parameters
    ----------
    news_items : List[dict], REQUIRED
        Structured news items with 'title', 'content', and 'url' fields.
        Must provide the exact news items you want to analyze.
        Do NOT call this tool with empty arguments {}

    Returns
    -------
    dict
        Contains:
        - direction: "INCREASES_YES" | "INCREASES_NO" | "NEUTRAL"
        - impact: "LOW" | "MEDIUM" | "HIGH" 
        - confidence: 0.0-1.0
        - reasoning: Brief explanation (1-2 sentences)
        - news_urls: List of source URLs
    """
    if not news_items or len(news_items) == 0:
        return {
            "direction": "NEUTRAL",
            "impact": "LOW",
            "confidence": 0.0,
            "reasoning": "No news items provided for analysis",
            "news_urls": []
        }
    
    return _analyze_market_impact_with_llm(news_items)


@tool  
def trade(action: Literal["BUY", "SELL"], position: Literal["YES", "NO"], usd: float) -> dict:
    """Execute a trade on the target market.
    
    This tool simulates trade execution and returns the trade details.
    Validation and balance checks are performed in apply_updates.
    
    Parameters:
    - action: BUY or SELL
    - position: YES or NO (which side to trade)
    - usd: Trade amount in USD
    
    Examples:
    - trade("BUY", "YES", 100.0) → Buy $100 of YES contracts
    - trade("BUY", "NO", 200.0) → Buy $200 of NO contracts  
    - trade("SELL", "YES", 50.0) → Sell $50 worth of YES holdings
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
    
    current_messages = getattr(trade, '_current_messages', [])
    analysis_dict = _extract_market_analysis(current_messages)
    
    market_context = _get_market_context()
    market_title = market_context.get('title', 'Unknown Market')
    
    market_data = _fetch_polymarket_data()
    price = market_data[f"{position.lower()}_price"]
    contracts = usd / price
    
    interrupt_data = {
        "type": "trade_approval_request",
        "trade": {
            "action": action,
            "position": position,
            "usd_amount": usd,
            "price": price,
            "contracts": round(contracts, 2)
        },
        "market": {
            "title": market_title
        },
        "market_analysis": analysis_dict,
        "message": "Do you approve this trade?",
        "options": ["yes", "no"]
    }
    
    response = interrupt(interrupt_data)
    
    # Handle user response
    if isinstance(response, str):
        approval = response.lower().strip()
    else:
        approval = str(response).lower().strip()
    
    if approval not in ["yes", "y", "approve", "ok"]:
        # Trade rejected by user
        return {
            "tool_name": "trade",
            "error": f"Trade cancelled by user: {approval}",
            "success": False,
            "user_cancelled": True,
            "action_summary": f"CANCELLED: {action} {position} ${usd:.2f} (User rejected)"
        }
    
    # market_data = _fetch_polymarket_data()
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


TOOLS = [get_market_data, get_news, analyze_market_impact, trade]
