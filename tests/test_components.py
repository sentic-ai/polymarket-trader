"""Test script to verify each component of the Polymarket agent."""
import json
from copy import deepcopy

def serialize_for_json(obj):
    """Convert datetime objects to ISO format strings in a dict."""
    if isinstance(obj, dict):
        obj = deepcopy(obj)
        for key, value in obj.items():
            if isinstance(value, dict):
                obj[key] = serialize_for_json(value)
            elif hasattr(value, 'isoformat'):
                obj[key] = value.isoformat()
    return obj
from datetime import datetime, timezone
from polymarket_agent.models import StateModel
from polymarket_agent.storage import save_state, load_state
from polymarket_agent.tools import get_market_odds, get_news, analyze_sentiment, trade

def test_storage():
    print("\n=== Testing Storage ===")
    # Create and save test state
    test_state = StateModel(
        balance=1000.0,
        holdings=0.0,
        last_5_actions=[],
        timestamp=datetime.now(timezone.utc)
    )
    save_state(test_state)
    
    # Load and verify
    loaded_state = load_state()
    print(f"Saved and loaded state matches: {loaded_state is not None}")
    print(f"State content: {loaded_state}")

def test_market_odds():
    print("\n=== Testing Market Odds ===")
    odds = get_market_odds()
    print(f"Market odds: {json.dumps(serialize_for_json(odds), indent=2)}")

def test_news():
    print("\n=== Testing News Feed ===")
    headlines = get_news()
    print("Headlines:")
    for h in headlines:
        print(f"- {h}")
    
    print("\nSentiment Analysis:")
    sentiments = analyze_sentiment(headlines)
    for s in sentiments:
        print(f"- {s['headline']}: {s['sentiment_score']}")

def test_trading():
    print("\n=== Testing Trading ===")
    # Try to buy
    buy_result = trade("BUY", 100.0)
    print(f"\nBuy result: {json.dumps(serialize_for_json(buy_result), indent=2)}")
    
    # Try to sell
    sell_result = trade("SELL", 50.0)
    print(f"\nSell result: {json.dumps(serialize_for_json(sell_result), indent=2)}")

if __name__ == "__main__":
    print("Starting component tests...")
    test_storage()
    test_market_odds()
    test_news()
    test_trading()
    print("\nTests completed!")