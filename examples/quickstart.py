"""Quick manual test for the Polymarket demo agent tools.

Run with:
    python quickstart.py

This will:
1. Print current market odds
2. Fetch news headlines and run sentiment analysis
3. Simulate a BUY trade ($50) and a SELL trade ($25)
4. Display the updated persistent state

The script uses JSON-based persistence in the ``data/`` directory (or the
path defined via the ``POLY_AGENT_DATA_DIR`` environment variable).
"""
from __future__ import annotations

from pprint import pprint

from polymarket_agent.tools import get_market_odds, get_news, analyze_sentiment, trade


def main() -> None:
    print("=== Current Market Odds ===")
    pprint(get_market_odds())

    print("\n=== News Headlines ===")
    headlines = get_news()
    pprint(headlines)

    print("\n=== Sentiment Analysis ===")
    pprint(analyze_sentiment(headlines))

    print("\n=== Executing Demo Trades ===")
    res_buy = trade("BUY", 50)
    print("BUY Result:")
    pprint(res_buy)

    res_sell = trade("SELL", 25)
    print("SELL Result:")
    pprint(res_sell)

    print("\nDone. Check the data directory for state.json and orders.json updates.")


if __name__ == "__main__":
    main()
