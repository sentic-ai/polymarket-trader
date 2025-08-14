# sentic-polymarket-trader

LLM-driven Polymarket trader demo with run trace, stats, and Sentic deployment badge.

A LangGraph-based agent that autonomously trades a fixed Polymarket contract based on market odds and news sentiment.

## Setup

1. Install the package in editable mode:

   ```bash
   pip install -e .
   ```

2. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   # edit .env with your keys
   ```

## Usage

### Quick Test

Run the example script to test the tools:

```bash
python examples/quickstart.py
```

### LangGraph Studio

To visualize and debug the agent graph:

1. Install LangGraph Studio:

   ```bash
   pip install langgraph-studio
   ```

2. Start the UI:

   ```bash
   langgraph studio
   ```

3. Open http://localhost:8000 and load the `trader` graph.

## Configuration

Environment variables:

- `POLY_AGENT_MODEL`: LLM to use (default: gpt-4o-mini)
- `POLY_AGENT_MAX_STEPS`: Maximum tool calls per tick (default: 2)
- `NEWS_API_KEY`: For real news (optional, falls back to demo data)
- `POLYMARKET_MARKET_ID`: Target market ID (default: btc-70k-2025-07-31)
- `POLY_AGENT_DATA_DIR`: Where to store state/orders (default: data/)
