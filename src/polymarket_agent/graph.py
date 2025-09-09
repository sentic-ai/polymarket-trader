"""LangGraph definition for the Polymarket demo agent.

Flow (Variant A):
    1. ContextBuilder – builds system prompt from saved state
    2. TraderBrain   – LLM with bound tools (get_market_odds, get_news, trade)
    3. ToolNode      – executes tool calls

The tick ends when the LLM no longer requests tool calls **or** the maximum
number of tool steps is reached (``MAX_TOOL_STEPS``).
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Dict, List, Literal, Optional
from typing_extensions import TypedDict, Annotated

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import TOOLS, get_market_data, get_news, _fetch_real_market_data, _increment_url_usage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("POLY_AGENT_MODEL", "gpt-5-mini")
MAX_TOOL_STEPS = int(os.getenv("POLY_AGENT_MAX_STEPS", "6"))

# Entry guard configuration
MIN_PRICE_CHANGE_BPS = int(os.getenv("POLY_AGENT_MIN_PRICE_CHANGE_BPS", "25"))  # 0.25%
SEC_TIME_BETWEEN_RUNS = int(os.getenv("POLY_AGENT_SEC_TIME_BETWEEN_RUNS", "300"))  # 5 minutes
REQUIRE_NEW_NEWS = os.getenv("POLY_AGENT_REQUIRE_NEW_NEWS", "true").lower() == "true"
DEBUG_MODE = os.getenv("POLY_AGENT_DEBUG_MODE", "false").lower() == "true"  # Skip guard rails

# Market ID for fetching real data
POLYMARKET_MARKET_ID = os.getenv("POLYMARKET_MARKET_ID", "516713")  # Default to USDT depeg market

# ---------------------------------------------------------------------------
# Graph State Definitions
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Shared graph state.

    - ``messages``: Conversation history (System, Human, AI, Tool)
    - ``step``:     How many tool steps have been executed?
    - ``balance``:  Current USD balance
    - ``yes_holdings``: Number of YES-contracts held
    - ``no_holdings``: Number of NO-contracts held
    - ``last_5_actions``: Recent trading actions history
    - ``news``: Latest news headlines from get_news tool
    - ``market_odds``: Latest market odds from get_market_data tool
    - ``last_run_timestamp``: Timestamp of last completed run
    - ``_skip_run``: Internal flag to skip run (set by entry_guard)
    - ``skip_reason``: Reason why run was skipped (if applicable)
    - ``news_usage_cache``: Cache of news URLs with usage counts
    - ``_user_cancelled_trade``: Internal flag when user cancels a trade (ends run)
    """

    messages: Annotated[List[BaseMessage], add_messages]
    step: int
    balance: float
    yes_holdings: float
    no_holdings: float
    last_5_actions: List[str]
    news: Optional[List[str]]
    market_odds: Optional[Dict[str, float]]
    last_run_timestamp: Optional[datetime]
    _skip_run: Optional[bool]
    skip_reason: Optional[str]
    news_usage_cache: Optional[Dict[str, int]]
    _user_cancelled_trade: Optional[bool]


# ---------------------------------------------------------------------------
# Node 1 – ContextBuilder
# ---------------------------------------------------------------------------

def build_context(state: AgentState) -> Dict[str, List[BaseMessage] | float | List[str] | datetime | bool | None]:
    print(f"CONTEXT: Starting context node at {datetime.now(tz=UTC).strftime('%H:%M:%S')}")
    """Creates the initial prompt based on persistent state."""
    balance = state.get("balance", 10000.0)
    yes_holdings = state.get("yes_holdings", 0.0)
    no_holdings = state.get("no_holdings", 0.0)
    last_actions = state.get("last_5_actions", [])
    news = state.get("news", None)
    market_odds = state.get("market_odds", None)
    last_run_timestamp = state.get("last_run_timestamp", None)
    skip_run = state.get("_skip_run", None)
    history = " | ".join(last_actions) or "(none)"
    
    # Fetch real market data
    market_data = _fetch_real_market_data(POLYMARKET_MARKET_ID)

    system_prompt = (
        "You are TraderGPT, an elite AI trading agent specializing in Polymarket prediction markets.\n\n"
    
        f"MARKET CONTEXT:\n"
        f"Title: {market_data['title']}\n"
        f"Description: {market_data['description']}\n"
        f"Closes: {market_data['end_date']}\n"
        f"Market ID: {market_data['id']}\n\n"

        "IDENTITY: You combine quantitative reasoning with market microstructure"
        " awareness and behavioral biases to find non-obvious edges in prediction markets.\n\n"
    "Inputs to value in your trading decision:\n\n"
    "   - Balance (USD)\n"
    "   - Holdings on both sides (YES shares, NO shares)\n"
    "   - Market odds and trading volume\n"
    "   - Your last 5 trades\n\n"
    "CORE STRATEGY FRAMEWORK:\n"
    "1. INFORMATION EDGE: Identify what the market is missing\n"
    "   - Detect narrative shifts before they're priced in\n"
    "   - Find contradictions between betting markets and fundamentals\n"
    "   - Spot overreactions to noise vs. signal\n\n"
    
    "2. MARKET PSYCHOLOGY: Exploit behavioral biases\n"
    "   - Recency bias: Markets overweight recent events\n"
    "   - Availability heuristic: Vivid news gets overpriced\n"
    "   - Herd behavior: Identify crowded trades to fade\n\n"
    
    "3. RISK MANAGEMENT:\n"
    "   - Per-trade sizing MUST obey: size_usd ≤ 15% of available cash.\n"
    "   - 15% is a absolut hard ceiling, not a default. Size your trades 0-15% of available cash\n"
    "   - SIZE EXAMPLES:\n"
    "     • Conviction 9-10/10 + HIGH impact: 10-15%\n"
    "     • Conviction 7-8/10 + MEDIUM impact: 5-10%\n"
    "     • Conviction 5-6/10 + LOW impact: 1-3%\n"
    "   - Incorporate your current book: avoid adding risk to an already crowded side;\n"
    "   - Start small; prefer HOLD or a small probe when signals are mixed or weak."
    "   - Choose the smallest trade size that clearly expresses your view; \n"
    "   - If the intended action conflicts with your existing exposure (e.g., want to buy NO while holding YES), consider selling/reducing the opposing side.\n"
    "   - Never justify size by 'adheres to 15%'; justify by evidence (sources + microstructure + your current book).\n"
    
    "TRADING OPTIONS:\n"
    "- BUY + YES: Buy YES contracts (bet outcome will happen)\n"
    "- BUY + NO: Buy NO contracts (bet outcome won't happen)\n"
    "- SELL + YES: Sell your YES contracts (if you own any)\n"
    "- SELL + NO: Sell your NO contracts (if you own any)\n\n"
    
    "TRADING SIGNALS (Combine multiple for conviction):\n"
    "- BUY_YES: Information asymmetry detected, market underpricing YES\n"
    "- BUY_NO: Euphoria detected, YES odds >80% on weak fundamentals\n"
    "- SELL: Take profits or cut losses on existing positions\n"
    "- HOLD: Mixed signals or fair value\n\n"
    
    "DECISION WORKFLOW:\n"
    "1. Scan market odds\n"
    "2. Fetch 1-2 news articles using get_news tool:\n"
    "   - Ignore generic news/unrelated news/spam\n"
    "   - What's priced in vs. what's new information\n"
    "   - Second-order effects markets might miss\n"
    "   - Sentiment extremes to fade\n"
    "3. Identify the market's blind spot\n"
    "4. Position Context - Your own book matters:\n"
    "   - Factor your past trades, holdings, and timing into your decision.\n"
    "4. Size position based on:\n"
    "   - Conviction level (1-10 scale)\n"
    "   - Risk/reward asymmetry\n"
    "   - Current holdings, past trades\n\n"
    "   - Use current holdings, average entry, and last 5 trades to decide whether to add, reduce, or flip rather than initiate a fresh opposite-side add.\n"
"   - If signals are mixed relative to your book, prefer HOLD or reduce instead of forcing a new position.\n"

    "OUTPUT FORMAT (strictly!):\n"
    "🎯 THESIS: [1-2 sentence insight the market is missing]\n"
    "📊 EDGE: [Specific mispricing or behavioral bias to exploit]\n"
    "🎲 CONVICTION: [X/10]\n"
    "💰 ACTION: [BUY/SELL + YES/NO X% of capital]\n"
    "📈 TARGET: [Expected odds in X timeframe]\n\n"
    "🔎 SIZING RATIONALE: [one line explaining why this size is justified given sources, microstructure, and your current book]\n"

    
    "Remember: The best trades are contrarian with a catalyst. Don't just follow news."
    "Find what others overlook. Your reputation depends on making non-obvious, profitable calls.\n\n"

        "You have up to 6 tool calls. End with your final decision.\n\n"
        "CRITICAL: When you decide to trade, you MUST actually call the trade(action='BUY'/'SELL', position='YES'/'NO', usd=amount) tool. "
        "Do not just say you will trade - execute it by calling the tool!"
    )
    system_msg = SystemMessage(content=system_prompt)

    user_msg = HumanMessage(
        content=(
            f"Current UTC: {datetime.now(tz=UTC).isoformat()}\n"
            f"Balance: ${balance:.2f}\n"
            f"YES Holdings: {yes_holdings:.4f} contracts\n"
            f"NO Holdings: {no_holdings:.4f} contracts\n"
            f"Last Trades: {history}"
        )
    )
    result = {
        "messages": [system_msg, user_msg],
        "step": 0,
    }
    
    # Only set defaults if these values don't exist in state (first run)
    if "balance" not in state:
        result["balance"] = balance
    if "yes_holdings" not in state:
        result["yes_holdings"] = yes_holdings
    if "no_holdings" not in state:
        result["no_holdings"] = no_holdings
    if "last_5_actions" not in state:
        result["last_5_actions"] = last_actions
    if "news" not in state:
        result["news"] = news
    if "market_odds" not in state:
        result["market_odds"] = market_odds
    if "last_run_timestamp" not in state:
        result["last_run_timestamp"] = last_run_timestamp
    if "_skip_run" not in state:
        result["_skip_run"] = skip_run
    if "news_usage_cache" not in state:
        result["news_usage_cache"] = {}
        
    return result


# ---------------------------------------------------------------------------
# Node 2 – TraderBrain (LLM call with tools)
# ---------------------------------------------------------------------------


def _is_last_step(step: int) -> bool:
    return step >= MAX_TOOL_STEPS


def call_model(state: AgentState) -> Dict[str, List[AIMessage] | int]:
    """Calls the LLM synchronously and returns the AIMessage."""

    # Set up news cache for the get_news tool
    current_cache = state.get('news_usage_cache', {})
    get_news._usage_cache = current_cache
    print(f"DEBUG: Set news cache with {len(current_cache)} URLs")
    
    # Set up current messages for trade tool rationale extraction
    from .tools import trade
    trade._current_messages = state.get("messages", [])
    print(f"DEBUG: Set current messages for trade tool ({len(state.get('messages', []))} messages)")

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=1).bind_tools(TOOLS)
    
    # Debug: Check what tools are available
    print(f"DEBUG: Available tools: {[tool.name for tool in TOOLS]}")

    response = llm.invoke(state["messages"])
    
    # Debug: Check if LLM tried to make tool calls
    if hasattr(response, 'tool_calls'):
        print(f"DEBUG: LLM tool calls: {len(response.tool_calls)}")
        for call in response.tool_calls:
            print(f"  - Tool: {call.get('name', 'unknown')}, Args: {call.get('args', {})}")
    if hasattr(response, 'invalid_tool_calls'):
        print(f"DEBUG: Invalid tool calls: {len(response.invalid_tool_calls)}")
        for call in response.invalid_tool_calls:
            print(f"  - Invalid: {call}")
    
    # Increment step if we just processed tool results
    new_step = state["step"]
    if state["messages"] and any(hasattr(msg, 'type') and msg.type == 'tool' 
                                for msg in state["messages"]):
        new_step += 1
    
    return {
        "messages": [response],
        "step": new_step,
    }


# ---------------------------------------------------------------------------
# Node 3 – Apply Updates (processes tool results and updates state)
# ---------------------------------------------------------------------------

def apply_updates(state: AgentState) -> Dict[str, float | List[str] | Dict[str, float] | datetime | bool]:
    """Apply state updates from tool results."""
    updates = {}
    
    if not state.get("messages"):
        return updates
    
    # Process ONLY ToolMessages from the most recent tool execution batch
    # Find the most recent AI message and process tool results that come after it
    last_ai_index = -1
    for i in range(len(state["messages"]) - 1, -1, -1):
        msg = state["messages"][i]
        if hasattr(msg, 'type') and msg.type == 'ai':
            last_ai_index = i
            break
    
    # Process only tool messages that come after the last AI message
    recent_tool_messages = []
    if last_ai_index >= 0:
        for i in range(last_ai_index + 1, len(state["messages"])):
            msg = state["messages"][i]
            if hasattr(msg, 'type') and msg.type == 'tool':
                recent_tool_messages.append(msg)
    
    for message in recent_tool_messages:
            
        tool_name = getattr(message, 'name', '')
        tool_content = message.content
        
        # Parse tool content
        if isinstance(tool_content, str):
            try:
                import json
                result = json.loads(tool_content)
            except (json.JSONDecodeError, TypeError):
                # For simple string results (like news headlines)
                result = tool_content
        elif isinstance(tool_content, (dict, list)):
            result = tool_content
        else:
            continue
        
        # Process different tool results - don't break, continue processing all tools
        if tool_name == 'trade':
            # Process trade tool result with validation
            if result and isinstance(result, dict) and result.get('tool_name') == 'trade':
                # Check if trade was successful at tool level (min/max validation)
                if result.get('success', True):  # Default to True for backward compatibility
                    side = result.get('side')
                    usd_amount = result.get('usd_amount', 0)
                    price = result.get('price', 1)
                    
                    current_balance = state.get('balance', 10000.0)
                    current_yes_holdings = state.get('yes_holdings', 0.0)
                    current_no_holdings = state.get('no_holdings', 0.0)
                    
                    # Additional validation: sufficient balance/holdings
                    trade_valid = True
                    error_msg = ""
                    
                    if side in ['BUY_YES', 'BUY_NO']:
                        # For buying, check if we have sufficient balance
                        if usd_amount > current_balance:
                            trade_valid = False
                            error_msg = f"Insufficient balance: ${current_balance:.2f} available, ${usd_amount:.2f} requested"
                    elif side == 'SELL_YES':
                        # For selling YES, check if we have enough YES contracts
                        max_sellable_usd = current_yes_holdings * price
                        if usd_amount > max_sellable_usd:
                            trade_valid = False
                            error_msg = f"Insufficient YES holdings: ${max_sellable_usd:.2f} worth available, ${usd_amount:.2f} requested"
                    elif side == 'SELL_NO':
                        # For selling NO, check if we have enough NO contracts
                        max_sellable_usd = current_no_holdings * price
                        if usd_amount > max_sellable_usd:
                            trade_valid = False
                            error_msg = f"Insufficient NO holdings: ${max_sellable_usd:.2f} worth available, ${usd_amount:.2f} requested"
                    
                    if trade_valid:
                        # Execute the trade - update balance and appropriate holdings
                        contracts = usd_amount / price
                        
                        if side == 'BUY_YES':
                            updates['balance'] = current_balance - usd_amount
                            updates['yes_holdings'] = current_yes_holdings + contracts
                        elif side == 'BUY_NO':
                            updates['balance'] = current_balance - usd_amount
                            updates['no_holdings'] = current_no_holdings + contracts
                        elif side == 'SELL_YES':
                            updates['balance'] = current_balance + usd_amount
                            updates['yes_holdings'] = current_yes_holdings - contracts
                        elif side == 'SELL_NO':
                            updates['balance'] = current_balance + usd_amount
                            updates['no_holdings'] = current_no_holdings - contracts
                        
                        # Update action history for successful trades
                        if 'action_summary' in result:
                            current_actions = state.get('last_5_actions', [])
                            new_actions = (current_actions + [result['action_summary']])[-5:]
                            updates['last_5_actions'] = new_actions
                    else:
                        # Trade failed validation - add error to action history
                        current_actions = state.get('last_5_actions', [])
                        new_actions = (current_actions + [f"FAILED: {error_msg}"])[-5:]
                        updates['last_5_actions'] = new_actions
                else:
                    # Trade failed at tool level - add error to action history
                    # Check if we have a custom action_summary (e.g., for user cancellations)
                    if 'action_summary' in result:
                        action_msg = result['action_summary']
                    else:
                        error_msg = result.get('error', 'Trade failed')
                        action_msg = f"FAILED: {error_msg}"
                    
                    current_actions = state.get('last_5_actions', [])
                    new_actions = (current_actions + [action_msg])[-5:]
                    updates['last_5_actions'] = new_actions
                    
                    # If user cancelled the trade, set flag to end the run
                    if result.get('user_cancelled'):
                        updates['_user_cancelled_trade'] = True
        
        elif tool_name == 'get_news':
            # Process news tool result and update usage cache
            if isinstance(result, list) and len(result) > 0:
                updates['news'] = result
                
                # Update news usage cache for the returned news item
                news_item = result[0]  # Should be exactly one item
                if isinstance(news_item, dict) and 'url' in news_item:
                    current_cache = state.get('news_usage_cache', {})
                    updated_cache = _increment_url_usage(current_cache, news_item['url'])
                    updates['news_usage_cache'] = updated_cache
                    print(f"DEBUG: Updated news cache, URL {news_item['url']} now used {updated_cache.get(news_item['url'], 1)} times")
        
        elif tool_name == 'get_market_data':
            # Process market odds tool result
            if isinstance(result, dict):
                updates['market_odds'] = result
    
    # Update last run timestamp to mark completion of this run
    updates['last_run_timestamp'] = datetime.now(tz=UTC)
    
    return updates


# ---------------------------------------------------------------------------
# Node 0 – Entry Guard (preflight checks)
# ---------------------------------------------------------------------------

def entry_guard(state: AgentState) -> Dict[str, List[str] | Dict[str, float] | datetime | bool | None]:
    """Performs preflight checks before running the trading agent.
    
    Checks:
    1. Price change threshold (basis points)
    2. Time between runs
    3. New news requirement
    
    If any check fails, routes to end. Otherwise clears stale data.
    DEBUG_MODE skips all checks.
    """
    now = datetime.now(tz=UTC)
    updates = {}
    
    print(f"ENTRY_GUARD: Starting checks at {now.strftime('%H:%M:%S')}")
    
    # DEBUG MODE: Skip all checks
    if DEBUG_MODE:
        print("ENTRY_GUARD: DEBUG_MODE enabled - skipping all checks")
        updates["_skip_run"] = False
        print(f"ENTRY_GUARD: Returning updates: {updates}")
        return updates
    
    # Get current market data and news (without storing in state)
    current_market_odds = get_market_data.invoke({})
    # For entry_guard, we don't need to pass cache since we're just checking for changes
    current_news = get_news.invoke({"search_query": "latest market news"})
    
    # Check 1: Time between runs
    last_run = state.get("last_run_timestamp")
    if last_run is not None:
        time_since_last = (now - last_run).total_seconds()
        print(f"ENTRY_GUARD: Time since last run: {time_since_last:.0f}s (need {SEC_TIME_BETWEEN_RUNS}s)")
        if time_since_last < SEC_TIME_BETWEEN_RUNS:
            reason = f"Time check: only {time_since_last:.0f}s since last run (need {SEC_TIME_BETWEEN_RUNS}s)"
            print(f"ENTRY_GUARD: Skipping run - {reason}")
            return {"_skip_run": True, "skip_reason": reason}
    else:
        print("ENTRY_GUARD: No previous run timestamp - time check passed")
    
    # Check 2: Price change threshold
    old_market_odds = state.get("market_odds")
    if old_market_odds is not None:
        old_yes_price = old_market_odds.get("yes_price", 0)
        new_yes_price = current_market_odds.get("yes_price", 0)
        
        price_change_bps = abs(new_yes_price - old_yes_price) * 10000  # Convert to basis points
        print(f"ENTRY_GUARD: Price change: {old_yes_price} → {new_yes_price} ({price_change_bps:.1f}bps, need {MIN_PRICE_CHANGE_BPS}bps)")
        if price_change_bps < MIN_PRICE_CHANGE_BPS:
            reason = f"Price check: change {price_change_bps:.1f}bps < {MIN_PRICE_CHANGE_BPS}bps threshold"
            print(f"ENTRY_GUARD: Skipping run - {reason}")
            return {"_skip_run": True, "skip_reason": reason}
    else:
        print("ENTRY_GUARD: No previous market odds - price check skipped")
    
    # Check 3: New news requirement
    if REQUIRE_NEW_NEWS:
        old_news = state.get("news")
        if old_news is not None:
            news_changed = old_news != current_news
            print(f"ENTRY_GUARD: News changed: {news_changed}")
            if not news_changed:
                reason = "News check: no new news available"
                print(f"ENTRY_GUARD: Skipping run - {reason}")
                return {"_skip_run": True, "skip_reason": reason}
        else:
            print("ENTRY_GUARD: No previous news - news check skipped")
    else:
        print("ENTRY_GUARD: News requirement disabled")
    
    # All checks passed - clear stale data and proceed
    print("ENTRY_GUARD: All checks passed, clearing stale data and proceeding")
    # Clear the _skip_run flag from previous runs
    updates["_skip_run"] = False
    
    print(f"ENTRY_GUARD: Returning updates: {updates}")
    return updates


# ---------------------------------------------------------------------------
# Edge routing function
# ---------------------------------------------------------------------------

def route_after_entry_guard(state: AgentState) -> Literal["context", "__end__"]:
    """Routes after entry guard - skip if checks failed, otherwise continue."""
    skip_run = state.get("_skip_run")
    print(f"ROUTING: _skip_run = {skip_run}")
    if skip_run:
        print("ROUTING: Going to __end__")
        return "__end__"
    print("ROUTING: Going to context")
    return "context"


def route_after_llm(state: AgentState) -> Literal["tools", "__end__"]:
    """Routes the flow based on tool calls or step limit."""

    if _is_last_step(state["step"]):
        return "__end__"

    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return "__end__"


def route_after_apply_updates(state: AgentState) -> Literal["llm", "__end__"]:
    """Routes after apply_updates - check for user cancellation."""
    
    if state.get("_user_cancelled_trade"):
        print("ROUTING: User cancelled trade - ending run")
        return "__end__"
    
    return "llm"


# ---------------------------------------------------------------------------
# Assemble graph
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("entry_guard", entry_guard)
builder.add_node("context", build_context)
builder.add_node("llm", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("apply_updates", apply_updates)

builder.add_edge("__start__", "entry_guard")
builder.add_conditional_edges("entry_guard", route_after_entry_guard)
builder.add_edge("context", "llm")

# After LLM, decide whether __end__ or tools
builder.add_conditional_edges("llm", route_after_llm)
# After tools, apply updates then check for cancellation
builder.add_edge("tools", "apply_updates")
builder.add_conditional_edges("apply_updates", route_after_apply_updates)


# from langgraph.checkpoint.memory import InMemorySaver
# agent_graph = builder.compile(checkpointer=InMemorySaver(), name="PolymarketTraderAgent")

# Langgraph API mode/studio compatible (local mode)
agent_graph = builder.compile(name="PolymarketTraderAgent")

__all__ = [
    "agent_graph",
    "builder",
    "AgentState",
]
