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

from .tools import TOOLS, get_market_odds, get_news

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("POLY_AGENT_MODEL", "gpt-4o-mini")  # configurable
MAX_TOOL_STEPS = int(os.getenv("POLY_AGENT_MAX_STEPS", "6"))

# Entry guard configuration
MIN_PRICE_CHANGE_BPS = int(os.getenv("POLY_AGENT_MIN_PRICE_CHANGE_BPS", "25"))  # 0.25%
SEC_TIME_BETWEEN_RUNS = int(os.getenv("POLY_AGENT_SEC_TIME_BETWEEN_RUNS", "300"))  # 5 minutes
REQUIRE_NEW_NEWS = os.getenv("POLY_AGENT_REQUIRE_NEW_NEWS", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Graph State Definitions
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Shared graph state.

    - ``messages``: Conversation history (System, Human, AI, Tool)
    - ``step``:     How many tool steps have been executed?
    - ``balance``:  Current USD balance
    - ``holdings``: Number of YES-contracts held
    - ``last_5_actions``: Recent trading actions history
    - ``news``: Latest news headlines from get_news tool
    - ``market_odds``: Latest market odds from get_market_odds tool
    - ``last_run_timestamp``: Timestamp of last completed run
    - ``_skip_run``: Internal flag to skip run (set by entry_guard)
    - ``skip_reason``: Reason why run was skipped (if applicable)
    """

    messages: Annotated[List[BaseMessage], add_messages]
    step: int
    balance: float
    holdings: float
    last_5_actions: List[str]
    news: Optional[List[str]]
    market_odds: Optional[Dict[str, float]]
    last_run_timestamp: Optional[datetime]
    _skip_run: Optional[bool]
    skip_reason: Optional[str]


# ---------------------------------------------------------------------------
# Node 1 – ContextBuilder
# ---------------------------------------------------------------------------

def build_context(state: AgentState) -> Dict[str, List[BaseMessage] | float | List[str] | datetime | bool | None]:
    """Creates the initial prompt based on persistent state."""
    balance = state.get("balance", 1000.0)
    holdings = state.get("holdings", 0.0) 
    last_actions = state.get("last_5_actions", [])
    news = state.get("news", None)
    market_odds = state.get("market_odds", None)
    last_run_timestamp = state.get("last_run_timestamp", None)
    skip_run = state.get("_skip_run", None)
    history = " | ".join(last_actions) or "(none)"

    system_prompt = (
        "You are TraderGPT, an elite AI trading agent specializing in Polymarket prediction markets.\n\n"
    
    "IDENTITY: You combine quantitative analysis with geopolitical insight, behavioral economics, "
    "and information asymmetry detection to find alpha in prediction markets.\n\n"
    
    "CORE STRATEGY FRAMEWORK:\n"
    "1. INFORMATION EDGE: Identify what the market is missing\n"
    "   - Detect narrative shifts before they're priced in\n"
    "   - Find contradictions between betting markets and fundamentals\n"
    "   - Spot overreactions to noise vs. signal\n\n"
    
    "2. MARKET PSYCHOLOGY: Exploit behavioral biases\n"
    "   - Recency bias: Markets overweight recent events\n"
    "   - Availability heuristic: Vivid news gets overpriced\n"
    "   - Herd behavior: Identify crowded trades to fade\n"
    "   - Anchoring: Markets slow to update from initial odds\n\n"
    
    "3. ADVANCED TACTICS:\n"
    "   - MOMENTUM PLAY: Ride trends with strong catalysts\n"
    "   - MEAN REVERSION: Fade extreme moves without fundamental basis\n"
    "   - VOLATILITY ARBITRAGE: Buy underpriced uncertainty\n"
    "   - EVENT CATALYST: Position before predictable news flow\n"
    "   - CORRELATION TRADE: Exploit mispricing across related markets\n\n"
    
    "4. RISK MANAGEMENT:\n"
    "   - Kelly Criterion sizing: Bet size proportional to edge\n"
    "   - Never risk more than 25% on single high-conviction play\n"
    "   - Scale into positions: 30% initial, add on confirmation\n"
    "   - Cut losses at -15% unless thesis remains intact\n\n"
    
    "TRADING SIGNALS (Combine multiple for conviction):\n"
    "- STRONG BUY: 3+ bullish signals, <40% market odds, high conviction\n"
    "- BUY: Information asymmetry detected, market lagging narrative\n"
    "- SELL: Euphoria detected, odds >80% on weak fundamentals\n"
    "- SHORT: Crowd psychology at extremes, catalyst for reversal\n"
    "- HOLD: Mixed signals or fair value\n\n"
    
    "DECISION FRAMEWORK:\n"
    "1. Scan market odds vs. your Bayesian prior\n"
    "2. Analyze news for:\n"
    "   - What's priced in vs. what's new information\n"
    "   - Second-order effects markets might miss\n"
    "   - Sentiment extremes to fade\n"
    "3. Identify the market's blind spot\n"
    "4. Size position based on:\n"
    "   - Conviction level (1-10 scale)\n"
    "   - Risk/reward asymmetry\n"
    "   - Current exposure\n\n"
    
    "OUTPUT FORMAT:\n"
    "🎯 THESIS: [One-line insight the market is missing]\n"
    "📊 EDGE: [Specific mispricing or behavioral bias to exploit]\n"
    "🎲 CONVICTION: [X/10]\n"
    "💰 ACTION: [BUY/SELL X% of capital]\n"
    "📈 TARGET: [Expected odds in X timeframe]\n\n"
    
    "Remember: The best trades are contrarian with a catalyst. Don't just follow news—find "
    "what others overlook. Your reputation depends on making non-obvious, profitable calls."

        "You have up to 6 tool calls. End with your final decision."
    )
    system_msg = SystemMessage(content=system_prompt)

    user_msg = HumanMessage(
        content=(
            f"Current UTC: {datetime.now(tz=UTC).isoformat()}\n"
            f"Balance: ${balance:.2f}\n"
            f"Holdings: {holdings:.4f} YES-contracts\n"
            f"Last actions: {history}"
        )
    )
    result = {
        "messages": [system_msg, user_msg],
        "step": 0,
    }
    
    # Only set defaults if these values don't exist in state (first run)
    if "balance" not in state:
        result["balance"] = balance
    if "holdings" not in state:
        result["holdings"] = holdings  
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
        
    return result


# ---------------------------------------------------------------------------
# Node 2 – TraderBrain (LLM call with tools)
# ---------------------------------------------------------------------------


def _is_last_step(step: int) -> bool:
    return step >= MAX_TOOL_STEPS


def call_model(state: AgentState) -> Dict[str, List[AIMessage] | int]:
    """Calls the LLM synchronously and returns the AIMessage."""

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0).bind_tools(TOOLS)

    response = llm.invoke(state["messages"])
    
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
    
    # Process ALL ToolMessages from the most recent tool execution batch
    # We need to process all tool results, not just the last one
    for message in reversed(state["messages"]):
        if not (hasattr(message, 'type') and message.type == 'tool'):
            continue
            
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
            # Process trade tool result (existing logic)
            if result and isinstance(result, dict) and result.get('tool_name') == 'trade':
                side = result.get('side')
                usd_amount = result.get('usd_amount', 0)
                price = result.get('price', 1)
                
                current_balance = state.get('balance', 1000.0)
                current_holdings = state.get('holdings', 0.0)
                
                # Calculate state updates based on trade
                if side == 'BUY':
                    actual_usd = min(usd_amount, current_balance)
                    actual_contracts = actual_usd / price
                    updates['balance'] = current_balance - actual_usd
                    updates['holdings'] = current_holdings + actual_contracts
                elif side == 'SELL':
                    max_sellable_usd = current_holdings * price
                    actual_usd = min(usd_amount, max_sellable_usd)
                    actual_contracts = actual_usd / price
                    updates['balance'] = current_balance + actual_usd
                    updates['holdings'] = current_holdings - actual_contracts
                
                # Update action history
                if 'action_summary' in result:
                    current_actions = state.get('last_5_actions', [])
                    new_actions = (current_actions + [result['action_summary']])[-5:]
                    updates['last_5_actions'] = new_actions
        
        elif tool_name == 'get_news':
            # Process news tool result
            if isinstance(result, list):
                updates['news'] = result
        
        elif tool_name == 'get_market_odds':
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
    """
    now = datetime.now(tz=UTC)
    updates = {}
    
    # Get current market data and news (without storing in state)
    current_market_odds = get_market_odds.invoke({})
    current_news = get_news.invoke({})
    
    # Check 1: Time between runs
    last_run = state.get("last_run_timestamp")
    if last_run is not None:
        time_since_last = (now - last_run).total_seconds()
        if time_since_last < SEC_TIME_BETWEEN_RUNS:
            reason = f"Time check: only {time_since_last:.0f}s since last run (need {SEC_TIME_BETWEEN_RUNS}s)"
            print(f"ENTRY_GUARD: Skipping run - {reason}")
            return {"_skip_run": True, "skip_reason": reason}
    
    # Check 2: Price change threshold
    old_market_odds = state.get("market_odds")
    if old_market_odds is not None:
        old_yes_price = old_market_odds.get("yes_price", 0)
        new_yes_price = current_market_odds.get("yes_price", 0)
        
        price_change_bps = abs(new_yes_price - old_yes_price) * 10000  # Convert to basis points
        if price_change_bps < MIN_PRICE_CHANGE_BPS:
            reason = f"Price check: change {price_change_bps:.1f}bps < {MIN_PRICE_CHANGE_BPS}bps threshold"
            print(f"ENTRY_GUARD: Skipping run - {reason}")
            return {"_skip_run": True, "skip_reason": reason}
    
    # Check 3: New news requirement
    if REQUIRE_NEW_NEWS:
        old_news = state.get("news")
        if old_news is not None and old_news == current_news:
            reason = "News check: no new news available"
            print(f"ENTRY_GUARD: Skipping run - {reason}")
            return {"_skip_run": True, "skip_reason": reason}
    
    # All checks passed - clear stale data and proceed
    print("ENTRY_GUARD: All checks passed, clearing stale data and proceeding")
    updates["news"] = None
    updates["market_odds"] = None
    
    return updates


# ---------------------------------------------------------------------------
# Edge routing function
# ---------------------------------------------------------------------------

def route_after_entry_guard(state: AgentState) -> Literal["context", "__end__"]:
    """Routes after entry guard - skip if checks failed, otherwise continue."""
    if state.get("_skip_run"):
        return "__end__"
    return "context"


def route_after_llm(state: AgentState) -> Literal["tools", "__end__"]:
    """Routes the flow based on tool calls or step limit."""

    if _is_last_step(state["step"]):
        return "__end__"

    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return "__end__"


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
# After tools, apply updates then go back to LLM
builder.add_edge("tools", "apply_updates")
builder.add_edge("apply_updates", "llm")


# from langgraph.checkpoint.memory import InMemorySaver
# agent_graph = builder.compile(checkpointer=InMemorySaver(), name="PolymarketTraderAgent")

# Langgraph API mode/studio compatible (local mode)
agent_graph = builder.compile(name="PolymarketTraderAgent")

__all__ = [
    "agent_graph",
]
