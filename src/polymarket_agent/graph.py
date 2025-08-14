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
from typing import Dict, List, Literal
from typing_extensions import TypedDict, Annotated

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import TOOLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("POLY_AGENT_MODEL", "gpt-4o-mini")  # configurable
MAX_TOOL_STEPS = int(os.getenv("POLY_AGENT_MAX_STEPS", "6"))

# ---------------------------------------------------------------------------
# Graph State Definitions
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Shared graph state.

    - ``messages``: Conversation history (System, Human, AI, Tool)
    - ``step``:     How many tool steps have been executed?
    - ``balance``:  Current USD balance
    - ``holdings``: Number of YES-contracts held
    - ``last_5_actions``: Recent trading actions history
    """

    messages: Annotated[List[BaseMessage], add_messages]
    step: int
    balance: float
    holdings: float
    last_5_actions: List[str]


# ---------------------------------------------------------------------------
# Node 1 – ContextBuilder
# ---------------------------------------------------------------------------

def build_context(state: AgentState) -> Dict[str, List[BaseMessage] | float | List[str]]:
    """Creates the initial prompt based on persistent state."""
    balance = state.get("balance", 1000.0)
    holdings = state.get("holdings", 0.0) 
    last_actions = state.get("last_5_actions", [])
    history = " | ".join(last_actions) or "(none)"

    system_prompt = (
        "You are TraderGPT, an autonomous trading agent for Polymarket prediction markets.\n\n"
        "OBJECTIVE: Maximize profit by trading YES/NO contracts based on market odds and news sentiment.\n\n"
        "TRADING RULES:\n"
        "- BUY when: Market undervalues YES probability vs. your analysis\n"
        "- SELL when: Market overvalues YES probability vs. your analysis  \n"
        "- HOLD when: Fair pricing or insufficient signal\n\n"
        "PROCESS:\n"
        "1. Get current market odds\n"
        "2. Get recent news headlines\n"
        "3. Analyze sentiment to gauge market direction\n"
        "4. Make trading decision based on combined analysis\n\n"
        "IMPORTANT:\n"
        "- Trade amounts are automatically adjusted to your maximum available funds/holdings\n"
        "- Consider position size relative to conviction level\n"
        "- Factor in current holdings when making decisions\n"
        "- Be concise in your reasoning\n\n"
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

def apply_updates(state: AgentState) -> Dict[str, float | List[str]]:
    """Apply state updates from tool results."""
    updates = {}
    
    if not state.get("messages"):
        return updates
    
    # Look for ToolMessages from the most recent tool execution
    for message in reversed(state["messages"]):
        # Only process trade tool results for state updates
        if (hasattr(message, 'type') and message.type == 'tool' and 
            getattr(message, 'name', '') == 'trade'):
            
            tool_content = message.content
            
            # Parse tool content
            if isinstance(tool_content, str):
                try:
                    import json
                    result = json.loads(tool_content)
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(tool_content, dict):
                result = tool_content
            else:
                continue
            
            # Process trade tool result
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
                
                break  # Only process most recent trade
    
    return updates


# ---------------------------------------------------------------------------
# Edge routing function
# ---------------------------------------------------------------------------

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

builder.add_node("context", build_context)
builder.add_node("llm", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("apply_updates", apply_updates)

builder.add_edge("__start__", "context")
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
