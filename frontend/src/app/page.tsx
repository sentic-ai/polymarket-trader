"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import Lottie from "lottie-react";
import { useState, useRef, useEffect } from "react";
import confetti from "canvas-confetti";
import logoAnimation from "../../public/logo_lottie.json";

interface MarketData {
  question: string;
  icon: string;
  outcomePrices: string[];
  volume: string;
  cached?: boolean;
  fetchedAt?: string;
}

interface AgentStats {
  balance?: number;
  runs_last_24h?: number;
  yes_holdings?: number;
  no_holdings?: number;
  last_actions?: unknown[];
}

// Run interface for /runs endpoint - simplified to only what we need
interface RunData {
  id: string;
  status: string;
  completed_at: string | null;
  logs: Array<{
    type: string;
    data?: unknown;
    node_name?: string;
    output?: unknown;
  }>;
}

// WebSocket message types
interface WebSocketMessage {
  type: string;
  data?: {
    runs?: Array<{ agent_id: string; [key: string]: unknown }>;
    agent_id?: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

// Helper function to check if a WebSocket message is relevant to this agent
function isMessageForThisAgent(
  message: WebSocketMessage,
  currentAgentId: string,
  runToAgentMap: Map<string, string>
): boolean {
  // Debug logging
  console.log(`🔍 Filtering message:`, {
    type: message.type,
    currentAgentId,
    hasData: !!message.data,
    dataAgentId: message.data?.agent_id,
    dataKeys: message.data ? Object.keys(message.data) : [],
  });

  // Always show system-wide messages (like execution_state updates)
  if (message.type === "execution_state") {
    console.log(`✅ Allowing execution_state message`);
    return true; // System messages relevant to all
  }

  // For run-related messages, check if any runs belong to this agent
  if (message.data?.runs) {
    // Message contains runs array - check if any runs belong to this agent
    const relevantRuns = message.data.runs.filter(
      (run) => run.agent_id === currentAgentId
    );
    const result = relevantRuns.length > 0;
    console.log(
      `🔍 Runs array message: ${result ? "✅ ALLOW" : "❌ BLOCK"} (relevant: ${
        relevantRuns.length
      })`
    );
    return result;
  }

  // For single run messages, check the run's agent_id
  if (message.data?.agent_id) {
    const result = message.data.agent_id === currentAgentId;
    console.log(
      `🔍 Single run message: ${result ? "✅ ALLOW" : "❌ BLOCK"} (${
        message.data.agent_id
      } vs ${currentAgentId})`
    );
    return result;
  }

  // For messages with run_id (like node_update), check if the run belongs to this agent
  const messageWithRunId = message as WebSocketMessage & { run_id?: string };
  const messageRunId = messageWithRunId.run_id;
  if (messageRunId && runToAgentMap.has(messageRunId)) {
    const runAgentId = runToAgentMap.get(messageRunId);
    const result = runAgentId === currentAgentId;
    console.log(
      `🔍 Run ID message: ${
        result ? "✅ ALLOW" : "❌ BLOCK"
      } (run ${messageRunId} belongs to ${runAgentId} vs ${currentAgentId})`
    );
    return result;
  }

  console.log(
    `❌ BLOCKING message - no agent identification (run_id: ${messageRunId}, mapped: ${runToAgentMap.has(
      messageRunId || ""
    )})`
  );
  return false;
}

export default function Home() {
  // Get agent configuration dynamically from URL or environment variables
  const [AGENT_ID, setAgentId] = useState<string>("");
  const [BACKEND_URL, setBackendUrl] = useState<string>("");

  // Track run_id to agent_id mapping for message filtering
  const runToAgentMapping = useRef<Map<string, string>>(new Map());

  useEffect(() => {
    // Extract agent ID from subdomain (e.g., polymarket-trader-u7i1502d.sentic.ai)
    const hostname =
      typeof window !== "undefined" ? window.location.hostname : "";
    let agentId = "";
    let backendUrl = "";

    if (hostname.includes(".sentic.ai")) {
      // Extract agent ID from subdomain
      agentId = hostname.split(".")[0];
      backendUrl =
        process.env.NEXT_PUBLIC_BACKEND_URL || "https://api.sentic.ai";
    } else {
      // Fallback to environment variables for localhost
      agentId = process.env.NEXT_PUBLIC_AGENT_ID || "polymarket-trader";
      backendUrl =
        process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    }

    console.log(`🎯 Dynamic Agent ID: ${agentId}, Backend: ${backendUrl}`);
    console.log(`🌐 Current hostname: ${hostname}`);
    setAgentId(agentId);
    setBackendUrl(backendUrl);
  }, []);

  const MARKET_DATA_API_URL = "/api/polymarket/market";
  const INITIAL_AMOUNT = 10000;
  const [isAgentRunning, setIsAgentRunning] = useState(false);
  const [runCounter, setRunCounter] = useState(235);
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [agentStats, setAgentStats] = useState<AgentStats | null>(null);
  const [totalAgents, setTotalAgents] = useState<number>(0);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  const isAutoScrolling = useRef(false);
  const [showAgentRequest, setShowAgentRequest] = useState(false);
  const [pendingTradeDetails, setPendingTradeDetails] = useState<{
    action: string;
    amount: string;
    price: string;
  } | null>(null);
  const [, setIsWaitingForApproval] = useState(false);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const approvalResolveRef = useRef<((approved: boolean) => void) | null>(null);
  const agentRequestRef = useRef<HTMLDivElement>(null);
  const [completedRuns, setCompletedRuns] = useState<
    Array<{
      id: number;
      runId: string;
      fullRunId: string;
      profit: string;
      action: string;
      tradePrice?: number;
      tradeAmount?: number;
      timestamp: string;
      cost: string;
      rationale: string;
      confidence: number;
      status: string;
      logs?: Array<{
        type: string;
        data?: unknown;
        node_name?: string;
        output?: unknown;
      }>;
    }>
  >([]);
  const [terminalContent, setTerminalContent] = useState<
    Array<{
      time: string;
      type: string;
      message: string;
      color: string;
      extra?: string;
    }>
  >([
    // Dummy data
    // {
    //   time: "[12:45:23]",
    //   type: "WARN",
    //   message: "Market volatility increasing",
    //   color: "text-amber-400",
    // },
    // {
    //   time: "[12:23:15]",
    //   type: "SUCCESS",
    //   message: "Sold 50 YES @ $0.48",
    //   extra: "+$12.00",
    //   color: "text-emerald-400",
    // },
    // {
    //   time: "[11:45:08]",
    //   type: "INFO",
    //   message: "Price trending upward",
    //   color: "text-blue-400",
    // },
    // {
    //   time: "[11:30:42]",
    //   type: "SUCCESS",
    //   message: "Bought 100 YES @ $0.41",
    //   color: "text-emerald-400",
    // },
    // {
    //   time: "[11:15:33]",
    //   type: "EXEC",
    //   message: "Running market analysis...",
    //   color: "text-cyan-400",
    // },
    // {
    //   time: "[11:14:12]",
    //   type: "AGENT",
    //   message: "Strategy updated: momentum_v2.1",
    //   color: "text-violet-400",
    // },
  ]);
  const missionControlRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<HTMLDivElement>(null);
  const generalWsRef = useRef<WebSocket | null>(null);
  const wsRetryCountRef = useRef(0);
  const wsConnectedTimeRef = useRef<number | null>(null);
  const wsRetryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const statsIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const marketDataIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const MAX_RETRIES = 5;

  // Auto-scroll helper functions
  const isNearBottom = (element: HTMLDivElement) => {
    const threshold = 5; // Allow 50px margin from bottom
    return (
      element.scrollTop + element.clientHeight >=
      element.scrollHeight - threshold
    );
  };

  const scrollToBottom = (element: HTMLDivElement) => {
    element.scrollTop = element.scrollHeight;
  };

  // Format volume with k/M/B suffixes
  const formatVolume = (volumeStr: string): string => {
    const volume = parseFloat(volumeStr);
    if (volume >= 1000000000) {
      return `$${(volume / 1000000000).toFixed(1)}B`;
    } else if (volume >= 1000000) {
      return `$${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `$${(volume / 1000).toFixed(0)}k`;
    } else {
      return `$${volume.toFixed(0)}`;
    }
  };

  const calculateTotalBalance = (): number => {
    if (!agentStats || !marketData) return INITIAL_AMOUNT;
    const cash = agentStats.balance || 0;
    const yesValue =
      (agentStats.yes_holdings || 0) *
      parseFloat(marketData.outcomePrices[0] || "0");
    const noValue =
      (agentStats.no_holdings || 0) *
      parseFloat(marketData.outcomePrices[1] || "0");

    return cash + yesValue + noValue;
  };

  const calculateProfit = () => {
    const totalBalance = calculateTotalBalance();
    const lifetimeProfit = totalBalance - INITIAL_AMOUNT;
    const profitPercentage = (lifetimeProfit / INITIAL_AMOUNT) * 100;

    return {
      profit: lifetimeProfit,
      percentage: profitPercentage,
      totalBalance: totalBalance,
    };
  };

  const getLatestRunInfo = (): string => {
    if (completedRuns.length === 0) {
      return "No runs completed yet";
    }

    const latestRun = completedRuns[0];

    const now = new Date();
    const runTime = new Date(`${now.toDateString()} ${latestRun.timestamp}`);
    const diffMs = now.getTime() - runTime.getTime();
    const diffMinutes = Math.floor(diffMs / (1000 * 60));

    let timeAgo: string;
    if (diffMinutes < 1) {
      timeAgo = "Just now";
    } else if (diffMinutes === 1) {
      timeAgo = "1 minute ago";
    } else if (diffMinutes < 60) {
      timeAgo = `${diffMinutes} minutes ago`;
    } else {
      const diffHours = Math.floor(diffMinutes / 60);
      if (diffHours === 1) {
        timeAgo = "1 hour ago";
      } else {
        timeAgo = `${diffHours} hours ago`;
      }
    }

    const displayAction =
      latestRun.action === "HOLD" ? "NO TRADE" : latestRun.action;
    return `Last run: ${timeAgo} (${displayAction})`;
  };

  const getActivityFeedItems = () => {
    if (completedRuns.length === 0) {
      return [
        {
          time: "--:--",
          action: "Waiting for runs...",
          color: "font-medium text-zinc-400",
        },
      ];
    }

    // Format all completed runs
    const realItems = completedRuns.map((run) => {
      const displayAction = run.action === "HOLD" ? "NO TRADE" : run.action;

      let color: string;
      if (displayAction === "NO TRADE") {
        color = "font-medium";
      } else if (displayAction.startsWith("BUY")) {
        color = "text-emerald-600 dark:text-emerald-400 font-medium";
      } else if (displayAction.startsWith("SELL")) {
        color = "text-red-500 dark:text-red-400 font-medium";
      } else {
        color = "font-medium";
      }

      return {
        time: run.timestamp.slice(0, 5),
        action: displayAction,
        color: color,
      };
    });

    const items = [];
    for (let i = 0; i < 8; i++) {
      items.push(realItems[i % realItems.length]);
    }

    return items;
  };

  // Fetch agent stats from API
  const fetchAgentStats = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/agents/${AGENT_ID}/status`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const stats = (await response.json()) as AgentStats;
      console.log("🔍 Agent stats:", stats);
      setAgentStats(stats);
    } catch (error) {
      console.error("❌ Error fetching agent stats:", error);
    }
  };

  // Fetch total agents count from API
  const fetchTotalAgents = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/agents`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = (await response.json()) as { total_agents: number };
      console.log("📊 Total agents:", data.total_agents);
      setTotalAgents(data.total_agents);
    } catch (error) {
      console.error("❌ Error fetching total agents:", error);
    }
  };

  const fetchMarketData = async () => {
    try {
      const response = await fetch(MARKET_DATA_API_URL);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = (await response.json()) as MarketData;
      console.log("📊 Market data:", data);
      setMarketData(data);
    } catch (error) {
      console.error("❌ Error fetching market data:", error);
    }
  };

  const extractEdgeFromLogs = (logs: RunData["logs"]): string => {
    const llmLogs = logs.filter(
      (log) => log.type === "node_update" && log.node_name === "llm"
    );

    for (const log of llmLogs) {
      try {
        const output = log.output as { messages?: Array<{ content?: string }> };
        const content = output?.messages?.[0]?.content;
        if (content && typeof content === "string") {
          const edgeMatch = content.match(/📊 EDGE: (.+?)(?=\n|$)/);
          if (edgeMatch && edgeMatch[1]) {
            return edgeMatch[1].trim();
          }
        }
      } catch {}
    }

    console.error("❌ EDGE not found in logs for run");
    return "Error: Could not extract trading rationale from logs";
  };

  const extractConvictionFromLogs = (logs: RunData["logs"]): number => {
    const llmLogs = logs.filter(
      (log) => log.type === "node_update" && log.node_name === "llm"
    );

    for (const log of llmLogs) {
      try {
        const output = log.output as { messages?: Array<{ content?: string }> };
        const content = output?.messages?.[0]?.content;
        if (content && typeof content === "string") {
          const convictionMatch = content.match(/🎲 CONVICTION: (\d+)\/10/);
          if (convictionMatch && convictionMatch[1]) {
            return parseInt(convictionMatch[1]) * 10; // Convert X/10 to percentage
          }
        }
      } catch {}
    }

    console.warn("❌ CONVICTION not found in logs for run");
    return 0;
  };

  const extractTradeActionFromLogs = (
    logs: RunData["logs"]
  ): { action: string; price?: number; amount?: number } => {
    const toolsLogs = logs.filter(
      (log) => log.type === "node_update" && log.node_name === "tools"
    );

    for (const log of toolsLogs) {
      try {
        const output = log.output as { messages?: Array<{ content?: string }> };
        const content = output?.messages?.[0]?.content;
        if (content && typeof content === "string") {
          const parsedContent = JSON.parse(content);
          if (
            parsedContent.tool_name === "trade" &&
            parsedContent.success === true
          ) {
            const side = parsedContent.side; // BUY_YES, BUY_NO, SELL_YES, SELL_NO
            const price = parsedContent.price;
            const amount = parsedContent.usd_amount;
            const contracts = parsedContent.contracts;

            let action: string;
            if (side.startsWith("BUY_")) {
              action = `BUY ${Math.round(contracts)} ${side.split("_")[1]}`;
            } else if (side.startsWith("SELL_")) {
              action = `SELL ${Math.round(contracts)} ${side.split("_")[1]}`;
            } else {
              action = side;
            }

            return { action, price, amount };
          }
        }
      } catch {}
    }

    return { action: "HOLD" };
  };

  const fetchRuns = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/agents/${AGENT_ID}/runs`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const runs = (await response.json()) as RunData[];
      console.log("🏃 Runs data:", runs);

      // Populate run_id to agent_id mapping for existing runs
      runs.forEach((run) => {
        if (run.id && !runToAgentMapping.current.has(run.id)) {
          runToAgentMapping.current.set(run.id, AGENT_ID);
          console.log(
            `🗺️ Added run mapping (from fetch): ${run.id} -> ${AGENT_ID}`
          );
        }
      });

      // Convert runs to completed runs format with short IDs and extracted data
      const convertedRuns = runs
        .filter((run) => run.status === "completed")
        .map((run, index) => {
          // Extract rationale, confidence, and trade action from THIS specific run's logs
          const rationale = extractEdgeFromLogs(run.logs);
          const confidence = extractConvictionFromLogs(run.logs);
          const tradeAction = extractTradeActionFromLogs(run.logs);

          return {
            id: index + 1,
            runId: `#${run.id.slice(0, 4)}`, // First 4 chars as display ID
            fullRunId: run.id, // Full UUID for the ID line
            profit: "+$12.50", // Keep dummy data for now
            action: tradeAction.action,
            tradePrice: tradeAction.price,
            tradeAmount: tradeAction.amount,
            timestamp: run.completed_at
              ? new Date(run.completed_at).toLocaleTimeString("en-US")
              : "Unknown",
            cost: "Free Run",
            rationale: rationale,
            confidence: confidence,
            status: run.status,
            logs: run.logs,
          };
        })
        .reverse(); // Most recent first

      setCompletedRuns(convertedRuns);
    } catch (error) {
      console.error("❌ Error fetching runs:", error);
    }
  };

  const translateWebSocketMessage = (
    message: unknown
  ): {
    entity: string;
    message: string;
    showConfetti?: boolean;
    updateAgentState?: boolean;
    showApprovalRequest?: boolean;
  } => {
    if (typeof message === "object" && message !== null && "type" in message) {
      const msg = message as {
        type: string;
        message?: string;
        data?: {
          execution_state?: string;
          node_name?: string;
          output?: {
            messages?: Array<{
              tool_calls?: Array<{ name?: string }>;
              content?: string;
            }>;
          };
          trade_details?: {
            trade?: {
              action?: string;
              position?: string;
              usd_amount?: number;
              price?: number;
              contracts?: number;
            };
          };
        };
      };

      if (msg.type === "ack") {
        return { entity: "SYSTEM", message: "Connected to Agent" };
      }

      if (msg.type === "trade_approval_request") {
        const msgWithRunId = msg as {
          type: string;
          run_id?: string;
          data?: {
            trade_details?: {
              trade?: {
                action?: string;
                position?: string;
                usd_amount?: number;
                price?: number;
                contracts?: number;
              };
            };
            run_id?: string;
          };
        };

        const tradeData = msgWithRunId.data?.trade_details?.trade;
        const runId = msgWithRunId.run_id || msgWithRunId.data?.run_id;

        if (tradeData && runId) {
          // Show approval request with trade details
          const tradeDetails = {
            action: tradeData.action || "UNKNOWN",
            amount: `${Math.round(tradeData.contracts || 0)} ${
              tradeData.position || ""
            }`,
            price: `$${tradeData.price || "0.00"}`,
          };

          // Trigger approval request UI with smooth transition
          setTimeout(() => {
            setCurrentRunId(runId);
            setPendingTradeDetails(tradeDetails);
            setIsWaitingForApproval(true);
            
            // Show the approval UI first
            setShowAgentRequest(true);
            
            // Wait a moment for the UI to render, then scroll smoothly
            setTimeout(() => {
              if (agentRequestRef.current) {
                agentRequestRef.current.scrollIntoView({
                  behavior: "smooth",
                  block: "center",
                });
              }
            }, 300); // Give time for the popup to appear
          }, 800); // Longer delay before showing popup
        }

        return {
          entity: "SYSTEM",
          message: "Trade approval required - waiting for user decision",
          showApprovalRequest: true,
        };
      }

      if (msg.type === "execution_state") {
        // Handle new agent_states structure
        const msgWithAgentStates = msg as {
          type: string;
          data?: {
            agent_states?: Record<string, string>;
            execution_state?: string; // fallback for old format
          };
        };

        const agentStates = msgWithAgentStates.data?.agent_states;

        if (agentStates && AGENT_ID && AGENT_ID in agentStates) {
          const executionState = agentStates[AGENT_ID];
          if (executionState === "idle") {
            return {
              entity: "SYSTEM",
              message: "Agent is now ready",
              updateAgentState: true,
            };
          } else {
            return {
              entity: "SYSTEM",
              message: `Agent state: ${executionState}`,
              updateAgentState: true,
            };
          }
        }

        // Fallback to old format
        if (msgWithAgentStates.data?.execution_state) {
          const executionState = msgWithAgentStates.data.execution_state;
          if (executionState === "idle") {
            return {
              entity: "SYSTEM",
              message: "Agent is now ready",
              updateAgentState: true,
            };
          } else {
            return {
              entity: "SYSTEM",
              message: `Agent state: ${executionState}`,
              updateAgentState: true,
            };
          }
        }

        // If no relevant state found, return generic message
        return {
          entity: "SYSTEM",
          message: "Agent state update received",
          updateAgentState: false,
        };
      }

      if (msg.type === "status") {
        const statusData = msg as {
          type: string;
          run_id?: string;
          data?: { status?: string };
        };
        if (statusData.data?.status === "running" && statusData.run_id) {
          return {
            entity: "SYSTEM",
            message: `Starting Agent Run #${statusData.run_id}`,
          };
        }
        if (statusData.data?.status === "completed") {
          return {
            entity: "SYSTEM",
            message: "Agent run sucessfully executed! Waiting for next run",
            showConfetti: true,
          };
        }
      }

      if (msg.type === "node_update" && msg.data) {
        const nodeName = msg.data.node_name;
        const output = msg.data.output;

        if (
          nodeName === "entry_guard" &&
          output &&
          typeof output === "object" &&
          "_skip_run" in output
        ) {
          const skipRun = String(
            (output as { _skip_run: unknown })._skip_run
          ).toLowerCase();
          if (skipRun === "false") {
            return {
              entity: "SYSTEM",
              message: "Agent has passed all check. Starting",
            };
          } else if (skipRun === "true") {
            return {
              entity: "SYSTEM",
              message: "Agent is not allowed to run. Entry check faild",
            };
          }
        }

        if (nodeName === "context") {
          return {
            entity: "SYSTEM",
            message: "Agent context sucessfully initialized",
          };
        }

        if (nodeName === "apply_updates") {
          if (output && typeof output === "object" && "market_odds" in output) {
            const marketOdds = (
              output as {
                market_odds?: {
                  yes_price?: string;
                  no_price?: string;
                  volume?: string;
                };
              }
            ).market_odds;
            if (marketOdds) {
              const yesPrice = marketOdds.yes_price || "N/A";
              const noPrice = marketOdds.no_price || "N/A";
              const volume = marketOdds.volume || "N/A";
              return {
                entity: "AGENT",
                message: `New Market data yes price: ${yesPrice}, no price: ${noPrice}, volume: ${volume}`,
              };
            }
          } else if (output && typeof output === "object" && "news" in output) {
            const news = (output as { news?: Array<unknown> }).news;
            if (news && Array.isArray(news)) {
              const newsCount = news.length;
              return {
                entity: "AGENT",
                message: `I have now ${newsCount} market news to analyze`,
              };
            }
          } else {
            console.log("apply element not found");
            return { entity: "SYSTEM", message: "Apply updates completed" };
          }
        }

        if (nodeName === "llm" && output?.messages) {
          const messages = output.messages as Array<{
            content?: string;
            tool_calls?: Array<{ name: string }>;
          }>;

          let result = "Agent: I am thinking";

          for (const message of messages) {
            const content = message.content?.trim();
            const toolCalls = message.tool_calls?.map((tc) => tc.name) || [];

            if (content) {
              result += ` and came to the conclusion ${content}`;
            }

            if (toolCalls.length > 0) {
              // Check for specific tool calls
              if (toolCalls.includes("analyze_market_impact")) {
                result += ` and I will now analyze how much impact the news have on the polymarket market`;
              } else {
                result += ` and should call tools ${toolCalls.join(", ")}`;
              }
            }
          }

          return { entity: "AGENT", message: result.replace("Agent: ", "") };
        }

        if (nodeName === "tools" && output?.messages) {
          const messages = output.messages as Array<{
            name?: string;
            content?: string;
            status?: string;
          }>;

          const message = messages[0];
          if (message) {
            const toolName = message.name;
            const status = String(message.status || "").toLowerCase();
            const content = message.content;

            if (toolName === "get_market_data") {
              if (status === "success") {
                return {
                  entity: "AGENT",
                  message: `Market Stats are ${content || ""}`,
                };
              } else {
                return {
                  entity: "AGENT",
                  message: "I couldn't get the market stats! (fail)",
                };
              }
            }

            if (toolName === "get_news") {
              if (status === "success") {
                if (content && content.trim()) {
                  return {
                    entity: "AGENT",
                    message: "I sucessfuly received market news!",
                  };
                } else {
                  return {
                    entity: "AGENT",
                    message: "There are no new market news yet..",
                  };
                }
              } else {
                return {
                  entity: "AGENT",
                  message: "I couldnt receive market news (fail)",
                };
              }
            }

            if (toolName === "analyze_market_impact") {
              if (status === "success" && content && content.trim()) {
                return {
                  entity: "AGENT",
                  message:
                    "I have sucessfully analyzed the news on market impact!",
                };
              } else {
                return {
                  entity: "AGENT",
                  message: "I couldnt analyze the market impact! (fail)",
                };
              }
            }

            if (toolName === "trade") {
              if (status === "success") {
                return {
                  entity: "AGENT",
                  message: "I will now execute the trade",
                };
              } else {
                return {
                  entity: "AGENT",
                  message: "I cant execute any trade now (fail)",
                };
              }
            }

            return {
              entity: "SYSTEM",
              message: "tool translation not implemented yet",
            };
          }

          return { entity: "SYSTEM", message: "Tools: No message data" };
        }

        return { entity: "SYSTEM", message: `${nodeName}: Processing...` };
      }

      return { entity: "SYSTEM", message: msg.type || "Unknown message" };
    }

    return { entity: "SYSTEM", message: "Invalid message format" };
  };

  const handleTerminalScroll = () => {
    // Ignore scroll events caused by our own auto-scroll
    if (isAutoScrolling.current) {
      // console.log("Ignoring scroll event (caused by auto-scroll)");
      return;
    }

    if (terminalRef.current) {
      const element = terminalRef.current;
      const isAtBottom = isNearBottom(element);

      // Debug the scroll measurements
      // console.log("USER scroll detected:", {
      //   scrollTop: element.scrollTop,
      //   clientHeight: element.clientHeight,
      //   scrollHeight: element.scrollHeight,
      //   distanceFromBottom:
      //     element.scrollHeight - (element.scrollTop + element.clientHeight),
      //   threshold: 50,
      //   isAtBottom: isAtBottom,
      // });

      setShouldAutoScroll(isAtBottom);
    }
  };

  // Auto-scroll effect when terminal content changes
  useEffect(() => {
    // console.log(
    //   "useEffect triggered, shouldAutoScroll:",
    //   shouldAutoScroll,
    //   "terminalContent.length:",
    //   terminalContent.length
    // );
    if (terminalRef.current && shouldAutoScroll) {
      // console.log("Auto-scrolling because shouldAutoScroll is true");
      // Set flag to ignore the scroll event we're about to cause
      isAutoScrolling.current = true;

      // Use setTimeout to ensure DOM is fully updated
      setTimeout(() => {
        if (terminalRef.current) {
          scrollToBottom(terminalRef.current);

          // Clear flag after scroll is complete
          setTimeout(() => {
            isAutoScrolling.current = false;
          }, 50);
        }
      }, 10);
    } else {
      // console.log("NOT auto-scrolling, shouldAutoScroll:", shouldAutoScroll);
    }
  }, [terminalContent, shouldAutoScroll]);

  useEffect(() => {
    if (!AGENT_ID || !BACKEND_URL) return; // Don't fetch until agent ID is loaded

    fetchAgentStats();
    fetchTotalAgents();

    statsIntervalRef.current = setInterval(fetchAgentStats, 5000);
    const totalAgentsIntervalRef = setInterval(fetchTotalAgents, 30000); // Update every 30 seconds

    return () => {
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current);
      }
      clearInterval(totalAgentsIntervalRef);
    };
  }, [AGENT_ID, BACKEND_URL]);

  useEffect(() => {
    fetchMarketData();

    marketDataIntervalRef.current = setInterval(fetchMarketData, 5000);

    return () => {
      if (marketDataIntervalRef.current) {
        clearInterval(marketDataIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!AGENT_ID || !BACKEND_URL) return; // Don't fetch until agent ID is loaded

    fetchRuns();

    const runsIntervalRef = setInterval(fetchRuns, 5000);

    return () => {
      clearInterval(runsIntervalRef);
    };
  }, [AGENT_ID, BACKEND_URL]);

  // Connect to general WebSocket on component mount
  useEffect(() => {
    if (!AGENT_ID || !BACKEND_URL) return; // Don't connect until agent ID is loaded

    const connectGeneralWebSocket = () => {
      if (wsRetryCountRef.current >= MAX_RETRIES) {
        console.log(
          "❌ Max WebSocket retry attempts reached. Stopping reconnection."
        );
        return;
      }

      console.log(
        `🔄 Connecting to WebSocket... (attempt ${
          wsRetryCountRef.current + 1
        }/${MAX_RETRIES})`
      );
      const ws = new WebSocket(
        `${BACKEND_URL.replace("http", "ws")}/ws/general`
      );
      generalWsRef.current = ws;

      ws.onopen = () => {
        console.log(
          "✅ General WebSocket connected - readyState:",
          ws.readyState
        );
        console.log("WebSocket URL:", ws.url);

        // Reset retry count and set connection time
        wsRetryCountRef.current = 0;
        wsConnectedTimeRef.current = Date.now();

        // Set timer to reset retry count after 10 seconds of stable connection
        setTimeout(() => {
          if (
            wsConnectedTimeRef.current &&
            Date.now() - wsConnectedTimeRef.current >= 10000
          ) {
            wsRetryCountRef.current = 0;
            console.log(
              "🔄 WebSocket stable for 10+ seconds, retry count reset"
            );
          }
        }, 10000);

        // Send a test message to confirm two-way communication
        ws.send("Frontend connected - test message");
      };

      ws.onmessage = (event) => {
        console.log("🟢 General WebSocket message received:", event.data);
        console.log("Message type:", typeof event.data);
        console.log("Message length:", event.data?.length);

        // Log WebSocket message to console only
        console.log("🔵 WebSocket raw message:", event.data);

        try {
          const message = JSON.parse(event.data);
          console.log("🟢 General parsed JSON message:", message);

          // Filter messages by agent ID - only process messages related to this agent
          const isRelevantMessage = isMessageForThisAgent(
            message,
            AGENT_ID,
            runToAgentMapping.current
          );
          if (!isRelevantMessage) {
            console.log(
              `🔇 Filtered out message - not for agent ${AGENT_ID}:`,
              message
            );
            return; // Skip processing this message
          }

          console.log(`✅ Processing message for agent ${AGENT_ID}:`, message);

          const translation = translateWebSocketMessage(message);
          const timestamp = new Date().toLocaleTimeString("en-US", {
            hour12: false,
          });

          let color = "text-green-400";
          switch (translation.entity.toLowerCase()) {
            case "system":
              color = "text-cyan-400";
              break;
            case "agent":
              color = "text-green-400";
              break;
          }

          setTerminalContent((prev) => [
            ...prev,
            {
              time: timestamp,
              type: translation.entity,
              message: translation.message,
              color: color,
            },
          ]);

          if (translation.updateAgentState) {
            const message = JSON.parse(event.data);
            if (message.type === "execution_state") {
              // Handle new agent_states structure
              if (message.data?.agent_states && AGENT_ID) {
                const agentStates = message.data.agent_states;
                if (AGENT_ID in agentStates) {
                  const executionState = agentStates[AGENT_ID];
                  setIsAgentRunning(executionState !== "idle");
                }
              }
              // Fallback to old format
              else if (message.data?.execution_state) {
                const executionState = message.data.execution_state;
                setIsAgentRunning(executionState !== "idle");
              }
            }
          }

          // Trigger confetti if specified
          if (translation.showConfetti) {
            triggerConfetti();
            fetchRuns();
          }
        } catch (error) {
          console.log("🟡 General raw message (not JSON):", event.data);
          console.log("Parse error:", error);
        }
      };

      ws.onclose = (event) => {
        console.log("🔴 General WebSocket closed:", {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
          readyState: ws.readyState,
        });

        wsConnectedTimeRef.current = null;

        // Only retry if it wasn't a clean close and we haven't exceeded max retries
        if (!event.wasClean && wsRetryCountRef.current < MAX_RETRIES) {
          wsRetryCountRef.current += 1;
          const retryDelay = Math.min(
            1000 * Math.pow(2, wsRetryCountRef.current - 1),
            10000
          ); // Exponential backoff, max 10s

          console.log(
            `🔄 Retrying WebSocket connection in ${retryDelay}ms... (${wsRetryCountRef.current}/${MAX_RETRIES})`
          );

          wsRetryTimeoutRef.current = setTimeout(() => {
            connectGeneralWebSocket();
          }, retryDelay);
        }
      };

      ws.onerror = (error) => {
        console.error("❌ General WebSocket error:", error);
        console.log("WebSocket readyState on error:", ws.readyState);
      };
    };

    connectGeneralWebSocket();

    // Cleanup on unmount
    return () => {
      if (wsRetryTimeoutRef.current) {
        clearTimeout(wsRetryTimeoutRef.current);
      }
      if (generalWsRef.current) {
        generalWsRef.current.close();
      }
    };
  }, [AGENT_ID, BACKEND_URL]);

  const handleApproval = async (approved: boolean) => {
    console.log(`🔔 User approval decision: ${approved ? "YES" : "NO"}`);

    if (currentRunId) {
      try {
        const response = await fetch(
          `${BACKEND_URL}/runs/${currentRunId}/approve`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              decision: approved ? "yes" : "no",
            }),
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log(`✅ Approval sent to backend:`, result);
      } catch (error) {
        console.error("❌ Error sending approval to backend:", error);
      }
    } else {
      console.error("❌ No run ID available for approval");
    }

    if (approvalResolveRef.current) {
      approvalResolveRef.current(approved);
      approvalResolveRef.current = null;
    }
    setShowAgentRequest(false);
    setPendingTradeDetails(null);
    setIsWaitingForApproval(false);
    setCurrentRunId(null);

    // Smooth scroll back to Mission Control terminal
    setTimeout(() => {
      if (missionControlRef.current) {
        missionControlRef.current.scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      }
    }, 100); // Small delay to ensure UI state is updated
  };

  const waitForUserApproval = (tradeDetails: {
    action: string;
    amount: string;
    price: string;
  }): Promise<boolean> => {
    return new Promise((resolve) => {
      setPendingTradeDetails(tradeDetails);
      setShowAgentRequest(true);
      setIsWaitingForApproval(true);
      approvalResolveRef.current = resolve;

      // Smooth scroll to the agent request box
      setTimeout(() => {
        if (agentRequestRef.current) {
          agentRequestRef.current.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
        }
      }, 100); // Small delay to ensure DOM is updated
    });
  };

  const handleRunAgent = async () => {
    // Agent state will be controlled by WebSocket messages

    try {
      // Make POST request to start the agent
      const response = await fetch(`${BACKEND_URL}/runs`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          agent_id: AGENT_ID,
          agent_type: "polymarket-trader",
          parameters: {
            debug_mode: true,
          },
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Agent started:", data);

      // Store run_id to agent_id mapping for WebSocket filtering
      if (data.id) {
        runToAgentMapping.current.set(data.id, AGENT_ID);
        console.log(`🗺️ Added run mapping: ${data.id} -> ${AGENT_ID}`);
      }

      // TODO: Establish WebSocket connection for specific run later
      // if (data.id) {
      //   const ws = new WebSocket(`ws://localhost:8000/ws/${data.id}`);
      //
      //   ws.onopen = () => {
      //     console.log('WebSocket connected to run:', data.id);
      //   };
      //
      //   ws.onmessage = (event) => {
      //     console.log('WebSocket message:', event.data);
      //     try {
      //       const message = JSON.parse(event.data);
      //       console.log('Parsed message:', message);
      //     } catch {
      //       console.log('Raw message (not JSON):', event.data);
      //     }
      //   };
      //
      //   ws.onclose = (event) => {
      //     console.log('WebSocket closed:', event.code, event.reason);
      //   };
      //
      //   ws.onerror = (error) => {
      //     console.error('WebSocket error:', error);
      //   };
      // }

      // After 1 second, scroll to Mission Control and clear terminal
      setTimeout(() => {
        if (missionControlRef.current) {
          missionControlRef.current.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
        }

        // Re-enable auto-scroll for new run
        setShouldAutoScroll(true);

        // Start streaming messages
        startAgentStreamingSequence();
      }, 1000);
    } catch (error) {
      console.error("Error starting agent:", error);
      setIsAgentRunning(false);
      // Optionally show error to user
      setTerminalContent([
        {
          time: `[${new Date().toLocaleTimeString("en-US")}]`,
          type: "ERROR",
          message: "Failed to start agent. Check connection to backend.",
          color: "text-red-400",
        },
      ]);
    }
  };

  const triggerConfetti = () => {
    // School pride effect colors matching our app theme
    const colors = [
      "#10b981",
      "#3b82f6",
      "#f59e0b",
      "#ef4444",
      "#8b5cf6",
      "#ec4899",
    ];
    const duration = 3000;
    const end = Date.now() + duration;

    // Initial center burst for immediate impact
    confetti({
      particleCount: 100,
      startVelocity: 30,
      spread: 360,
      origin: { x: 0.5, y: 0.5 },
      colors: colors,
    });

    // School pride continuous side cannons
    (function frame() {
      confetti({
        particleCount: 2,
        angle: 60,
        spread: 55,
        origin: { x: 0 },
        colors: colors,
      });
      confetti({
        particleCount: 2,
        angle: 120,
        spread: 55,
        origin: { x: 1 },
        colors: colors,
      });

      if (Date.now() < end) {
        requestAnimationFrame(frame);
      }
    })();

    // Additional burst from center after 500ms
    setTimeout(() => {
      confetti({
        particleCount: 75,
        startVelocity: 25,
        spread: 180,
        origin: { x: 0.5, y: 0.3 },
        colors: colors,
      });
    }, 500);
  };

  const startAgentStreamingSequence = async () => {
    const currentRun = runCounter;
    // const preExecuteMessages = [
    //   {
    //     delay: 0,
    //     time: "[12:45:00]",
    //     type: "SYSTEM",
    //     message: `Initializing Run #${currentRun}...`,
    //     color: "text-cyan-400",
    //   },
    //   {
    //     delay: 500,
    //     time: "[12:45:01]",
    //     type: "SYSTEM",
    //     message: "Assigned to your session",
    //     color: "text-cyan-400",
    //   },
    //   {
    //     delay: 1000,
    //     time: "[12:45:02]",
    //     type: "CONNECT",
    //     message: "Accessing Polymarket API...",
    //     color: "text-blue-400",
    //   },
    //   {
    //     delay: 1500,
    //     time: "[12:45:03]",
    //     type: "FETCH",
    //     message: 'Loading market "BTC $100k by 2025"',
    //     color: "text-blue-400",
    //   },
    //   {
    //     delay: 2500,
    //     time: "[12:45:05]",
    //     type: "ANALYZE",
    //     message: "Current YES price: $0.4523",
    //     color: "text-yellow-400",
    //   },
    //   {
    //     delay: 4000,
    //     time: "[12:45:08]",
    //     type: "ANALYZE",
    //     message: "Volume spike detected",
    //     color: "text-yellow-400",
    //   },
    //   {
    //     delay: 6000,
    //     time: "[12:45:12]",
    //     type: "STRATEGY",
    //     message: "Momentum indicator positive",
    //     color: "text-purple-400",
    //   },
    //   {
    //     delay: 7500,
    //     time: "[12:45:15]",
    //     type: "DECISION",
    //     message: "Evaluating position...",
    //     color: "text-orange-400",
    //   },
    // ];

    // preExecuteMessages.forEach((msg) => {
    //   setTimeout(() => {
    //     setTerminalContent((prev) => [
    //       ...prev,
    //       {
    //         time: msg.time,
    //         type: msg.type,
    //         message: msg.message,
    //         color: msg.color,
    //       },
    //     ]);
    //   }, msg.delay);
    // });

    // // Skip pre-execute messages and go straight to approval
    // setTimeout(async () => {
    //   setTerminalContent((prev) => [
    //     ...prev,
    //     {
    //       time: "[12:45:18]",
    //       type: "EXECUTE",
    //       message: "Buying 150 YES @ $0.4523",
    //       color: "text-emerald-400",
    //     },
    //   ]);

    //   // Wait for user approval
    //   const approved = await waitForUserApproval({
    //     action: "BUY",
    //     amount: "150 YES",
    //     price: "$0.4523",
    //   });

    //   if (approved) {
    //     // Continue with execution
    //     const postExecuteMessages = [
    //       {
    //         delay: 500,
    //         time: "[12:45:20]",
    //         type: "CONFIRM",
    //         message: "Order filled successfully",
    //         color: "text-emerald-400",
    //       },
    //       {
    //         delay: 1000,
    //         time: "[12:45:21]",
    //         type: "RESULT",
    //         message: "Position opened: 150 YES @ $0.4523 (PnL = $0.00) ",
    //         color: "text-emerald-400",
    //         extra: "",
    //       },
    //     ];

    //     postExecuteMessages.forEach((msg) => {
    //       setTimeout(() => {
    //         setTerminalContent((prev) => [
    //           ...prev,
    //           {
    //             time: msg.time,
    //             type: msg.type,
    //             message: msg.message,
    //             color: msg.color,
    //             extra: msg.extra,
    //           },
    //         ]);
    //       }, msg.delay);
    //     });

    //     // Add completed run card and reset button after all messages are done
    //     setTimeout(() => {
    //       const generateRunId = () => {
    //         const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
    //         let result = "";
    //         for (let i = 0; i < 4; i++) {
    //           result += chars.charAt(Math.floor(Math.random() * chars.length));
    //         }
    //         return `#${currentRun}-${result}`;
    //       };

    //       const getCurrentCost = () => {
    //         const freeRuns = 5;
    //         if (currentRun <= 235 + freeRuns - 1) {
    //           const runNumber = currentRun - 234;
    //           return `Free Run (${runNumber}/5)`;
    //         } else {
    //           return "$0.10";
    //         }
    //       };

    //       const newRun = {
    //         id: currentRun,
    //         runId: generateRunId(),
    //         fullRunId: `simulated-run-${Date.now()}`, // Generate a fake UUID for simulated runs
    //         profit: "+$12.50",
    //         action: "BUY YES",
    //         tradePrice: 0.4523,
    //         tradeAmount: 150,
    //         timestamp: new Date().toLocaleTimeString(),
    //         cost: getCurrentCost(),
    //         rationale:
    //           "Volume has increased 30% in the last 5 minutes, signaling strong buy momentum.",
    //         confidence: 85,
    //         status: "completed",
    //         logs: [],
    //       };
    //       setCompletedRuns((prev) => [newRun, ...prev]);
    //       setRunCounter((prev) => prev + 1);

    //       // Show confetti celebration
    //       triggerConfetti();
    //     }, 2000);
    //   } else {
    //     // User declined - show cancelled message
    //     setTimeout(() => {
    //       setTerminalContent((prev) => [
    //         ...prev,
    //         {
    //           time: "[12:45:19]",
    //           type: "CANCEL",
    //           message: "Trade cancelled by user",
    //           color: "text-red-400",
    //         },
    //       ]);
    //       // Agent state will be controlled by WebSocket messages
    //     }, 500);
    //   }
    // }, 500); // Changed from 9000ms to 500ms since we skip pre-execute messages
  };

  // Don't render anything until agent ID is loaded
  if (!AGENT_ID || !BACKEND_URL) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p className="text-zinc-600 dark:text-zinc-400">
            Loading agent configuration...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-50 via-neutral-50 to-stone-50 dark:from-zinc-950 dark:via-neutral-950 dark:to-stone-950">
      {/* Header */}
      <header className="border-b border-zinc-200/60 dark:border-zinc-800/60 bg-white/70 dark:bg-zinc-950/70 backdrop-blur-xl sticky top-0 z-10">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex flex-col items-center gap-1">
            <div className="w-10 h-10">
              <Lottie
                animationData={logoAnimation}
                loop={true}
                autoplay={true}
                style={{ width: "100%", height: "100%" }}
              />
            </div>
            <div className="text-zinc-500 dark:text-zinc-400 text-xs font-medium">
              Flux trades for you
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="gap-2 border-zinc-300 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
          >
            <svg
              viewBox="0 0 24 24"
              className="w-4 h-4 fill-current"
              aria-hidden="true"
            >
              <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
            </svg>
            Share
          </Button>
        </div>
      </header>

      {/* Agent Status - Under Header */}
      <div className="bg-white/50 dark:bg-zinc-950/50 backdrop-blur-sm border-b border-zinc-200/30 dark:border-zinc-800/30">
        <div className="container mx-auto px-6 py-3 flex justify-center">
          <div
            className={`flex items-center gap-2 text-sm font-medium ${
              isAgentRunning
                ? "text-orange-600 dark:text-orange-400"
                : "text-emerald-600 dark:text-emerald-400"
            }`}
          >
            <div
              className={`w-2 h-2 rounded-full ${
                isAgentRunning
                  ? "bg-orange-500 animate-pulse"
                  : "bg-emerald-500 animate-pulse"
              }`}
            ></div>
            <span className="font-semibold">
              AGENT STATUS: {isAgentRunning ? "Running" : "Ready"}
            </span>
            <span className="text-zinc-400 dark:text-zinc-500 mx-2">•</span>
            <span className="text-zinc-600 dark:text-zinc-400">
              {isAgentRunning
                ? "Executing trade strategy..."
                : getLatestRunInfo()}
            </span>
          </div>
        </div>
      </div>

      {/* Hero Section - One Viewport */}
      <div className="min-h-[calc(100vh-160px)] flex items-center justify-center py-8">
        <div className="container mx-auto px-6">
          <div className="text-center space-y-8">
            {/* Main Title with Animated Gradient */}
            <div className="space-y-6">
              <div className="space-y-4">
                <h1 className="text-6xl font-bold leading-tight text-zinc-900 dark:text-zinc-100">
                  Can this Agent beat{" "}
                  <motion.span
                    className="bg-gradient-to-r from-emerald-600 via-teal-500 to-cyan-500 dark:from-emerald-400 dark:via-teal-400 dark:to-cyan-400 bg-clip-text text-transparent bg-[length:200%_100%]"
                    animate={{
                      backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
                    }}
                    transition={{
                      duration: 2,
                      ease: "easeInOut",
                      repeat: Infinity,
                      repeatDelay: 2,
                    }}
                  >
                    Polymarket
                  </motion.span>
                  ?
                </h1>
                <p className="text-xl text-zinc-600 dark:text-zinc-400 font-medium max-w-2xl mx-auto leading-relaxed">
                  Every click runs the same agent: one wallet, one trade at a
                  time.
                </p>
              </div>

              {/* Side by Side Layout */}
              <div className="grid lg:grid-cols-2 gap-8 items-start pt-2">
                {/* Left: Portfolio Card */}
                <div className="flex justify-center">
                  <div className="relative">
                    {/* Glow effect */}
                    <div className="absolute -inset-2 bg-gradient-to-r from-emerald-500/20 via-teal-500/20 to-cyan-500/20 rounded-2xl blur-xl"></div>

                    <Card className="relative bg-gradient-to-r from-zinc-950/95 to-zinc-900/95 dark:from-zinc-900/95 dark:to-zinc-800/95 border border-zinc-700/50 dark:border-zinc-600/50 shadow-2xl backdrop-blur-sm">
                      <CardContent className="px-12 py-8 text-center">
                        {/* Giant P&L */}
                        <div className="mb-3">
                          <span
                            className={`text-7xl font-bold tracking-tight ${
                              calculateProfit().profit >= 0
                                ? "text-emerald-400"
                                : "text-red-400"
                            }`}
                          >
                            {calculateProfit().profit >= 0 ? "+" : ""}$
                            {calculateProfit().profit.toFixed(0)}
                          </span>
                          <span className="text-2xl text-zinc-400 dark:text-zinc-500 ml-3 font-medium">
                            lifetime P&L
                          </span>
                        </div>

                        {/* Runs & Trades */}
                        <div className="flex items-center justify-center gap-6 text-zinc-400 dark:text-zinc-500">
                          <span className="text-lg">
                            <span className="font-semibold text-white">
                              {agentStats?.runs_last_24h || 0}
                            </span>{" "}
                            runs today
                          </span>
                          <span className="text-zinc-600 dark:text-zinc-600">
                            •
                          </span>
                          <span className="text-lg">
                            <span className="font-semibold text-white">
                              {agentStats?.last_actions?.length || 0}
                            </span>{" "}
                            total trades
                          </span>
                        </div>

                        {/* Portfolio & Holdings - 2 Column Split */}
                        <div className="mt-6 pt-6 border-t border-zinc-700/30">
                          <div className="grid gap-4 md:grid-cols-2">
                            {/* Portfolio */}
                            <div className="bg-zinc-800/30 border border-zinc-600/20 rounded-lg p-4">
                              <h3 className="text-sm font-semibold text-white mb-3">
                                PORTFOLIO
                              </h3>
                              <div className="space-y-2">
                                <div className="flex justify-between items-center">
                                  <span className="text-zinc-400 text-sm">
                                    Balance:
                                  </span>
                                  <span className="font-semibold text-white">
                                    $
                                    {calculateProfit().totalBalance.toLocaleString(
                                      "en-US"
                                    )}{" "}
                                    <span
                                      className={`text-xs font-normal ${
                                        calculateProfit().percentage >= 0
                                          ? "text-emerald-400"
                                          : "text-red-400"
                                      }`}
                                    >
                                      (
                                      {calculateProfit().percentage >= 0
                                        ? "+"
                                        : ""}
                                      {calculateProfit().percentage.toFixed(1)}
                                      %)
                                    </span>
                                  </span>
                                </div>
                                <div className="flex justify-between items-center">
                                  <span className="text-zinc-400 text-sm">
                                    Available:
                                  </span>
                                  <span className="font-semibold text-white">
                                    $
                                    {agentStats?.balance?.toLocaleString(
                                      "en-US"
                                    ) || "9,959"}
                                  </span>
                                </div>
                              </div>
                            </div>

                            {/* Holdings */}
                            <div className="bg-zinc-800/30 border border-zinc-600/20 rounded-lg p-4">
                              <h3 className="text-sm font-semibold text-white mb-3">
                                HOLDINGS
                              </h3>
                              <div className="space-y-2">
                                <div className="text-zinc-400 text-sm space-y-1">
                                  <div>
                                    YES:{" "}
                                    {agentStats?.yes_holdings?.toFixed(2) ||
                                      "500.00"}
                                  </div>
                                  <div>
                                    NO:{" "}
                                    {agentStats?.no_holdings?.toFixed(2) ||
                                      "0.00"}
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Market Info */}
                        <div className="mt-4 pt-4 border-t border-zinc-700/30">
                          <p className="text-xs text-zinc-500 dark:text-zinc-400">
                            All activity is on the market:{" "}
                            {marketData?.question ||
                              "Will BTC hit $100k by 2025?"}
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>

                {/* Right: 3-Step Process */}
                <div className="flex justify-center">
                  <div className="relative" ref={agentRequestRef}>
                    {/* Glow effect */}
                    <div className="absolute -inset-2 bg-gradient-to-r from-emerald-500/20 via-teal-500/20 to-cyan-500/20 rounded-2xl blur-xl"></div>

                    <Card className="relative bg-gradient-to-r from-white to-zinc-50 dark:from-zinc-900 dark:to-zinc-800 border-2 border-emerald-200/50 dark:border-emerald-700/50 shadow-xl w-full max-w-md">
                      <CardContent className="px-8 py-6 min-h-[480px]">
                        {showAgentRequest && pendingTradeDetails ? (
                          // AGENT REQUEST UI - Replace all content with smooth animation
                          <motion.div 
                            className="w-full flex flex-col items-center justify-center space-y-6"
                            initial={{ opacity: 0, scale: 0.95, y: 10 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            transition={{ duration: 0.4, ease: "easeOut" }}
                          >
                            <div className="text-center space-y-3">
                              <div className="w-12 h-12 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold text-lg mx-auto">
                                ⚡
                              </div>
                              <div className="space-y-1">
                                <h2 className="text-lg font-bold text-zinc-800 dark:text-zinc-200">
                                  Agent Request
                                </h2>
                                <p className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-wide font-medium">
                                  Approval Required
                                </p>
                              </div>
                              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                                The agent wants to execute this trade:
                              </p>
                            </div>

                            <div className="bg-white dark:bg-zinc-800 rounded-lg p-4 border-2 border-orange-200 dark:border-orange-700/50 w-full space-y-3">
                              <div className="text-center">
                                <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400">
                                  {pendingTradeDetails.action}{" "}
                                  {pendingTradeDetails.amount}
                                </div>
                                <div className="text-sm text-zinc-600 dark:text-zinc-400">
                                  @ {pendingTradeDetails.price}
                                </div>
                              </div>

                              <div className="border-t border-zinc-200 dark:border-zinc-700 pt-3 space-y-2">
                                <div className="text-left">
                                  <div className="text-xs font-medium text-zinc-700 dark:text-zinc-300 mb-1">
                                    Rationale:
                                  </div>
                                  <div className="text-xs text-zinc-600 dark:text-zinc-400">
                                    After processing the latest market
                                    information, the agent has identified a
                                    potential trading opportunity.
                                  </div>
                                </div>

                                {/* <div className="flex items-center justify-between">
                                  <div className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                                    Confidence:
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <div className="text-xs font-bold text-emerald-600 dark:text-emerald-400">
                                      85%
                                    </div>
                                    <div className="w-16 h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                                      <div className="w-[85%] h-full bg-emerald-500 rounded-full"></div>
                                    </div>
                                  </div>
                                </div> */}
                              </div>
                            </div>

                            <div className="text-center space-y-3">
                              <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                                Do you approve this trade?
                              </p>
                              <div className="flex gap-3 justify-center">
                                <Button
                                  size="default"
                                  onClick={() => handleApproval(true)}
                                  className="bg-emerald-600 hover:bg-emerald-700 text-white px-8 py-2"
                                >
                                  ✓ Yes
                                </Button>
                                <Button
                                  size="default"
                                  variant="outline"
                                  onClick={() => handleApproval(false)}
                                  className="border-red-300 text-red-600 hover:bg-red-50 dark:border-red-700 dark:text-red-400 dark:hover:bg-red-900/20 px-8 py-2"
                                >
                                  ✗ No
                                </Button>
                              </div>
                            </div>
                          </motion.div>
                        ) : (
                          // NORMAL 3-STEP UI
                          <>
                            {/* STEP 1: SEE THE MARKET */}
                            <div className="mb-6">
                              <div className="flex items-center gap-2 mb-3">
                                <div className="w-6 h-6 bg-emerald-600 text-white rounded-full flex items-center justify-center font-semibold text-xs">
                                  1
                                </div>
                                <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 uppercase tracking-wide">
                                  SEE THE MARKET
                                </h3>
                              </div>
                              <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-lg p-3 space-y-2">
                                <div className="flex items-center justify-center gap-2">
                                  {marketData?.icon ? (
                                    <img
                                      src={marketData.icon}
                                      alt="Market icon"
                                      className="w-6 h-6 rounded"
                                      onError={(e) => {
                                        (
                                          e.target as HTMLImageElement
                                        ).style.display = "none";
                                      }}
                                    />
                                  ) : (
                                    <span className="text-xl">₿</span>
                                  )}
                                  <h4 className="text-sm text-zinc-800 dark:text-zinc-200 font-medium">
                                    &quot;
                                    {marketData?.question ||
                                      "Will BTC hit $100k by 2025?"}
                                    &quot;
                                  </h4>
                                </div>
                                <div className="flex items-center justify-center gap-3 text-xs text-zinc-600 dark:text-zinc-400">
                                  <span>
                                    Volume:{" "}
                                    <span className="font-semibold text-zinc-800 dark:text-zinc-200">
                                      {marketData?.volume
                                        ? formatVolume(marketData.volume)
                                        : "$255k"}
                                    </span>
                                  </span>
                                  <span>•</span>
                                  <span>via Polymarket</span>
                                </div>
                                <div className="grid grid-cols-2 gap-2">
                                  <div className="bg-emerald-50 dark:bg-emerald-900/20 p-2 rounded text-center">
                                    <div className="text-emerald-700 dark:text-emerald-300 font-bold text-sm">
                                      $
                                      {marketData?.outcomePrices?.[0] || "0.45"}
                                    </div>
                                    <div className="text-emerald-600 dark:text-emerald-400 text-xs">
                                      YES
                                    </div>
                                  </div>
                                  <div className="bg-red-50 dark:bg-red-900/20 p-2 rounded text-center">
                                    <div className="text-red-700 dark:text-red-300 font-bold text-sm">
                                      $
                                      {marketData?.outcomePrices?.[1] || "0.55"}
                                    </div>
                                    <div className="text-red-600 dark:text-red-400 text-xs">
                                      NO
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* STEP 2: RUN THE AGENT */}
                            <div className="mb-6">
                              <div className="flex items-center gap-2 mb-3">
                                <div className="w-6 h-6 bg-emerald-600 text-white rounded-full flex items-center justify-center font-semibold text-xs">
                                  2
                                </div>
                                <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 uppercase tracking-wide">
                                  RUN THE AGENT
                                </h3>
                              </div>
                              <div className="text-center">
                                <Button
                                  size="default"
                                  onClick={handleRunAgent}
                                  disabled={isAgentRunning}
                                  className={`h-10 px-8 mb-3 text-sm font-bold text-white shadow-lg transition-all duration-300 ${
                                    isAgentRunning
                                      ? "bg-zinc-600 cursor-not-allowed animate-pulse"
                                      : "bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-700 hover:via-teal-700 hover:to-cyan-700 hover:scale-105"
                                  }`}
                                >
                                  {isAgentRunning
                                    ? "⟳ STARTING..."
                                    : "▶️ RUN AGENT"}
                                </Button>
                                <p className="text-xs text-zinc-600 dark:text-zinc-400 mb-1">
                                  <span className="text-emerald-600 dark:text-emerald-400 font-semibold">
                                    5 free runs
                                  </span>{" "}
                                  • No card required
                                </p>
                                <div className="text-xs text-zinc-500 dark:text-zinc-400 mb-3">
                                  <p className="font-medium">
                                    Every run improves the Agent
                                  </p>
                                  {/* <p className="text-xs opacity-75">
                                    Join 234 contributors
                                  </p> */}
                                </div>

                                {/* Live Activity Feed */}
                                <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-lg p-2">
                                  <div className="h-4 overflow-hidden relative">
                                    <motion.div
                                      className="absolute inset-0"
                                      animate={{
                                        y: [
                                          0, -16, -32, -48, -64, -80, -96, -112,
                                        ],
                                      }}
                                      transition={{
                                        duration: 24,
                                        ease: "easeInOut",
                                        repeat: Infinity,
                                      }}
                                    >
                                      {getActivityFeedItems().map(
                                        (item, index) => (
                                          <div
                                            key={index}
                                            className="h-4 flex items-center justify-center text-xs text-zinc-500 dark:text-zinc-400"
                                          >
                                            <span>{item.time} - visitor →</span>
                                            <span
                                              className={`${item.color} mx-1`}
                                            >
                                              {item.action}
                                            </span>
                                          </div>
                                        )
                                      )}
                                    </motion.div>
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* STEP 3: VIEW THE RESULT */}
                            <div className="mb-3">
                              <div className="flex items-center gap-2 mb-3">
                                <div className="w-6 h-6 bg-emerald-600 text-white rounded-full flex items-center justify-center font-semibold text-xs">
                                  3
                                </div>
                                <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 uppercase tracking-wide">
                                  VIEW THE RESULT
                                </h3>
                              </div>
                              <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-lg p-3 text-center">
                                <p className="text-xs text-zinc-600 dark:text-zinc-400">
                                  The agent&apos;s actions will appear in the
                                  Mission Control terminal below
                                </p>
                              </div>
                            </div>
                          </>
                        )}
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-16 space-y-12">
        {/* Mission Control Terminal */}
        <Card
          ref={missionControlRef}
          className="bg-zinc-900 border-zinc-700/60 shadow-2xl relative overflow-hidden"
        >
          <CardHeader className="bg-zinc-900/90 border-b border-zinc-700/60 relative py-0.5 px-3">
            <CardTitle className="flex items-center justify-between text-zinc-200 text-xs leading-none">
              <span className="text-xs font-medium flex items-center gap-1 leading-none">
                ⭐ MISSION CONTROL
              </span>
              <Badge
                variant="outline"
                className="bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
              >
                <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse mr-1"></span>
                OPERATIONAL
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 relative">
            <div
              ref={terminalRef}
              onScroll={handleTerminalScroll}
              className="bg-zinc-950/90 text-emerald-400 font-mono text-sm p-4 h-64 overflow-y-auto relative backdrop-blur-sm"
            >
              <div className="space-y-2 -mt-2">
                {terminalContent.map((log, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <span className="text-zinc-500">{log.time}</span>
                    <span className={log.color}>{log.type}</span>
                    <span>{log.message}</span>
                    {log.extra && (
                      <span className="text-emerald-400 font-semibold">
                        {log.extra}
                      </span>
                    )}
                  </div>
                ))}
                <div className="flex items-center">
                  <span className="text-emerald-400">agent@polymarket:~$ </span>
                  <span className="animate-pulse">_</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Results Cards */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold text-zinc-800 dark:text-zinc-200">
            Recent Runs
          </h3>
          {completedRuns.length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {completedRuns.map((run) => (
                <div
                  key={run.id}
                  className="bg-zinc-50 dark:bg-zinc-800/50 border-2 border-dashed border-zinc-300 dark:border-zinc-600 rounded-lg p-4 font-mono text-sm"
                >
                  <div className="border border-zinc-400 dark:border-zinc-500 rounded p-3">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-emerald-600 dark:text-emerald-400">
                          ✅
                        </span>
                        <span className="font-semibold">
                          Run #{run.id} Complete
                        </span>
                      </div>

                      <div className="text-xs text-zinc-500 dark:text-zinc-400 font-mono break-all">
                        Run-ID: {run.fullRunId} • Cost: {run.cost}
                      </div>

                      <div className="border-t border-zinc-300 dark:border-zinc-600 my-2"></div>

                      <div className="space-y-1">
                        <div>
                          <span className="text-zinc-600 dark:text-zinc-400">
                            Action:{" "}
                          </span>
                          <span className="font-medium">{run.action}</span>
                        </div>
                        <div>
                          <span className="text-zinc-600 dark:text-zinc-400">
                            Price:{" "}
                          </span>
                          <span className="font-medium">
                            {run.tradePrice
                              ? `$${run.tradePrice.toFixed(3)}`
                              : run.action === "HOLD"
                              ? "N/A"
                              : "$0.4523"}
                          </span>
                        </div>
                        <div>
                          <span className="text-zinc-600 dark:text-zinc-400">
                            P&L:{" "}
                          </span>
                          <span className="font-medium text-zinc-500 dark:text-zinc-400">
                            $0.00
                          </span>
                        </div>

                        <div className="mt-2 p-2 bg-zinc-50 dark:bg-zinc-800/50 rounded text-xs">
                          <div className="text-zinc-700 dark:text-zinc-300 font-medium mb-1">
                            Rationale:
                          </div>
                          <div className="text-zinc-600 dark:text-zinc-400 text-xs leading-relaxed">
                            {run.rationale}
                          </div>

                          <div className="flex items-center justify-between mt-2">
                            <span className="text-zinc-700 dark:text-zinc-300 font-medium">
                              Confidence:
                            </span>
                            <div className="flex items-center gap-2">
                              <span className="text-emerald-600 dark:text-emerald-400 font-bold text-xs">
                                {run.confidence}%
                              </span>
                              <div className="w-12 h-1 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-emerald-500 rounded-full"
                                  style={{ width: `${run.confidence}%` }}
                                ></div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="border-t border-zinc-300 dark:border-zinc-600 my-2"></div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="w-full text-xs gap-1 border-zinc-300 hover:bg-zinc-100 dark:border-zinc-600 dark:hover:bg-zinc-700"
                      >
                        Share Result
                        <svg
                          viewBox="0 0 24 24"
                          className="w-3 h-3 fill-current"
                          aria-hidden="true"
                        >
                          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.80l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                        </svg>
                      </Button>
                    </div>
                  </div>
                  <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 text-center">
                    Completed at {run.timestamp}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-zinc-500 dark:text-zinc-400">
              No runs completed yet. Click &quot;RUN AGENT&quot; to get started!
            </div>
          )}
        </div>

        <style jsx>{`
          @keyframes drift {
            0% {
              transform: translateX(0) translateY(0);
            }
            100% {
              transform: translateX(-200px) translateY(-100px);
            }
          }
        `}</style>
      </div>

      {/* Footer */}
      <footer className="border-t border-zinc-200/60 dark:border-zinc-800/60 bg-white/70 dark:bg-zinc-950/70 backdrop-blur-xl mt-16">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between text-sm text-zinc-600 dark:text-zinc-400">
          <div className="flex items-center gap-4">
            <span>Paper Trading</span>
            <span className="text-zinc-300 dark:text-zinc-600">|</span>
            <span>Agents deployed: {totalAgents}</span>
          </div>
          <div className="font-semibold">
            via{" "}
            <a
              href="https://sentic.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-gradient-to-r from-zinc-700 to-zinc-600 dark:from-zinc-300 dark:to-zinc-400 bg-clip-text text-transparent hover:from-emerald-600 hover:to-teal-600 dark:hover:from-emerald-400 dark:hover:to-teal-400 transition-all duration-200"
            >
              Sentic.AI
            </a>
          </div>
        </div>
      </footer>

      {/* Deployed via Badge - Framer Style */}
      <div className="fixed top-1/2 right-6 z-50 -translate-y-1/2">
        <a
          href="https://sentic.ai"
          target="_blank"
          rel="noopener noreferrer"
          className="group flex items-center gap-2 bg-zinc-900/95 hover:bg-zinc-900 text-white px-3 py-2 rounded-full text-xs font-medium transition-all duration-200 hover:scale-105 shadow-lg backdrop-blur-sm border border-zinc-700/50"
        >
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full animate-pulse"></div>
            <span>Deployed via</span>
            <span className="font-semibold bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
              Sentic.ai
            </span>
          </div>
          <svg
            className="w-3 h-3 transition-transform group-hover:translate-x-0.5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
            />
          </svg>
        </a>
      </div>
    </div>
  );
}
