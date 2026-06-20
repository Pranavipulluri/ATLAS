"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export interface LaneState {
  lane: string;
  density: number;
  wait_time: number;
  has_emergency: boolean;
  is_green: boolean;
  reasoning: string;
  detection_source: string;
  vehicle_types: Record<string, number>;
  emergency_type: string | null;
  has_pedestrians?: boolean;
}

export interface LaneScoreDetail {
  lane: string;
  score: number;
  density: number;
  wait_time: number;
  has_emergency: boolean;
  green_wave_boost: number;
  formula: string;
}

export interface DecisionBreakdown {
  type: string;
  icon: string;
  switched: boolean;
  winner: string;
  winner_score: number;
  reason: string;
  current_green: string;
  time_in_phase: number;
  all_scores: LaneScoreDetail[];
}

export interface IntersectionState {
  intersection_id: string;
  current_green: string;
  time_in_phase: number;
  interrupted_lane: string | null;
  green_wave_active: boolean;
  green_wave_lanes: string[];
  decision_reason: string;
  scores: Record<string, number>;
  lanes: Record<string, LaneState>;
  lane_reasonings: Record<string, string>;
  decision_breakdown: DecisionBreakdown;
  ai_lane_reasonings: Record<string, string>;
  ai_master_reasoning: string;
}

export interface EmergencyState {
  vehicle_id: string;
  route_key: string;
  route: Array<{ intersection: string; lane: string }>;
  current_step: number;
  ticks_at_step: number;
  ticks_per_step: number;
  completed: boolean;
  current_intersection: string | null;
  current_lane: string | null;
}

export interface A2AMessage {
  type: string;
  from?: string;
  to?: string;
  vehicle_id?: string;
  target_lane?: string;
  message: string;
}

export interface XAILog {
  timestamp: number;
  node_id: string;
  explanation: string;
  type: "normal" | "emergency" | "green_wave" | "starvation" | "auction";
  fallback?: boolean;
}

export interface AgentHistoryEntry {
  id: string;
  timestamp: number;
  node_id: string;
  type: "master" | "lane";
  lane?: string;
  reasoning: string;
}

export interface GridState {
  tick: number;
  tick_rate: number;
  intersections: Record<string, IntersectionState>;
  emergencies: EmergencyState[];
  a2a_messages: A2AMessage[];
  green_wave_active: boolean;
  xai_logs: XAILog[];
  traffic_multipliers: Record<string, number>;
  detection_mode: string;
  detection_interval: number;
  severe_rain?: string | null;
  flood_active?: string | null;
}

export function useTrafficSocket(url?: string) {
  const [gridState, setGridState] = useState<GridState | null>(null);
  const [connected, setConnected] = useState(false);
  const [logs, setLogs] = useState<XAILog[]>([]);
  const [agentHistory, setAgentHistory] = useState<AgentHistoryEntry[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const prevAiStateRef = useRef<Record<string, { master: string; lanes: Record<string, string> }>>({});

  // Resolve default socket URL dynamically from env var or window location
  let resolvedUrl = url;
  if (!resolvedUrl) {
    resolvedUrl = "ws://localhost:8000/ws";
    if (typeof window !== "undefined") {
      const envApiUrl = process.env.NEXT_PUBLIC_API_URL;
      if (envApiUrl) {
        const baseUrl = envApiUrl.replace(/\/$/, "");
        resolvedUrl = baseUrl.replace(/^http/, "ws") + "/ws";
      }
    }
  }

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(resolvedUrl);

      ws.onopen = () => {
        setConnected(true);
        console.log("[WS] Connected to backend");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as GridState | { type: string };
          if ("type" in data && data.type === "command_response") return;
          const state = data as GridState;
          setGridState(state);

          // Update general XAI logs (Master Explainer)
          if (state.xai_logs && state.xai_logs.length > 0) {
            setLogs((prev) => {
              const newLogs = [...prev, ...state.xai_logs.filter(
                (log) => !prev.some((p) => p.timestamp === log.timestamp && p.node_id === log.node_id)
              )];
              return newLogs.slice(-50);
            });
          }

          // Accumulate AI Agent Reasoning History (Kimi)
          const newEntries: AgentHistoryEntry[] = [];
          const now = Date.now();

          Object.entries(state.intersections).forEach(([nodeId, nodeState]) => {
            const prevNode = prevAiStateRef.current[nodeId] || { master: "", lanes: {} };
            const mReason = nodeState.ai_master_reasoning;

            // Master changed?
            if (mReason && mReason !== prevNode.master) {
              newEntries.push({ id: `m-${nodeId}-${now}`, timestamp: now, node_id: nodeId, type: "master", reasoning: mReason });
              prevNode.master = mReason;
            }

            // Lanes changed?
            if (nodeState.ai_lane_reasonings) {
              Object.entries(nodeState.ai_lane_reasonings).forEach(([lane, lReason]) => {
                if (lReason && lReason !== prevNode.lanes[lane]) {
                  newEntries.push({ id: `l-${nodeId}-${lane}-${now}`, timestamp: now, node_id: nodeId, type: "lane", lane, reasoning: lReason });
                  prevNode.lanes[lane] = lReason;
                }
              });
            }

            prevAiStateRef.current[nodeId] = prevNode;
          });

          if (newEntries.length > 0) {
            setAgentHistory((prev) => {
              const combined = [...newEntries, ...prev]; // newer first
              return combined.slice(0, 200); // keep last 200 reasonings
            });
          }
        } catch {
          // ignore parse errors
        }
      };

      ws.onclose = () => {
        setConnected(false);
        console.log("[WS] Disconnected, reconnecting in 2s...");
        reconnectRef.current = setTimeout(connect, 2000);
      };

      ws.onerror = () => { ws.close(); };
      wsRef.current = ws;
    } catch {
      reconnectRef.current = setTimeout(connect, 2000);
    }
  }, [resolvedUrl]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const sendCommand = useCallback(
    (command: Record<string, unknown>) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(command));
      }
    },
    []
  );

  return { gridState, connected, logs, agentHistory, sendCommand };
}
