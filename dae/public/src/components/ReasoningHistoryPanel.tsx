"use client";

import React, { useState, useMemo } from "react";
import { AgentHistoryEntry } from "@/hooks/useTrafficSocket";

interface ReasoningHistoryPanelProps {
  nodeId: string;
  history: AgentHistoryEntry[];
  onClose: () => void;
}

export default function ReasoningHistoryPanel({ nodeId, history, onClose }: ReasoningHistoryPanelProps) {
  const [filter, setFilter] = useState<"all" | "master" | "lane">("all");

  const filteredHistory = useMemo(() => {
    return history
      .filter((h) => h.node_id === nodeId)
      .filter((h) => filter === "all" || h.type === filter);
  }, [history, nodeId, filter]);

  return (
    <div className="h-full flex flex-col bg-white rounded-2xl shadow-sm border border-indigo-100 overflow-hidden relative">
      <div className="p-3 bg-indigo-50/50 border-b border-indigo-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-200">
            <span className="text-xl">🤖</span>
          </div>
          <div>
            <h2 className="text-sm font-bold text-indigo-900 tracking-tight">
              Node {nodeId} AI History
            </h2>
            <p className="text-[10px] text-indigo-500 font-semibold uppercase tracking-wider">
              Agent Reasoning Logs
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="w-7 h-7 rounded-full bg-white border border-gray-200 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-200 flex items-center justify-center transition-colors"
        >
          ✖
        </button>
      </div>

      <div className="px-3 py-2 bg-white border-b border-gray-100 flex gap-2">
        {(["all", "master", "lane"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`flex-1 py-1.5 rounded-lg text-[10px] font-bold uppercase transition-all ${
              filter === f
                ? "bg-indigo-500 text-white shadow-md shadow-indigo-200"
                : "bg-gray-50 text-gray-500 hover:bg-gray-100"
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-gray-50/50">
        {filteredHistory.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-400 gap-2">
            <span className="text-3xl opacity-50">📭</span>
            <span className="text-xs font-semibold">No AI logs yet for Node {nodeId}</span>
          </div>
        ) : (
          filteredHistory.map((entry) => (
            <div
              key={entry.id}
              className={`p-3 rounded-xl border shadow-sm transition-all ${
                entry.type === "master"
                  ? "bg-white border-indigo-100 shadow-indigo-50"
                  : "bg-white border-purple-100 shadow-purple-50"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5">
                  <span
                    className={`px-1.5 py-0.5 rounded text-[8px] font-bold text-white ${
                      entry.type === "master" ? "bg-indigo-500" : "bg-purple-500"
                    }`}
                  >
                    {entry.type === "master" ? "👑 MASTER" : `👁 ${entry.lane?.toUpperCase()} LANE`}
                  </span>
                  <span className="text-[10px] font-mono text-gray-400">
                    {new Date(entry.timestamp).toLocaleTimeString([], {
                      hour12: false,
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    })}
                  </span>
                </div>
              </div>
              <p className="text-[11px] leading-relaxed text-gray-700">
                {entry.reasoning}
              </p>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
