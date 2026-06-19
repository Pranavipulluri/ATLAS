"use client";

import React from "react";
import { XAILog, A2AMessage } from "@/hooks/useTrafficSocket";

interface RationaleLogProps {
  logs: XAILog[];
  a2aMessages: A2AMessage[];
}

function getLogStyle(type: string) {
  switch (type) {
    case "emergency": return { border: "border-l-red-400", bg: "bg-red-50" };
    case "green_wave": return { border: "border-l-emerald-400", bg: "bg-emerald-50" };
    case "starvation": return { border: "border-l-amber-400", bg: "bg-amber-50" };
    case "auction": return { border: "border-l-indigo-400", bg: "bg-indigo-50" };
    default: return { border: "border-l-gray-300", bg: "bg-gray-50" };
  }
}

function getLogIcon(type: string) {
  switch (type) {
    case "emergency": return "🚨";
    case "green_wave": return "🌊";
    case "starvation": return "⏰";
    case "auction": return "🔄";
    default: return "📋";
  }
}

function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function RationaleLog({ logs, a2aMessages }: RationaleLogProps) {
  const scrollRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => { scrollRef.current && (scrollRef.current.scrollTop = scrollRef.current.scrollHeight); }, [logs, a2aMessages]);

  const entries = [
    ...logs.map((l) => ({ key: `x-${l.timestamp}-${l.node_id}`, time: l.timestamp, type: l.type, content: l.explanation, isA2A: false })),
    ...a2aMessages.map((m, i) => ({ key: `a-${i}`, time: Date.now() / 1000, type: m.type === "GREEN_WAVE_ALERT" ? "green_wave" as const : "normal" as const, content: m.message, isA2A: true })),
  ].sort((a, b) => a.time - b.time).slice(-30);

  return (
    <div className="glass-card flex flex-col h-full">
      <div className="px-4 py-3 border-b border-gray-100 flex items-center gap-2">
        <span className="text-lg">🧠</span>
        <h3 className="text-sm font-bold text-gray-800 tracking-wide">XAI Rationale Log</h3>
        <span className="text-[10px] text-gray-400 ml-auto">{logs.length} events</span>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-1.5">
        {entries.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">Waiting for events...</div>
        ) : entries.map((e) => {
          const s = getLogStyle(e.type);
          return (
            <div key={e.key} className={`log-entry border-l-2 ${s.border} ${s.bg} rounded-r-lg px-3 py-2 text-[11px]`}>
              <div className="flex items-center gap-1.5 mb-0.5">
                <span className="text-xs">{getLogIcon(e.type)}</span>
                {e.isA2A && <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-cyan-100 text-cyan-700 font-semibold">A2A</span>}
                <span className="text-gray-400 font-mono text-[9px] ml-auto">{formatTime(e.time)}</span>
              </div>
              <p className="text-gray-700 leading-relaxed">{e.content}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
