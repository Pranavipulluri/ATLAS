"use client";

import React, { useState, lazy, Suspense } from "react";
import { useTrafficSocket } from "@/hooks/useTrafficSocket";
import RationaleLog from "@/components/RationaleLog";
import ControlPanel from "@/components/ControlPanel";
import ReasoningHistoryPanel from "@/components/ReasoningHistoryPanel";
import MetricsPanel from "@/components/MetricsPanel";

const Scene3D = lazy(() => import("@/components/Scene3D"));

export default function Dashboard() {
  const { gridState, connected, logs, agentHistory, sendCommand } = useTrafficSocket();
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-gray-50 overflow-y-auto" style={{ fontFamily: "'Inter', sans-serif" }}>

      {/* ── Header ── */}
      <header className="sticky top-0 z-50 mx-3 mt-3 px-5 py-2.5 flex items-center justify-between bg-white/95 backdrop-blur-sm rounded-2xl shadow-sm border border-gray-100">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center text-base shadow-md shadow-indigo-200">
            🚦
          </div>
          <div>
            <h1 className="text-base font-bold text-gray-800 leading-tight">
              Agentic Edge Traffic Management
            </h1>
            <p className="text-[9px] text-gray-400 tracking-widest uppercase">
              Phase 2 • Multi-Agent System • God&apos;s Eye Dashboard
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {gridState && (
            <div className="flex items-center gap-3 text-right">
              <div>
                <div className="text-[9px] text-gray-400">TICK</div>
                <div className="text-sm font-bold font-mono text-indigo-500">#{gridState.tick}</div>
              </div>
              <div className="w-px h-7 bg-gray-200" />
              <div>
                <div className="text-[9px] text-gray-400">NODES</div>
                <div className="text-sm font-bold font-mono text-cyan-500">
                  {Object.keys(gridState.intersections).length}
                </div>
              </div>
            </div>
          )}
          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[10px] font-semibold ${
            connected
              ? "bg-green-50 text-green-600 border border-green-200"
              : "bg-red-50 text-red-500 border border-red-200"
          }`}>
            <div className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
            {connected ? "LIVE" : "CONNECTING..."}
          </div>
        </div>
      </header>

      {/* ── TOP: 3D Scene (16:9) + Control Panel side by side ── */}
      <div className="flex gap-3 px-3 pt-3">

        {/* 3D Viewport — 16:9 aspect ratio */}
        <div className="flex-1 min-w-0">
          <div
            className="w-full bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden"
            style={{ aspectRatio: "16 / 9" }}
          >
            {gridState ? (
              <Suspense
                fallback={
                  <div className="w-full h-full flex items-center justify-center bg-gray-50">
                    <div className="text-center space-y-2">
                      <div className="text-3xl animate-spin">⚙️</div>
                      <div className="text-gray-400 text-xs">Loading 3D Scene...</div>
                    </div>
                  </div>
                }
              >
                <Scene3D
                  gridState={gridState}
                  selectedNode={selectedNode}
                  onSelectNode={setSelectedNode}
                />
              </Suspense>
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-gray-50">
                <div className="text-center space-y-3">
                  <div className="text-4xl animate-pulse">🚦</div>
                  <div className="text-gray-400 text-sm">
                    {connected ? "Waiting for simulation data..." : "Connecting to backend..."}
                  </div>
                  <div className="text-[10px] text-gray-300">
                    Make sure FastAPI backend is running on port 8000
                  </div>
                </div>
              </div>
            )}
          </div>
          <div className="flex gap-3 px-1 mt-1 text-[9px] text-gray-400 tracking-wide">
            <span>🖱 LEFT DRAG: Rotate</span>
            <span>RIGHT DRAG: Pan</span>
            <span>SCROLL: Zoom</span>
            <span>CLICK NODE: Focus</span>
            <span>ESC: Overview</span>
          </div>
        </div>

        {/* Control Panel — fixed width */}
        <div className="w-[290px] flex-shrink-0">
          <ControlPanel
            gridState={gridState}
            connected={connected}
            sendCommand={sendCommand}
          />
        </div>
      </div>

      {/* ── BOTTOM: Metrics + Logs (scroll down to see) ── */}
      <div className="flex gap-3 px-3 pt-3 pb-6">

        {/* Metrics */}
        <div className="w-[290px] flex-shrink-0">
          <MetricsPanel gridState={gridState} />
        </div>

        {/* XAI Log / Reasoning */}
        <div className="flex-1 min-w-0" style={{ minHeight: "350px" }}>
          {selectedNode ? (
            <ReasoningHistoryPanel
              nodeId={selectedNode}
              history={agentHistory}
              onClose={() => setSelectedNode(null)}
            />
          ) : (
            <RationaleLog
              logs={logs}
              a2aMessages={gridState?.a2a_messages || []}
            />
          )}
        </div>
      </div>

    </div>
  );
}
