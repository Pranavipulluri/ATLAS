"use client";

import React, { useState } from "react";
import { GridState } from "@/hooks/useTrafficSocket";

interface ControlPanelProps {
  gridState: GridState | null;
  connected: boolean;
  sendCommand: (cmd: Record<string, unknown>) => void;
}

const ROUTES = [
  { key: "A_to_D", label: "A → B → D", desc: "Diagonal" },
  { key: "A_to_C", label: "A → C", desc: "South" },
  { key: "A_to_B", label: "A → B", desc: "East" },
  { key: "D_to_A", label: "D → B → A", desc: "Reverse" },
  { key: "B_to_C", label: "B → D → C", desc: "Long" },
  { key: "C_to_B", label: "C → D → B", desc: "Long Rev" },
];

export default function ControlPanel({ gridState, connected, sendCommand }: ControlPanelProps) {
  const [route, setRoute] = useState("A_to_D");
  const [spawning, setSpawning] = useState(false);
  const [chaosNode, setChaosNode] = useState("A");
  const prevFlood = React.useRef<string | null>(null);

  React.useEffect(() => {
    if (gridState?.flood_active && gridState.flood_active !== prevFlood.current) {
      // Play 3-second warning siren
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(600, ctx.currentTime);
      
      // Siren wobble effect over 3 seconds
      for (let i = 0; i < 6; i++) {
        osc.frequency.linearRampToValueAtTime(800, ctx.currentTime + i * 0.5 + 0.25);
        osc.frequency.linearRampToValueAtTime(600, ctx.currentTime + i * 0.5 + 0.5);
      }
      
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 3.0);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 3.0);
      
      // TTS Alert
      const utterance = new SpeechSynthesisUtterance(`water logging is detected at node ${gridState.flood_active}`);
      utterance.rate = 1.0;
      utterance.pitch = 1.2;
      window.speechSynthesis.speak(utterance);
      
      // Delayed browser alert so it doesn't block TTS immediately
      setTimeout(() => alert(`🚨 FLOOD DETECTED AT NODE ${gridState.flood_active}! Escalating to manual human control immediately.`), 3100);
    }
    prevFlood.current = gridState?.flood_active || null;
  }, [gridState?.flood_active]);

  const spawn = () => { setSpawning(true); sendCommand({ type: "spawn_ambulance", route }); setTimeout(() => setSpawning(false), 1000); };
  const active = gridState?.emergencies?.filter((e) => !e.completed) || [];

  return (
    <div className="glass-card flex flex-col">
      <div className="px-4 py-3 border-b border-gray-100 flex items-center gap-2">
        <span className="text-lg">🎮</span>
        <h3 className="text-sm font-bold text-gray-800 tracking-wide">Control Panel</h3>
        <div className="ml-auto flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`} />
          <span className="text-[10px] text-gray-400">{connected ? "LIVE" : "OFFLINE"}</span>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Ambulance */}
        <div>
          <label className="text-[10px] uppercase tracking-widest text-gray-400 font-semibold mb-2 block">🚑 Spawn Emergency</label>
          <select value={route} onChange={(e) => setRoute(e.target.value)}
            className="w-full bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-700 px-3 py-2 mb-2 focus:outline-none focus:border-indigo-400 transition-colors">
            {ROUTES.map((r) => <option key={r.key} value={r.key}>{r.label} — {r.desc}</option>)}
          </select>
          <button onClick={spawn} disabled={spawning}
            className={`w-full py-2.5 rounded-lg text-sm font-bold tracking-wide transition-all ${
              spawning ? "bg-red-100 text-red-400 cursor-wait" : "bg-red-500 text-white hover:bg-red-600 active:scale-[0.98] shadow-md shadow-red-200"
            }`}>
            {spawning ? "🚨 Dispatching..." : "🚑 Deploy Ambulance"}
          </button>

          {active.length > 0 && (
            <div className="mt-2 space-y-1">
              {active.map((em) => (
                <div key={em.vehicle_id} className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-red-50 border border-red-100">
                  <span className="ambulance-icon text-sm">🚑</span>
                  <div className="flex-1">
                    <span className="text-[11px] font-bold text-red-500">{em.vehicle_id}</span>
                    <span className="text-[10px] text-gray-400 ml-2">
                      Step {em.current_step + 1}/{em.route.length} • {em.current_intersection ? `Node ${em.current_intersection}` : "Transit"}
                    </span>
                  </div>
                  <div className="w-14 h-1.5 bg-red-100 rounded-full overflow-hidden">
                    <div className="h-full bg-red-400 rounded-full transition-all duration-500"
                      style={{ width: `${((em.current_step + em.ticks_at_step / em.ticks_per_step) / em.route.length) * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="h-px bg-gray-100" />

        {/* Traffic */}
        <div>
          <label className="text-[10px] uppercase tracking-widest text-gray-400 font-semibold mb-2 block">🚗 Traffic Multiplier</label>
          <div className="grid grid-cols-2 gap-2">
            {["A", "B", "C", "D"].map((n) => {
              const m = gridState?.traffic_multipliers?.[n] || 1.0;
              return (
                <div key={n} className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-gray-50">
                  <span className="text-xs font-bold text-gray-500 w-4">{n}</span>
                  <input type="range" min="0.5" max="3" step="0.5" value={m}
                    onChange={(e) => sendCommand({ type: "set_traffic", node: n, multiplier: parseFloat(e.target.value) })}
                    className="flex-1 h-1 accent-indigo-500" />
                  <span className="text-[10px] font-mono text-gray-400 w-6">{m}×</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="h-px bg-gray-100" />

        {/* Chaos Toggles */}
        <div>
          <label className="text-[10px] uppercase tracking-widest text-gray-400 font-semibold mb-2 block">🌪️ Chaos Toggles</label>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-[11px] font-bold text-gray-500">Target Node:</span>
            <select value={chaosNode} onChange={(e) => setChaosNode(e.target.value)}
              className="bg-gray-50 border border-gray-200 rounded px-2 py-1 text-xs font-bold text-gray-700 outline-none focus:border-indigo-400">
              <option value="A">Node A</option>
              <option value="B">Node B</option>
              <option value="C">Node C</option>
              <option value="D">Node D</option>
            </select>
          </div>
          <div className="space-y-2">
            <button
              onClick={() => sendCommand({ type: "toggle_severe_rain", node: chaosNode, state: gridState?.severe_rain !== chaosNode })}
              className={`w-full py-2 rounded-lg text-sm font-bold tracking-wide transition-all border ${
                gridState?.severe_rain === chaosNode
                  ? "bg-blue-500 text-white border-blue-600 shadow-md shadow-blue-200" 
                  : "bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100"
              }`}
            >
              {gridState?.severe_rain === chaosNode ? `🌧️ Rain Active on ${chaosNode}` : `🌧️ Simulate Rain at ${chaosNode}`}
            </button>
            <button
              onClick={() => sendCommand({ type: "trigger_flood", node: chaosNode })}
              disabled={!!gridState?.flood_active}
              className={`w-full py-2 rounded-lg text-sm font-bold tracking-wide transition-all border ${
                gridState?.flood_active
                  ? "bg-red-500 text-white border-red-600 shadow-md shadow-red-200"
                  : "bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100"
              }`}
            >
              {gridState?.flood_active ? `🚨 Flood Active at ${gridState.flood_active}` : `🌊 Trigger Flood at ${chaosNode}`}
            </button>
            <button
              onClick={() => {
                sendCommand({ type: "spawn_pedestrians", node: chaosNode });
              }}
              className="w-full py-2 rounded-lg text-sm font-bold tracking-wide transition-all border bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100"
            >
              🚶 Spawn Pedestrians at {chaosNode}
            </button>
          </div>
        </div>

        <div className="h-px bg-gray-100" />

        {/* Speed */}
        <div>
          <label className="text-[10px] uppercase tracking-widest text-gray-400 font-semibold mb-2 block">⚡ Speed</label>
          <div className="flex gap-1.5">
            {[{ l: "0.5×", r: 2 }, { l: "1×", r: 1 }, { l: "2×", r: 0.5 }, { l: "4×", r: 0.25 }].map((o) => (
              <button key={o.l} onClick={() => sendCommand({ type: "set_tick_rate", rate: o.r })}
                className={`flex-1 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  gridState?.tick_rate === o.r ? "bg-indigo-500 text-white shadow-sm" : "bg-gray-100 text-gray-500 hover:bg-gray-200"
                }`}>{o.l}</button>
            ))}
          </div>
        </div>

        <div className="h-px bg-gray-100" />

        <button onClick={() => sendCommand({ type: "reset" })}
          className="w-full py-2 rounded-lg text-sm font-semibold text-gray-500 bg-gray-50 hover:bg-gray-100 transition-all border border-gray-200">
          ↺ Reset Simulation
        </button>

        {gridState && (
          <div className="grid grid-cols-2 gap-2 pt-1">
            <div className="text-center px-2 py-1.5 rounded-lg bg-indigo-50">
              <div className="text-[10px] text-gray-400">Tick</div>
              <div className="text-sm font-bold text-indigo-600 font-mono">#{gridState.tick}</div>
            </div>
            <div className="text-center px-2 py-1.5 rounded-lg bg-red-50">
              <div className="text-[10px] text-gray-400">Active 🚑</div>
              <div className="text-sm font-bold text-red-500 font-mono">{active.length}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
