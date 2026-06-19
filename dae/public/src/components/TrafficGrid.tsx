"use client";

import React from "react";
import { GridState, EmergencyState } from "@/hooks/useTrafficSocket";
import IntersectionNode from "./IntersectionNode";
import { useRoadVehicles, RoadCar } from "@/hooks/useRoadVehicles";

interface TrafficGridProps {
  gridState: GridState;
  selectedNode: string | null;
  onSelectNode: (nodeId: string) => void;
}

const GRID_POSITIONS: Record<string, { row: number; col: number; x: number; y: number }> = {
  A: { row: 0, col: 0, x: 60, y: 60 },
  B: { row: 0, col: 1, x: 500, y: 60 },
  C: { row: 1, col: 0, x: 60, y: 480 },
  D: { row: 1, col: 1, x: 500, y: 480 },
};

const CONNECTIONS: Array<{ from: string; to: string }> = [
  { from: "A", to: "B" },
  { from: "A", to: "C" },
  { from: "B", to: "D" },
  { from: "C", to: "D" },
];

function getEmergencyForNode(emergencies: EmergencyState[], nodeId: string): EmergencyState | null {
  return emergencies.find((e) => !e.completed && e.current_intersection === nodeId) || null;
}

function isConnectionOnEmergencyRoute(emergencies: EmergencyState[], from: string, to: string): boolean {
  for (const em of emergencies) {
    if (em.completed) continue;
    for (let i = 0; i < em.route.length - 1; i++) {
      if (
        (em.route[i].intersection === from && em.route[i + 1].intersection === to) ||
        (em.route[i].intersection === to && em.route[i + 1].intersection === from)
      ) return true;
    }
  }
  return false;
}

export default function TrafficGrid({ gridState, selectedNode, onSelectNode }: TrafficGridProps) {
  const { intersections, emergencies, green_wave_active } = gridState;
  const { cars: roadCars, getCarPosition } = useRoadVehicles(gridState);

  // Helpers that handle both boolean and string backend values
  const isRainAt = (nodeId: string) => {
    const v: any = gridState.severe_rain;
    if (v === true || v === "true") return true;
    return v === nodeId;
  };
  const isFloodAt = (nodeId: string) => {
    const v: any = gridState.flood_active;
    if (v === true || v === "true") return true;
    return v === nodeId;
  };

  return (
    <div className="relative" style={{ width: "880px", height: "860px" }}>
      <style>{`
        @keyframes rainfall-drop {
          0% { transform: translateY(-60px) rotate(15deg); opacity: 0; }
          10% { opacity: 0.6; }
          80% { opacity: 0.6; }
          100% { transform: translateY(320px) rotate(15deg); opacity: 0; }
        }
      `}</style>
      
      {/* Light background grid dots */}
      <svg className="absolute inset-0 pointer-events-none opacity-30" width="880" height="860">
        <defs>
          <pattern id="dot-grid" width="30" height="30" patternUnits="userSpaceOnUse">
            <circle cx="15" cy="15" r="0.8" fill="#94a3b8" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#dot-grid)" />
      </svg>

      {/* Road connections SVG */}
      <svg className="absolute inset-0 pointer-events-none" width="880" height="860" style={{ zIndex: 0 }}>
        <defs>
          <linearGradient id="road-fill-v" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#d1d5db" />
            <stop offset="10%" stopColor="#9ca3af" />
            <stop offset="50%" stopColor="#6b7280" />
            <stop offset="90%" stopColor="#9ca3af" />
            <stop offset="100%" stopColor="#d1d5db" />
          </linearGradient>
          <linearGradient id="road-fill-h" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#d1d5db" />
            <stop offset="10%" stopColor="#9ca3af" />
            <stop offset="50%" stopColor="#6b7280" />
            <stop offset="90%" stopColor="#9ca3af" />
            <stop offset="100%" stopColor="#d1d5db" />
          </linearGradient>
          <filter id="road-shadow">
            <feDropShadow dx="0" dy="2" stdDeviation="4" floodColor="rgba(0,0,0,0.1)" />
          </filter>
          <filter id="glow-g">
            <feGaussianBlur stdDeviation="5" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {CONNECTIONS.map(({ from, to }) => {
          const fp = GRID_POSITIONS[from];
          const tp = GRID_POSITIONS[to];
          const x1 = fp.x + 160, y1 = fp.y + 160;
          const x2 = tp.x + 160, y2 = tp.y + 160;
          const isV = x1 === x2;
          const isEmRoute = isConnectionOnEmergencyRoute(emergencies, from, to);

          return (
            <g key={`${from}-${to}`}>
              {/* Road surface */}
              <line x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={`url(#road-fill-${isV ? "v" : "h"})`}
                strokeWidth="48" strokeLinecap="butt" filter="url(#road-shadow)" />

              {/* Sidewalk edges */}
              <line x1={x1 + (isV ? -24 : 0)} y1={y1 + (isV ? 0 : -24)}
                x2={x2 + (isV ? -24 : 0)} y2={y2 + (isV ? 0 : -24)}
                stroke="#e5e7eb" strokeWidth="3" />
              <line x1={x1 + (isV ? 24 : 0)} y1={y1 + (isV ? 0 : 24)}
                x2={x2 + (isV ? 24 : 0)} y2={y2 + (isV ? 0 : 24)}
                stroke="#e5e7eb" strokeWidth="3" />

              {/* Center dashes */}
              <line x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="rgba(250, 204, 21, 0.45)" strokeWidth="1.5" strokeDasharray="10 8" />

              {/* Green Wave overlay */}
              {isEmRoute && green_wave_active && (
                <>
                  <line x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke="rgba(16,185,129,0.2)" strokeWidth="48" filter="url(#glow-g)" />
                  <line x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke="#10b981" strokeWidth="3" strokeDasharray="12 6"
                    strokeLinecap="round" filter="url(#glow-g)"
                    style={{ animation: "dash-march 0.5s linear infinite" }} />
                  <line x1={x1 + (isV ? -24 : 0)} y1={y1 + (isV ? 0 : -24)}
                    x2={x2 + (isV ? -24 : 0)} y2={y2 + (isV ? 0 : -24)}
                    stroke="rgba(16,185,129,0.5)" strokeWidth="3" filter="url(#glow-g)" />
                  <line x1={x1 + (isV ? 24 : 0)} y1={y1 + (isV ? 0 : 24)}
                    x2={x2 + (isV ? 24 : 0)} y2={y2 + (isV ? 0 : 24)}
                    stroke="rgba(16,185,129,0.5)" strokeWidth="3" filter="url(#glow-g)" />
                </>
              )}

              {/* Label */}
              <text x={(x1 + x2) / 2 + (isV ? 32 : 0)} y={(y1 + y2) / 2 + (isV ? 0 : -32)}
                fill="#94a3b8" fontSize="10" textAnchor="middle" fontFamily="monospace">
                {from}↔{to}
              </text>
            </g>
          );
        })}

        {/* Ambulance radar */}
        {emergencies.filter((e) => !e.completed && e.current_intersection).map((em) => {
          const pos = GRID_POSITIONS[em.current_intersection!];
          if (!pos) return null;
          const cx = pos.x + 160, cy = pos.y + 160;
          return (
            <g key={em.vehicle_id}>
              <circle cx={cx} cy={cy} r="40" fill="none" stroke="rgba(220,38,38,0.35)" strokeWidth="1.5">
                <animate attributeName="r" from="25" to="65" dur="1.5s" repeatCount="indefinite" />
                <animate attributeName="opacity" from="0.7" to="0" dur="1.5s" repeatCount="indefinite" />
              </circle>
              <circle cx={cx} cy={cy} r="25" fill="none" stroke="rgba(220,38,38,0.5)" strokeWidth="1">
                <animate attributeName="r" from="15" to="50" dur="1.5s" begin="0.5s" repeatCount="indefinite" />
                <animate attributeName="opacity" from="0.5" to="0" dur="1.5s" begin="0.5s" repeatCount="indefinite" />
              </circle>
            </g>
          );
        })}

        {/* ═══ ROAD VEHICLES ═══ */}
        {roadCars.map((car) => {
          const pos = getCarPosition(car);
          if (!pos) return null;
          const w = car.type === "truck" ? 16 : 10;
          const h = car.type === "truck" ? 8 : 6;
          // Determine road direction for rotation
          const fromC = GRID_POSITIONS[car.fromNode];
          const toC = GRID_POSITIONS[car.toNode];
          const isVertical = fromC && toC && fromC.x === toC.x;
          const angle = isVertical ? 90 : 0;
          return (
            <g key={car.id} style={{ transition: "none" }}>
              {/* Shadow */}
              <ellipse
                cx={pos.x + 1} cy={pos.y + 2}
                rx={w / 2 - 1} ry={h / 2 - 1}
                fill="rgba(0,0,0,0.15)"
                transform={`rotate(${angle}, ${pos.x + 1}, ${pos.y + 2})`}
              />
              {/* Car body */}
              <rect
                x={pos.x - w / 2} y={pos.y - h / 2}
                width={w} height={h}
                rx={2} ry={2}
                fill={car.color}
                stroke="rgba(0,0,0,0.2)" strokeWidth={0.5}
                transform={`rotate(${angle}, ${pos.x}, ${pos.y})`}
              />
              {/* Windshield highlight */}
              <rect
                x={pos.x + (isVertical ? -1.5 : w / 2 - 4)} y={pos.y + (isVertical ? (h > 6 ? -6 : -4) : -1)}
                width={isVertical ? 3 : 3} height={isVertical ? 3 : 2}
                rx={1}
                fill="rgba(255,255,255,0.5)"
                transform={`rotate(${angle}, ${pos.x}, ${pos.y})`}
              />
              {/* Brake lights when stopped */}
              {car.stopped && (
                <circle cx={pos.x - (isVertical ? 0 : w / 2)} cy={pos.y - (isVertical ? h / 2 : 0)} r={1.5} fill="#ef4444">
                  <animate attributeName="opacity" values="1;0.4;1" dur="0.5s" repeatCount="indefinite" />
                </circle>
              )}
            </g>
          );
        })}
      </svg>

      {/* Intersection nodes */}
      {Object.entries(GRID_POSITIONS).map(([nodeId, pos]) => {
        const nodeState = intersections[nodeId];
        if (!nodeState) return null;
        return (
          <div key={nodeId} className="absolute transition-transform duration-300"
            style={{ left: `${pos.x}px`, top: `${pos.y}px`, zIndex: selectedNode === nodeId ? 10 : 1 }}>
            <IntersectionNode
              state={nodeState}
              emergency={getEmergencyForNode(emergencies, nodeId)}
              onClick={() => onSelectNode(nodeId)}
              isSelected={selectedNode === nodeId}
              gridPosition={{ row: pos.row, col: pos.col }}
            />
            {isFloodAt(nodeId) && (
              <div className="absolute top-[-30px] left-1/2 -translate-x-1/2 z-30 animate-pulse flex flex-col items-center pointer-events-none">
                <div className="text-5xl filter drop-shadow-[0_0_12px_rgba(59,130,246,0.9)]">🌊</div>
                <div className="text-xs font-bold text-blue-600 bg-blue-50 border border-blue-200 px-2 py-1 rounded-md shadow flex items-center gap-1 whitespace-nowrap mt-2">
                  <div className="w-2 h-2 rounded-full bg-blue-500 animate-ping" />
                  FLOOD WARNING
                </div>
              </div>
            )}
            {isRainAt(nodeId) && (
              <>
                <div className="absolute top-[-30px] left-1/2 -translate-x-1/2 z-30 flex flex-col items-center pointer-events-none">
                  <div className="text-5xl filter drop-shadow-[0_0_8px_rgba(59,130,246,0.6)] animate-pulse">🌧️</div>
                  <div className="text-xs font-bold text-blue-600 bg-blue-50 border border-blue-200 px-2 py-1 rounded-md shadow flex items-center gap-1 whitespace-nowrap mt-2">
                    SEVERE RAIN
                  </div>
                </div>
                {/* Rain Drops Overlay */}
                <div className="absolute w-[320px] h-[320px] pointer-events-none z-40 overflow-hidden rounded-[16px]" style={{ left: 0, top: 0 }}>
                  {Array.from({ length: 40 }).map((_, i) => {
                    // Use index-based deterministic positions instead of Math.random()
                    const leftPct = ((i * 37 + 13) % 130) - 10;
                    const dur = 0.4 + ((i * 7) % 4) * 0.1;
                    const delay = (i * 0.05) % 2;
                    return (
                      <div key={i} className="absolute bg-blue-400 rounded-full blur-[0.5px]"
                        style={{
                          left: `${leftPct}%`,
                          top: `-10%`,
                          width: '2px', height: '24px',
                          opacity: 0,
                          animation: `rainfall-drop ${dur}s linear infinite`,
                          animationDelay: `${delay}s`
                        }} />
                    );
                  })}
                </div>
              </>
            )}
          </div>
        );
      })}

      {/* Emergency Overlays (Green Wave & Flood) */}
      <div className="absolute top-3 w-full pointer-events-none z-20 flex flex-col items-center gap-2">
        {green_wave_active && (
            <div className="green-wave-path px-5 py-2.5 rounded-full text-sm font-bold text-white flex items-center gap-2 shadow-xl">
              🌊 GREEN WAVE ACTIVE 🚑
            </div>
        )}
        {gridState.flood_active && (
            <div className="bg-red-600 px-5 py-2.5 rounded-full text-sm font-bold text-white flex items-center gap-2 shadow-xl animate-pulse">
              🚨 MANUAL OVERRIDE: FLOOD DETECTED
            </div>
        )}
      </div>
    </div>
  );
}
