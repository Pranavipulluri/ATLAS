"use client";

import React, { useMemo } from "react";
import { IntersectionState, EmergencyState } from "@/hooks/useTrafficSocket";

interface IntersectionNodeProps {
  state: IntersectionState;
  emergency?: EmergencyState | null;
  onClick?: () => void;
  isSelected?: boolean;
  gridPosition: { row: number; col: number };
}

function densityColor(d: number) {
  if (d <= 5) return "#22c55e";
  if (d <= 15) return "#eab308";
  if (d <= 25) return "#f97316";
  return "#ef4444";
}

function carColor(i: number) {
  const colors = ["#3b82f6", "#6366f1", "#8b5cf6", "#ec4899", "#f97316", "#14b8a6"];
  return colors[i % colors.length];
}

type CarState = { id: number; colorIdx: number; status: "queued" | "crossing"; crossTime: number };

let _carIdGen = 0;

function useTrafficCars(density: number, isGreen: boolean) {
  const carsRef = React.useRef<CarState[]>([]);
  const [, forceUpdate] = React.useState(0);
  const prevDensityRef = React.useRef(density);

  React.useEffect(() => {
    const target = Math.min(density, 7);
    const cars = carsRef.current;

    // Remove expired crossing cars
    const now = Date.now();
    carsRef.current = cars.filter((c) => c.status === "queued" || now - c.crossTime < 1500);

    const queued = carsRef.current.filter((c) => c.status === "queued");

    if (target > queued.length) {
      // Need more cars: spawn at tail of queue
      for (let i = 0; i < target - queued.length; i++) {
        carsRef.current.push({
          id: _carIdGen++,
          colorIdx: Math.floor(Math.random() * 6),
          status: "queued",
          crossTime: 0,
        });
      }
    } else if (target < queued.length) {
      // Fewer cars: mark front-of-queue as "crossing"
      let toRemove = queued.length - target;
      for (const car of carsRef.current) {
        if (toRemove <= 0) break;
        if (car.status === "queued") {
          car.status = "crossing";
          car.crossTime = Date.now();
          toRemove--;
        }
      }
    }

    prevDensityRef.current = density;
    forceUpdate((n) => n + 1);
  }, [density, isGreen]);

  // Cleanup timer for crossing cars
  React.useEffect(() => {
    const t = setInterval(() => {
      const now = Date.now();
      const before = carsRef.current.length;
      carsRef.current = carsRef.current.filter((c) => c.status === "queued" || now - c.crossTime < 1500);
      if (carsRef.current.length !== before) forceUpdate((n) => n + 1);
    }, 500);
    return () => clearInterval(t);
  }, []);

  return carsRef.current;
}

export default function IntersectionNode({
  state,
  emergency,
  onClick,
  isSelected,
}: IntersectionNodeProps) {
  const hasEmergency =
    emergency && !emergency.completed && emergency.current_intersection === state.intersection_id;
  const isGreenWave = state.green_wave_active;

  // Memoize random windows so they don't flicker
  const windowStates = useMemo(() => Array.from({ length: 24 }, () => Math.random() > 0.5), []);

  const nCars = useTrafficCars(state.lanes.North?.density || 0, !!state.lanes.North?.is_green);
  const sCars = useTrafficCars(state.lanes.South?.density || 0, !!state.lanes.South?.is_green);
  const eCars = useTrafficCars(state.lanes.East?.density || 0, !!state.lanes.East?.is_green);
  const wCars = useTrafficCars(state.lanes.West?.density || 0, !!state.lanes.West?.is_green);

  return (
    <div
      onClick={onClick}
      className={`relative cursor-pointer select-none transition-transform duration-300 ${isSelected ? "scale-[1.03]" : ""}`}
      style={{ width: "320px", height: "320px" }}
    >
      {/* ====== ISOMETRIC 3D VIEW ====== */}
      <div className="absolute inset-0" style={{ perspective: "900px", perspectiveOrigin: "50% 35%" }}>
        <div
          className="relative w-full h-full"
          style={{
            transformStyle: "preserve-3d",
            transform: "rotateX(60deg) rotateZ(45deg) scale(0.68)",
            transformOrigin: "50% 50%",
          }}
        >
          {/* ── Ground plane ── */}
          <div
            className="absolute"
            style={{
              width: "340px", height: "340px", left: "-20px", top: "-20px",
              background: "#d4d9de",
              transform: "translateZ(-1px)",
              borderRadius: "12px",
              boxShadow: "0 20px 60px rgba(0,0,0,0.15)",
            }}
          />

          {/* ── Sidewalk / grass patches (corners) ── */}
          {[
            { l: "0px", t: "0px" },
            { l: "195px", t: "0px" },
            { l: "0px", t: "195px" },
            { l: "195px", t: "195px" },
          ].map((pos, i) => (
            <div
              key={i}
              className="absolute"
              style={{
                left: pos.l, top: pos.t,
                width: "105px", height: "105px",
                background: "linear-gradient(135deg, #bbf7d0, #86efac)",
                transform: "translateZ(1px)",
                borderRadius: "6px",
                boxShadow: "inset 0 2px 6px rgba(0,0,0,0.06)",
              }}
            />
          ))}

          {/* ── Road: North-South ── */}
          <div
            className="absolute"
            style={{
              width: "90px", height: "300px", left: "105px", top: "0px",
              background: "linear-gradient(90deg, #4b5563, #6b7280 20%, #9ca3af 50%, #6b7280 80%, #4b5563)",
              transform: "translateZ(2px)",
              boxShadow: "0 4px 15px rgba(0,0,0,0.12)",
            }}
          >
            {/* Center dashes */}
            <div className="absolute left-1/2 -translate-x-1/2 top-2 bottom-2 w-px flex flex-col items-center justify-around">
              {[...Array(10)].map((_, i) => (
                <div key={i} className="w-0.5 h-3 bg-yellow-400/60 rounded-full" />
              ))}
            </div>
            {/* Curb lines */}
            <div className="absolute left-1 top-0 bottom-0 w-px bg-white/30" />
            <div className="absolute right-1 top-0 bottom-0 w-px bg-white/30" />
          </div>

          {/* ── Road: East-West ── */}
          <div
            className="absolute"
            style={{
              width: "300px", height: "90px", left: "0px", top: "105px",
              background: "linear-gradient(180deg, #4b5563, #6b7280 20%, #9ca3af 50%, #6b7280 80%, #4b5563)",
              transform: "translateZ(2px)",
              boxShadow: "0 4px 15px rgba(0,0,0,0.12)",
            }}
          >
            <div className="absolute top-1/2 -translate-y-1/2 left-2 right-2 h-px flex items-center justify-around">
              {[...Array(10)].map((_, i) => (
                <div key={i} className="w-3 h-0.5 bg-yellow-400/60 rounded-full" />
              ))}
            </div>
            <div className="absolute top-1 left-0 right-0 h-px bg-white/30" />
            <div className="absolute bottom-1 left-0 right-0 h-px bg-white/30" />
          </div>

          {/* ── Intersection center (crossroads) ── */}
          <div
            className={hasEmergency ? "emergency-active" : ""}
            style={{
              position: "absolute",
              width: "90px", height: "90px", left: "105px", top: "105px",
              background: isGreenWave
                ? "linear-gradient(135deg, #6ee7b7, #34d399)"
                : "linear-gradient(135deg, #9ca3af, #6b7280)",
              transform: "translateZ(3px)",
              boxShadow: isGreenWave
                ? "0 6px 25px rgba(52,211,153,0.4)"
                : "0 6px 20px rgba(0,0,0,0.15)",
              borderRadius: "2px",
            }}
          >
            {/* Zebra crossings */}
            <div className="absolute top-1/2 left-1 right-1 -translate-y-1/2 flex flex-col gap-[2px]">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-[2px] bg-white/50 rounded-full" />
              ))}
            </div>
            <div className="absolute left-1/2 top-1 bottom-1 -translate-x-1/2 flex flex-row gap-[2px]">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="w-[2px] bg-white/50 rounded-full" />
              ))}
            </div>
          </div>

          {/* ── Traffic Signal Poles ── */}
          {(["North", "South", "East", "West"] as const).map((lane) => {
            const laneState = state.lanes[lane];
            if (!laneState) return null;
            const isGreen = laneState.is_green;

            const polePositions: Record<string, { left: string; top: string }> = {
              North: { left: "96px", top: "96px" },
              South: { left: "200px", top: "200px" },
              East: { left: "200px", top: "96px" },
              West: { left: "96px", top: "200px" },
            };

            return (
              <div key={lane} className="absolute" style={polePositions[lane]}>
                {/* Pole base */}
                <div style={{
                  width: "8px", height: "8px", borderRadius: "50%",
                  background: "#94a3b8", transform: "translateZ(3px)",
                  boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
                }} />
                {/* Signal housing */}
                <div style={{
                  position: "absolute", left: "-3px", top: "-3px",
                  width: "14px", height: "22px", borderRadius: "3px",
                  background: "#1e293b", transform: "translateZ(30px)",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
                  display: "flex", flexDirection: "column", alignItems: "center",
                  justifyContent: "center", gap: "2px", padding: "2px",
                }}>
                  {/* Red light */}
                  <div style={{
                    width: "8px", height: "8px", borderRadius: "50%",
                    background: isGreen ? "#7f1d1d" : "#ef4444",
                    boxShadow: isGreen ? "none" : "0 0 8px rgba(239,68,68,0.6)",
                  }} />
                  {/* Green light */}
                  <div style={{
                    width: "8px", height: "8px", borderRadius: "50%",
                    background: isGreen ? "#22c55e" : "#14532d",
                    boxShadow: isGreen ? "0 0 8px rgba(34,197,94,0.6)" : "none",
                  }} />
                </div>
              </div>
            );
          })}

          {/* ── Cars physically tracked and animated ── */}
          {/* North */}
          {nCars.map((c, idx) => {
            const qIdx = nCars.filter(x => x.status === "queued").findIndex(x => x.id === c.id);
            const top = c.status === "crossing" ? "340px" : `${105 - 18 - (qIdx >= 0 ? qIdx : 0) * 21}px`;
            return (
              <div key={c.id} className="absolute transition-all" style={{
                left: "125px", top, width: "12px", height: "18px", borderRadius: "3px",
                background: carColor(c.colorIdx), boxShadow: "0 2px 4px rgba(0,0,0,0.15)",
                transform: "translateZ(4px)",
                transitionDuration: c.status === "crossing" ? "1200ms" : "300ms",
                transitionTimingFunction: c.status === "crossing" ? "linear" : "ease-out"
              }} />
            );
          })}
          {/* South */}
          {sCars.map((c, idx) => {
            const qIdx = sCars.filter(x => x.status === "queued").findIndex(x => x.id === c.id);
            const top = c.status === "crossing" ? "-40px" : `${215 + (qIdx >= 0 ? qIdx : 0) * 21}px`;
            return (
              <div key={c.id} className="absolute transition-all" style={{
                left: "160px", top, width: "12px", height: "18px", borderRadius: "3px",
                background: carColor(c.colorIdx), boxShadow: "0 2px 4px rgba(0,0,0,0.15)",
                transform: "translateZ(4px)",
                transitionDuration: c.status === "crossing" ? "1200ms" : "300ms",
                transitionTimingFunction: c.status === "crossing" ? "linear" : "ease-out"
              }} />
            );
          })}
          {/* West */}
          {wCars.map((c, idx) => {
            const qIdx = wCars.filter(x => x.status === "queued").findIndex(x => x.id === c.id);
            const left = c.status === "crossing" ? "340px" : `${105 - 18 - (qIdx >= 0 ? qIdx : 0) * 21}px`;
            return (
              <div key={c.id} className="absolute transition-all" style={{
                top: "130px", left, width: "18px", height: "12px", borderRadius: "3px",
                background: carColor(c.colorIdx), boxShadow: "0 2px 4px rgba(0,0,0,0.15)",
                transform: "translateZ(4px)",
                transitionDuration: c.status === "crossing" ? "1200ms" : "300ms",
                transitionTimingFunction: c.status === "crossing" ? "linear" : "ease-out"
              }} />
            );
          })}
          {/* East */}
          {eCars.map((c, idx) => {
            const qIdx = eCars.filter(x => x.status === "queued").findIndex(x => x.id === c.id);
            const left = c.status === "crossing" ? "-40px" : `${215 + (qIdx >= 0 ? qIdx : 0) * 21}px`;
            return (
              <div key={c.id} className="absolute transition-all" style={{
                top: "160px", left, width: "18px", height: "12px", borderRadius: "3px",
                background: carColor(c.colorIdx), boxShadow: "0 2px 4px rgba(0,0,0,0.15)",
                transform: "translateZ(4px)",
                transitionDuration: c.status === "crossing" ? "1200ms" : "300ms",
                transitionTimingFunction: c.status === "crossing" ? "linear" : "ease-out"
              }} />
            );
          })}

          {/* ── Corner buildings (3D extruded) ── */}
          {[
            { l: "8px", t: "8px", w: 90, h: 90, bh: 40, col: "#e0e7ff", roof: "#c7d2fe" },
            { l: "202px", t: "8px", w: 88, h: 90, bh: 28, col: "#dbeafe", roof: "#bfdbfe" },
            { l: "8px", t: "202px", w: 90, h: 88, bh: 34, col: "#d1fae5", roof: "#a7f3d0" },
            { l: "202px", t: "202px", w: 88, h: 88, bh: 22, col: "#fef3c7", roof: "#fde68a" },
          ].map((b, bi) => (
            <React.Fragment key={bi}>
              {/* Building base (side face illusion) */}
              <div className="absolute" style={{
                left: b.l, top: b.t, width: `${b.w}px`, height: `${b.h}px`,
                background: b.col, transform: "translateZ(1px)",
                borderRadius: "4px", border: "1px solid rgba(0,0,0,0.05)",
              }} />
              {/* Building top face */}
              <div className="absolute" style={{
                left: b.l, top: b.t, width: `${b.w}px`, height: `${b.h}px`,
                background: b.roof, transform: `translateZ(${b.bh}px)`,
                borderRadius: "4px", border: "1px solid rgba(0,0,0,0.06)",
                boxShadow: "0 -4px 16px rgba(0,0,0,0.08)",
              }}>
                {/* Windows grid */}
                <div className="grid grid-cols-3 gap-1 p-2.5">
                  {windowStates.slice(bi * 6, bi * 6 + 6).map((lit, wi) => (
                    <div key={wi} className="h-2.5 rounded-sm" style={{
                      background: lit ? "rgba(99,102,241,0.2)" : "rgba(0,0,0,0.06)",
                    }} />
                  ))}
                </div>
              </div>
            </React.Fragment>
          ))}

          {/* ── Ambulance on intersection ── */}
          {hasEmergency && (
            <div className="absolute ambulance-icon" style={{
              left: "132px", top: "132px",
              transform: "translateZ(45px)", fontSize: "30px",
              filter: "drop-shadow(0 4px 10px rgba(220,38,38,0.6))",
            }}>
              🚑
            </div>
          )}
        </div>
      </div>

      {/* ====== 2D HUD OVERLAY ====== */}
      {/* Node label */}
      <div className="absolute -top-2 left-1/2 -translate-x-1/2 z-20">
        <div className={`
          px-3 py-1 rounded-full text-[11px] font-bold tracking-wider shadow-md
          ${hasEmergency
            ? "bg-red-500 text-white"
            : isGreenWave
            ? "bg-emerald-500 text-white"
            : "bg-white text-gray-700 border border-gray-200"
          }
        `}>
          Node {state.intersection_id}
        </div>
      </div>

      {/* Pedestrians 2D Overlay */}
      {(["North", "South", "East", "West"] as const).map((lane) => {
        if (!state.lanes[lane]?.has_pedestrians) return null;
        const pos2d: Record<string, { top: string; left: string }> = {
          North: { top: "22%", left: "37%" },
          South: { top: "68%", left: "55%" },
          East: { top: "35%", left: "68%" },
          West: { top: "55%", left: "22%" },
        };
        return (
          <div key={lane} className="absolute z-30 animate-bounce" style={{ top: pos2d[lane].top, left: pos2d[lane].left }}>
            <div className="text-4xl filter drop-shadow-lg" title={`Pedestrians waiting at ${lane} lane`}>
              🚶‍♂️
            </div>
          </div>
        );
      })}

      {/* Lane stats bar */}
      <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 z-20">
        <div className="flex gap-0.5 px-2 py-1.5 rounded-xl bg-white/95 shadow-md border border-gray-100 backdrop-blur-sm">
          {(["N", "S", "E", "W"] as const).map((short) => {
            const full = short === "N" ? "North" : short === "S" ? "South" : short === "E" ? "East" : "West";
            const lane = state.lanes[full];
            if (!lane) return null;
            return (
              <div key={short} className={`
                flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[9px]
                ${lane.is_green ? "bg-green-50" : "bg-gray-50"}
              `}>
                <div className={`w-1.5 h-1.5 rounded-full ${lane.is_green ? "bg-green-500" : "bg-red-400"}`} />
                <span className="font-bold text-gray-500">{short}</span>
                <span className="font-mono font-semibold text-gray-700">{lane.density}</span>
                <div className="w-5 h-1 bg-gray-200 rounded-full overflow-hidden">
                  <div className="h-full rounded-full density-bar-fill"
                    style={{ width: `${Math.min(100, (lane.density / 50) * 100)}%`, background: densityColor(lane.density) }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Phase timer */}
      <div className="absolute top-0 right-0 z-20 px-1.5 py-0.5 rounded-bl-lg bg-white/80 text-[8px] font-mono text-gray-400 shadow-sm">
        T:{state.time_in_phase}s
      </div>
    </div>
  );
}
