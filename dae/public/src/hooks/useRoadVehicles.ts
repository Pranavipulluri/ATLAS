"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { GridState } from "@/hooks/useTrafficSocket";

/* ─── Road topology ─── */
const GRID_CENTERS: Record<string, { x: number; y: number }> = {
  A: { x: 220, y: 220 },
  B: { x: 660, y: 220 },
  C: { x: 220, y: 640 },
  D: { x: 660, y: 640 },
};

// Each lane departure maps to a road segment
// When density drops on a lane (green light departure), cars travel along that road
const LANE_TO_ROAD: Record<string, Record<string, string>> = {
  A: { East: "B", South: "C" },           // A→B (East), A→C (South)
  B: { West: "A", South: "D" },           // B→A (West), B→D (South)
  C: { North: "A", East: "D" },           // C→A (North), C→D (East)
  D: { North: "B", West: "C" },           // D→B (North), D→C (West)
};

// Opposite mapping: arriving at a node from a direction
const INCOMING_LANE: Record<string, Record<string, string>> = {
  A: { B: "East", C: "South" },
  B: { A: "West", D: "South" },
  C: { A: "North", D: "East" },
  D: { B: "North", C: "West" },
};

/* ─── Types ─── */
export interface RoadCar {
  id: number;
  fromNode: string;
  toNode: string;
  progress: number;      // 0 = at origin, 1 = at destination
  speed: number;         // progress per second (0.2 = 5s journey)
  color: string;
  laneOffset: number;    // px offset from center line (-10 or +10 for two lanes)
  type: "car" | "truck";
  stopped: boolean;
}

const CAR_COLORS = [
  "#3b82f6", "#ef4444", "#f59e0b", "#10b981", "#8b5cf6",
  "#ec4899", "#06b6d4", "#f97316", "#6366f1", "#14b8a6",
  "#e11d48", "#7c3aed", "#0ea5e9", "#d97706", "#059669",
];

let _roadCarId = 0;

/* ─── Hook ─── */
export function useRoadVehicles(gridState: GridState | null) {
  const carsRef = useRef<RoadCar[]>([]);
  const prevDensitiesRef = useRef<Record<string, Record<string, number>>>({});
  const [renderTick, setRenderTick] = useState(0);
  
  // Keep latest gridState in a ref so the animation loop can access it without restarting
  const gridStateRef = useRef<GridState | null>(gridState);
  useEffect(() => {
    gridStateRef.current = gridState;
  }, [gridState]);

  // Spawn cars when density drops (departure from green lane)
  useEffect(() => {
    if (!gridState?.intersections) return;

    const prevDens = prevDensitiesRef.current;
    const newDens: Record<string, Record<string, number>> = {};

    for (const nodeId of ["A", "B", "C", "D"]) {
      const nodeState = gridState.intersections[nodeId];
      if (!nodeState) continue;
      newDens[nodeId] = {};

      for (const lane of ["North", "South", "East", "West"]) {
        const density = nodeState.lanes?.[lane]?.density ?? 0;
        newDens[nodeId][lane] = density;

        // Only spawn road cars for lanes that have a road connection
        const destNode = LANE_TO_ROAD[nodeId]?.[lane];
        if (!destNode) continue;

        const prevD = prevDens[nodeId]?.[lane] ?? density;
        const departed = prevD - density;

        if (departed > 0 && nodeState.lanes?.[lane]?.is_green) {
          // Spawn 'departed' cars on the road
          const carsToSpawn = Math.min(departed, 3); // Cap to avoid flooding
          for (let i = 0; i < carsToSpawn; i++) {
            const isTruck = Math.random() < 0.2;
            carsRef.current.push({
              id: _roadCarId++,
              fromNode: nodeId,
              toNode: destNode,
              progress: Math.random() * 0.05, // slight stagger
              speed: 0.15 + Math.random() * 0.1, // ~5-7 seconds for full journey
              color: CAR_COLORS[Math.floor(Math.random() * CAR_COLORS.length)],
              laneOffset: -10, // departing lane (right side of road)
              type: isTruck ? "truck" : "car",
              stopped: false,
            });
          }
        }
      }
    }

    prevDensitiesRef.current = newDens;
  }, [gridState?.tick]); // Only on tick changes

  // Also constantly spawn some ambient traffic for visual life
  useEffect(() => {
    if (!gridState?.intersections) return;

    const interval = setInterval(() => {
      // Spawn 1-2 ambient cars on random roads
      const roads = [
        { from: "A", to: "B" }, { from: "B", to: "A" },
        { from: "A", to: "C" }, { from: "C", to: "A" },
        { from: "B", to: "D" }, { from: "D", to: "B" },
        { from: "C", to: "D" }, { from: "D", to: "C" },
      ];
      const road = roads[Math.floor(Math.random() * roads.length)];
      const isTruck = Math.random() < 0.15;

      carsRef.current.push({
        id: _roadCarId++,
        fromNode: road.from,
        toNode: road.to,
        progress: 0,
        speed: 0.12 + Math.random() * 0.08,
        color: CAR_COLORS[Math.floor(Math.random() * CAR_COLORS.length)],
        laneOffset: -10,
        type: isTruck ? "truck" : "car",
        stopped: false,
      });
    }, 800); // Spawn ambient car every 800ms

    return () => clearInterval(interval);
  }, [!!gridState]);

  // Animation loop (runs continuously, references gridState via ref to avoid restarts)
  useEffect(() => {
    let frameId: number;
    let lastTime = performance.now();

    const animate = (now: number) => {
      const dt = (now - lastTime) / 1000; // seconds
      lastTime = now;

      let changed = false;
      const toRemove: number[] = [];
      const currentGridState = gridStateRef.current;

      for (const car of carsRef.current) {
        if (car.progress >= 1) {
          toRemove.push(car.id);
          changed = true;
          continue;
        }

        // Check if should stop at destination (red light)
        if (car.progress > 0.85 && currentGridState?.intersections) {
          const destState = currentGridState.intersections[car.toNode];
          const incomingLane = INCOMING_LANE[car.toNode]?.[car.fromNode];
          if (destState && incomingLane) {
            const isGreen = destState.lanes?.[incomingLane]?.is_green;
            if (!isGreen && car.progress > 0.88) {
              car.stopped = true;
              // Don't advance — car waits at red
              continue;
            } else {
              car.stopped = false;
            }
          }
        }

        car.progress += car.speed * dt;
        if (car.progress > 1) car.progress = 1;
        changed = true;
      }

      if (toRemove.length > 0) {
        carsRef.current = carsRef.current.filter((c) => !toRemove.includes(c.id));
      }

      // Cap total road cars
      if (carsRef.current.length > 60) {
        carsRef.current = carsRef.current.slice(-50);
      }

      if (changed) {
        setRenderTick((n) => n + 1);
      }

      frameId = requestAnimationFrame(animate);
    };

    frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, []); // Empty dependency array so animation loop never resets

  // Compute car screen positions
  const getCarPosition = useCallback((car: RoadCar) => {
    const from = GRID_CENTERS[car.fromNode];
    const to = GRID_CENTERS[car.toNode];
    if (!from || !to) return null;

    const x = from.x + (to.x - from.x) * car.progress;
    const y = from.y + (to.y - from.y) * car.progress;

    // Offset for lane separation (perpendicular to road direction)
    const isVertical = from.x === to.x;
    const ox = isVertical ? car.laneOffset : 0;
    const oy = isVertical ? 0 : car.laneOffset;

    return { x: x + ox, y: y + oy };
  }, []);

  return { cars: carsRef.current, getCarPosition };
}
