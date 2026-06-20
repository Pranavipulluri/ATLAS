"use client";

import React, { Suspense, useRef, useEffect, useState, useCallback } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Html, Environment } from "@react-three/drei";
import * as THREE from "three";
import { GridState, EmergencyState } from "@/hooks/useTrafficSocket";
import Intersection3D from "./Intersection3D";

interface Scene3DProps {
  gridState: GridState;
  selectedNode: string | null;
  onSelectNode: (nodeId: string) => void;
}

const NODE_POSITIONS: Record<string, [number, number]> = {
  A: [-16, -16],
  B: [16, -16],
  C: [-16, 16],
  D: [16, 16],
};

const NODE_ORDER = ["A", "B", "C", "D"];

const ROAD_CONNECTIONS: Array<{ from: string; to: string }> = [
  { from: "A", to: "B" },
  { from: "A", to: "C" },
  { from: "B", to: "D" },
  { from: "C", to: "D" },
];

// ─── Connecting Road ─── 
function ConnectingRoad({
  from, to, isEmergencyRoute, isGreenWave,
}: {
  from: [number, number]; to: [number, number]; isEmergencyRoute: boolean; isGreenWave: boolean;
}) {
  const cx = (from[0] + to[0]) / 2;
  const cz = (from[1] + to[1]) / 2;
  const dx = to[0] - from[0];
  const dz = to[1] - from[1];
  const length = Math.sqrt(dx * dx + dz * dz);
  const angle = Math.atan2(dx, dz);

  return (
    <group position={[cx, 0, cz]} rotation={[0, angle, 0]}>
      <mesh position={[0, 0.02, 0]} receiveShadow>
        <boxGeometry args={[3.8, 0.08, length - 10]} />
        <meshStandardMaterial color="#555555" roughness={0.9} metalness={0.1} />
      </mesh>
      <mesh position={[-2, 0.1, 0]}>
        <boxGeometry args={[0.25, 0.2, length - 10]} />
        <meshStandardMaterial color="#cccccc" roughness={0.8} />
      </mesh>
      <mesh position={[2, 0.1, 0]}>
        <boxGeometry args={[0.25, 0.2, length - 10]} />
        <meshStandardMaterial color="#cccccc" roughness={0.8} />
      </mesh>
      {Array.from({ length: Math.floor((length - 12) / 1.8) }).map((_, i) => (
        <mesh key={i} position={[0, 0.08, -((length - 12) / 2) + i * 1.8 + 0.4]}>
          <boxGeometry args={[0.08, 0.02, 0.8]} />
          <meshStandardMaterial color="#f5c542" roughness={0.5} />
        </mesh>
      ))}
      {isEmergencyRoute && isGreenWave && (
        <mesh position={[0, 0.09, 0]}>
          <boxGeometry args={[4, 0.01, length - 10]} />
          <meshStandardMaterial color="#00ff88" emissive="#00ff88" emissiveIntensity={0.4} transparent opacity={0.2} />
        </mesh>
      )}
    </group>
  );
}

function isOnEmergencyRoute(emergencies: EmergencyState[], from: string, to: string): boolean {
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

// ─── Camera Animation ───
interface CameraTarget {
  position: THREE.Vector3;
  lookAt: THREE.Vector3;
}

function CameraController({ target }: { target: CameraTarget | null }) {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);
  const animRef = useRef<number | null>(null);

  const flyTo = useCallback(
    (dest: CameraTarget) => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      const startPos = camera.position.clone();
      const startTarget = controlsRef.current
        ? controlsRef.current.target.clone()
        : new THREE.Vector3(0, 0, 0);
      const startTime = Date.now();
      const duration = 1000;

      const animate = () => {
        const t = Math.min(1, (Date.now() - startTime) / duration);
        const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        camera.position.lerpVectors(startPos, dest.position, ease);
        if (controlsRef.current) {
          controlsRef.current.target.lerpVectors(startTarget, dest.lookAt, ease);
          controlsRef.current.update();
        }
        if (t < 1) { animRef.current = requestAnimationFrame(animate); }
        else { animRef.current = null; }
      };
      animRef.current = requestAnimationFrame(animate);
    },
    [camera]
  );

  useEffect(() => { if (target) flyTo(target); }, [target, flyTo]);

  return (
    <OrbitControls ref={controlsRef}
      enablePan enableZoom enableRotate
      minDistance={5} maxDistance={150}
      maxPolarAngle={Math.PI / 2.05} minPolarAngle={0.05}
      target={[0, 0, 0]} enableDamping dampingFactor={0.08}
    />
  );
}

// ─── Node Label ───
function NodeLabel({
  nodeId, position, state, hasAmbulance, isGreenWave, onClick, isSelected,
}: {
  nodeId: string; position: [number, number, number]; state: { current_green: string; time_in_phase: number };
  hasAmbulance: boolean; isGreenWave: boolean; onClick: () => void; isSelected: boolean;
}) {
  return (
    <Html position={position} center style={{ pointerEvents: "auto" }}>
      <div onClick={onClick} className="cursor-pointer select-none" style={{ transform: "translateY(-20px)" }}>
        <div className={`
          px-3 py-1.5 rounded-xl text-xs font-bold tracking-wide shadow-lg whitespace-nowrap transition-all
          ${hasAmbulance ? "bg-red-500 text-white"
            : isGreenWave ? "bg-emerald-500 text-white"
            : isSelected ? "bg-indigo-500 text-white"
            : "bg-white text-gray-700 border border-gray-200"}
        `}>
          Node {nodeId}
          <span className="ml-2 text-[10px] opacity-70">
            {state.current_green[0]}↑ {state.time_in_phase}s
          </span>
        </div>
      </div>
    </Html>
  );
}

// ─── 2D Signal Light Component ───
function SignalLight({ isGreen, label }: { isGreen: boolean; label: string }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <span className="text-[9px] font-bold text-gray-400 tracking-wider">{label}</span>
      {/* Signal housing */}
      <div className="bg-gray-900 rounded-lg p-1 flex flex-col gap-1 shadow-lg" style={{ width: "28px" }}>
        {/* Red */}
        <div className={`w-5 h-5 rounded-full mx-auto transition-all duration-500 ${
          !isGreen
            ? "bg-red-500 shadow-[0_0_12px_rgba(239,68,68,0.8),0_0_24px_rgba(239,68,68,0.4)]"
            : "bg-red-950"
        }`} />
        {/* Yellow */}
        <div className="w-5 h-5 rounded-full mx-auto bg-yellow-950" />
        {/* Green */}
        <div className={`w-5 h-5 rounded-full mx-auto transition-all duration-500 ${
          isGreen
            ? "bg-green-500 shadow-[0_0_12px_rgba(34,197,94,0.8),0_0_24px_rgba(34,197,94,0.4)]"
            : "bg-green-950"
        }`} />
      </div>
      {/* Status text */}
      <span className={`text-[8px] font-bold ${isGreen ? "text-green-600" : "text-red-500"}`}>
        {isGreen ? "GO" : "STOP"}
      </span>
    </div>
  );
}

// ─── Signal Light WITH Eye Icon for AI Reasoning ───
function SignalWithReasoning({
  isGreen, label, lane, aiReasoning,
}: {
  isGreen: boolean; label: string;
  lane?: import("@/hooks/useTrafficSocket").LaneState;
  aiReasoning?: string;
}) {
  const [showReasoning, setShowReasoning] = useState(false);

  return (
    <div className="flex flex-col items-center gap-1 relative">
      <div className="flex items-center gap-1">
        <span className="text-[9px] font-bold text-gray-400 tracking-wider">{label}</span>
        {lane && (
          <button
            onClick={() => setShowReasoning(!showReasoning)}
            className={`w-4 h-4 rounded-full flex items-center justify-center text-[8px] transition-all ${
              showReasoning ? "bg-indigo-500 text-white" : "bg-gray-200 text-gray-400 hover:bg-indigo-100"
            }`}
            title="View AI agent reasoning"
          >
            👁
          </button>
        )}
      </div>

      <div className="bg-gray-900 rounded-lg p-1 flex flex-col gap-1 shadow-lg" style={{ width: "28px" }}>
        <div className={`w-5 h-5 rounded-full mx-auto transition-all duration-500 ${
          !isGreen ? "bg-red-500 shadow-[0_0_12px_rgba(239,68,68,0.8),0_0_24px_rgba(239,68,68,0.4)]" : "bg-red-950"
        }`} />
        <div className="w-5 h-5 rounded-full mx-auto bg-yellow-950" />
        <div className={`w-5 h-5 rounded-full mx-auto transition-all duration-500 ${
          isGreen ? "bg-green-500 shadow-[0_0_12px_rgba(34,197,94,0.8),0_0_24px_rgba(34,197,94,0.4)]" : "bg-green-950"
        }`} />
      </div>

      <span className={`text-[8px] font-bold ${isGreen ? "text-green-600" : "text-red-500"}`}>
        {isGreen ? "GO" : "STOP"}
      </span>

      {/* AI Reasoning Tooltip */}
      {showReasoning && lane && (
        <div className="absolute top-full mt-1 z-50 w-64 bg-gray-900 text-white rounded-lg shadow-xl p-2.5 text-[9px] leading-tight"
          style={{ left: "50%", transform: "translateX(-50%)" }}>
          <div className="flex items-center gap-1.5 mb-1.5">
            <span className={`px-1.5 py-0.5 rounded text-[8px] font-bold ${
              lane.detection_source === "yolo" ? "bg-purple-500" : lane.detection_source === "mock" ? "bg-amber-500" : "bg-gray-500"
            }`}>
              {lane.detection_source === "yolo" ? "🎥 YOLO" : lane.detection_source === "mock" ? "📊 MOCK" : "📈 SIM"}
            </span>
            <span className="text-gray-300 font-semibold">{label} Lane Agent</span>
          </div>

          {/* AI reasoning (from kimi-k2.5:cloud) */}
          {aiReasoning && (
            <div className="mb-1.5 p-1.5 rounded bg-indigo-900/50 border border-indigo-700/50">
              <div className="flex items-center gap-1 mb-0.5">
                <span className="text-[7px] bg-indigo-600 px-1 py-0.5 rounded font-bold">🤖 AI Agent</span>
                <span className="text-[7px] text-indigo-300">kimi-k2.5</span>
              </div>
              <div className="text-indigo-100 text-[9px]">{aiReasoning}</div>
            </div>
          )}

          {/* Template reasoning */}
          <div className="text-gray-300 mb-1 text-[8px] opacity-80">{lane.reasoning || "—"}</div>

          {lane.vehicle_types && Object.keys(lane.vehicle_types).length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1">
              {Object.entries(lane.vehicle_types).map(([type, count]) => (
                <span key={type} className="bg-gray-700 px-1.5 py-0.5 rounded text-[8px]">
                  {count}× {type}
                </span>
              ))}
            </div>
          )}
          <div className="flex items-center gap-2 mt-1.5 pt-1.5 border-t border-gray-700">
            <span>📊 {lane.density} vehicles</span>
            <span>⏰ {lane.wait_time}s wait</span>
          </div>
          <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-gray-900 rotate-45" />
        </div>
      )}
    </div>
  );
}

// ─── Main Scene ───
export default function Scene3D({ gridState, selectedNode, onSelectNode }: Scene3DProps) {
  const { intersections, emergencies, green_wave_active } = gridState;
  const [focusedNode, setFocusedNode] = useState<string | null>(null);
  const [cameraTarget, setCameraTarget] = useState<CameraTarget | null>(null);

  const goOverview = useCallback(() => {
    setFocusedNode(null);
    onSelectNode("");
    setCameraTarget({
      position: new THREE.Vector3(0, 70, 8),
      lookAt: new THREE.Vector3(0, 0, 0),
    });
  }, [onSelectNode]);

  const focusIntersection = useCallback(
    (nodeId: string) => {
      const pos = NODE_POSITIONS[nodeId];
      if (!pos) return;
      setFocusedNode(nodeId);
      onSelectNode(nodeId);
      setCameraTarget({
        position: new THREE.Vector3(pos[0] + 14, 22, pos[1] + 14),
        lookAt: new THREE.Vector3(pos[0], 0, pos[1]),
      });
    },
    [onSelectNode]
  );

  useEffect(() => {
    if (selectedNode && selectedNode !== "" && selectedNode !== focusedNode) {
      focusIntersection(selectedNode);
    }
  }, [selectedNode]); // eslint-disable-line react-hooks/exhaustive-deps

  // Keyboard
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const gridMap: Record<string, Record<string, string>> = {
        A: { ArrowRight: "B", ArrowDown: "C" },
        B: { ArrowLeft: "A", ArrowDown: "D" },
        C: { ArrowRight: "D", ArrowUp: "A" },
        D: { ArrowLeft: "C", ArrowUp: "B" },
      };
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
        e.preventDefault();
        if (!focusedNode) { focusIntersection("A"); return; }
        const next = gridMap[focusedNode]?.[e.key];
        if (next) focusIntersection(next);
      }
      if (e.key === "1") focusIntersection("A");
      if (e.key === "2") focusIntersection("B");
      if (e.key === "3") focusIntersection("C");
      if (e.key === "4") focusIntersection("D");
      if (e.key === "Escape") goOverview();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [focusedNode, focusIntersection, goOverview]);

  // Get focused intersection data
  const focusedState = focusedNode ? intersections[focusedNode] : null;

  return (
    <div className="w-full h-full relative">
      <Canvas
        shadows
        camera={{ position: [0, 70, 8], fov: 45, near: 0.1, far: 300 }}
        style={{ background: "#f0f4f8" }}
      >
        <Suspense fallback={null}>
          <ambientLight intensity={0.6} />
          <directionalLight
            position={[30, 40, 20]} intensity={1.2} castShadow
            shadow-mapSize={[2048, 2048]} shadow-camera-far={100}
            shadow-camera-left={-40} shadow-camera-right={40}
            shadow-camera-top={40} shadow-camera-bottom={-40}
          />
          <directionalLight position={[-20, 15, -10]} intensity={0.3} />
          <hemisphereLight args={["#bfe8ff", "#8abe6a", 0.4]} />
          <Environment preset="city" />
          <fog attach="fog" args={["#f0f4f8", 100, 250]} />

          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, 0]} receiveShadow>
            <planeGeometry args={[200, 200]} />
            <meshStandardMaterial color="#a8d5a2" roughness={1} />
          </mesh>
          <gridHelper args={[200, 80, "#88bb88", "#88bb88"]} position={[0, -0.05, 0]} />

          {ROAD_CONNECTIONS.map(({ from, to }) => (
            <ConnectingRoad key={`${from}-${to}`}
              from={NODE_POSITIONS[from]} to={NODE_POSITIONS[to]}
              isEmergencyRoute={isOnEmergencyRoute(emergencies, from, to)}
              isGreenWave={green_wave_active}
            />
          ))}

          {Object.entries(NODE_POSITIONS).map(([nodeId, [x, z]]) => {
            const nodeState = intersections[nodeId];
            if (!nodeState) return null;
            const hasAmb = emergencies.some((e) => !e.completed && e.current_intersection === nodeId);
            return (
              <React.Fragment key={nodeId}>
                <Intersection3D state={nodeState} offset={[x, z]} hasAmbulance={hasAmb} />
                <NodeLabel nodeId={nodeId} position={[x, 6, z]}
                  state={nodeState} hasAmbulance={hasAmb}
                  isGreenWave={nodeState.green_wave_active}
                  onClick={() => focusIntersection(nodeId)}
                  isSelected={focusedNode === nodeId}
                />
              </React.Fragment>
            );
          })}

          {/* ── Moving Ambulances on roads ── */}
          {emergencies.filter(e => !e.completed).map((em) => {
            // Compute ambulance 3D position by interpolating between route nodes
            const step = em.current_step;
            const progress = em.ticks_per_step > 0 ? em.ticks_at_step / em.ticks_per_step : 0;

            if (step >= em.route.length) return null;
            const currentNode = em.route[step]?.intersection;
            const nextNode = step + 1 < em.route.length ? em.route[step + 1]?.intersection : null;

            const fromPos = NODE_POSITIONS[currentNode];
            if (!fromPos) return null;

            let ax = fromPos[0];
            let az = fromPos[1];

            if (nextNode) {
              const toPos = NODE_POSITIONS[nextNode];
              if (toPos) {
                ax = fromPos[0] + (toPos[0] - fromPos[0]) * progress;
                az = fromPos[1] + (toPos[1] - fromPos[1]) * progress;
              }
            }

            return (
              <group key={em.vehicle_id}>
                {/* Ambulance body on road */}
                <group position={[ax, 0, az]}>
                  <mesh position={[0, 0.35, 0]} castShadow>
                    <boxGeometry args={[0.6, 0.45, 1.1]} />
                    <meshStandardMaterial color="#ffffff" metalness={0.3} roughness={0.4} />
                  </mesh>
                  {/* Red cross */}
                  <mesh position={[0.31, 0.4, 0]}>
                    <boxGeometry args={[0.01, 0.2, 0.08]} />
                    <meshStandardMaterial color="#ff0000" emissive="#ff0000" emissiveIntensity={0.8} />
                  </mesh>
                  <mesh position={[0.31, 0.4, 0]}>
                    <boxGeometry args={[0.01, 0.08, 0.2]} />
                    <meshStandardMaterial color="#ff0000" emissive="#ff0000" emissiveIntensity={0.8} />
                  </mesh>
                  {/* Siren beacon */}
                  <mesh position={[0, 0.65, 0]}>
                    <boxGeometry args={[0.2, 0.1, 0.15]} />
                    <meshStandardMaterial color="#ff3333" emissive="#ff0000" emissiveIntensity={3} />
                  </mesh>
                  {/* Ground glow */}
                  <mesh position={[0, 0.06, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                    <ringGeometry args={[0.6, 1.5, 16]} />
                    <meshStandardMaterial color="#ff3333" emissive="#ff0000" emissiveIntensity={0.5} transparent opacity={0.25} />
                  </mesh>
                  <pointLight position={[0, 0.8, 0]} color="#ff0000" intensity={8} distance={15} decay={2} />
                </group>

                {/* Route trail line */}
                <line>
                  <bufferGeometry>
                    <bufferAttribute
                      attach="attributes-position"
                      args={[
                        new Float32Array(
                          em.route.flatMap(r => {
                            const p = NODE_POSITIONS[r.intersection];
                            return p ? [p[0], 0.15, p[1]] : [0, 0.15, 0];
                          })
                        ),
                        3
                      ]}
                      count={em.route.length}
                    />
                  </bufferGeometry>
                  <lineBasicMaterial color="#ff4444" linewidth={2} transparent opacity={0.5} />
                </line>
              </group>
            );
          })}

          {green_wave_active && (
            <Html position={[0, 12, 0]} center>
              <div className="green-wave-path px-5 py-2 rounded-full text-sm font-bold text-white flex items-center gap-2 shadow-xl">
                🌊 GREEN WAVE ACTIVE 🚑
              </div>
            </Html>
          )}

          <CameraController target={cameraTarget} />
        </Suspense>
      </Canvas>

      {/* ══════════════════════════════════════════════════
          SIGNAL STATUS HUD + AGENT REASONING
          ══════════════════════════════════════════════════ */}
      {focusedNode && focusedState && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10" style={{ maxWidth: "520px" }}>
          <div className="bg-white/95 rounded-2xl shadow-lg border border-gray-200 px-4 py-3 backdrop-blur-sm">
            {/* Header */}
            <div className="text-center mb-2 flex items-center justify-center gap-2">
              <span className="text-sm font-bold text-gray-700">
                🚦 Node {focusedNode} — Signal Status
              </span>
              <span className="text-[10px] text-gray-400">
                Phase: {focusedState.current_green} | T: {focusedState.time_in_phase}s
              </span>
              {focusedState.decision_breakdown && (
                <span className={`text-[9px] px-2 py-0.5 rounded-full font-bold ${
                  focusedState.decision_breakdown.type === "emergency" ? "bg-red-100 text-red-600" :
                  focusedState.decision_breakdown.type === "green_wave" ? "bg-emerald-100 text-emerald-600" :
                  focusedState.decision_breakdown.type === "auction" ? "bg-blue-100 text-blue-600" :
                  focusedState.decision_breakdown.type === "starvation" ? "bg-orange-100 text-orange-600" :
                  "bg-gray-100 text-gray-500"
                }`}>
                  {focusedState.decision_breakdown.icon} {focusedState.decision_breakdown.type.toUpperCase()}
                </span>
              )}
            </div>

            {/* 4 Signal Lights with Eye Icons */}
            <div className="flex flex-col items-center gap-0">
              <SignalWithReasoning
                isGreen={focusedState.lanes.North?.is_green || false}
                label="NORTH"
                lane={focusedState.lanes.North}
                aiReasoning={focusedState.ai_lane_reasonings?.North}
              />
              <div className="flex items-center gap-4 -my-1">
                <SignalWithReasoning
                  isGreen={focusedState.lanes.West?.is_green || false}
                  label="WEST"
                  lane={focusedState.lanes.West}
                  aiReasoning={focusedState.ai_lane_reasonings?.West}
                />
                <div className="w-10 h-10 rounded-lg bg-gray-100 border border-gray-200 flex items-center justify-center">
                  <span className="text-lg">✚</span>
                </div>
                <SignalWithReasoning
                  isGreen={focusedState.lanes.East?.is_green || false}
                  label="EAST"
                  lane={focusedState.lanes.East}
                  aiReasoning={focusedState.ai_lane_reasonings?.East}
                />
              </div>
              <SignalWithReasoning
                isGreen={focusedState.lanes.South?.is_green || false}
                label="SOUTH"
                lane={focusedState.lanes.South}
                aiReasoning={focusedState.ai_lane_reasonings?.South}
              />
            </div>

            {/* Lane density summary */}
            <div className="flex gap-2 mt-2 justify-center">
              {(["North", "South", "East", "West"] as const).map((dir) => {
                const lane = focusedState.lanes[dir];
                if (!lane) return null;
                return (
                  <div key={dir} className={`px-2 py-1 rounded-lg text-[9px] font-mono ${
                    lane.is_green ? "bg-green-50 text-green-700" : "bg-red-50 text-red-600"
                  }`}>
                    {dir[0]}: {lane.density} cars
                    <span className="ml-1 opacity-60">
                      ({lane.detection_source === "yolo" ? "🎥" : lane.detection_source === "mock" ? "📊" : "📈"})
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Master Decision Panel */}
            {focusedState.decision_breakdown && focusedState.decision_breakdown.all_scores && (
              <div className="mt-2 pt-2 border-t border-gray-100">
                <div className="text-[9px] text-gray-400 font-bold uppercase mb-1">
                  🧠 Master Agent Decision
                </div>
                {focusedState.ai_master_reasoning && (
                  <div className="p-1.5 mb-1.5 rounded bg-indigo-50 border border-indigo-200">
                    <div className="flex items-center gap-1 mb-0.5">
                      <span className="text-[7px] bg-indigo-500 text-white px-1 py-0.5 rounded font-bold">🤖 AI</span>
                      <span className="text-[7px] text-indigo-400">kimi-k2.5</span>
                    </div>
                    <div className="text-[9px] text-indigo-800">{focusedState.ai_master_reasoning}</div>
                  </div>
                )}
                <div className="text-[10px] text-gray-600 mb-1.5">
                  {focusedState.decision_breakdown.reason}
                </div>
                <div className="grid grid-cols-4 gap-1">
                  {focusedState.decision_breakdown.all_scores.map((s) => (
                    <div key={s.lane} className={`px-1.5 py-1 rounded text-[8px] font-mono ${
                      s.lane === focusedState.decision_breakdown.winner
                        ? "bg-indigo-50 border border-indigo-200 text-indigo-700"
                        : "bg-gray-50 text-gray-500"
                    }`}>
                      <div className="font-bold text-[9px]">{s.lane[0]}: {s.score}</div>
                      <div className="opacity-70">D:{s.density} W:{s.wait_time}s</div>
                      {s.has_emergency && <div className="text-red-500">🚨 EMG</div>}
                      {s.green_wave_boost > 0 && <div className="text-emerald-500">🌊 +{s.green_wave_boost}</div>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* View Angle Selector — Top Right */}
      <div className="absolute top-4 right-4 z-10 bg-white/95 rounded-2xl shadow-lg border border-gray-200 overflow-hidden" style={{ minWidth: "140px" }}>
        <div className="px-3 py-2 bg-gray-50 border-b border-gray-100 flex items-center gap-2">
          <span className="text-sm">👁️</span>
          <span className="text-[11px] font-bold text-gray-600 tracking-wide">VIEW</span>
        </div>
        <div className="p-2">
          <div className="text-[9px] text-gray-400 font-semibold uppercase mb-1.5 px-1">Intersection</div>
          <div className="grid grid-cols-5 gap-1">
            <button onClick={goOverview}
              className={`py-1.5 rounded-lg text-[10px] font-bold transition-all ${
                !focusedNode ? "bg-indigo-500 text-white shadow-sm" : "bg-gray-100 text-gray-500 hover:bg-gray-200"
              }`}>ALL</button>
            {NODE_ORDER.map((n) => (
              <button key={n} onClick={() => focusIntersection(n)}
                className={`py-1.5 rounded-lg text-[11px] font-bold transition-all ${
                  focusedNode === n ? "bg-indigo-500 text-white shadow-sm" : "bg-gray-100 text-gray-500 hover:bg-gray-200"
                }`}>{n}</button>
            ))}
          </div>
        </div>
      </div>

      {/* Navigation D-pad — Bottom Left */}
      <div className="absolute bottom-4 left-4 z-10 flex flex-col gap-1 bg-white/90 rounded-xl p-2 shadow-sm border border-gray-100">
        <div className="text-[9px] text-gray-400 font-semibold text-center mb-1">NAVIGATE</div>
        <div className="flex justify-center">
          <button onClick={() => {
            const m: Record<string, string> = { C: "A", D: "B" };
            focusIntersection(focusedNode ? (m[focusedNode] || "A") : "A");
          }} className="w-7 h-7 rounded bg-gray-100 text-gray-500 text-sm hover:bg-indigo-100 hover:text-indigo-600 flex items-center justify-center">↑</button>
        </div>
        <div className="flex gap-1 justify-center">
          <button onClick={() => {
            const m: Record<string, string> = { B: "A", D: "C" };
            focusIntersection(focusedNode ? (m[focusedNode] || "A") : "A");
          }} className="w-7 h-7 rounded bg-gray-100 text-gray-500 text-sm hover:bg-indigo-100 hover:text-indigo-600 flex items-center justify-center">←</button>
          <button onClick={goOverview}
            className="w-7 h-7 rounded bg-gray-100 text-[8px] text-gray-400 hover:bg-indigo-100 hover:text-indigo-600 flex items-center justify-center font-bold">ALL</button>
          <button onClick={() => {
            const m: Record<string, string> = { A: "B", C: "D" };
            focusIntersection(focusedNode ? (m[focusedNode] || "B") : "B");
          }} className="w-7 h-7 rounded bg-gray-100 text-gray-500 text-sm hover:bg-indigo-100 hover:text-indigo-600 flex items-center justify-center">→</button>
        </div>
        <div className="flex justify-center">
          <button onClick={() => {
            const m: Record<string, string> = { A: "C", B: "D" };
            focusIntersection(focusedNode ? (m[focusedNode] || "C") : "C");
          }} className="w-7 h-7 rounded bg-gray-100 text-gray-500 text-sm hover:bg-indigo-100 hover:text-indigo-600 flex items-center justify-center">↓</button>
        </div>
      </div>
    </div>
  );
}
