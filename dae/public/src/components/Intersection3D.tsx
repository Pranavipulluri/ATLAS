"use client";

import React, { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { IntersectionState } from "@/hooks/useTrafficSocket";
import TrafficSignalPole from "./TrafficSignalPole";

interface Intersection3DProps {
  state: IntersectionState;
  offset: [number, number];
  hasAmbulance: boolean;
}

const CAR_COLORS = ["#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6", "#10b981", "#ec4899", "#06b6d4", "#f97316"];

// ─── Single Car: queues on red, drives through on green, despawns ───
function QueueCar({
  basePosition, color, rotationY, isGreen, queueIndex, direction,
}: {
  basePosition: [number, number, number];
  color: string;
  rotationY: number;
  isGreen: boolean;
  queueIndex: number; // 0 = closest to intersection
  direction: [number, number, number];
}) {
  const meshRef = useRef<THREE.Group>(null);
  const progressRef = useRef(0);
  const departedRef = useRef(false);
  const speed = 0.012 + queueIndex * 0.002; // further back = slightly faster to catch up

  useFrame(() => {
    if (!meshRef.current) return;

    if (isGreen && !departedRef.current) {
      // Drive forward through intersection
      progressRef.current += speed;
      if (progressRef.current > 2.5) {
        // Car drove out of view — reset it back to queue (simulates new car arriving)
        progressRef.current = 0;
        departedRef.current = false;
      }
    } else if (!isGreen) {
      // Red: glide back to queue position smoothly
      if (progressRef.current > 0) {
        progressRef.current = Math.max(0, progressRef.current - 0.04);
      }
      departedRef.current = false;
    }

    const p = progressRef.current;
    meshRef.current.position.set(
      basePosition[0] + direction[0] * p * 3,
      basePosition[1],
      basePosition[2] + direction[2] * p * 3
    );
  });

  return (
    <group ref={meshRef} position={basePosition} rotation={[0, rotationY, 0]}>
      {/* Car body */}
      <mesh position={[0, 0.15, 0]} castShadow>
        <boxGeometry args={[0.4, 0.2, 0.7]} />
        <meshStandardMaterial color={color} metalness={0.4} roughness={0.5} />
      </mesh>
      {/* Cabin */}
      <mesh position={[0, 0.32, -0.02]}>
        <boxGeometry args={[0.34, 0.16, 0.36]} />
        <meshStandardMaterial color={color} metalness={0.2} roughness={0.3} transparent opacity={0.85} />
      </mesh>
      {/* Wheels */}
      {[[-0.22, 0.06, 0.22], [0.22, 0.06, 0.22], [-0.22, 0.06, -0.22], [0.22, 0.06, -0.22]].map((p, i) => (
        <mesh key={i} position={p as [number, number, number]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.06, 0.06, 0.05, 8]} />
          <meshStandardMaterial color="#1a1a1a" />
        </mesh>
      ))}
      {/* Headlights */}
      <mesh position={[0.14, 0.15, 0.35]}>
        <sphereGeometry args={[0.03, 6, 6]} />
        <meshStandardMaterial color="#ffffcc" emissive="#ffffcc" emissiveIntensity={0.3} />
      </mesh>
      <mesh position={[-0.14, 0.15, 0.35]}>
        <sphereGeometry args={[0.03, 6, 6]} />
        <meshStandardMaterial color="#ffffcc" emissive="#ffffcc" emissiveIntensity={0.3} />
      </mesh>
    </group>
  );
}

// ─── Ambulance with flashing sirens, positioned at intersection center ───
function Ambulance3D({ position }: { position: [number, number, number] }) {
  const sirenRef = useRef<THREE.PointLight>(null);
  const groupRef = useRef<THREE.Group>(null);
  const timeRef = useRef(0);

  useFrame((_, delta) => {
    timeRef.current += delta;
    if (sirenRef.current) {
      const flash = Math.sin(timeRef.current * 8) > 0;
      sirenRef.current.color.set(flash ? "#ff0000" : "#0000ff");
      sirenRef.current.intensity = 5 + Math.sin(timeRef.current * 12) * 3;
    }
    // Slight pulsing glow effect
    if (groupRef.current) {
      groupRef.current.position.y = position[1] + Math.sin(timeRef.current * 2) * 0.03;
    }
  });

  return (
    <group ref={groupRef} position={position}>
      {/* Body */}
      <mesh position={[0, 0.3, 0]} castShadow>
        <boxGeometry args={[0.55, 0.4, 1.0]} />
        <meshStandardMaterial color="#ffffff" metalness={0.3} roughness={0.4} />
      </mesh>
      {/* Red cross on side */}
      <mesh position={[0.28, 0.35, 0]}>
        <boxGeometry args={[0.01, 0.18, 0.07]} />
        <meshStandardMaterial color="#ff0000" emissive="#ff0000" emissiveIntensity={0.5} />
      </mesh>
      <mesh position={[0.28, 0.35, 0]}>
        <boxGeometry args={[0.01, 0.07, 0.18]} />
        <meshStandardMaterial color="#ff0000" emissive="#ff0000" emissiveIntensity={0.5} />
      </mesh>
      {/* Siren lights */}
      <mesh position={[0, 0.55, 0.18]}>
        <boxGeometry args={[0.12, 0.07, 0.1]} />
        <meshStandardMaterial color="#ff0000" emissive="#ff0000" emissiveIntensity={2} />
      </mesh>
      <mesh position={[0, 0.55, -0.18]}>
        <boxGeometry args={[0.12, 0.07, 0.1]} />
        <meshStandardMaterial color="#2222ff" emissive="#0000ff" emissiveIntensity={2} />
      </mesh>
      <pointLight ref={sirenRef} position={[0, 0.7, 0]} color="#ff0000" intensity={6} distance={12} decay={2} />
      {/* Ground glow ring */}
      <mesh position={[0, 0.05, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.8, 1.2, 16]} />
        <meshStandardMaterial color="#ff3333" emissive="#ff0000" emissiveIntensity={1} transparent opacity={0.3} />
      </mesh>
      {/* Wheels */}
      {[[-0.28, 0.06, 0.32], [0.28, 0.06, 0.32], [-0.28, 0.06, -0.32], [0.28, 0.06, -0.32]].map((p, i) => (
        <mesh key={i} position={p as [number, number, number]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.07, 0.07, 0.05, 8]} />
          <meshStandardMaterial color="#1a1a1a" />
        </mesh>
      ))}
    </group>
  );
}

// ─── How many cars to show per lane (scales with density) ───
function visibleCars(density: number): number {
  if (density <= 0) return 0;
  if (density <= 3) return 1;
  if (density <= 6) return 2;
  if (density <= 10) return 3;
  if (density <= 18) return 4;
  if (density <= 30) return 5;
  return 6; // max 6 visible cars
}

const CAR_SPACING = 1.3;

export default function Intersection3D({ state, offset, hasAmbulance }: Intersection3DProps) {
  const [ox, oz] = offset;

  const roadMat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#555555", roughness: 0.9, metalness: 0.1 }), []);
  const sidewalkMat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#cccccc", roughness: 0.8 }), []);
  const grassMat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#6abf69", roughness: 1.0 }), []);
  const markingMat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.5 }), []);
  const centerMat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#f5c542", roughness: 0.5 }), []);

  const isGreenWave = state.green_wave_active;

  const northGreen = state.lanes.North?.is_green || false;
  const southGreen = state.lanes.South?.is_green || false;
  const eastGreen = state.lanes.East?.is_green || false;
  const westGreen = state.lanes.West?.is_green || false;

  const nCars = visibleCars(state.lanes.North?.density || 0);
  const sCars = visibleCars(state.lanes.South?.density || 0);
  const eCars = visibleCars(state.lanes.East?.density || 0);
  const wCars = visibleCars(state.lanes.West?.density || 0);

  return (
    <group position={[ox, 0, oz]}>
      {/* Grass corners */}
      {[[-4.5, 0, -4.5], [4.5, 0, -4.5], [-4.5, 0, 4.5], [4.5, 0, 4.5]].map((p, i) => (
        <mesh key={`g-${i}`} position={p as [number, number, number]} receiveShadow>
          <boxGeometry args={[5, 0.15, 5]} />
          <primitive object={grassMat} attach="material" />
        </mesh>
      ))}

      {/* Sidewalk edges */}
      {[
        [-2, 0.1, -4.5], [2, 0.1, -4.5], [-2, 0.1, 4.5], [2, 0.1, 4.5],
        [-4.5, 0.1, -2], [-4.5, 0.1, 2], [4.5, 0.1, -2], [4.5, 0.1, 2],
      ].map((p, i) => (
        <mesh key={`sw-${i}`} position={p as [number, number, number]}>
          <boxGeometry args={[i < 4 ? 0.3 : 5, 0.22, i < 4 ? 5 : 0.3]} />
          <primitive object={sidewalkMat} attach="material" />
        </mesh>
      ))}

      {/* N-S road */}
      <mesh position={[0, 0.02, 0]} receiveShadow>
        <boxGeometry args={[3.8, 0.08, 14]} />
        <primitive object={roadMat} attach="material" />
      </mesh>
      {/* E-W road */}
      <mesh position={[0, 0.02, 0]} receiveShadow>
        <boxGeometry args={[14, 0.08, 3.8]} />
        <primitive object={roadMat} attach="material" />
      </mesh>

      {/* Center line dashes */}
      {[-5.5, -4.2, -3.0, 3.0, 4.2, 5.5].map((z, i) => (
        <mesh key={`cn-${i}`} position={[0, 0.08, z]}>
          <boxGeometry args={[0.08, 0.02, 0.7]} />
          <primitive object={centerMat} attach="material" />
        </mesh>
      ))}
      {[-5.5, -4.2, -3.0, 3.0, 4.2, 5.5].map((x, i) => (
        <mesh key={`ce-${i}`} position={[x, 0.08, 0]}>
          <boxGeometry args={[0.7, 0.02, 0.08]} />
          <primitive object={centerMat} attach="material" />
        </mesh>
      ))}

      {/* Crosswalks */}
      {[-1.2, -0.6, 0, 0.6, 1.2].map((v, i) => (
        <React.Fragment key={`xw-${i}`}>
          <mesh position={[v, 0.08, -2.2]}><boxGeometry args={[0.3, 0.02, 0.6]} /><primitive object={markingMat} attach="material" /></mesh>
          <mesh position={[v, 0.08, 2.2]}><boxGeometry args={[0.3, 0.02, 0.6]} /><primitive object={markingMat} attach="material" /></mesh>
          <mesh position={[2.2, 0.08, v]}><boxGeometry args={[0.6, 0.02, 0.3]} /><primitive object={markingMat} attach="material" /></mesh>
          <mesh position={[-2.2, 0.08, v]}><boxGeometry args={[0.6, 0.02, 0.3]} /><primitive object={markingMat} attach="material" /></mesh>
        </React.Fragment>
      ))}

      {/* Green Wave glow */}
      {isGreenWave && (
        <mesh position={[0, 0.09, 0]}>
          <boxGeometry args={[4, 0.01, 14]} />
          <meshStandardMaterial color="#00ff88" emissive="#00ff88" emissiveIntensity={0.3} transparent opacity={0.15} />
        </mesh>
      )}

      {/* Traffic Signal Poles */}
      <TrafficSignalPole position={[-2.3, 0, -2.3]} rotation={[0, Math.PI / 2, 0]} isGreen={northGreen} hasEmergency={state.lanes.North?.has_emergency} />
      <TrafficSignalPole position={[2.3, 0, -2.3]} rotation={[0, Math.PI, 0]} isGreen={eastGreen} hasEmergency={state.lanes.East?.has_emergency} />
      <TrafficSignalPole position={[2.3, 0, 2.3]} rotation={[0, -Math.PI / 2, 0]} isGreen={southGreen} hasEmergency={state.lanes.South?.has_emergency} />
      <TrafficSignalPole position={[-2.3, 0, 2.3]} rotation={[0, 0, 0]} isGreen={westGreen} hasEmergency={state.lanes.West?.has_emergency} />

      {/* ─── Cars per lane (scaled to density) ─── */}
      {/* North (approaching from -Z) */}
      {Array.from({ length: nCars }).map((_, i) => (
        <QueueCar key={`cn-${i}`}
          basePosition={[0.7, 0, -2.8 - i * CAR_SPACING]}
          color={CAR_COLORS[i % CAR_COLORS.length]}
          rotationY={0} isGreen={northGreen} queueIndex={i}
          direction={[0, 0, 1]}
        />
      ))}
      {/* South (approaching from +Z) */}
      {Array.from({ length: sCars }).map((_, i) => (
        <QueueCar key={`cs-${i}`}
          basePosition={[-0.7, 0, 2.8 + i * CAR_SPACING]}
          color={CAR_COLORS[(i + 2) % CAR_COLORS.length]}
          rotationY={Math.PI} isGreen={southGreen} queueIndex={i}
          direction={[0, 0, -1]}
        />
      ))}
      {/* East (approaching from +X) */}
      {Array.from({ length: eCars }).map((_, i) => (
        <QueueCar key={`ce-${i}`}
          basePosition={[2.8 + i * CAR_SPACING, 0, -0.7]}
          color={CAR_COLORS[(i + 4) % CAR_COLORS.length]}
          rotationY={Math.PI / 2} isGreen={eastGreen} queueIndex={i}
          direction={[-1, 0, 0]}
        />
      ))}
      {/* West (approaching from -X) */}
      {Array.from({ length: wCars }).map((_, i) => (
        <QueueCar key={`cw-${i}`}
          basePosition={[-2.8 - i * CAR_SPACING, 0, 0.7]}
          color={CAR_COLORS[(i + 6) % CAR_COLORS.length]}
          rotationY={-Math.PI / 2} isGreen={westGreen} queueIndex={i}
          direction={[1, 0, 0]}
        />
      ))}

      {/* Density indicators — subtle text showing count when density is high */}
      {nCars > 0 && (state.lanes.North?.density || 0) > 6 && (
        <mesh position={[1.6, 0.3, -4.5]}>
          <boxGeometry args={[0.8, 0.3, 0.3]} />
          <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.3} transparent opacity={0.7} />
        </mesh>
      )}
      {sCars > 0 && (state.lanes.South?.density || 0) > 6 && (
        <mesh position={[-1.6, 0.3, 4.5]}>
          <boxGeometry args={[0.8, 0.3, 0.3]} />
          <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.3} transparent opacity={0.7} />
        </mesh>
      )}

      {/* ─── Ambulance at this intersection ─── */}
      {hasAmbulance && <Ambulance3D position={[0, 0, 0]} />}

      {/* Small corner buildings */}
      {[
        { p: [-5.2, 0, -5.2] as [number, number, number], h: 1.8, c: "#e8e0f0" },
        { p: [5.2, 0, -5.2] as [number, number, number], h: 1.3, c: "#dce8f0" },
        { p: [-5.2, 0, 5.2] as [number, number, number], h: 1.5, c: "#d8f0dc" },
        { p: [5.2, 0, 5.2] as [number, number, number], h: 1.0, c: "#f0e8d8" },
      ].map((b, i) => (
        <group key={`bldg-${i}`} position={b.p}>
          <mesh position={[0, b.h / 2, 0]} castShadow>
            <boxGeometry args={[2.5, b.h, 2.5]} />
            <meshStandardMaterial color={b.c} roughness={0.7} metalness={0.1} />
          </mesh>
          <mesh position={[0, b.h + 0.02, 0]}>
            <boxGeometry args={[2.6, 0.04, 2.6]} />
            <meshStandardMaterial color="#999" roughness={0.5} />
          </mesh>
        </group>
      ))}
    </group>
  );
}
