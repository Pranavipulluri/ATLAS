"use client";

import React, { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface TrafficSignalPoleProps {
  position: [number, number, number];
  rotation?: [number, number, number];
  isGreen: boolean;
  hasEmergency?: boolean;
}

/**
 * Realistic 3D Traffic Signal Pole with DYNAMIC animated lights.
 * - Lights pulse and glow with varying intensity
 * - Smooth color transitions
 * - Emergency mode makes red light strobe
 */
export default function TrafficSignalPole({
  position,
  rotation = [0, 0, 0],
  isGreen,
  hasEmergency = false,
}: TrafficSignalPoleProps) {
  const redLightRef = useRef<THREE.Mesh>(null);
  const greenLightRef = useRef<THREE.Mesh>(null);
  const redGlowRef = useRef<THREE.PointLight>(null);
  const greenGlowRef = useRef<THREE.PointLight>(null);
  const timeRef = useRef(0);

  // Animate lights every frame for dynamic pulsing
  useFrame((_, delta) => {
    timeRef.current += delta;
    const t = timeRef.current;

    // Red light
    if (redLightRef.current) {
      const mat = redLightRef.current.material as THREE.MeshStandardMaterial;
      if (!isGreen) {
        // Active red: pulsing glow
        const pulse = 0.8 + Math.sin(t * 3) * 0.2;
        mat.color.set("#ff1a1a");
        mat.emissive.set("#ff0000");
        mat.emissiveIntensity = hasEmergency ? 2.5 + Math.sin(t * 10) * 2.5 : 1.8 * pulse;
      } else {
        // Inactive red: dim
        mat.color.lerp(new THREE.Color("#3a0808"), 0.08);
        mat.emissive.set("#000000");
        mat.emissiveIntensity = 0;
      }
    }

    // Red point light
    if (redGlowRef.current) {
      if (!isGreen) {
        const pulse = hasEmergency ? 3 + Math.sin(t * 10) * 3 : 1.5 + Math.sin(t * 3) * 0.5;
        redGlowRef.current.intensity = pulse;
        redGlowRef.current.color.set(hasEmergency && Math.sin(t * 10) > 0 ? "#ff0000" : "#ff2200");
      } else {
        redGlowRef.current.intensity = Math.max(0, redGlowRef.current.intensity - delta * 3);
      }
    }

    // Green light
    if (greenLightRef.current) {
      const mat = greenLightRef.current.material as THREE.MeshStandardMaterial;
      if (isGreen) {
        const pulse = 0.8 + Math.sin(t * 2.5) * 0.2;
        mat.color.set("#00ff55");
        mat.emissive.set("#00ff44");
        mat.emissiveIntensity = 2.0 * pulse;
      } else {
        mat.color.lerp(new THREE.Color("#083a15"), 0.08);
        mat.emissive.set("#000000");
        mat.emissiveIntensity = 0;
      }
    }

    // Green point light
    if (greenGlowRef.current) {
      if (isGreen) {
        greenGlowRef.current.intensity = 1.5 + Math.sin(t * 2.5) * 0.5;
      } else {
        greenGlowRef.current.intensity = Math.max(0, greenGlowRef.current.intensity - delta * 3);
      }
    }
  });

  return (
    <group position={position} rotation={rotation}>
      {/* === Vertical Pole === */}
      <mesh position={[0, 0.05, 0]}>
        <cylinderGeometry args={[0.18, 0.22, 0.1, 16]} />
        <meshStandardMaterial color="#555555" metalness={0.8} roughness={0.3} />
      </mesh>
      <mesh position={[0, 2.5, 0]}>
        <cylinderGeometry args={[0.08, 0.1, 5, 12]} />
        <meshStandardMaterial color="#666666" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* === Horizontal Arm === */}
      <mesh position={[1.5, 4.8, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.06, 0.07, 3, 10]} />
        <meshStandardMaterial color="#666666" metalness={0.8} roughness={0.3} />
      </mesh>
      <mesh position={[0.15, 4.7, 0]}>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshStandardMaterial color="#555555" metalness={0.7} roughness={0.3} />
      </mesh>

      {/* === Signal Housing === */}
      <group position={[2.2, 4.2, 0]}>
        {/* Mounting bracket */}
        <mesh position={[0, 0.5, 0]}>
          <cylinderGeometry args={[0.03, 0.03, 0.6, 6]} />
          <meshStandardMaterial color="#555555" metalness={0.7} roughness={0.3} />
        </mesh>

        {/* Signal box */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[0.55, 1.5, 0.35]} />
          <meshStandardMaterial color="#1a1a1a" metalness={0.5} roughness={0.4} />
        </mesh>
        {/* Back plate */}
        <mesh position={[-0.29, 0, 0]}>
          <boxGeometry args={[0.02, 1.6, 0.45]} />
          <meshStandardMaterial color="#1a1a1a" metalness={0.5} roughness={0.4} />
        </mesh>

        {/* === RED LIGHT (top) === */}
        <group position={[0.15, 0.45, 0]}>
          <mesh position={[0.1, 0.08, 0]} rotation={[0, 0, -0.3]}>
            <boxGeometry args={[0.15, 0.35, 0.4]} />
            <meshStandardMaterial color="#222222" metalness={0.3} roughness={0.6} />
          </mesh>
          <mesh ref={redLightRef} position={[0.13, 0, 0]}>
            <sphereGeometry args={[0.14, 16, 16]} />
            <meshStandardMaterial
              color={isGreen ? "#3a0808" : "#ff1a1a"}
              emissive={isGreen ? "#000000" : "#ff0000"}
              emissiveIntensity={isGreen ? 0 : 2}
              transparent opacity={0.95}
            />
          </mesh>
          <pointLight ref={redGlowRef}
            position={[0.3, 0, 0]} color="#ff0000"
            intensity={isGreen ? 0 : 2} distance={5} decay={2}
          />
        </group>

        {/* === YELLOW LIGHT (middle) === */}
        <group position={[0.15, 0, 0]}>
          <mesh position={[0.1, 0.08, 0]} rotation={[0, 0, -0.3]}>
            <boxGeometry args={[0.15, 0.35, 0.4]} />
            <meshStandardMaterial color="#222222" metalness={0.3} roughness={0.6} />
          </mesh>
          <mesh position={[0.13, 0, 0]}>
            <sphereGeometry args={[0.14, 16, 16]} />
            <meshStandardMaterial color="#3a3a00" emissive="#000000" emissiveIntensity={0} transparent opacity={0.95} />
          </mesh>
        </group>

        {/* === GREEN LIGHT (bottom) === */}
        <group position={[0.15, -0.45, 0]}>
          <mesh position={[0.1, 0.08, 0]} rotation={[0, 0, -0.3]}>
            <boxGeometry args={[0.15, 0.35, 0.4]} />
            <meshStandardMaterial color="#222222" metalness={0.3} roughness={0.6} />
          </mesh>
          <mesh ref={greenLightRef} position={[0.13, 0, 0]}>
            <sphereGeometry args={[0.14, 16, 16]} />
            <meshStandardMaterial
              color={isGreen ? "#00ff55" : "#083a15"}
              emissive={isGreen ? "#00ff44" : "#000000"}
              emissiveIntensity={isGreen ? 2 : 0}
              transparent opacity={0.95}
            />
          </mesh>
          <pointLight ref={greenGlowRef}
            position={[0.3, 0, 0]} color="#00ff44"
            intensity={isGreen ? 2 : 0} distance={5} decay={2}
          />
        </group>
      </group>
    </group>
  );
}
