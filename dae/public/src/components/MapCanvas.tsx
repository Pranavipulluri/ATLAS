"use client";

import React, { useRef, useState, useCallback, useEffect } from "react";

interface MapCanvasProps {
  children: React.ReactNode;
  contentWidth?: number;
  contentHeight?: number;
}

const MIN_ZOOM = 0.4;
const MAX_ZOOM = 3.0;
const ZOOM_STEP = 0.15;

export default function MapCanvas({ children, contentWidth = 880, contentHeight = 860 }: MapCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(0.85);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [focusedNode, setFocusedNode] = useState<string | null>(null);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((p) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, p + (e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP))));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsPanning(true);
    setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  }, [pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning) return;
    setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
  }, [isPanning, panStart]);

  const handleMouseUp = useCallback(() => setIsPanning(false), []);

  const zoomIn = () => setZoom((p) => Math.min(MAX_ZOOM, p + ZOOM_STEP * 2));
  const zoomOut = () => setZoom((p) => Math.max(MIN_ZOOM, p - ZOOM_STEP * 2));
  const fitAll = () => { setZoom(0.85); setPan({ x: 0, y: 0 }); setFocusedNode(null); };

  const navigateToNode = useCallback((nodeId: string) => {
    const positions: Record<string, { x: number; y: number }> = {
      A: { x: 220, y: 220 }, B: { x: 660, y: 220 }, C: { x: 220, y: 640 }, D: { x: 660, y: 640 },
    };
    const pos = positions[nodeId];
    if (!pos || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const tz = 1.8;
    setZoom(tz);
    setPan({ x: rect.width / 2 - pos.x * tz, y: rect.height / 2 - pos.y * tz });
    setFocusedNode(nodeId);
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "+" || e.key === "=") zoomIn();
      if (e.key === "-") zoomOut();
      if (e.key === "0" || e.key === "Escape") fitAll();
      if ("abcd".includes(e.key.toLowerCase())) navigateToNode(e.key.toUpperCase());
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [navigateToNode]);

  return (
    <div className="relative w-full h-full overflow-hidden bg-gray-50/50">
      {/* Zoom controls */}
      <div className="absolute top-4 right-4 z-30 flex flex-col gap-1.5">
        {[
          { label: "+", fn: zoomIn, title: "Zoom In (+)" },
          { label: "−", fn: zoomOut, title: "Zoom Out (-)" },
          { label: "FIT", fn: fitAll, title: "Fit All (0)" },
        ].map((b) => (
          <button key={b.label} onClick={b.fn} title={b.title}
            className="w-8 h-8 bg-white rounded-lg shadow-sm border border-gray-200 flex items-center justify-center text-sm font-bold text-gray-500 hover:text-indigo-600 hover:border-indigo-300 transition-all">
            {b.label === "FIT" ? <span className="text-[8px]">{b.label}</span> : b.label}
          </button>
        ))}
      </div>

      {/* Node nav */}
      <div className="absolute bottom-4 right-4 z-30 flex gap-1.5">
        {["A", "B", "C", "D"].map((n) => (
          <button key={n} onClick={() => focusedNode === n ? fitAll() : navigateToNode(n)}
            className={`w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold transition-all shadow-sm ${
              focusedNode === n ? "bg-indigo-500 text-white" : "bg-white text-gray-400 border border-gray-200 hover:text-indigo-500"
            }`}>{n}</button>
        ))}
      </div>

      {/* Zoom level */}
      <div className="absolute bottom-4 left-4 z-30 px-2 py-1 bg-white rounded-lg shadow-sm border border-gray-100 text-[10px] text-gray-400 font-mono">
        {Math.round(zoom * 100)}%
      </div>

      {/* Minimap */}
      <div className="absolute top-4 left-4 z-30 w-[90px] h-[90px] bg-white rounded-xl shadow-sm border border-gray-100 p-1.5 opacity-60 hover:opacity-100 transition-opacity">
        <div className="relative w-full h-full">
          {[{ id: "A", x: 8, y: 8 }, { id: "B", x: 55, y: 8 }, { id: "C", x: 8, y: 55 }, { id: "D", x: 55, y: 55 }].map((n) => (
            <div key={n.id} className={`absolute w-2.5 h-2.5 rounded-sm ${focusedNode === n.id ? "bg-indigo-500" : "bg-gray-300"}`}
              style={{ left: n.x, top: n.y }} />
          ))}
          <svg className="absolute inset-0" width="90" height="90">
            <line x1={13} y1={13} x2={60} y2={13} stroke="#d1d5db" strokeWidth="1" />
            <line x1={13} y1={13} x2={13} y2={60} stroke="#d1d5db" strokeWidth="1" />
            <line x1={60} y1={13} x2={60} y2={60} stroke="#d1d5db" strokeWidth="1" />
            <line x1={13} y1={60} x2={60} y2={60} stroke="#d1d5db" strokeWidth="1" />
          </svg>
          <div className="minimap-viewport absolute rounded-sm"
            style={{
              left: `${Math.max(0, Math.min(70, 50 - (pan.x / contentWidth) * 100 / zoom))}%`,
              top: `${Math.max(0, Math.min(70, 50 - (pan.y / contentHeight) * 100 / zoom))}%`,
              width: `${Math.min(100, 100 / zoom)}%`, height: `${Math.min(100, 100 / zoom)}%`,
            }} />
        </div>
      </div>

      {/* Canvas */}
      <div ref={containerRef} className="w-full h-full map-canvas"
        onWheel={handleWheel} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
        <div className="map-transform origin-top-left" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}>
          {children}
        </div>
      </div>
    </div>
  );
}
