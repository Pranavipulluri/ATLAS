"""
AI Agent Reasoners using LangChain + Ollama (kimi-k2.5:cloud)

Provides LLM-powered reasoning for:
- Lane Agents: Analyze detection data and explain observations
- Master Agent: Explain auction decisions and signal phase choices

These run ASYNCHRONOUSLY and never block the real-time simulation loop.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class LaneAgentReasoner:
    """
    LangChain-powered AI reasoner for Lane Agents.
    Analyzes detection data and generates observations.
    """

    def __init__(self):
        self._available = True
        try:
            self.llm = ChatOllama(
                base_url="http://localhost:11434",
                model="qwen2.5:0.5b",
                temperature=0.4,
                num_predict=250,
            )
            self.chain = self._build_chain()
            print("[AI] Lane Agent Reasoner initialized (qwen2.5:0.5b on localhost)")
        except Exception as e:
            print(f"[AI] Lane Reasoner unavailable: {e}")
            self._available = False

    def _build_chain(self):
        prompt = PromptTemplate(
            template="""You are a Lane Agent AI for a traffic intersection. Analyze the sensor data and provide a highly detailed, comprehensive observation.

Lane: {lane} direction
Detection Source: {detection_source}
Vehicles Detected: {vehicle_count}
Vehicle Breakdown: {vehicle_types}
Emergency Vehicle: {has_emergency} {emergency_type}
Current Signal: {signal_state}
Wait Time: {wait_time} seconds

Write a detailed paragraph (3-4 sentences) thoroughly analyzing the traffic conditions on this lane. 
Explain exactly what the numbers mean for traffic flow, highlight any urgency (especially if there are emergency vehicles or high wait times), and provide a clear, logical recommendation for the master agent on whether the signal should change. 
Start with the lane direction name.""",
            input_variables=[
                "lane", "detection_source", "vehicle_count", "vehicle_types",
                "has_emergency", "emergency_type", "signal_state", "wait_time"
            ]
        )
        return prompt | self.llm | StrOutputParser()

    async def reason(
        self,
        lane: str,
        density: int,
        wait_time: float,
        has_emergency: bool,
        is_green: bool,
        detection_source: str = "mock",
        vehicle_types: Optional[Dict[str, int]] = None,
        emergency_type: Optional[str] = None,
    ) -> str:
        """Generate AI reasoning for a lane agent."""
        if not self._available:
            return self._fallback_reasoning(
                lane, density, wait_time, has_emergency,
                is_green, detection_source, vehicle_types, emergency_type
            )

        try:
            result = await asyncio.wait_for(
                self.chain.ainvoke({
                    "lane": lane,
                    "detection_source": detection_source.upper(),
                    "vehicle_count": density,
                    "vehicle_types": str(vehicle_types or {}),
                    "has_emergency": "YES ⚠️" if has_emergency else "No",
                    "emergency_type": emergency_type or "N/A",
                    "signal_state": "GREEN 🟢" if is_green else "RED 🔴",
                    "wait_time": f"{wait_time:.1f}",
                }),
                timeout=120.0,
            )
            return result.strip()
        except Exception:
            return self._fallback_reasoning(
                lane, density, wait_time, has_emergency,
                is_green, detection_source, vehicle_types, emergency_type
            )

    def _fallback_reasoning(
        self, lane, density, wait_time, has_emergency,
        is_green, detection_source, vehicle_types, emergency_type,
    ) -> str:
        """Deterministic fallback when LLM is unavailable."""
        src = "🎥 YOLO" if detection_source == "yolo" else "📊 MOCK"
        parts = [f"{lane}: [{src}] {density} vehicles detected"]
        if vehicle_types:
            parts.append("(" + ", ".join(f"{v} {k}" for k, v in vehicle_types.items()) + ")")
        if has_emergency:
            parts.append(f"⚠️ {emergency_type or 'Emergency vehicle'} — max priority!")
        if not is_green and wait_time > 20:
            parts.append(f"Waiting {wait_time:.0f}s — urgent signal needed")
        elif is_green:
            parts.append("Signal GREEN — traffic flowing")
        return " | ".join(parts)


class MasterAgentReasoner:
    """
    LangChain-powered AI reasoner for Master Agent.
    Explains signal phase decisions and auction results.
    """

    def __init__(self):
        self._available = True
        try:
            self.llm = ChatOllama(
                base_url="http://localhost:11434",
                model="qwen2.5:0.5b",
                temperature=0.4,
                num_predict=350,
            )
            self.chain = self._build_chain()
            print("[AI] Master Agent Reasoner initialized (qwen2.5:0.5b on localhost)")
        except Exception as e:
            print(f"[AI] Master Reasoner unavailable: {e}")
            self._available = False

    def _build_chain(self):
        prompt = PromptTemplate(
            template="""You are the Master Agent AI controlling traffic signals at intersection Node {node_id}. 
Your job is to provide a highly detailed, comprehensive explanation of your most recent signal phase decision.

Current Green Signal: {current_green} direction
Time in Phase: {time_in_phase} seconds
Decision Type: {decision_type}

Lane Auction Results (Priority Formula: P = 10×E + 2×D + 0.5×T_wait + GreenWaveBoost):
{auction_results}

Primary Reason for Decision: {decision_reason}

Emergency Vehicles Present: {has_any_emergency}
Green Wave Active: {green_wave_active}

Write a detailed, analytical paragraph (3-5 sentences) fully explaining your decision process. 
1. Explain exactly which lane won the auction and why, referencing specific scores, densities, or wait times.
2. If an emergency vehicle or green wave preempted normal traffic, describe exactly how that influenced the switch.
3. Conclude by explicitly stating the action taken (e.g., maintaining the current green phase, or switching to a new lane).
Be comprehensive and speak clearly as an intelligent traffic controller. Start your response with "Node {node_id}:".""",
            input_variables=[
                "node_id", "current_green", "time_in_phase", "decision_type",
                "auction_results", "decision_reason", "has_any_emergency", "green_wave_active"
            ]
        )
        return prompt | self.llm | StrOutputParser()

    async def reason(
        self,
        node_id: str,
        current_green: str,
        time_in_phase: float,
        decision_type: str,
        all_scores: List[Dict[str, Any]],
        decision_reason: str,
        has_any_emergency: bool,
        green_wave_active: bool,
    ) -> str:
        """Generate AI reasoning for a master agent decision."""
        if not self._available:
            return self._fallback_reasoning(
                node_id, decision_reason, all_scores
            )

        try:
            auction_str = "\n".join(
                f"  {s['lane']}: Score={s['score']} (D={s['density']}, T={s['wait_time']}s"
                + (", EMERGENCY" if s.get('has_emergency') else "")
                + (f", GreenWave+{s['green_wave_boost']}" if s.get('green_wave_boost', 0) > 0 else "")
                + ")"
                for s in all_scores
            )

            result = await asyncio.wait_for(
                self.chain.ainvoke({
                    "node_id": node_id,
                    "current_green": current_green,
                    "time_in_phase": f"{time_in_phase:.1f}",
                    "decision_type": decision_type.upper(),
                    "auction_results": auction_str,
                    "decision_reason": decision_reason,
                    "has_any_emergency": "YES ⚠️" if has_any_emergency else "No",
                    "green_wave_active": "YES 🌊" if green_wave_active else "No",
                }),
                timeout=120.0,
            )
            return result.strip()
        except Exception:
            return self._fallback_reasoning(node_id, decision_reason, all_scores)

    def _fallback_reasoning(
        self, node_id: str, decision_reason: str, all_scores: List[Dict[str, Any]]
    ) -> str:
        """Deterministic fallback."""
        winner = all_scores[0] if all_scores else {"lane": "?", "score": 0}
        return f"Node {node_id}: {decision_reason}"


# Singleton instances
lane_reasoner = LaneAgentReasoner()
master_reasoner = MasterAgentReasoner()
