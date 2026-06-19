"""
XAI Explainer Service
Uses LangChain + ChatOllama (kimi-k2.5:cloud) to translate state changes
into human-readable rationale logs. Runs in PARALLEL and never blocks
the real-time signal decisions.
"""

import asyncio
import time
from typing import Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class XAIExplainer:
    """
    Explainable AI service using LangChain.
    Translates complex state changes into human-readable rationale.
    Fire-and-forget: never blocks the main simulation loop.
    """
    
    def __init__(self, ws_manager=None):
        self.ws_manager = ws_manager
        self._available = True
        
        try:
            self.llm = ChatOllama(
                model="kimi-k2.5:cloud",
                temperature=0.3,
                num_predict=150,  # keep responses concise
            )
            self.chain = self._build_chain()
        except Exception as e:
            print(f"[XAI] Warning: Could not initialize LLM: {e}")
            self._available = False
    
    def _build_chain(self):
        prompt = PromptTemplate(
            template="""You are an explainable AI assistant for a traffic management system.
Translate this signal decision into a clear, concise, one-sentence explanation for a traffic control dashboard.

Intersection: Node {node_id}
Decision: {decision_reason}
Current Green: {current_green}
Lane Scores: {scores}
Green Wave Active: {green_wave_active}
Green Wave Lanes: {green_wave_lanes}

Write ONE sentence explaining what happened and why, as if narrating for a traffic control operator. Be specific about lane names and numbers. Start with "Node {node_id}:".
Example: "Node B: Preemptively switching South lane to green due to incoming ambulance from Node A, priority boosted to 1012.5."
""",
            input_variables=["node_id", "decision_reason", "current_green", "scores", "green_wave_active", "green_wave_lanes"]
        )
        return prompt | self.llm | StrOutputParser()
    
    async def explain(self, node_id: str, state: Dict[str, Any], grid_state: Dict[str, Any]):
        """
        Generate an XAI explanation for a signal decision.
        This runs asynchronously and adds the result to the WS manager's log.
        """
        if not self._available:
            self._add_fallback_log(node_id, state)
            return
        
        try:
            result = await asyncio.wait_for(
                self.chain.ainvoke({
                    "node_id": node_id,
                    "decision_reason": state.get("decision_reason", ""),
                    "current_green": state.get("current_green", ""),
                    "scores": str(state.get("scores", {})),
                    "green_wave_active": state.get("green_wave_active", False),
                    "green_wave_lanes": str(state.get("green_wave_lanes", [])),
                }),
                timeout=10.0  # don't wait more than 10s
            )
            
            log_entry = {
                "timestamp": time.time(),
                "node_id": node_id,
                "explanation": result.strip(),
                "type": self._classify_event(state),
            }
            
            if self.ws_manager:
                self.ws_manager.add_xai_log(log_entry)
                
        except asyncio.TimeoutError:
            self._add_fallback_log(node_id, state)
        except Exception as e:
            self._add_fallback_log(node_id, state, str(e))
    
    def _add_fallback_log(self, node_id: str, state: Dict[str, Any], error: str = ""):
        """Add a deterministic fallback explanation when LLM is unavailable."""
        reason = state.get("decision_reason", "No reason available")
        log_entry = {
            "timestamp": time.time(),
            "node_id": node_id,
            "explanation": f"Node {node_id}: {reason}",
            "type": self._classify_event(state),
            "fallback": True,
        }
        if self.ws_manager:
            self.ws_manager.add_xai_log(log_entry)
    
    def _classify_event(self, state: Dict[str, Any]) -> str:
        """Classify the event type for frontend color coding."""
        reason = state.get("decision_reason", "").upper()
        if "EMERGENCY" in reason:
            return "emergency"
        elif "GREEN_WAVE" in reason:
            return "green_wave"
        elif "MAX_GREEN" in reason:
            return "starvation"
        elif "AUCTION_SWITCH" in reason:
            return "auction"
        return "normal"
