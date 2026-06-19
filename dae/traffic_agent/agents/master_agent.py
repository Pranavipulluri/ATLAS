"""
Master Agent (Level 2 - Intersection Brain)
One per intersection. Gathers data from 4 Lane Agents,
runs the deterministic Priority Auction, and decides signal phase.
Now includes detailed decision breakdown and lane reasoning aggregation.
"""

from typing import Dict, Any, Optional, List
from agents.lane_agent import LaneAgent, LaneReport
from engine.priority_auction import (
    compute_lane_priority, run_auction, should_switch_phase,
    LaneScore, MIN_GREEN_TIME, MAX_GREEN_TIME
)


LANE_NAMES = ["North", "South", "East", "West"]


class MasterAgent:
    """
    Level 2 Agent - the intersection brain.
    Manages 4 Lane Agents and runs the priority auction.
    Generates detailed decision breakdown with reasoning.
    """

    def __init__(self, intersection_id: str):
        self.intersection_id = intersection_id
        self.lane_agents: Dict[str, LaneAgent] = {
            name: LaneAgent(name) for name in LANE_NAMES
        }
        self.current_green: str = "North"
        self.time_in_phase: float = 0.0
        self.interrupted_lane: Optional[str] = None
        self.green_wave_boosts: Dict[str, float] = {}
        self.last_decision_reason: str = ""
        self.last_scores: Dict[str, float] = {}
        # New: detailed reasoning
        self.lane_reasonings: Dict[str, str] = {}
        self.decision_breakdown: Dict[str, Any] = {}

    def receive_green_wave_alert(self, target_lane: str, boost: float):
        self.green_wave_boosts[target_lane] = boost

    def clear_green_wave(self, target_lane: str):
        self.green_wave_boosts.pop(target_lane, None)

    def tick(
        self,
        lane_densities: Dict[str, int],
        lane_emergencies: Dict[str, bool],
        dt: float = 1.0,
        detection_data: Optional[Dict[str, Dict[str, Any]]] = None,
        lane_pedestrians: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Process one simulation tick.

        Args:
            lane_densities: {"North": 5, ...}
            lane_emergencies: {"North": False, ...}
            dt: time step
            detection_data: optional YOLO detection data per lane
            lane_pedestrians: optional Pedestrian presence per lane
        """
        self.time_in_phase += dt
        lane_pedestrians = lane_pedestrians or {}

        # 1. Update all Lane Agents
        for name in LANE_NAMES:
            is_green = (name == self.current_green)
            det = (detection_data or {}).get(name, {})
            self.lane_agents[name].update(
                density=lane_densities.get(name, 0),
                has_emergency=lane_emergencies.get(name, False),
                is_green=is_green,
                dt=dt,
                detection_source=det.get("source", "simulation"),
                vehicle_types=det.get("vehicle_types", {}),
                emergency_type=det.get("emergency_type"),
                detection_reasoning=det.get("reasoning", ""),
                has_pedestrians=lane_pedestrians.get(name, False),
            )

        # 2. Collect reports and compute priority scores
        lane_scores: List[LaneScore] = []
        for name in LANE_NAMES:
            report = self.lane_agents[name].generate_report()
            boost = self.green_wave_boosts.get(name, 0.0)
            score = compute_lane_priority(
                lane=report.lane,
                density=report.density,
                wait_time=report.wait_time,
                has_emergency=report.has_emergency,
                green_wave_boost=boost,
                has_pedestrians=report.has_pedestrians,
            )
            lane_scores.append(score)

        # 3. Run the auction
        ranked = run_auction(lane_scores)

        # 4. Decide phase
        switch, target, reason = should_switch_phase(
            self.current_green, ranked, self.time_in_phase
        )

        # 5. Handle post-emergency recovery
        if switch and target != self.current_green:
            winner = ranked[0]
            if (winner.has_emergency or winner.green_wave_boost > 0) and self.interrupted_lane is None:
                self.interrupted_lane = self.current_green
            elif target == self.interrupted_lane:
                self.interrupted_lane = None
            self.current_green = target
            self.time_in_phase = 0.0

        # 6. Store reasoning
        self.last_decision_reason = reason
        self.last_scores = {s.lane: round(s.score, 1) for s in ranked}
        self.lane_reasonings = {
            name: agent.reasoning for name, agent in self.lane_agents.items()
        }

        # 7. Build decision breakdown
        self._build_decision_breakdown(ranked, switch, target, reason)

        # 8. Update lane agent green states
        for name in LANE_NAMES:
            self.lane_agents[name].is_green = (name == self.current_green)

        return self.get_state()

    def _build_decision_breakdown(
        self, ranked: List[LaneScore], switch: bool, target: str, reason: str
    ):
        """Build detailed decision breakdown for the frontend."""
        winner = ranked[0]

        # Determine decision type
        if "EMERGENCY_PREEMPT" in reason:
            decision_type = "emergency"
            decision_icon = "🚨"
        elif "GREEN_WAVE" in reason:
            decision_type = "green_wave"
            decision_icon = "🌊"
        elif "MAX_GREEN" in reason:
            decision_type = "starvation"
            decision_icon = "⏰"
        elif "AUCTION_SWITCH" in reason:
            decision_type = "auction"
            decision_icon = "🔄"
        else:
            decision_type = "maintain"
            decision_icon = "✅"

        self.decision_breakdown = {
            "type": decision_type,
            "icon": decision_icon,
            "switched": switch,
            "winner": winner.lane,
            "winner_score": round(winner.score, 1),
            "reason": reason,
            "current_green": self.current_green,
            "time_in_phase": round(self.time_in_phase, 1),
            "all_scores": [
                {
                    "lane": s.lane,
                    "score": round(s.score, 1),
                    "density": s.density,
                    "wait_time": round(s.wait_time, 1),
                    "has_emergency": s.has_emergency,
                    "green_wave_boost": s.green_wave_boost,
                    "has_pedestrians": s.has_pedestrians,
                    "formula": (
                        f"P = ({10.0}×{'1000' if s.has_emergency else '0'}) + "
                        f"({2.0}×{s.density}) + ({0.5}×{s.wait_time:.1f})"
                        + (f" + {s.green_wave_boost:.0f}" if s.green_wave_boost > 0 else "")
                        + (f" + (50×{s.wait_time:.0f})" if s.has_pedestrians else "")
                        + f" = {s.score:.1f}"
                    ),
                }
                for s in ranked
            ],
        }

    def get_state(self) -> Dict[str, Any]:
        """Get full intersection state for WebSocket broadcast."""
        return {
            "intersection_id": self.intersection_id,
            "current_green": self.current_green,
            "time_in_phase": round(self.time_in_phase, 1),
            "interrupted_lane": self.interrupted_lane,
            "green_wave_active": bool(self.green_wave_boosts),
            "green_wave_lanes": list(self.green_wave_boosts.keys()),
            "decision_reason": self.last_decision_reason,
            "scores": self.last_scores,
            "lanes": {
                name: agent.to_dict() for name, agent in self.lane_agents.items()
            },
            # New reasoning fields
            "lane_reasonings": self.lane_reasonings,
            "decision_breakdown": self.decision_breakdown,
        }
