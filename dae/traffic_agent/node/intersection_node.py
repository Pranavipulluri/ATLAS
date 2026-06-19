"""
Autonomous intersection node.
Can run standalone without GridCoordinator.
Ready for MQTT later, works in-memory now.
"""
from typing import Dict, Any, List, Optional
from agents.master_agent import MasterAgent
from engine.priority_auction import GREEN_WAVE_BOOST
from logger import logger_node


class IntersectionNode:
    """Single intersection with independent MAS (Multi-Agent System)."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.master = MasterAgent(node_id)
        self.tick_count = 0
        self.last_state = {}

        # Simulated lane densities (will be replaced by YOLO or sensor input later)
        self.lane_densities = {
            "North": 5,
            "South": 3,
            "East": 8,
            "West": 2,
        }

        # Green-wave alerts from upstream (empty now, filled by MQTT later)
        self.green_wave_alerts = {}  # {lane: boost_value}

        # Local emergency vehicles
        self.active_emergencies: List[Dict[str, Any]] = []

        logger_node.info(f"[{self.node_id}] Node initialized with MasterAgent")

    def set_lane_density(self, lane: str, density: int):
        """Update perceived lane density (from YOLO or simulation)."""
        if lane in self.lane_densities:
            self.lane_densities[lane] = density

    def receive_green_wave_alert(self, lane: str, boost: float, from_node: str):
        """Receive green-wave alert from upstream node."""
        logger_node.info(
            f"[{self.node_id}] Received green-wave for {lane} from {from_node} (+{boost})"
        )
        self.master.receive_green_wave_alert(lane, boost)
        self.green_wave_alerts[lane] = boost

    def add_emergency(self, emergency_data: Dict[str, Any]):
        """Register an emergency vehicle at this node."""
        self.active_emergencies.append(emergency_data)
        logger_node.info(
            f"[{self.node_id}] Emergency {emergency_data['vehicle_id']} registered"
        )

    def tick(self, dt: float = 1.0) -> Dict[str, Any]:
        """
        Run one simulation tick.
        Node makes local decisions autonomously.

        Returns: intersection state dict (compatible with dashboard)
        """
        self.tick_count += 1

        # Build lane data from current densities
        lane_data = {
            lane: self.lane_densities[lane] for lane in ["North", "South", "East", "West"]
        }

        # Build emergency flags
        emergency_flags = {lane: False for lane in ["North", "South", "East", "West"]}
        for em in self.active_emergencies:
            if em["current_step"] < len(em["route"]):
                step = em["route"][em["current_step"]]
                emergency_flags[step["lane"]] = True

        # Advance emergencies
        completed = []
        for em in self.active_emergencies:
            em["ticks_at_step"] += 1
            if em["ticks_at_step"] >= em["ticks_per_step"]:
                em["current_step"] += 1
                em["ticks_at_step"] = 0
                if em["current_step"] >= len(em["route"]):
                    completed.append(em["vehicle_id"])

        # Remove completed emergencies
        for vehicle_id in completed:
            self.active_emergencies = [
                em for em in self.active_emergencies if em["vehicle_id"] != vehicle_id
            ]
            logger_node.info(f"[{self.node_id}] Emergency {vehicle_id} completed")

        # Tick master agent (autonomous local decision-making)
        state = self.master.tick(lane_data, emergency_flags, dt=dt)

        # Store for later retrieval
        self.last_state = state
        self.last_state["tick"] = self.tick_count

        logger_node.debug(
            f"[{self.node_id}] Tick {self.tick_count}: green={state.get('current_green')} "
            f"time={state.get('time_in_phase'):.1f}s"
        )

        return state

    def get_state(self) -> Dict[str, Any]:
        """Return last computed state."""
        if not self.last_state:
            # Return empty state if tick hasn't run yet
            return {
                "intersection_id": self.node_id,
                "current_green": "North",
                "time_in_phase": 0.0,
                "tick": 0,
                "lanes": {},
            }
        return self.last_state

    def reset(self):
        """Reset node to initial state."""
        self.tick_count = 0
        self.last_state = {}
        self.active_emergencies = []
        self.master = MasterAgent(self.node_id)
        logger_node.info(f"[{self.node_id}] Node reset")
