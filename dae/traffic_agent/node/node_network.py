"""
Network of autonomous nodes.
Now uses in-memory calls, but interface will work with MQTT later.
Nodes make local decisions independently, network just coordinates.
"""
from typing import Dict, List, Optional, Any
from .intersection_node import IntersectionNode
from engine.priority_auction import GREEN_WAVE_BOOST
from logger import logger_network


class NodeNetwork:
    """
    Manages multiple autonomous nodes and their coordination.
    Can work in:
    - Simulation mode (in-memory, current laptop)
    - MQTT mode (distributed, for Jetson later)
    """

    TOPOLOGY = {
        "A": {"neighbors": ["B", "C"]},
        "B": {"neighbors": ["A", "D"]},
        "C": {"neighbors": ["A", "D"]},
        "D": {"neighbors": ["B", "C"]},
    }

    AMBULANCE_ROUTES = {
        "A_to_D": [
            {"intersection": "A", "lane": "East"},
            {"intersection": "B", "lane": "South"},
            {"intersection": "D", "lane": "North"},
        ],
        "A_to_C": [
            {"intersection": "A", "lane": "South"},
            {"intersection": "C", "lane": "North"},
        ],
        "A_to_B": [
            {"intersection": "A", "lane": "East"},
            {"intersection": "B", "lane": "West"},
        ],
        "B_to_C": [
            {"intersection": "B", "lane": "South"},
            {"intersection": "D", "lane": "West"},
            {"intersection": "C", "lane": "East"},
        ],
        "C_to_B": [
            {"intersection": "C", "lane": "East"},
            {"intersection": "D", "lane": "North"},
            {"intersection": "B", "lane": "South"},
        ],
        "D_to_A": [
            {"intersection": "D", "lane": "North"},
            {"intersection": "B", "lane": "West"},
            {"intersection": "A", "lane": "East"},
        ],
    }

    def __init__(self, mode: str = "simulation"):
        """
        Args:
            mode: 'simulation' (in-memory) or 'mqtt' (distributed, for Jetson)
        """
        self.mode = mode
        self.nodes: Dict[str, IntersectionNode] = {
            node_id: IntersectionNode(node_id) for node_id in ["A", "B", "C", "D"]
        }
        self.active_ambulances: Dict[str, Dict[str, Any]] = {}
        self._ambulance_counter = 0

        logger_network.info(f"NodeNetwork initialized in {mode} mode")

    def tick_all(self, dt: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """
        Tick all nodes.
        Each node makes local decisions autonomously.

        Returns: state from all nodes
        """
        states = {}
        for node_id, node in self.nodes.items():
            states[node_id] = node.tick(dt=dt)

        # Update ambulance progress
        self._update_ambulances()

        return states

    def _update_ambulances(self):
        """Advance ambulances through route steps."""
        completed = []

        for ambulance_id, amb in self.active_ambulances.items():
            amb["ticks"] += 1
            # Move to next step every 8 ticks (8 simulation seconds per intersection)
            if amb["ticks"] >= 8:
                amb["current_step"] += 1
                amb["ticks"] = 0
                if amb["current_step"] >= len(amb["route"]):
                    completed.append(ambulance_id)
                else:
                    # Send green-wave alert to next step
                    self._send_green_wave_alert(ambulance_id)

        # Remove completed ambulances
        for ambulance_id in completed:
            route = self.active_ambulances[ambulance_id]["route"]
            logger_network.info(
                f"Ambulance {ambulance_id} completed route {route[0]['intersection']}->"
                f"{route[-1]['intersection']}"
            )
            del self.active_ambulances[ambulance_id]

    def _send_green_wave_alert(self, ambulance_id: str):
        """Send green-wave alert from current step to next steps."""
        amb = self.active_ambulances[ambulance_id]
        route = amb["route"]
        current_step = amb["current_step"]

        # Alert all remaining steps
        for step_idx in range(current_step + 1, len(route)):
            target_step = route[step_idx]
            target_node_id = target_step["intersection"]
            target_lane = target_step["lane"]
            from_node = route[current_step]["intersection"]

            # In simulation mode: direct call
            if self.mode == "simulation":
                self.nodes[target_node_id].receive_green_wave_alert(
                    target_lane, GREEN_WAVE_BOOST, from_node=from_node
                )
            # In MQTT mode: would publish to MQTT topic (implemented later)

    def spawn_ambulance(self, route_key: str) -> Optional[str]:
        """
        Spawn ambulance on a predefined route.
        Trigger upstream nodes to send green-wave alerts downstream.
        """
        if route_key not in self.AMBULANCE_ROUTES:
            logger_network.warning(f"Unknown route: {route_key}")
            return None

        route = self.AMBULANCE_ROUTES[route_key]
        self._ambulance_counter += 1
        ambulance_id = f"AMB-{self._ambulance_counter:03d}"

        self.active_ambulances[ambulance_id] = {
            "route": route,
            "route_key": route_key,
            "current_step": 0,
            "ticks": 0,
            "completed": False,
        }

        # Register at starting node
        start_node_id = route[0]["intersection"]
        self.nodes[start_node_id].add_emergency({
            "vehicle_id": ambulance_id,
            "route": route,
            "current_step": 0,
            "ticks_at_step": 0,
            "ticks_per_step": 8,
        })

        # Send initial green-wave alerts
        self._send_green_wave_alert(ambulance_id)

        logger_network.info(
            f"Ambulance {ambulance_id} spawned on route {route_key}: "
            f"{route[0]['intersection']}->{route[-1]['intersection']}"
        )

        return ambulance_id

    def get_grid_state(self) -> Dict[str, Any]:
        """
        Aggregate all node states into one dashboard-compatible dict.
        Same format as current system, but sourced from autonomous nodes.
        """
        intersections = {}
        emergencies = []

        for node_id, node in self.nodes.items():
            intersections[node_id] = node.get_state()

        for amb_id, amb in self.active_ambulances.items():
            route = amb["route"]
            current_step = amb["current_step"]
            emergencies.append({
                "vehicle_id": amb_id,
                "route_key": amb["route_key"],
                "route": route,
                "current_step": current_step,
                "ticks_at_step": amb["ticks"],
                "ticks_per_step": 8,
                "completed": False,
                "current_intersection": (
                    route[current_step]["intersection"] if current_step < len(route) else None
                ),
                "current_lane": (
                    route[current_step]["lane"] if current_step < len(route) else None
                ),
            })

        return {
            "tick": max((n.tick_count for n in self.nodes.values()), default=0),
            "intersections": intersections,
            "emergencies": emergencies,
            "mode": self.mode,
            "active_ambulances": len(self.active_ambulances),
        }

    def reset_all(self):
        """Reset all nodes."""
        for node in self.nodes.values():
            node.reset()
        self.active_ambulances.clear()
        self._ambulance_counter = 0
        logger_network.info("All nodes reset")

    def get_node(self, node_id: str) -> Optional[IntersectionNode]:
        """Get a specific node for direct access (for testing)."""
        return self.nodes.get(node_id)
