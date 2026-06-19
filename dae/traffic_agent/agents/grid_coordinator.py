"""
Grid Coordinator (Level 3 - Inter-Master Communication / A2A)
This is THE MOAT: Master Agents talk to each other.

Manages the 2x2 grid topology and routes emergency vehicles
through multiple intersections with predictive Green Wave alerts.

Now accepts detection data from YOLO VideoFeedManager.

Grid Layout:
  A ──── B
  │      │
  │      │
  C ──── D
  
Adjacency:
  A ↔ B (East/West)
  A ↔ C (South/North)
  B ↔ D (South/North)
  C ↔ D (East/West)
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from agents.master_agent import MasterAgent
from engine.priority_auction import GREEN_WAVE_BOOST


GRID_TOPOLOGY: Dict[str, Dict[str, Tuple[str, str]]] = {
    "A": {"B": ("East", "West"), "C": ("South", "North")},
    "B": {"A": ("West", "East"), "D": ("South", "North")},
    "C": {"A": ("North", "South"), "D": ("East", "West")},
    "D": {"C": ("West", "East"), "B": ("North", "South")},
}

AMBULANCE_ROUTES: Dict[str, List[Dict[str, str]]] = {
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


@dataclass
class ActiveEmergency:
    """Tracks an active emergency vehicle traversing the grid."""
    vehicle_id: str
    route_key: str
    route: List[Dict[str, str]]
    current_step: int = 0
    ticks_at_step: int = 0
    ticks_per_step: int = 8
    completed: bool = False


class GridCoordinator:
    """
    Level 3 A2A Communication Layer.
    Orchestrates inter-master communication for the Green Wave.
    """

    def __init__(self):
        self.masters: Dict[str, MasterAgent] = {
            node_id: MasterAgent(node_id) for node_id in ["A", "B", "C", "D"]
        }
        self.active_emergencies: List[ActiveEmergency] = []
        self.a2a_messages: List[Dict[str, Any]] = []
        self._emergency_counter = 0

    def spawn_ambulance(self, route_key: str) -> Optional[str]:
        if route_key not in AMBULANCE_ROUTES:
            return None
        self._emergency_counter += 1
        vehicle_id = f"AMB-{self._emergency_counter:03d}"
        route = AMBULANCE_ROUTES[route_key]
        emergency = ActiveEmergency(
            vehicle_id=vehicle_id, route_key=route_key,
            route=route, current_step=0, ticks_at_step=0,
        )
        self.active_emergencies.append(emergency)
        self._send_green_wave_alerts(emergency)
        return vehicle_id

    def _send_green_wave_alerts(self, emergency: ActiveEmergency):
        for step_idx in range(emergency.current_step + 1, len(emergency.route)):
            future_step = emergency.route[step_idx]
            target_intersection = future_step["intersection"]
            target_lane = future_step["lane"]
            if target_intersection in self.masters:
                self.masters[target_intersection].receive_green_wave_alert(
                    target_lane=target_lane, boost=GREEN_WAVE_BOOST
                )
                current_intersection = emergency.route[emergency.current_step]["intersection"]
                msg = {
                    "type": "GREEN_WAVE_ALERT",
                    "from": current_intersection,
                    "to": target_intersection,
                    "vehicle_id": emergency.vehicle_id,
                    "target_lane": target_lane,
                    "message": (
                        f"Node {current_intersection} → Node {target_intersection}: "
                        f"Incoming {emergency.vehicle_id} heading to {target_lane} lane. "
                        f"Pre-clear requested (boost +{GREEN_WAVE_BOOST})."
                    )
                }
                self.a2a_messages.append(msg)

    def tick(
        self,
        lane_data: Dict[str, Dict[str, Dict[str, Any]]],
        detection_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Process one simulation tick for the entire grid.

        Args:
            lane_data: {node: {lane: {density: int}}}
            detection_data: optional YOLO detection results per lane
        """
        # 1. Advance ambulance positions
        self._advance_emergencies()

        # 2. Build emergency flags
        emergency_flags = {
            node: {lane: False for lane in ["North", "South", "East", "West"]}
            for node in self.masters
        }
        for em in self.active_emergencies:
            if not em.completed and em.current_step < len(em.route):
                step = em.route[em.current_step]
                emergency_flags[step["intersection"]][step["lane"]] = True

        # Also merge YOLO-detected emergencies
        if detection_data:
            for node_id in ["A", "B", "C", "D"]:
                for lane in ["North", "South", "East", "West"]:
                    det = (detection_data.get(node_id, {}).get(lane, {}))
                    if det.get("has_emergency", False):
                        emergency_flags[node_id][lane] = True

        # 3. Tick each Master Agent
        intersection_states = {}
        for node_id, master in self.masters.items():
            densities = {}
            lane_pedestrians = {}
            for lane in ["North", "South", "East", "West"]:
                l_data = lane_data.get(node_id, {}).get(lane, {})
                densities[lane] = l_data.get("density", 0)
                lane_pedestrians[lane] = l_data.get("has_pedestrians", False)

            # Pass detection metadata to master agent
            det_meta = {}
            if detection_data:
                for lane in ["North", "South", "East", "West"]:
                    det = detection_data.get(node_id, {}).get(lane, {}).get("detection", {})
                    det_meta[lane] = {
                        "source": det.get("source", "simulation"),
                        "vehicle_types": det.get("vehicle_types", {}),
                        "emergency_type": det.get("emergency_type"),
                        "reasoning": det.get("reasoning", ""),
                    }

            emergencies = emergency_flags[node_id]
            state = master.tick(
                densities, emergencies, dt=1.0,
                detection_data=det_meta if det_meta else None,
                lane_pedestrians=lane_pedestrians,
            )
            intersection_states[node_id] = state

        # 4. Build full grid state
        return {
            "intersections": intersection_states,
            "emergencies": [self._emergency_to_dict(em) for em in self.active_emergencies],
            "a2a_messages": self.a2a_messages[-10:],
            "green_wave_active": any(not em.completed for em in self.active_emergencies),
        }

    def _advance_emergencies(self):
        for em in self.active_emergencies:
            if em.completed:
                continue
            em.ticks_at_step += 1
            if em.ticks_at_step >= em.ticks_per_step:
                current_step = em.route[em.current_step]
                current_node = current_step["intersection"]
                current_lane = current_step["lane"]
                if current_node in self.masters:
                    self.masters[current_node].clear_green_wave(current_lane)
                em.current_step += 1
                em.ticks_at_step = 0
                if em.current_step >= len(em.route):
                    em.completed = True
                    for master in self.masters.values():
                        master.green_wave_boosts.clear()
                    self.a2a_messages.append({
                        "type": "EMERGENCY_COMPLETE",
                        "vehicle_id": em.vehicle_id,
                        "message": f"{em.vehicle_id} has completed route {em.route_key}. All signals returning to normal auction."
                    })
                else:
                    self._send_green_wave_alerts(em)

    def _emergency_to_dict(self, em: ActiveEmergency) -> Dict[str, Any]:
        return {
            "vehicle_id": em.vehicle_id,
            "route_key": em.route_key,
            "route": em.route,
            "current_step": em.current_step,
            "ticks_at_step": em.ticks_at_step,
            "ticks_per_step": em.ticks_per_step,
            "completed": em.completed,
            "current_intersection": em.route[em.current_step]["intersection"] if em.current_step < len(em.route) else None,
            "current_lane": em.route[em.current_step]["lane"] if em.current_step < len(em.route) else None,
        }

    def get_available_routes(self) -> List[str]:
        return list(AMBULANCE_ROUTES.keys())
