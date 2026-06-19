"""
Simulation Loop
Manages the real-time traffic simulation with YOLO detection integration,
AI reasoning via LangChain agents, and ambulance progression through the grid.
"""

import asyncio
import random
import time
from typing import Dict, Any, Optional, Callable, Awaitable
from detection.yolo_detector import VideoFeedManager
from agents.ai_reasoner import lane_reasoner, master_reasoner


class SimulationLoop:
    """
    Async simulation loop that ticks every ~1 second.
    Integrates YOLO detection + AI reasoning via LangChain.
    """

    def __init__(
        self,
        grid_coordinator,
        ws_broadcast: Optional[Callable] = None,
        xai_explain: Optional[Callable] = None,
        video_feed_manager: Optional[VideoFeedManager] = None,
        ws_manager = None,
    ):
        self.grid = grid_coordinator
        self.ws_broadcast = ws_broadcast
        self.xai_explain = xai_explain
        self.video_feeds = video_feed_manager or VideoFeedManager(detection_interval=60.0)
        self.ws_manager = ws_manager
        self.running = False
        self.tick_rate = 1.0
        self.tick_count = 0

        self.lane_densities: Dict[str, Dict[str, int]] = {
            "A": {"North": 5, "South": 3, "East": 8, "West": 2},
            "B": {"North": 4, "South": 6, "East": 3, "West": 7},
            "C": {"North": 7, "South": 2, "East": 4, "West": 5},
            "D": {"North": 3, "South": 8, "East": 6, "West": 4},
        }

        self.arrival_range = (0, 3)
        self.departure_range = (0, 2)
        self.traffic_multipliers: Dict[str, float] = {
            "A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0
        }
        self.detection_mode: str = "yolo"
        self.severe_rain: Optional[str] = None
        self.flood_active: Optional[str] = None
        self.pedestrians: Dict[str, Dict[str, bool]] = {
            "A": {}, "B": {}, "C": {}, "D": {}
        }

        # AI reasoning cache (updated asynchronously by LangChain agents)
        self.ai_lane_reasonings: Dict[str, Dict[str, str]] = {
            node: {lane: "" for lane in ["North", "South", "East", "West"]}
            for node in ["A", "B", "C", "D"]
        }
        self.ai_master_reasonings: Dict[str, str] = {
            "A": "", "B": "", "C": "", "D": ""
        }

        # AI reasoning interval (run every N ticks, not every tick for performance)
        self.ai_reasoning_interval = 25  # Increased to 25 for Raspberry Pi Edge Node
        self._pending_reasoning_tasks: list = []
        self.ai_is_busy = False  # Global lock to prevent concurrent Raspberry Pi overloads

    async def start(self):
        self.running = True
        self.video_feeds.run_detection_cycle()
        while self.running:
            start_time = time.time()
            await self.run_single_tick()
            elapsed = time.time() - start_time
            sleep_time = max(0, self.tick_rate - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self):
        self.running = False

    async def run_single_tick(self):
        """Execute one simulation tick."""
        self.tick_count += 1

        # 1. YOLO detection cycle
        detection_ran = False
        if self.video_feeds.should_run_detection():
            self.video_feeds.run_detection_cycle()
            detection_ran = True

        detection_lane_data = self.video_feeds.get_lane_data()

        # 2. Build lane data with YOLO counts and inject trucks
        lane_data = {}
        for node_id in ["A", "B", "C", "D"]:
            lane_data[node_id] = {}
            for lane in ["North", "South", "East", "West"]:
                det = detection_lane_data[node_id][lane]
                
                # Introduce Trucks
                if "vehicle_types" not in det:
                    det["vehicle_types"] = {}
                trucks = random.randint(0, 2)
                det["vehicle_types"]["truck"] = trucks
                det["vehicle_types"]["car"] = max(1, det.get("density", 1) - trucks)

                if detection_ran:
                    density = det["density"]
                    multiplier = self.traffic_multipliers.get(node_id, 1.0)
                    density = int(density * multiplier)
                else:
                    density = self.lane_densities[node_id].get(lane, int(det["density"]) if "density" in det else 3)
                
                # Steady arrivals (no negative jitter that causes fake departures!)
                arrivals = random.randint(0, 1)
                density = max(0, min(50, density + arrivals))
                
                master = self.grid.masters[node_id]
                if master.current_green == lane:
                    if self.severe_rain == node_id:
                        departures = random.randint(0, 1)
                    else:
                        departures = random.randint(2, 5) # Faster vehicles
                    density = max(0, density - departures)
                    # Clear pedestrians if lane is green
                    if self.pedestrians[node_id].get(lane):
                        self.pedestrians[node_id][lane] = False

                has_ped = self.pedestrians[node_id].get(lane, False)
                lane_data[node_id][lane] = {"density": density, "has_pedestrians": has_ped}
                self.lane_densities[node_id][lane] = density

        # 3. Build detection metadata
        detection_data = {}
        for node_id in ["A", "B", "C", "D"]:
            detection_data[node_id] = {}
            for lane in ["North", "South", "East", "West"]:
                detection_data[node_id][lane] = detection_lane_data[node_id][lane]

        # 4. Tick the grid coordinator
        grid_state = self.grid.tick(lane_data, detection_data=detection_data)

        # 5. Fire AI reasoning (async, every N ticks)
        if self.tick_count % self.ai_reasoning_interval == 0:
            self._fire_ai_reasoning(grid_state, detection_lane_data)

        # 6. Inject cached AI reasoning into grid state
        for node_id, state in grid_state.get("intersections", {}).items():
            state["ai_lane_reasonings"] = self.ai_lane_reasonings.get(node_id, {})
            state["ai_master_reasoning"] = self.ai_master_reasonings.get(node_id, "")

        # 7. Add simulation metadata
        grid_state["tick"] = self.tick_count
        grid_state["tick_rate"] = self.tick_rate
        grid_state["traffic_multipliers"] = self.traffic_multipliers
        grid_state["detection_mode"] = self.detection_mode
        grid_state["detection_interval"] = self.video_feeds.detection_interval
        grid_state["severe_rain"] = self.severe_rain
        grid_state["flood_active"] = self.flood_active

        # 8. Broadcast via WebSocket
        if self.ws_broadcast:
            await self.ws_broadcast(grid_state)

        # 9. XAI explanation for interesting events
        if self.xai_explain:
            for node_id, state in grid_state.get("intersections", {}).items():
                reason = state.get("decision_reason", "")
                if any(kw in reason for kw in ["EMERGENCY", "GREEN_WAVE", "MAX_GREEN", "AUCTION_SWITCH"]):
                    asyncio.create_task(self._safe_explain(node_id, state, grid_state))

        return grid_state

    def _fire_ai_reasoning(self, grid_state: Dict, detection_data: Dict):
        """Fire-and-forget AI reasoning task that processes the grid sequentially to protect the Edge Node."""
        if self.ai_is_busy:
            return
            
        self.ai_is_busy = True
        asyncio.create_task(self._process_grid_reasoning_sequentially(grid_state, detection_data))

    async def _process_grid_reasoning_sequentially(self, grid_state: Dict, detection_data: Dict):
        try:
            for node_id, state in grid_state.get("intersections", {}).items():
                # Process Master Agent sequentially
                breakdown = state.get("decision_breakdown", {})
                await self._run_master_reasoning(
                    node_id=node_id,
                    current_green=state.get("current_green", ""),
                    time_in_phase=state.get("time_in_phase", 0),
                    decision_type=breakdown.get("type", "maintain"),
                    all_scores=breakdown.get("all_scores", []),
                    decision_reason=state.get("decision_reason", ""),
                    has_any_emergency=any(l.get("has_emergency", False) for l in state.get("lanes", {}).values()),
                    green_wave_active=state.get("green_wave_active", False),
                )

                # Process Lane Agents sequentially to prevent Pi overload
                for lane_name, lane_state in state.get("lanes", {}).items():
                    det = detection_data.get(node_id, {}).get(lane_name, {})
                    detection = det.get("detection", {})
                    await self._run_lane_reasoning(
                        node_id, lane_name,
                        density=lane_state.get("density", 0),
                        wait_time=lane_state.get("wait_time", 0),
                        has_emergency=lane_state.get("has_emergency", False),
                        is_green=lane_state.get("is_green", False),
                        detection_source=detection.get("source", "mock"),
                        vehicle_types=detection.get("vehicle_types", {}),
                        emergency_type=detection.get("emergency_type"),
                    )
        finally:
            self.ai_is_busy = False

    async def _run_lane_reasoning(self, node_id: str, lane_name: str, **kwargs):
        """Run lane agent AI reasoning and cache the result."""
        try:
            reasoning = await lane_reasoner.reason(lane=lane_name, **kwargs)
            self.ai_lane_reasonings[node_id][lane_name] = reasoning
        except Exception:
            pass

    async def _run_master_reasoning(self, **kwargs):
        """Run master agent AI reasoning and cache the result."""
        try:
            node_id = kwargs.get("node_id", "A")
            reasoning = await master_reasoner.reason(**kwargs)
            self.ai_master_reasonings[node_id] = reasoning
        except Exception:
            pass

    async def _safe_explain(self, node_id: str, state: Dict, grid_state: Dict):
        try:
            if self.xai_explain:
                await self.xai_explain(node_id, state, grid_state)
        except Exception:
            pass

    def handle_command(self, command: Dict[str, Any]):
        cmd_type = command.get("type")

        if cmd_type == "spawn_ambulance":
            route_key = command.get("route", "A_to_D")
            vehicle_id = self.grid.spawn_ambulance(route_key)
            return {"status": "ok", "vehicle_id": vehicle_id}

        elif cmd_type == "toggle_severe_rain":
            state = command.get("state", False)
            node_id = command.get("node", "A")
            self.severe_rain = node_id if state else None
            if self.severe_rain and self.ws_manager:
                self.ws_manager.add_xai_log({
                    "timestamp": time.time(),
                    "node_id": "System",
                    "explanation": f"Severe Rain Detected at Node {node_id}. Slower vehicle departure rates. Green lights auto-adapting.",
                    "type": "emergency"
                })
            return {"status": "ok", "severe_rain": self.severe_rain}

        elif cmd_type == "trigger_flood":
            node_id = command.get("node", "B")
            self.flood_active = node_id
            if self.ws_manager:
                self.ws_manager.add_xai_log({
                    "timestamp": time.time(),
                    "node_id": "A",
                    "explanation": f"Node A Master: Received Flood Alert from Node {node_id}. Rerouting all Northbound traffic to East/West corridors. Escalating to MANUAL human control.",
                    "type": "emergency"
                })
            return {"status": "ok", "flood_active": self.flood_active}

        elif cmd_type == "spawn_pedestrians":
            node_id = command.get("node", "C")
            for lane in ["North", "South", "East", "West"]:
                self.pedestrians[node_id][lane] = True
            return {"status": "ok", "node": node_id}

        elif cmd_type == "set_traffic":
            node_id = command.get("node")
            multiplier = command.get("multiplier", 1.0)
            if node_id in self.traffic_multipliers:
                self.traffic_multipliers[node_id] = multiplier
            return {"status": "ok"}

        elif cmd_type == "set_tick_rate":
            self.tick_rate = max(0.2, min(5.0, command.get("rate", 1.0)))
            return {"status": "ok"}

        elif cmd_type == "assign_video":
            node_id = command.get("node", "")
            lane = command.get("lane", "")
            path = command.get("path", "")
            success = self.video_feeds.assign_video(node_id, lane, path)
            return {"status": "ok" if success else "error"}

        elif cmd_type == "set_detection_interval":
            interval = max(10, min(300, command.get("interval", 60)))
            self.video_feeds.detection_interval = float(interval)
            return {"status": "ok", "interval": interval}

        elif cmd_type == "force_detection":
            self.video_feeds.run_detection_cycle()
            return {"status": "ok", "message": "Detection cycle forced"}

        elif cmd_type == "set_ai_interval":
            interval = max(1, min(30, command.get("interval", 5)))
            self.ai_reasoning_interval = interval
            return {"status": "ok", "interval": interval}

        elif cmd_type == "reset":
            self.lane_densities = {
                "A": {"North": 5, "South": 3, "East": 8, "West": 2},
                "B": {"North": 4, "South": 6, "East": 3, "West": 7},
                "C": {"North": 7, "South": 2, "East": 4, "West": 5},
                "D": {"North": 3, "South": 8, "East": 6, "West": 4},
            }
            self.traffic_multipliers = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0}
            self.grid.active_emergencies.clear()
            self.grid.a2a_messages.clear()
            for master in self.grid.masters.values():
                master.green_wave_boosts.clear()
                master.interrupted_lane = None
            self.video_feeds.run_detection_cycle()
            return {"status": "ok"}

        return {"status": "error", "message": f"Unknown command: {cmd_type}"}
