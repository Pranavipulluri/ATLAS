"""
Agentic Edge Traffic Management System
FastAPI Backend — uses GridCoordinator + SimulationLoop (full-featured)
for real simulation (rain, flood, pedestrians, ambulance movement, arrivals/departures).

The NodeNetwork / IntersectionNode decentralized architecture exists in node/
and represents the Jetson deployment target (TRAFFIC_MODE=mqtt).
"""

import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agents.grid_coordinator import GridCoordinator
from simulation.sim_loop import SimulationLoop
from ws.ws_manager import WSManager
from xai.explainer import XAIExplainer
from config import API_HOST, API_PORT, TICK_RATE
from logger import logger_api, logger_sim


# --- Global instances ---
ws_manager = WSManager()
grid = GridCoordinator()
xai_explainer = XAIExplainer(ws_manager=ws_manager)

sim_loop = SimulationLoop(
    grid_coordinator=grid,
    ws_broadcast=ws_manager.broadcast,
    xai_explain=xai_explainer.explain,
    ws_manager=ws_manager,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start simulation on app startup, stop on shutdown."""
    logger_api.info("Starting Agentic Edge Traffic Management (Full Simulation Mode)")
    logger_api.info(f"Tick rate: {TICK_RATE}s")
    task = asyncio.create_task(sim_loop.start())
    yield
    logger_api.info("Stopping simulation")
    sim_loop.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Agentic Edge Traffic Management",
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Command Handler ---
def handle_command(command: dict) -> dict:
    """Route WebSocket commands to sim_loop."""
    return sim_loop.handle_command(command)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        await ws_manager.receive_commands(websocket, handle_command)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# --- REST Endpoints ---
@app.get("/")
def root():
    return {
        "status": "Agentic Edge Traffic Management v3.1 Running",
        "version": "3.1.0",
        "websocket": "ws://localhost:8000/ws",
        "intersections": ["A", "B", "C", "D"],
    }


@app.get("/api/state")
def get_state():
    """Get current grid state (REST fallback — live state from last tick)."""
    states = {}
    for node_id, master in grid.masters.items():
        states[node_id] = master.get_state()
    return {
        "intersections": states,
        "emergencies": [grid._emergency_to_dict(e) for e in grid.active_emergencies],
    }


@app.post("/api/spawn_ambulance/{route_key}")
def spawn_ambulance(route_key: str):
    vehicle_id = grid.spawn_ambulance(route_key)
    if vehicle_id:
        return {"status": "ok", "vehicle_id": vehicle_id}
    return {"status": "error", "message": f"Invalid route: {route_key}"}


@app.get("/api/routes")
def get_routes():
    return {"routes": grid.get_available_routes()}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "mode": "simulation",
        "active_nodes": ["A", "B", "C", "D"],
        "active_ambulances": len(grid.active_emergencies),
        "tick": sim_loop.tick_count,
    }


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info",
    )
