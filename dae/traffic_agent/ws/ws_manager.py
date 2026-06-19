"""
WebSocket Manager
Handles WebSocket connections for real-time state sync 
between the FastAPI backend and Next.js frontend.
"""

import json
import asyncio
from typing import List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect


class WSManager:
    """
    Manages WebSocket connections.
    Broadcasts grid state to all connected clients every tick.
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._xai_logs: List[Dict[str, Any]] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, state: Dict[str, Any]):
        """
        Broadcast the full grid state to all connected clients.
        Includes XAI logs if available.
        """
        # Attach XAI logs
        state["xai_logs"] = self._xai_logs[-20:]  # last 20 explanations
        
        message = json.dumps(state)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    def add_xai_log(self, log: Dict[str, Any]):
        """Add an XAI explanation to the log buffer."""
        self._xai_logs.append(log)
        # Keep buffer bounded
        if len(self._xai_logs) > 100:
            self._xai_logs = self._xai_logs[-50:]
    
    async def receive_commands(self, websocket: WebSocket, command_handler):
        """
        Listen for commands from a client and dispatch them.
        Runs until the client disconnects.
        """
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    command = json.loads(data)
                    result = command_handler(command)
                    await websocket.send_text(json.dumps({
                        "type": "command_response",
                        "result": result
                    }))
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
        except WebSocketDisconnect:
            self.disconnect(websocket)
