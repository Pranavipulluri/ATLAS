"""
Configuration management for traffic system.
Easy switching between laptop simulation and Jetson deployment.
"""
import os
from enum import Enum


class Mode(Enum):
    """Deployment modes."""

    SIMULATION = "simulation"  # Laptop: in-memory, no MQTT
    MQTT = "mqtt"  # Distributed: MQTT mesh (Jetson)
    HYBRID = "hybrid"  # Laptop: MQTT-ready with local broker


# Load from environment variables
TRAFFIC_MODE = os.getenv("TRAFFIC_MODE", "simulation")
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost:1883")
NODE_ID = os.getenv("NODE_ID", None)  # None = run all nodes, else run single node

# Detection mode
DETECTION_MODE = os.getenv("DETECTION_MODE", "mock")  # "mock" or "yolo"
YOLO_DETECTION_INTERVAL = int(os.getenv("YOLO_DETECTION_INTERVAL", "60"))

# Logging
DEBUG = os.getenv("DEBUG", "0") == "1"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Simulation
TICK_RATE = float(os.getenv("TICK_RATE", "1.0"))

# Validation
if TRAFFIC_MODE not in ["simulation", "mqtt", "hybrid"]:
    raise ValueError(f"Invalid TRAFFIC_MODE: {TRAFFIC_MODE}")

if DETECTION_MODE not in ["mock", "yolo"]:
    raise ValueError(f"Invalid DETECTION_MODE: {DETECTION_MODE}")
