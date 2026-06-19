"""
Abstract detection interface.
Allows different detection backends (mock, YOLO, sensor input).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import random
import time


class VehicleDetector(ABC):
    """Abstract interface for vehicle detection."""

    @abstractmethod
    def detect(self, source: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect vehicles from source.

        Returns:
            {
                "vehicle_count": int,
                "vehicle_types": {"car": int, "truck": int, ...},
                "emergency_detected": bool,
                "emergency_type": str | None,
                "source": "mock" | "yolo" | "sensor",
                "reasoning": str,
                "confidence": float
            }
        """
        pass

    @abstractmethod
    def detect_lane(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Detect on a specific lane/direction."""
        pass


class MockDetector(VehicleDetector):
    """Simulated detection for development/testing."""

    def detect(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Return realistic mock detection data."""
        cars = random.randint(2, 15)
        trucks = random.randint(0, 3)
        buses = random.randint(0, 2)
        motorcycles = random.randint(0, 4)
        total = cars + trucks + buses + motorcycles

        types = {}
        if cars > 0:
            types["car"] = cars
        if trucks > 0:
            types["truck"] = trucks
        if buses > 0:
            types["bus"] = buses
        if motorcycles > 0:
            types["motorcycle"] = motorcycles

        # 5% chance of emergency vehicle
        emergency = random.random() < 0.05
        emergency_type = (
            random.choice(["ambulance", "fire truck"]) if emergency else None
        )

        reasoning = f"[MOCK] Detected {total} vehicles: " + ", ".join(
            f"{v} {k}(s)" for k, v in types.items()
        )
        if emergency:
            reasoning += f" | ⚠️ EMERGENCY: {emergency_type} detected!"

        return {
            "vehicle_count": total,
            "vehicle_types": types,
            "emergency_detected": emergency,
            "emergency_type": emergency_type,
            "source": "mock",
            "confidence": round(random.uniform(0.65, 0.95), 2),
            "reasoning": reasoning,
            "timestamp": time.time(),
        }

    def detect_lane(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Same as detect for mock."""
        return self.detect(source)


class YOLODetector(VehicleDetector):
    """Real YOLO detection (YOLOv8)."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        """Initialize YOLO model."""
        self.model = None
        self.available = False

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            self.available = True
            print(f"[YOLO] Model loaded: {model_path}")
        except ImportError:
            print("[YOLO] ultralytics not installed. Falling back to mock.")
            self.fallback = MockDetector()
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}. Using mock detections.")
            self.fallback = MockDetector()

    def detect(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Run YOLO detection on video/camera."""
        if not self.available or source is None:
            return self.fallback.detect(source)

        try:
            from pathlib import Path

            if not Path(source).exists():
                return self.fallback.detect(source)

            # Run inference (simplified)
            results = self.model(source, verbose=False, conf=0.3)

            vehicle_counts = {}
            emergency_detected = False
            emergency_type = None

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = r.names.get(cls_id, "").lower()

                    # Map to vehicle types
                    if cls_id == 2 or "car" in cls_name:
                        vehicle_counts["car"] = vehicle_counts.get("car", 0) + 1
                    elif cls_id == 7 or "truck" in cls_name:
                        vehicle_counts["truck"] = vehicle_counts.get("truck", 0) + 1
                    elif cls_id == 5 or "bus" in cls_name:
                        vehicle_counts["bus"] = vehicle_counts.get("bus", 0) + 1
                    elif cls_id == 3 or "motorcycle" in cls_name:
                        vehicle_counts["motorcycle"] = (
                            vehicle_counts.get("motorcycle", 0) + 1
                        )

                    # Check emergency
                    if "ambulance" in cls_name or "emergency" in cls_name:
                        emergency_detected = True
                        emergency_type = "ambulance"

            total = sum(vehicle_counts.values())
            reasoning = (
                f"[YOLO] Detected {total} vehicles: "
                + ", ".join(f"{v} {k}(s)" for k, v in vehicle_counts.items())
            )
            if emergency_detected:
                reasoning += f" | ⚠️ EMERGENCY: {emergency_type} detected!"

            return {
                "vehicle_count": total,
                "vehicle_types": vehicle_counts,
                "emergency_detected": emergency_detected,
                "emergency_type": emergency_type,
                "source": "yolo",
                "confidence": 0.85,  # Average confidence
                "reasoning": reasoning,
                "timestamp": time.time(),
            }

        except Exception as e:
            print(f"[YOLO] Detection error: {e}. Using mock.")
            return self.fallback.detect(source)

    def detect_lane(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Same as detect for now."""
        return self.detect(source)


def get_detector(mode: str) -> VehicleDetector:
    """
    Factory: return appropriate detector.

    Args:
        mode: "mock" or "yolo"

    Returns:
        VehicleDetector instance
    """
    if mode == "mock":
        return MockDetector()
    elif mode == "yolo":
        return YOLODetector()
    else:
        raise ValueError(f"Unknown detector mode: {mode}")
