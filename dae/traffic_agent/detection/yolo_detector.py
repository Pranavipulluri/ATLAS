"""
YOLO Detection Service
Processes video clips using YOLOv8 to detect vehicles and emergency vehicles.
Falls back to realistic mock detections when no video clips are assigned.

Detection interval: configurable (default 60 seconds / 1 minute)
"""

import random
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


# YOLO vehicle class IDs (COCO dataset)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
EMERGENCY_CLASSES = {
    # Custom-trained or class name matching
    "ambulance": True,
    "fire truck": True,
}

# All vehicle-like COCO class IDs
ALL_VEHICLE_CLASS_IDS = [2, 3, 5, 7]


@dataclass
class DetectionResult:
    """Result from YOLO detection on a single video feed."""
    vehicle_count: int = 0
    vehicle_types: Dict[str, int] = field(default_factory=dict)
    emergency_detected: bool = False
    emergency_type: Optional[str] = None
    confidence: float = 0.0
    source: str = "mock"  # "yolo" or "mock"
    timestamp: float = field(default_factory=time.time)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_count": self.vehicle_count,
            "vehicle_types": self.vehicle_types,
            "emergency_detected": self.emergency_detected,
            "emergency_type": self.emergency_type,
            "confidence": round(self.confidence, 2),
            "source": self.source,
            "timestamp": self.timestamp,
            "reasoning": self.reasoning,
        }


class YOLODetector:
    """
    YOLOv8 vehicle detection engine.
    Uses ultralytics when available, falls back to mock data.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = None
        self.model_path = model_path
        self._available = False

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self._available = True
            print(f"[YOLO] Model loaded: {model_path}")
        except ImportError:
            print("[YOLO] ultralytics not installed. Using mock detections.")
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}. Using mock detections.")

    def detect_from_video(self, video_path: str, sample_frames: int = 5) -> DetectionResult:
        """
        Run YOLO detection on a video clip.
        Samples N frames evenly distributed through the clip.
        Returns aggregated detection result.
        """
        if not self._available or not Path(video_path).exists():
            return self._mock_detection()

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return self._mock_detection()

            frame_indices = [int(i * total_frames / sample_frames) for i in range(sample_frames)]

            all_counts: Dict[str, int] = {}
            total_conf = 0.0
            conf_count = 0
            emergency_found = False
            emergency_type = None

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                results = self.model(frame, verbose=False, conf=0.3)

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        total_conf += conf
                        conf_count += 1

                        # Check vehicle classes
                        if cls_id in VEHICLE_CLASSES:
                            vtype = VEHICLE_CLASSES[cls_id]
                            all_counts[vtype] = all_counts.get(vtype, 0) + 1

                        # Check class names for emergency vehicles
                        cls_name = r.names.get(cls_id, "").lower()
                        if cls_name in EMERGENCY_CLASSES:
                            emergency_found = True
                            emergency_type = cls_name

            cap.release()

            # Average across sampled frames
            avg_counts = {k: max(1, v // sample_frames) for k, v in all_counts.items()}
            total_vehicles = sum(avg_counts.values())
            avg_conf = total_conf / conf_count if conf_count > 0 else 0.0

            reasoning = (
                f"YOLO detected {total_vehicles} vehicles across {sample_frames} sampled frames: "
                + ", ".join(f"{v} {k}(s)" for k, v in avg_counts.items())
            )
            if emergency_found:
                reasoning += f" | ⚠️ EMERGENCY: {emergency_type} detected!"

            return DetectionResult(
                vehicle_count=total_vehicles,
                vehicle_types=avg_counts,
                emergency_detected=emergency_found,
                emergency_type=emergency_type,
                confidence=avg_conf,
                source="yolo",
                reasoning=reasoning,
            )

        except Exception as e:
            print(f"[YOLO] Detection error: {e}")
            return self._mock_detection()

    def _mock_detection(self) -> DetectionResult:
        """Generate realistic mock detection data."""
        cars = random.randint(2, 15)
        trucks = random.randint(0, 3)
        buses = random.randint(0, 2)
        motorcycles = random.randint(0, 4)
        total = cars + trucks + buses + motorcycles

        types = {}
        if cars > 0: types["car"] = cars
        if trucks > 0: types["truck"] = trucks
        if buses > 0: types["bus"] = buses
        if motorcycles > 0: types["motorcycle"] = motorcycles

        # 5% chance of emergency vehicle in mock mode
        emergency = random.random() < 0.05
        emergency_type = random.choice(["ambulance", "fire truck"]) if emergency else None

        reasoning = (
            f"[MOCK] Detected {total} vehicles: "
            + ", ".join(f"{v} {k}(s)" for k, v in types.items())
        )
        if emergency:
            reasoning += f" | ⚠️ EMERGENCY: {emergency_type} detected!"

        return DetectionResult(
            vehicle_count=total,
            vehicle_types=types,
            emergency_detected=emergency,
            emergency_type=emergency_type,
            confidence=round(random.uniform(0.65, 0.95), 2),
            source="mock",
            reasoning=reasoning,
        )


class VideoFeedManager:
    """
    Manages video feeds for all 16 lanes (4 intersections × 4 directions).
    Runs YOLO detection at configurable intervals.
    """

    def __init__(self, detection_interval: float = 60.0):
        self.detector = YOLODetector()
        self.detection_interval = detection_interval  # seconds between YOLO runs

        # Video clip assignments: {intersection_id: {lane: path}}
        self.video_clips: Dict[str, Dict[str, str]] = {
            node: {lane: "" for lane in ["North", "South", "East", "West"]}
            for node in ["A", "B", "C", "D"]
        }

        # Cached detection results
        self.cached_results: Dict[str, Dict[str, DetectionResult]] = {
            node: {lane: DetectionResult() for lane in ["North", "South", "East", "West"]}
            for node in ["A", "B", "C", "D"]
        }

        self.last_detection_time: float = 0.0

    def assign_video(self, intersection_id: str, lane: str, video_path: str) -> bool:
        """Assign a video clip to a specific intersection-lane."""
        if intersection_id in self.video_clips and lane in self.video_clips[intersection_id]:
            self.video_clips[intersection_id][lane] = video_path
            return True
        return False

    def should_run_detection(self) -> bool:
        """Check if enough time has passed for a new detection cycle."""
        return (time.time() - self.last_detection_time) >= self.detection_interval

    def run_detection_cycle(self) -> Dict[str, Dict[str, DetectionResult]]:
        """
        Run YOLO detection on all assigned video feeds.
        Uses mock data for lanes without video clips.
        """
        self.last_detection_time = time.time()

        for node_id in ["A", "B", "C", "D"]:
            for lane in ["North", "South", "East", "West"]:
                clip_path = self.video_clips[node_id][lane]
                if clip_path and Path(clip_path).exists():
                    result = self.detector.detect_from_video(clip_path)
                else:
                    result = self.detector._mock_detection()
                self.cached_results[node_id][lane] = result

        return self.cached_results

    def get_lane_data(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Convert detection results into lane data format expected by GridCoordinator.
        Returns: {node: {lane: {density, has_emergency, detection}}}
        """
        result = {}
        for node_id in ["A", "B", "C", "D"]:
            result[node_id] = {}
            for lane in ["North", "South", "East", "West"]:
                det = self.cached_results[node_id][lane]
                result[node_id][lane] = {
                    "density": det.vehicle_count,
                    "has_emergency": det.emergency_detected,
                    "detection": det.to_dict(),
                }
        return result

    def get_status(self) -> Dict[str, Any]:
        """Return status of all video feeds and detection results."""
        status = {}
        for node_id in ["A", "B", "C", "D"]:
            status[node_id] = {}
            for lane in ["North", "South", "East", "West"]:
                clip = self.video_clips[node_id][lane]
                det = self.cached_results[node_id][lane]
                status[node_id][lane] = {
                    "has_video": bool(clip),
                    "video_path": clip,
                    "last_detection": det.to_dict(),
                }
        return status
