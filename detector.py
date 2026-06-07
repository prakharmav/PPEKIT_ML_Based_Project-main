"""
SafeGuard AI — PPE Detection Engine
Wraps the YOLOv8 model with violation logic, visual annotation,
and incident screenshot capture.
"""

import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Violation / Safe class sets  (lower-cased, matched against model.names)
# ---------------------------------------------------------------------------
VIOLATION_KEYWORDS = {
    "no-helmet", "no_helmet", "no helmet",
    "no-hardhat", "no_hardhat", "no hardhat",
    "no-vest",   "no_vest",   "no vest",
    "no-safety-vest", "no safety vest",
    "no-jacket", "no_jacket", "no jacket",
    "without-helmet", "without-vest",
}

# Colors (BGR)
COL_VIOLATION = (82,  82,  255)   # neon red (#FF5252)
COL_SAFE      = (0,   230, 118)   # neon green (#00E676)
COL_NEUTRAL   = (7,   193, 255)   # amber (#FFC107)


class PPEDetector:
    def __init__(self, model_path: str = "best.pt", conf: float = 0.4):
        self.model      = YOLO(model_path)
        self.conf       = conf
        self.class_names: dict = self.model.names
        os.makedirs("incidents", exist_ok=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _is_violation(self, name: str) -> bool:
        return name.lower() in VIOLATION_KEYWORDS

    def _is_person(self, name: str) -> bool:
        return "person" in name.lower()

    def _pretty_name(self, name: str) -> str:
        name_lower = name.lower()
        if "no-helmet" in name_lower or "no_helmet" in name_lower or "no helmet" in name_lower or "without-helmet" in name_lower:
            return "NO HELMET"
        if "no-vest" in name_lower or "no_vest" in name_lower or "no vest" in name_lower or "no safety vest" in name_lower or "without-vest" in name_lower:
            return "NO VEST"
        if "helmet" in name_lower or "hardhat" in name_lower:
            return "Helmet"
        if "vest" in name_lower or "jacket" in name_lower:
            return "Vest"
        return name.replace("-", " ").replace("_", " ").title()

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> dict:
        """
        Run inference on one frame.
        Returns dict with annotated frame + per-frame statistics.
        """
        results = self.model(frame, conf=self.conf, verbose=False)

        detections      = []
        violation_count = 0
        person_count    = 0
        violation_types = []

        for r in results:
            for box in r.boxes:
                cls_id  = int(box.cls[0])
                name    = self.class_names[cls_id]
                conf_   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                is_viol = self._is_violation(name)
                is_pers = self._is_person(name)

                if is_viol:
                    violation_count += 1
                    violation_types.append(self._pretty_name(name))
                    color = COL_VIOLATION
                elif is_pers:
                    person_count += 1
                    color = COL_SAFE
                else:
                    # check if the class itself is helmet or vest to color green
                    name_lower = name.lower()
                    if "helmet" in name_lower or "vest" in name_lower or "hardhat" in name_lower or "jacket" in name_lower:
                        color = COL_SAFE
                    else:
                        color = COL_NEUTRAL

                detections.append({
                    "class":        name,
                    "confidence":   conf_,
                    "bbox":         (x1, y1, x2, y2),
                    "is_violation": is_viol,
                    "color":        color,
                })

        annotated = self._annotate(frame.copy(), detections)

        # Estimate worker count: at minimum the violation boxes are people
        worker_count = max(person_count, violation_count)

        return {
            "frame":           annotated,
            "raw_frame":       frame,
            "detections":      detections,
            "violation_count": violation_count,
            "worker_count":    worker_count,
            "violation_types": list(set(violation_types)),
            "has_violation":   violation_count > 0,
            "compliance":      (
                round((worker_count - violation_count) / worker_count * 100, 1)
                if worker_count > 0 else 100.0
            ),
        }

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------
    def _annotate(self, frame: np.ndarray, detections: list) -> np.ndarray:
        import time
        h, w = frame.shape[:2]

        # Draw small neon L-shaped corner brackets (cybersecurity HUD ticks)
        length = 15
        color_hud = (255, 194, 0)  # Electric Blue/Cyan BGR (#00C2FF)
        
        # Top-left corner
        cv2.line(frame, (10, 10), (10 + length, 10), color_hud, 1)
        cv2.line(frame, (10, 10), (10, 10 + length), color_hud, 1)
        # Top-right corner
        cv2.line(frame, (w - 10, 10), (w - 10 - length, 10), color_hud, 1)
        cv2.line(frame, (w - 10, 10), (w - 10, 10 + length), color_hud, 1)
        # Bottom-left corner
        cv2.line(frame, (10, h - 10), (10 + length, h - 10), color_hud, 1)
        cv2.line(frame, (10, h - 10), (10, h - 10 - length), color_hud, 1)
        # Bottom-right corner
        cv2.line(frame, (w - 10, h - 10), (w - 10 - length, h - 10), color_hud, 1)
        cv2.line(frame, (w - 10, h - 10), (w - 10, h - 10 - length), color_hud, 1)

        # Draw Camera Name Overlay in top-left
        cv2.putText(frame, "CAM-01 | FACTORY FLOOR", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 193, 7), 1, cv2.LINE_AA) # Warning Amber

        # Draw Blinking REC indicator in top-right
        rec_on = (int(time.time() * 2) % 2 == 0)
        if rec_on:
            cv2.circle(frame, (w - 75, 20), 5, (82, 82, 255), -1) # Danger Red BGR (#FF4D4F)
            cv2.putText(frame, "REC", (w - 63, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (82, 82, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (w - 75, 20), 5, (40, 40, 100), -1)
            cv2.putText(frame, "REC", (w - 63, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 150), 1, cv2.LINE_AA)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color     = det["color"]
            thickness = 2
            label     = self._pretty_name(det["class"])

            # bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # label text size & drawing
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            # draw small fill box above bounding box for label
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # timestamp in bottom-left
        ts = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
        cv2.putText(frame, ts, (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------
    def save_screenshot(self, frame: np.ndarray) -> str:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = f"incidents/incident_{ts}.jpg"
        cv2.imwrite(path, frame)
        return path

    # ------------------------------------------------------------------
    # Static image helper
    # ------------------------------------------------------------------
    def detect_pil(self, pil_image: Image.Image) -> dict:
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        result = self.detect(frame)
        result["frame_rgb"] = cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB)
        return result
