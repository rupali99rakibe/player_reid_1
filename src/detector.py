# src/detector.py
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_players(self, frame):
        results = self.model(frame)
        detections = []

        for r in results[0].boxes:
            cls = int(r.cls[0])
            if cls == 0:  # assuming 0 = player, 1 = ball
                bbox = r.xyxy[0].tolist()
                conf = float(r.conf[0])
                detections.append((bbox, conf))

        return detections