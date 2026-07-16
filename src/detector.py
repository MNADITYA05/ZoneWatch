import torch

from src.types import Detection


class Detector:
    def __init__(self, model_path: str | None = None, confidence: float = 0.5, target_classes: list[int] | None = None):
        if model_path:
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
        else:
            self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        self.model.conf = confidence
        self.target_classes = target_classes or [0]

    def detect(self, frame) -> list[Detection]:
        results = self.model(frame)
        detections = []
        for *box, conf, cls in results.pred[0]:
            if int(cls) in self.target_classes:
                x1, y1, x2, y2 = map(int, box)
                detections.append(Detection(x1, y1, x2, y2, float(conf), int(cls)))
        return detections
