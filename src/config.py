import json
from dataclasses import dataclass, field


@dataclass
class Config:
    video_path: str = "3452304387-preview.mp4"
    model_path: str = "yolov5s.pt"
    confidence: float = 0.5
    target_classes: list = field(default_factory=lambda: [0])
    grid_rows: int = 1
    grid_cols: int = 2
    segmentation_method: str = "intensity_weighted"
    roi: list = field(default_factory=lambda: [100, 100, 500, 400])
    iou_threshold: float = 0.3
    max_missed_frames: int = 30
    output_csv: str = "output/metrics.csv"
    output_video: str = ""
    display: bool = True

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))
