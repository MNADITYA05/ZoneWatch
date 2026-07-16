import cv2
import numpy as np

from src.types import Detection


def draw_detections(frame: np.ndarray, detections: list[Detection], splits: list[int]) -> np.ndarray:
    for det in detections:
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)

    h, w = frame.shape[:2]
    for split_x in splits:
        cv2.line(frame, (split_x, 0), (split_x, h), (255, 0, 0), 2)

    prev = 0
    labels = [f"Segment {i+1}" for i in range(len(splits) + 1)]
    for i, split_x in enumerate(splits + [w]):
        cx = (prev + split_x) // 2
        cv2.putText(frame, labels[i], (cx - 60, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        prev = split_x

    return frame


def draw_roi(frame: np.ndarray, roi_points: list[tuple[int, int]]) -> np.ndarray:
    pts = np.array(roi_points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
    return frame
