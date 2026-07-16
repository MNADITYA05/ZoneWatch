from dataclasses import dataclass, field

from src.types import Detection


@dataclass
class Track:
    id: int
    bbox: tuple = (0, 0, 0, 0)
    segment: str = ""
    start_frame: int = 0
    end_frame: int = 0
    missed: int = 0


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 30):
        self.tracks: list[Track] = []
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union else 0

    def update(self, detections: list[Detection], frame_number: int) -> dict[int, Track]:
        used = set()
        matched = []

        for det in detections:
            bbox = (det.x1, det.y1, det.x2, det.y2)
            best_track = None
            best_iou = self.iou_threshold

            for t in self.tracks:
                if t.id in used or t.missed:
                    continue
                iou = self._iou(bbox, t.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = t

            if best_track is not None:
                best_track.bbox = bbox
                best_track.end_frame = frame_number
                best_track.segment = det.segment
                best_track.missed = 0
                used.add(best_track.id)
                matched.append(best_track)
            else:
                t = Track(self.next_id, bbox, det.segment, frame_number, frame_number)
                self.tracks.append(t)
                self.next_id += 1
                matched.append(t)
                used.add(t.id)

        for t in self.tracks:
            if t.id not in used:
                t.missed += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return {t.id: t for t in self.tracks if not t.missed}
