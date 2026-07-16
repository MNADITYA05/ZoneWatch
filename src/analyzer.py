from collections import defaultdict

from src.tracker import Track


class Analyzer:
    def __init__(self):
        self.segment_frames: dict[str, int] = defaultdict(int)
        self.person_segment_frames: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def update(self, tracks: dict[int, Track]):
        for tid, track in tracks.items():
            seg = track.segment or "Unknown"
            self.segment_frames[seg] += 1
            self.person_segment_frames[tid][seg] += 1

    def results(self, fps: float, room_area_px: float, total_frames: int) -> dict:
        segments = {
            seg: {"total_detection_seconds": round(frames / fps, 2), "frame_count": frames}
            for seg, frames in sorted(self.segment_frames.items())
        }
        persons = {
            str(tid): dict(dict(seg_counts))
            for tid, seg_counts in self.person_segment_frames.items()
        }
        return {
            "total_frames": total_frames,
            "fps": fps,
            "room_area_px": room_area_px,
            "segments": segments,
            "persons": persons,
        }
