from src.tracker import Tracker
from src.types import Detection


def make_det(x1, y1, x2, y2, seg=""):
    return Detection(x1, y1, x2, y2, confidence=0.9, class_id=0, segment=seg)


def test_new_track_created():
    tracker = Tracker(iou_threshold=0.3)
    dets = [make_det(0, 0, 50, 50)]
    tracks = tracker.update(dets, 1)
    assert len(tracks) == 1


def test_iou_match():
    tracker = Tracker(iou_threshold=0.3)
    dets1 = [make_det(0, 0, 50, 50)]
    tracks1 = tracker.update(dets1, 1)
    tid = list(tracks1.keys())[0]

    dets2 = [make_det(5, 5, 55, 55)]
    tracks2 = tracker.update(dets2, 2)
    assert tid in tracks2, "Same person should match via IoU"


def test_no_match_low_iou():
    tracker = Tracker(iou_threshold=0.3)
    dets1 = [make_det(0, 0, 50, 50)]
    tracker.update(dets1, 1)

    dets2 = [make_det(200, 200, 250, 250)]
    tracks2 = tracker.update(dets2, 2)
    assert len(tracker.tracks) == 2
    assert len(tracks2) == 1


def test_segment_carries_through():
    tracker = Tracker(iou_threshold=0.3)
    dets1 = [make_det(0, 0, 50, 50, seg="Segment 1")]
    tracks1 = tracker.update(dets1, 1)
    tid = list(tracks1.keys())[0]
    assert tracks1[tid].segment == "Segment 1"

    dets2 = [make_det(5, 5, 55, 55, seg="Segment 1")]
    tracks2 = tracker.update(dets2, 2)
    assert tracks2[tid].segment == "Segment 1"


def test_missed_frames_removed():
    tracker = Tracker(iou_threshold=0.3, max_missed=2)
    dets1 = [make_det(0, 0, 50, 50)]
    tracker.update(dets1, 1)

    tracker.update([], 2)
    tracker.update([], 3)
    tracks4 = tracker.update([], 4)
    assert len(tracks4) == 0
