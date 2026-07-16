import numpy as np
from src.segmenter import Segmenter


def test_geometric_split():
    seg = Segmenter(rows=1, cols=2, method="geometric")
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    splits = seg.compute_splits(frame)
    assert splits == [100], f"Expected [100], got {splits}"


def test_geometric_three_cols():
    seg = Segmenter(rows=1, cols=3, method="geometric")
    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    splits = seg.compute_splits(frame)
    assert splits == [100, 200], f"Expected [100, 200], got {splits}"


def test_intensity_split_uniform():
    seg = Segmenter(method="intensity_weighted")
    frame = np.ones((100, 200, 3), dtype=np.uint8) * 128
    splits = seg.compute_splits(frame)
    assert len(splits) == 1
    assert splits[0] == 100, f"Expected 100, got {splits[0]}"


def test_intensity_split_skewed():
    seg = Segmenter(method="intensity_weighted")
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :120] = [255, 255, 255]
    splits = seg.compute_splits(frame)
    assert splits[0] == 60, f"Expected 60 (intensity midpoint), got {splits[0]}"


def test_assign():
    seg = Segmenter()
    assert seg.assign(50, [100]) == "Segment 1"
    assert seg.assign(100, [100]) == "Segment 2"
    assert seg.assign(150, [100]) == "Segment 2"


def test_assign_three_segments():
    seg = Segmenter()
    assert seg.assign(30, [100, 200]) == "Segment 1"
    assert seg.assign(150, [100, 200]) == "Segment 2"
    assert seg.assign(250, [100, 200]) == "Segment 3"


def test_grayscale_input():
    seg = Segmenter(method="intensity_weighted")
    frame = np.ones((50, 100), dtype=np.uint8) * 200
    splits = seg.compute_splits(frame)
    assert splits == [50]
