import cv2
import numpy as np


class Segmenter:
    def __init__(self, rows: int = 1, cols: int = 2, method: str = "intensity_weighted"):
        self.rows = rows
        self.cols = cols
        self.method = method

    def compute_splits(self, frame: np.ndarray) -> list[int]:
        width = frame.shape[1]
        if self.method == "geometric":
            return [width * (i + 1) // self.cols for i in range(self.cols - 1)]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        col_sums = gray.sum(axis=0).astype(np.float64)
        total = col_sums.sum()
        if total == 0:
            return [width // 2]

        target = total / self.cols
        splits = []
        acc = 0.0
        for i in range(width):
            acc += col_sums[i]
            if acc >= target * (len(splits) + 1):
                splits.append(i + 1)
                if len(splits) == self.cols - 1:
                    break
        while len(splits) < self.cols - 1:
            splits.append(width - 1)
        return splits

    def assign(self, x_center: int, splits: list[int]) -> str:
        for i, split in enumerate(splits):
            if x_center < split:
                return f"Segment {i + 1}"
        return f"Segment {len(splits) + 1}"
