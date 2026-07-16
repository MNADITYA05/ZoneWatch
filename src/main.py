import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

from src.config import Config
from src.detector import Detector
from src.segmenter import Segmenter
from src.tracker import Tracker
from src.analyzer import Analyzer
from src import visualizer


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Area mapping and segmentation using YOLOv5")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--model", type=str, help="Path to model weights")
    parser.add_argument("--confidence", type=float, help="Detection confidence threshold")
    parser.add_argument("--output-csv", type=str, help="Path to output CSV")
    parser.add_argument("--output-video", type=str, help="Path to output annotated video")
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    parser.add_argument("--method", type=str, choices=["geometric", "intensity_weighted"], help="Segmentation method")
    args = parser.parse_args()

    cfg = Config()
    if args.config:
        cfg = Config.from_json(args.config)
    if args.video:
        cfg.video_path = args.video
    if args.model:
        cfg.model_path = args.model
    if args.confidence is not None:
        cfg.confidence = args.confidence
    if args.output_csv:
        cfg.output_csv = args.output_csv
    if args.output_video:
        cfg.output_video = args.output_video
    if args.no_display:
        cfg.display = False
    if args.method:
        cfg.segmentation_method = args.method
    return cfg


def main():
    cfg = parse_args()

    if not os.path.exists(cfg.video_path):
        print(f"Error: video not found at {cfg.video_path}", file=sys.stderr)
        sys.exit(1)

    detector = Detector(cfg.model_path, cfg.confidence, cfg.target_classes)
    segmenter = Segmenter(cfg.grid_rows, cfg.grid_cols, cfg.segmentation_method)
    tracker = Tracker(cfg.iou_threshold, cfg.max_missed_frames)
    analyzer = Analyzer()

    cap = cv2.VideoCapture(cfg.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    pts = np.array(cfg.roi, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(roi_mask, [pts], 255)
    room_area_px = cv2.countNonZero(roi_mask)

    video_writer = None
    if cfg.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(cfg.output_video, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        detections = detector.detect(frame)
        splits = segmenter.compute_splits(frame)

        for det in detections:
            x_center = (det.x1 + det.x2) // 2
            det.segment = segmenter.assign(x_center, splits)

        tracks = tracker.update(detections, frame_number)
        analyzer.update(tracks)

        if cfg.display or video_writer:
            annotated = frame.copy()
            annotated = visualizer.draw_detections(annotated, detections, splits)
            annotated = visualizer.draw_roi(annotated, [(pts[i][0][0], pts[i][0][1]) for i in range(len(pts))])

            if cfg.display:
                cv2.imshow("ZoneWatch", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if video_writer:
                video_writer.write(annotated)

        if frame_number % 100 == 0:
            print(f"Processed {frame_number}/{total_frames} frames", end="\r")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    results = analyzer.results(fps, room_area_px, total_frames)

    os.makedirs(os.path.dirname(cfg.output_csv) or ".", exist_ok=True)
    with open(cfg.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "detection_frames", "detection_seconds", "room_area_px"])
        for seg, data in results["segments"].items():
            writer.writerow([seg, data["frame_count"], data["total_detection_seconds"], room_area_px])

    print(f"\nResults saved to {cfg.output_csv}")
    print(f"Room area: {room_area_px} px\u00b2")
    for seg, data in results["segments"].items():
        print(f"  {seg}: {data['total_detection_seconds']}s ({data['frame_count']} frames)")
    print(f"  Unique persons tracked: {len(results['persons'])}")


if __name__ == "__main__":
    main()
