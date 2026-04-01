import os
import cv2
import pandas as pd
from collections import deque

import config
from utils import ensure_dir, clip_box
from tracker import FixedROIBallTracker
from visualization import draw_overlay


def process_video(video_path, detector):
    name = os.path.splitext(os.path.basename(video_path))[0]

    out_video_dir = os.path.join(config.OUTPUT_ROOT, "videos")
    out_csv_dir = os.path.join(config.OUTPUT_ROOT, "annotations")
    out_debug_dir = os.path.join(config.OUTPUT_ROOT, "debug")

    ensure_dir(out_video_dir)
    ensure_dir(out_csv_dir)
    ensure_dir(out_debug_dir)

    out_video = os.path.join(out_video_dir, f"{name}.mp4")
    out_csv = os.path.join(out_csv_dir, f"{name}.csv")
    out_debug = os.path.join(out_debug_dir, f"{name}_debug.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"could not open {video_path}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if FPS <= 1:
        FPS = 30.0

    roi_x1 = int(config.ROI_LEFT * W)
    roi_y1 = int(config.ROI_TOP * H)
    roi_x2 = int(config.ROI_RIGHT * W)
    roi_y2 = int(config.ROI_BOTTOM * H)
    roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)

    tracker = FixedROIBallTracker(W, H, FPS, roi_box)
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    official_rows = []
    debug_rows = []
    trail = deque(maxlen=config.TRAIL_LEN)
    render_smooth = deque(maxlen=4)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        dets = detector.predict(crop)

        full_dets = []
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            x1 += roi_x1
            y1 += roi_y1
            x2 += roi_x1
            y2 += roi_y1

            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, W, H)
            bw = x2 - x1
            bh = y2 - y1

            if bw < 1 or bh < 1:
                continue

            full_dets.append({
                "xyxy": [x1, y1, x2, y2],
                "conf": d["conf"],
                "cls": d["cls"],
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "area": bw * bh,
                "aspect": bw / max(bh, 1e-6)
            })

        result = tracker.update(full_dets)

        if result["visible"] == 1:
            out_x = round(float(result["out_x"]), 2)
            out_y = round(float(result["out_y"]), 2)
        else:
            out_x, out_y = -1, -1

        official_rows.append({
            "frame": frame_idx,
            "x": out_x,
            "y": out_y,
            "visible": int(result["visible"])
        })

        debug_rows.append({
            "frame": frame_idx,
            "visible": int(result["visible"]),
            "x": out_x,
            "y": out_y,
            "draw_x": round(float(result["draw_x"]), 2) if result["draw_x"] != -1 else -1,
            "draw_y": round(float(result["draw_y"]), 2) if result["draw_y"] != -1 else -1,
            "num_candidates": len(full_dets),
            "score": result["score"],
            "missed": result["missed"],
            "reason": result["reason"]
        })

        if result["draw_x"] != -1 and result["draw_y"] != -1:
            render_smooth.append((result["draw_x"], result["draw_y"]))
            sx = sum(p[0] for p in render_smooth) / len(render_smooth)
            sy = sum(p[1] for p in render_smooth) / len(render_smooth)

            if result["visible"] == 1:
                trail.append((int(sx), int(sy)))
            elif config.DRAW_PREDICTED_TRAIL:
                trail.append((int(sx), int(sy)))
            else:
                trail.append(None)
        else:
            trail.append(None)

        frame = draw_overlay(frame, result, trail, roi_box)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    pd.DataFrame(official_rows).to_csv(out_csv, index=False)
    pd.DataFrame(debug_rows).to_csv(out_debug, index=False)

    print(f"{name} done")
    print(f"  video      : {out_video}")
    print(f"  annotations: {out_csv}")
    print(f"  debug      : {out_debug}")
