import os
import glob


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def gather_videos(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    exts = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV")
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(input_path, ext)))
    return sorted(videos)


def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    return x1, y1, x2, y2


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def nms_xyxy(dets, iou_thr=0.5):
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    kept = []

    while dets:
        best = dets.pop(0)
        kept.append(best)
        remain = [d for d in dets if iou_xyxy(best["xyxy"], d["xyxy"]) < iou_thr]
        dets = remain

    return kept
