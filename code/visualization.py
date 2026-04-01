import cv2
import config

def draw_overlay(frame, result, trail, roi_box):
    if config.DRAW_ROI:
        rx1, ry1, rx2, ry2 = roi_box
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

    pred = result["pred"]
    if config.DRAW_PRED_POINT and pred is not None:
        cv2.circle(frame, (int(pred["x"]), int(pred["y"])), 4, (255, 0, 0), -1)

    if result["accepted_det"] is not None:
        x1, y1, x2, y2 = result["accepted_det"]["xyxy"]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 2)

    dx, dy = result["draw_x"], result["draw_y"]
    if dx != -1 and dy != -1:
        if result["visible"] == 1:
            cv2.circle(frame, (int(dx), int(dy)), 6, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (int(dx), int(dy)), 5, (120, 120, 255), 1)

    for i in range(1, len(trail)):
        p1 = trail[i - 1]
        p2 = trail[i]
        if p1 is None or p2 is None:
            continue
        cv2.line(frame, p1, p2, (0, 255, 255), 2)

    if config.DRAW_DEBUG_TEXT:
        cv2.putText(frame, f"Visible: {result['visible']}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Missed: {result['missed']}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame
