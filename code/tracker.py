import math
import numpy as np
from collections import deque

from kalman import BallKalman
import config


class FixedROIBallTracker:
    def __init__(self, frame_w, frame_h, fps, roi_box):
        self.w = frame_w
        self.h = frame_h
        self.fps = fps if fps and fps > 1 else 30.0
        self.roi_box = roi_box

        ref_w, ref_h, ref_fps = 1920.0, 1080.0, 30.0
        res_scale = math.sqrt((frame_w * frame_h) / (ref_w * ref_h))
        fps_scale = ref_fps / self.fps

        self.base_gate = config.BASE_GATE_PX * res_scale * fps_scale
        self.max_gate = config.MAX_GATE_PX * res_scale * fps_scale
        self.gate_expand_per_miss = config.GATE_EXPAND_PER_MISS * res_scale

        area_scale = (frame_w * frame_h) / (ref_w * ref_h)
        self.min_area = max(4, int(16 * area_scale))
        self.max_area = max(self.min_area + 10, int(1400 * area_scale))

        self.kf = BallKalman(dt=1.0)
        self.missed = 0
        self.measure_hist = deque(maxlen=config.MEASURE_HIST_LEN)
        self.vel_hist = deque(maxlen=config.VELOCITY_HIST_LEN)
        self.tracked_frames = 0

    def reset(self):
        self.kf.reset()
        self.missed = 0
        self.measure_hist.clear()
        self.vel_hist.clear()
        self.tracked_frames = 0

    def _median_velocity(self):
        if not self.vel_hist:
            return 0.0, 0.0
        arr = np.array(self.vel_hist, dtype=np.float32)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

    def predict(self):
        kpred = self.kf.predict() if self.kf.initialized else None

        if len(self.measure_hist) >= 2:
            last_x, last_y = self.measure_hist[-1]
            mvx, mvy = self._median_velocity()
            inertial_x = last_x + mvx
            inertial_y = last_y + mvy

            if kpred is None:
                return {"x": inertial_x, "y": inertial_y, "vx": mvx, "vy": mvy, "source": "inertial"}

            bx = 0.65 * kpred[0] + 0.35 * inertial_x
            by = 0.65 * kpred[1] + 0.35 * inertial_y
            bvx = 0.65 * kpred[2] + 0.35 * mvx
            bvy = 0.65 * kpred[3] + 0.35 * mvy
            return {"x": bx, "y": by, "vx": bvx, "vy": bvy, "source": "blend"}

        if kpred is not None:
            return {"x": kpred[0], "y": kpred[1], "vx": kpred[2], "vy": kpred[3], "source": "kalman"}

        return None

    def _adaptive_gate(self, pred):
        if pred is None:
            return self.max_gate
        speed = math.hypot(pred["vx"], pred["vy"])
        gate = self.base_gate + 0.55 * speed + self.missed * self.gate_expand_per_miss
        return min(self.max_gate, gate)

    def _size_score(self, area):
        if area < self.min_area:
            return -2.0 * (self.min_area - area) / max(self.min_area, 1.0)
        if area > self.max_area:
            return -2.0 * (area - self.max_area) / max(self.max_area, 1.0)

        mid = (self.min_area + self.max_area) / 2.0
        spread = max((self.max_area - self.min_area) / 2.0, 1.0)
        return 1.0 - abs(area - mid) / spread

    def _direction_consistency(self, cx, cy):
        if len(self.measure_hist) < 2:
            return 0.0

        x1, y1 = self.measure_hist[-2]
        x2, y2 = self.measure_hist[-1]

        vx1, vy1 = x2 - x1, y2 - y1
        vx2, vy2 = cx - x2, cy - y2

        n1 = math.hypot(vx1, vy1)
        n2 = math.hypot(vx2, vy2)

        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0

        cosang = (vx1 * vx2 + vy1 * vy2) / (n1 * n2)
        return max(-1.0, min(1.0, cosang))

    def _dynamic_conf_ok(self, conf, area):
        thresh = config.CONF
        if area < 90:
            thresh -= 0.04
        if self.missed > 0:
            thresh -= 0.02
        thresh = max(0.05, thresh)
        return conf >= thresh

    def _score_candidate(self, det, pred):
        cx, cy = det["cx"], det["cy"]
        conf = det["conf"]
        area = det["area"]
        aspect = det["aspect"]

        if not self._dynamic_conf_ok(conf, area):
            return -1e9, {"reason": "low_conf"}

        score = 0.0
        score += 4.6 * conf
        score += 1.5 * self._size_score(area)
        score -= 0.8 * abs(math.log(max(aspect, 1e-6)))

        dbg = {"reason": "", "dist": None, "gate": None}

        if pred is not None:
            px, py = pred["x"], pred["y"]
            pvx, pvy = pred["vx"], pred["vy"]

            dist = math.hypot(cx - px, cy - py)
            gate = self._adaptive_gate(pred)

            dbg["dist"] = dist
            dbg["gate"] = gate

            if dist > gate:
                dbg["reason"] = f"outside_gate_{dist:.1f}>{gate:.1f}"
                return -1e9, dbg

            score -= 0.030 * dist
            score -= 0.013 * math.hypot(cx - (px + pvx), cy - (py + pvy))
            score += 0.45 * self._direction_consistency(cx, cy)

            if self.missed > 0 and dist < 0.45 * gate:
                score += 0.35

        return score, dbg

    def update(self, detections):
        pred = self.predict()

        best_det = None
        best_score = -1e9
        best_dbg = {"reason": "no_candidates", "dist": None, "gate": None}

        for det in detections:
            score, dbg = self._score_candidate(det, pred)
            if score > best_score:
                best_score = score
                best_det = det
                best_dbg = dbg

        if best_det is not None and best_score >= config.ACCEPT_SCORE:
            mx, my = best_det["cx"], best_det["cy"]

            if self.measure_hist:
                last_x, last_y = self.measure_hist[-1]
                self.vel_hist.append((mx - last_x, my - last_y))

            self.measure_hist.append((mx, my))

            if not self.kf.initialized:
                self.kf.init(mx, my)
            else:
                self.kf.correct(mx, my)

            self.missed = 0
            self.tracked_frames += 1

            return {
                "visible": 1,
                "out_x": mx,
                "out_y": my,
                "draw_x": mx,
                "draw_y": my,
                "accepted_det": best_det,
                "pred": pred,
                "score": best_score,
                "missed": self.missed,
                "reason": best_dbg["reason"]
            }

        self.missed += 1

        if self.kf.initialized and self.missed <= config.MAX_MISSED:
            state = self.kf.get_state()
            draw_x, draw_y = state[0], state[1]
        else:
            draw_x, draw_y = -1, -1

        if self.missed > config.MAX_MISSED:
            self.reset()

        return {
            "visible": 0,
            "out_x": -1,
            "out_y": -1,
            "draw_x": draw_x,
            "draw_y": draw_y,
            "accepted_det": None,
            "pred": pred,
            "score": best_score,
            "missed": self.missed,
            "reason": best_dbg["reason"]
        }
