from ultralytics import YOLO
from utils import nms_xyxy


class BallDetector:
    def __init__(self, model_path, ball_class_id=0, conf=0.18, iou=0.45, imgsz=1280, device=None):
        self.model = YOLO(model_path)
        self.ball_class_id = ball_class_id
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def predict(self, image_bgr):
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        result = results[0]

        dets = []
        if result.boxes is None or len(result.boxes) == 0:
            return dets

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, clss):
            cls = int(cls)
            if cls != self.ball_class_id:
                continue

            x1, y1, x2, y2 = map(float, box.tolist())
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 1 or h < 1:
                continue

            dets.append({
                "xyxy": [x1, y1, x2, y2],
                "conf": float(conf),
                "cls": cls,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "area": w * h,
                "aspect": w / max(h, 1e-6)
            })

        return nms_xyxy(dets, iou_thr=0.5)
