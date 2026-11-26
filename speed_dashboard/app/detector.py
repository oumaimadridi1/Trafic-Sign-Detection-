"""Detectors: the real YOLO model, and a MOCK one used only for demos/tests."""
import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_NAMES = [
    "give way", "no entry", "no parking", "no stopping", "no trucks",
    "no vehicles carrying dangerous goods", "no vehicles carrying explosives",
    "no vehicles carrying water pollutants", "parking zone", "pay parking zone",
    "pedestrian crossing ahead", "road works", "roundabout",
    "speed limit 30", "speed limit 50", "turn right",
]


@dataclass
class Det:
    cls_id: int
    label: str
    conf: float
    x1: float  # all coordinates are normalised 0..1 (independent of image size)
    y1: float
    x2: float
    y2: float


class YoloDetector:
    is_mock = False

    def __init__(self, path, imgsz=960):
        from ultralytics import YOLO  # imported here so MOCK mode works without it

        self.model = YOLO(str(path))
        self.imgsz = imgsz
        self.names = [self.model.names[i] for i in sorted(self.model.names)]
        try:
            import torch
            self.device = "cuda:0 (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "unknown"

    def predict(self, img_bgr, conf):
        r = self.model.predict(img_bgr, conf=conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            xyxyn = r.boxes.xyxyn.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxyn, confs, clss):
                out.append(Det(int(k), self.names[int(k)], float(c), float(x1), float(y1), float(x2), float(y2)))
        out.sort(key=lambda d: -d.conf)
        return out


class MockDetector:
    """Fake detector: derives 'detections' from the image content. NOT a real model."""
    is_mock = True
    device = "mock (no GPU)"

    def __init__(self, names=None):
        self.names = list(names or DEFAULT_NAMES)

    def predict(self, img_bgr, conf):
        small = cv2.resize(img_bgr, (8, 8), interpolation=cv2.INTER_AREA)
        h = hashlib.md5(small.tobytes()).digest()
        # very dark / empty frames -> no detection (lets us test "ignore empty frames")
        if small.mean() < 12:
            return []
        k = int(small[:, :, 2].mean()) % len(self.names)       # red channel decides the class
        c = 0.55 + (h[0] / 255) * 0.44
        out = []
        if c >= conf:
            out.append(Det(k, self.names[k], c, 0.55, 0.25, 0.70, 0.50))
        if h[1] % 3 == 0:
            k2 = (k + 5) % len(self.names)
            c2 = 0.45 + (h[2] / 255) * 0.4
            if c2 >= conf:
                out.append(Det(k2, self.names[k2], c2, 0.15, 0.35, 0.27, 0.62))
        out.sort(key=lambda d: -d.conf)
        return out


def class_color_bgr(cls_id):
    """Same colour rule as the dashboard (hsl hue = id*47 mod 360)."""
    hsv = np.uint8([[[int(((cls_id * 47) % 360) / 2), 190, 235]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_detections(img, dets, wrong_ids=()):
    """Draw boxes on a copy of img. dets: list of dicts (label, conf, x1..y2 normalised, id, cls_id)."""
    out = img.copy()
    H, W = out.shape[:2]
    th = max(2, round(W / 400))
    fs = max(0.5, W / 1100)
    for d in dets:
        p1 = (int(d["x1"] * W), int(d["y1"] * H))
        p2 = (int(d["x2"] * W), int(d["y2"] * H))
        col = (60, 60, 230) if d.get("id") in wrong_ids else class_color_bgr(d["cls_id"])
        cv2.rectangle(out, p1, p2, col, th)
        text = f'{d["label"]} {d["conf"]:.2f}' + ("  (WRONG)" if d.get("id") in wrong_ids else "")
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y0 = max(p1[1] - tht - 8, 0)
        x0 = max(0, min(p1[0], W - tw - 8))   # keep the label inside the picture
        cv2.rectangle(out, (x0, y0), (x0 + tw + 8, y0 + tht + 8), col, -1)
        cv2.putText(out, text, (x0 + 4, y0 + tht + 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out
