"""
Person (body) detection via YOLO11n on ONNX Runtime.

Runs on the same onnxruntime-gpu as ArcFace — no torch at runtime. torch and
ultralytics are needed only to produce the .onnx (tools/export_person_model.py)
and can be uninstalled afterwards; one CUDA runtime instead of two also avoids
both libraries reserving separate VRAM pools.

Why bodies at all: a face-only pipeline cannot see someone who covers up or turns
around, and cannot track at a low analysis rate. Measured on the live camera, a
person facing away scored 0.06 against his own enrollment — below the impostor
floor, so no threshold can rescue it. The body carries the identity instead; the
face only has to establish it once.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DATA_DIR

MODEL_PATH = DATA_DIR / "models" / "yolo11n.onnx"
PERSON_CLASS = 0          # COCO class 0 is "person"
INPUT_SIZE = 640

# The default conf_threshold is deliberately below pipeline.BODY_SPAWN_THRESHOLD.
# Boxes in between are not junk to be filtered — they are ByteTrack's low
# confidence bucket, which continues the track of someone who has walked behind
# a desk. They can never start a new track, so a false positive there is inert.


@dataclass
class PersonBox:
    bbox: np.ndarray      # [x1, y1, x2, y2] in original frame coordinates
    score: float

    @property
    def foot_point(self) -> tuple[float, float]:
        """
        Where this person is standing: bottom-centre of the body box.

        This is the real reason to detect bodies for zone membership. The face-only
        version had to use the chin as a stand-in, which sits at head height and
        put people in the wrong zone depending on how they leaned.
        """
        return (float(self.bbox[0] + self.bbox[2]) / 2.0, float(self.bbox[3]))


def _letterbox(frame: np.ndarray, size: int = INPUT_SIZE):
    """Resize preserving aspect ratio, pad to square. Returns (image, scale, pad)."""
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


class PersonDetector:
    def __init__(self, model_path: Path = MODEL_PATH, conf_threshold: float = 0.25,
                 nms_threshold: float = 0.5, providers=None):
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run: .venv/bin/python tools/export_person_model.py")

        if providers is None:
            available = ort.get_available_providers()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.providers = self.session.get_providers()
        self.on_gpu = "CUDAExecutionProvider" in self.providers
        print(f"[YOLO person] running on {'GPU' if self.on_gpu else 'CPU'} ({self.providers[0]})")

    def detect(self, frame: np.ndarray) -> list[PersonBox]:
        img, scale, (pad_x, pad_y) = _letterbox(frame)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0

        out = self.session.run(None, {self.input_name: blob})[0]

        # YOLO11 emits (1, 84, 8400): 4 box values then 80 class scores, one column
        # per anchor. Transposed relative to YOLOv5, and with no objectness column —
        # the class score IS the confidence.
        preds = out[0].T                              # (8400, 84)
        scores = preds[:, 4 + PERSON_CLASS]
        keep = scores > self.conf_threshold
        if not keep.any():
            return []
        preds, scores = preds[keep], scores[keep]

        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        # Undo the letterbox: remove padding first, then the scale.
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        fh, fw = frame.shape[:2]
        x1, x2 = np.clip(x1, 0, fw), np.clip(x2, 0, fw)
        y1, y2 = np.clip(y1, 0, fh), np.clip(y2, 0, fh)

        boxes_wh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        idxs = cv2.dnn.NMSBoxes(boxes_wh.tolist(), scores.tolist(),
                                self.conf_threshold, self.nms_threshold)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()

        return [PersonBox(bbox=np.array([x1[i], y1[i], x2[i], y2[i]], dtype=np.float32),
                          score=float(scores[i]))
                for i in idxs]


def face_to_person(face_bbox: np.ndarray, persons: list[PersonBox]) -> int | None:
    """
    Index of the person whose box contains this face, or None.

    Scored by how much of the FACE falls inside the body box rather than IoU: a
    face is a small fraction of a body, so IoU is tiny even for a perfect match and
    would rank a wrong-but-similar-sized box higher.
    """
    if not persons:
        return None
    fx1, fy1, fx2, fy2 = face_bbox[:4]
    face_area = max((fx2 - fx1) * (fy2 - fy1), 1e-6)

    best, best_frac = None, 0.0
    for i, p in enumerate(persons):
        px1, py1, px2, py2 = p.bbox
        iw = max(0.0, min(fx2, px2) - max(fx1, px1))
        ih = max(0.0, min(fy2, py2) - max(fy1, py1))
        frac = (iw * ih) / face_area
        if frac > best_frac:
            best, best_frac = i, frac
    # Over half the face inside the body box; below that it is an overlap, not a
    # containment, and stamping a name on it would attach it to the wrong person.
    return best if best_frac >= 0.5 else None


if __name__ == "__main__":
    import sys
    import time

    photos = [f"data/test_faces/pic{i}.jpeg" for i in (1, 3, 4)]
    det = PersonDetector()

    for path in photos:
        img = cv2.imread(path)
        if img is None:
            print(f"skip {path}")
            continue
        det.detect(img)                      # warmup
        t = time.perf_counter()
        for _ in range(5):
            people = det.detect(img)
        ms = (time.perf_counter() - t) / 5 * 1000
        widths = sorted(int(p.bbox[2] - p.bbox[0]) for p in people)
        print(f"{path.split('/')[-1]:<12} {img.shape[1]}x{img.shape[0]}  "
              f"persons={len(people):<3} {ms:6.1f}ms  "
              f"body widths: min={widths[0] if widths else 0} max={widths[-1] if widths else 0}")

    # Containment must beat IoU here: a face is a few percent of a body by area.
    body = PersonBox(bbox=np.array([100, 100, 200, 400], dtype=np.float32), score=0.9)
    other = PersonBox(bbox=np.array([300, 100, 400, 400], dtype=np.float32), score=0.9)
    inside = np.array([130, 110, 170, 150], dtype=np.float32)
    outside = np.array([0, 0, 40, 40], dtype=np.float32)
    assert face_to_person(inside, [body, other]) == 0
    assert face_to_person(outside, [body, other]) is None
    assert face_to_person(inside, []) is None
    assert body.foot_point == (150.0, 400.0), body.foot_point
    print("persons.py self-check passed")
