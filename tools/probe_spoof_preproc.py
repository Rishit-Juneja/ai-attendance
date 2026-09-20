"""
Work out what MiniFASNetV2 actually wants, by trying the combinations.

Scoring every genuine face at p=[0.000 0.005 0.995] — near-identical across a
close-up portrait and a blurry CCTV torso — looks less like a strict model than
one that is ignoring its input. This script checks that first (does the output
move at all when the image changes?), then sweeps the preprocessing choices the
model card left ambiguous: /255 vs raw 0-255, BGR vs RGB, and the crop scale.

    .venv/bin/python tools/probe_spoof_preproc.py

Whichever combination makes real faces separate from noise is the right one.
If none do, the export is bad and the model card is not worth arguing with.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_CONFIG, ENROLLMENT_DIR, LOGS_DIR

MODEL = DEFAULT_CONFIG.spoof_model_path


def run(sess, name, img, scale_255: bool, rgb: bool):
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if rgb else img
    x = x.astype(np.float32)
    if scale_255:
        x /= 255.0
    x = x.transpose(2, 0, 1)[np.newaxis]
    out = sess.run(None, {name: x})[0].ravel()
    e = np.exp(out - out.max())
    return e / e.sum()


def main():
    if not Path(MODEL).exists():
        raise SystemExit(f"No model at {MODEL}")
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    faces = sorted(ENROLLMENT_DIR.glob("*.jp*g"))[:3]
    faces += sorted(LOGS_DIR.glob("*/unresolved/person2.jpg"))[:1]
    if not faces:
        raise SystemExit("No face images found to probe with.")

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    black = np.zeros((80, 80, 3), np.uint8)
    white = np.full((80, 80, 3), 255, np.uint8)

    print("Does the output move at all?  (/255, BGR — what the code does now)\n")
    for label, img in [("random noise", noise), ("black", black), ("white", white)]:
        p = run(sess, name, img, True, False)
        print(f"  {label:14s} p=[{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]")
    for f in faces:
        img = cv2.resize(cv2.imread(str(f)), (80, 80))
        p = run(sess, name, img, True, False)
        print(f"  {f.name:14.14s} p=[{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]")

    print("\nPreprocessing sweep — spread = how much the 3 outputs differ between")
    print("a real face and random noise. A dead combination scores ~0.\n")
    best = []
    for scale_255 in (True, False):
        for rgb in (False, True):
            face_ps = [run(sess, name, cv2.resize(cv2.imread(str(f)), (80, 80)),
                           scale_255, rgb) for f in faces]
            noise_p = run(sess, name, noise, scale_255, rgb)
            face_mean = np.mean(face_ps, axis=0)
            spread = float(np.abs(face_mean - noise_p).sum())
            tag = f"{'/255' if scale_255 else 'raw '} {'RGB' if rgb else 'BGR'}"
            print(f"  {tag}  face=[{face_mean[0]:.3f} {face_mean[1]:.3f} "
                  f"{face_mean[2]:.3f}]  noise=[{noise_p[0]:.3f} {noise_p[1]:.3f} "
                  f"{noise_p[2]:.3f}]  spread={spread:.3f}")
            best.append((spread, tag, face_mean))

    spread, tag, face_mean = max(best)
    print(f"\n  most responsive: {tag} (spread {spread:.3f})")
    if spread < 0.1:
        print("  Nothing separates a face from noise under any combination. The "
              "export is not\n  usable — the model is returning a constant.")
    else:
        print(f"  real faces land on class {int(np.argmax(face_mean))} under that "
              f"combination.")


if __name__ == "__main__":
    main()
