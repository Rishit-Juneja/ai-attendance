"""
Check that the liveness model is wired correctly before trusting a single verdict.

The failure mode this exists for: MiniFASNetV2 accepts any 80x80x3 tensor and
always returns three numbers. Feed it RGB when it wants BGR, or a tight crop
when it wants 2.7x, and it still answers confidently — just wrongly. Nothing
raises. So the only way to know the wiring is right is to push real images
through it and see whether the scores land where they should.

    .venv/bin/python tools/verify_spoof_model.py                # signature + real faces
    .venv/bin/python tools/verify_spoof_model.py --image a.jpg  # score one image

A live face should score high. If real faces come out low, suspect the crop
scale or the channel order before touching any threshold.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_CONFIG
from src.antispoof import SilentFaceLiveness, CROP_SCALE, INPUT_SIZE


def signature(model_path: str):
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"model: {model_path}")
    for i in sess.get_inputs():
        print(f"  in   {i.name:12s} {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"  out  {o.name:12s} {o.shape} {o.type}")

    out_shape = sess.get_outputs()[0].shape
    classes = out_shape[-1] if len(out_shape) > 1 else None
    if classes != 3:
        print(f"\n  WARNING: expected a 3-class head [live, print, replay], got "
              f"{classes}.\n  src/antispoof.py reads index 0 as live — that "
              f"assumption no longer holds.")
    in_shape = sess.get_inputs()[0].shape
    if list(in_shape[-2:]) != [INPUT_SIZE, INPUT_SIZE]:
        print(f"\n  WARNING: expected {INPUT_SIZE}x{INPUT_SIZE} input, got "
              f"{in_shape}.")
    return sess


def score_image(liveness: SilentFaceLiveness, path: Path, detector=None) -> float | None:
    """Detect the face, then score it the way the pipeline would."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"  {path.name:28s} unreadable")
        return None
    faces = detector.detect(img) if detector else []
    if not faces:
        # No face found: score the whole image as if it were the crop, so a
        # pre-cropped face still tells us something.
        h, w = img.shape[:2]
        side = min(h, w) / CROP_SCALE
        bbox = [w / 2 - side / 2, h / 2 - side / 2, w / 2 + side / 2, h / 2 + side / 2]
        note = "(no face detected, scored centre)"
    else:
        bbox = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])).bbox
        note = ""
    s = liveness.predict(bbox, img)
    verdict = "LIVE " if s >= 0.5 else "SPOOF"
    # Full distribution, because "index 0 is live" is an assumption taken from a
    # model card, and the whole point of this script is to stop trusting it.
    patch = cv2.resize(liveness._crop(bbox, img), (INPUT_SIZE, INPUT_SIZE))
    patch = patch.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
    raw = liveness.model.run(None, {liveness._input: patch})[0].ravel()
    e = np.exp(raw - raw.max())
    p = e / e.sum()
    print(f"  {path.name:28s} liveness={s:.3f}  {verdict}  "
          f"p=[{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}] {note}")
    return s


def live_stream(liveness: SilentFaceLiveness, url: str, seconds: float):
    """
    Score faces off the camera, at the distance and resolution they will really
    be seen at. Saved crops cannot substitute: they are already tight, so the
    2.7x expansion has nothing left to expand into and the model sees a framing
    it was never trained on.
    """
    import os
    import time
    from src.antispoof import LIVENESS_MIN_FACE_PX
    from src.pipeline import ArcFaceEmbedder

    if url.startswith("rtsp"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(int(url) if url.isdigit() else url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {url}")

    det = ArcFaceEmbedder(det_size=DEFAULT_CONFIG.profile.det_size, det_thresh=0.4)
    interval = 1.0 / DEFAULT_CONFIG.profile.analysis_fps
    scores, widths, small = [], [], []
    started = last = time.time()
    print(f"\nscoring live faces for {seconds:.0f}s at {url}")
    while time.time() - started < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        if time.time() - last < interval:
            continue
        last = time.time()
        for f in det.detect(frame):
            w = f.bbox[2] - f.bbox[0]
            if w < LIVENESS_MIN_FACE_PX:
                small.append(w)
                print(f"\r  too small to judge: {len(small)} faces, "
                      f"{np.median(small):.0f}px median", end="", flush=True)
                continue
            s = liveness.predict(f.bbox, frame)
            scores.append(s)
            widths.append(w)
            print(f"\r  n={len(scores):3d}  face={w:5.1f}px  liveness={s:.3f}  "
                  f"median={np.median(scores):.3f}  too-small={len(small)}",
                  end="", flush=True)
    cap.release()
    print()
    if not scores:
        # Which of the two this is decides completely different things, so say.
        if small:
            raise SystemExit(
                f"\n{len(small)} faces seen, ALL under the {LIVENESS_MIN_FACE_PX}px "
                f"floor (median {np.median(small):.0f}px, max {max(small):.0f}px).\n"
                "Liveness cannot run at this distance — the model would be judging "
                "upscaled blur.\nThat is a camera/seating fact, not a bug: at this "
                "range spoof checking is off\nregardless of which model is loaded.")
        raise SystemExit("No face detected at all — was anyone in frame?")
    live = sum(s >= 0.5 for s in scores)
    print(f"\n  {live}/{len(scores)} read LIVE   median={np.median(scores):.3f}   "
          f"face width {min(widths):.0f}-{max(widths):.0f}px")
    print("\n  If that was a real person, anything below ~80% LIVE means real "
          "students will be\n  blocked — raise LIVENESS_MIN_FACE_PX or leave the "
          "model off.\n  If that was a photo or a screen, anything above ~20% "
          "LIVE means the spoof walks through.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", action="append", default=[],
                    help="score this image (repeatable)")
    ap.add_argument("--model", default=DEFAULT_CONFIG.spoof_model_path)
    ap.add_argument("--rtsp", nargs="?", const="rtsp://CAMERA-IP:554/stream1",
                    help="score live frames instead of files — the only test "
                         "that means anything, since saved crops have already "
                         "lost the 2.7x context the model needs")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(
            f"No model at {args.model}\n"
            "  curl -L -o data/models/minifasnet_v2.onnx \\\n"
            "    https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx"
            "/resolve/main/minifasnet_v2.onnx")

    signature(args.model)
    liveness = SilentFaceLiveness(args.model)

    if args.rtsp:
        return live_stream(liveness, args.rtsp, args.seconds)

    paths = [Path(p) for p in args.image]
    if not paths:
        # Real faces this camera and this enrolment flow have already produced.
        # All were captured from actual people, so all should read LIVE.
        from src.config import ENROLLMENT_DIR, LOGS_DIR
        paths = sorted(ENROLLMENT_DIR.glob("*.jp*g"))[:6]
        paths += sorted(LOGS_DIR.glob("*/unresolved/*.jpg"))[:6]
    if not paths:
        raise SystemExit("No images to score — pass --image.")

    detector = None
    try:
        from src.pipeline import ArcFaceEmbedder
        detector = ArcFaceEmbedder(det_size=DEFAULT_CONFIG.profile.det_size,
                                   det_thresh=0.4)
    except Exception as e:      # scoring still works, just on the centre crop
        print(f"\n(no face detector: {e})")

    print("\nscoring real captures — these are all genuine people, so anything "
          "reading SPOOF\nmeans the wiring is wrong, not that the model is strict:")
    scores = [s for p in paths if (s := score_image(liveness, p, detector)) is not None]

    if scores:
        live = sum(s >= 0.5 for s in scores)
        print(f"\n  {live}/{len(scores)} genuine faces read LIVE "
              f"(median {np.median(scores):.3f})")
        if live < len(scores) * 0.6:
            print("  Most real faces are being called spoofs. Check the channel "
                  "order (model\n  wants BGR) and CROP_SCALE before adjusting any "
                  "threshold.")


if __name__ == "__main__":
    main()
