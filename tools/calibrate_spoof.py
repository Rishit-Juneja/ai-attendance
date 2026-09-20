"""
Measure the anti-spoof motion threshold instead of guessing it.

`spoof_pixel_movement_thresh` separates a live face from a printed photo by how
much the face crop changes between analysed frames. That number is tied to the
sampling interval: it was calibrated when frames were 50 ms apart and analysis
now runs at 3 fps (333 ms), so it is stale by an unknown factor. Micro-motion
does not scale linearly with the gap, so it cannot be rescaled on paper — it has
to be measured on this camera, at this frame rate, against a real photo.

Two runs and a report:

    .venv/bin/python tools/calibrate_spoof.py --label live  --seconds 40
        ... sit in front of the camera and behave normally

    .venv/bin/python tools/calibrate_spoof.py --label spoof --seconds 40
        ... hold a printed photo (or a phone showing a face) up to the camera

    .venv/bin/python tools/calibrate_spoof.py --report

The report prints the two distributions and a recommended threshold. It also
prints the overlap, which is the part that matters: if the live floor sits below
the spoof ceiling, no threshold separates them and the heuristic needs replacing
rather than retuning (see SilentFaceLiveness in src/antispoof.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Run as a plain script (tools/ is not a package), so put the repo root on the
# path before importing src.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_CONFIG, LOGS_DIR
from src.pipeline import ArcFaceEmbedder

OUT_DIR = LOGS_DIR / "spoof_calibration"
DEFAULT_URL = "rtsp://CAMERA-IP:554/stream1"


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    areas = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areas - inter)


def collect(url: str, label: str, seconds: float, padding: float = 0.2) -> Path:
    """Sample face-crop motion at the configured analysis rate."""
    interval = 1.0 / DEFAULT_CONFIG.profile.analysis_fps
    if url.startswith("rtsp"):
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cap = cv2.VideoCapture(int(url) if url.isdigit() else url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {url}")

    embedder = ArcFaceEmbedder(
        det_size=DEFAULT_CONFIG.profile.det_size,
        det_thresh=0.4,
    )
    print(f"[{label}] sampling every {interval * 1000:.0f} ms for {seconds:.0f}s — "
          f"keep ONE face in shot")

    motions: list[float] = []
    prev_crop = None
    prev_bbox = None
    switches = 0
    started = time.time()
    last = 0.0
    while time.time() - started < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()
        if now - last < interval:
            continue
        last = now

        faces = embedder.detect(frame)
        if len(faces) != 1:
            # Two faces means two motion sources in one number. Skip rather than
            # quietly average a real person together with the photo they hold.
            continue

        bbox = faces[0].bbox.astype(int)
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        px, py = int((x2 - x1) * padding), int((y2 - y1) * padding)
        crop = frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
        if crop.size == 0:
            continue
        # Same maths as SpoofChecker.check, deliberately: calibrating anything
        # else would produce a number that does not transfer.
        crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64))

        # One face per frame is not enough to guarantee it is the SAME face. With
        # a real person and a photo both in shot, the detector finds one or the
        # other from frame to frame and this loop happily diffs a face against a
        # photograph — which reads as enormous "motion" and silently poisons the
        # distribution. Measured on this camera: alternating samples of 0.04 and
        # 21.94, which is not a thing a human body does in 333 ms. Require the
        # box to overlap the previous one, so a subject switch drops the sample
        # instead of becoming one.
        if prev_bbox is not None and _iou(bbox, prev_bbox) < 0.3:
            switches += 1
            prev_crop, prev_bbox = crop, bbox
            continue

        if prev_crop is not None:
            motions.append(float(np.mean(cv2.absdiff(crop, prev_crop))))
            print(f"\r  samples={len(motions):4d}  last={motions[-1]:6.2f}  "
                  f"mean={np.mean(motions):6.2f}  switches={switches:3d}",
                  end="", flush=True)
        prev_crop, prev_bbox = crop, bbox

    cap.release()
    print()
    if len(motions) < 10:
        raise SystemExit(f"Only {len(motions)} samples — was a single face in shot?")
    if switches > len(motions) * 0.2:
        raise SystemExit(
            f"Rejected: {switches} subject switches against {len(motions)} usable "
            f"samples.\nThe detector kept finding a DIFFERENT face between samples, "
            f"so more than one\nface was in shot — a photo on a screen behind you "
            f"counts. Clear the frame\nof every face except the one you are "
            f"measuring and run this again.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{label}.json"
    path.write_text(json.dumps({
        "label": label,
        "url": url,
        "analysis_fps": DEFAULT_CONFIG.profile.analysis_fps,
        "interval_ms": round(interval * 1000),
        "switches": switches,
        "motions": motions,
    }, indent=2))
    print(f"[{label}] {len(motions)} samples → {path}")
    return path


def report():
    live_path, spoof_path = OUT_DIR / "live.json", OUT_DIR / "spoof.json"
    missing = [p.name for p in (live_path, spoof_path) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} — run both collections first.")

    live = np.array(json.loads(live_path.read_text())["motions"])
    spoof = np.array(json.loads(spoof_path.read_text())["motions"])

    def describe(name, a):
        print(f"  {name:6s} n={len(a):4d}  min={a.min():6.2f}  p5={np.percentile(a, 5):6.2f}  "
              f"median={np.median(a):6.2f}  p95={np.percentile(a, 95):6.2f}  max={a.max():6.2f}")

    print(f"\nmotion between samples ({json.loads(live_path.read_text())['interval_ms']} ms apart)")
    describe("live", live)
    describe("spoof", spoof)

    # The decision boundary that matters is the live FLOOR against the spoof
    # CEILING — the tails, not the means. A threshold set from the means flags
    # every live person who briefly holds still.
    live_floor, spoof_ceiling = np.percentile(live, 5), np.percentile(spoof, 95)
    print(f"\n  live 5th percentile : {live_floor:6.2f}   <- flagging starts below this")
    print(f"  spoof 95th percentile: {spoof_ceiling:6.2f}   <- spoofs must stay under this")

    if live_floor <= spoof_ceiling:
        print("\n  OVERLAP. No threshold separates these two. The motion heuristic "
              "cannot do this job on this camera —\n  use a real liveness model "
              "(SilentFaceLiveness in src/antispoof.py) rather than retuning.")
        return

    recommended = (live_floor + spoof_ceiling) / 2
    print(f"\n  separation: {live_floor - spoof_ceiling:.2f}")
    print(f"  spoof_pixel_movement_thresh = {recommended:.2f}   (currently "
          f"{DEFAULT_CONFIG.spoof_pixel_movement_thresh})")
    print("  Set it in src/config.py. Re-run this if analysis_fps changes — the "
          "number is only\n  valid for the interval it was measured at.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL, help="stream URL or webcam index")
    ap.add_argument("--label", choices=["live", "spoof"], help="which run this is")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        report()
    elif args.label:
        collect(args.url, args.label, args.seconds)
        other = "spoof" if args.label == "live" else "live"
        if (OUT_DIR / f"{other}.json").exists():
            report()
        else:
            print(f"\nNow run the '{other}' pass, then --report.")
    else:
        ap.error("pass --label live / --label spoof, or --report")


if __name__ == "__main__":
    main()
