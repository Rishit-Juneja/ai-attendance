"""
Measure what det_size actually buys, in faces you can identify per millisecond.

§16 reasons that SCRFD at 640x640 scales a 1920-wide frame by 3x, so a 34px
face reaches the detector as 11px and is "simply not found" — and concludes
det_size=(1280,1280) is the first thing to change. That is an inference, not a
measurement, and on this project's own crowd photos it does not hold up.

The number that matters is not total detections. It is detections wide enough
to identify. §16 measured genuine-vs-impostor overlap below ~25px, so a face
under that is guesswork no matter how confidently it was detected — raising
det_size mostly manufactures more of them, and each one costs embedding time
and lands a phantom in the unresolved queue for a teacher to look at.

Run it against real classroom footage before trusting either conclusion:

    python tools/measure_det_size.py frame1.jpg frame2.jpg
    python tools/measure_det_size.py --rtsp rtsp://CAMERA-IP:554/stream1
"""
import argparse
import sys
import time

import cv2

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.pipeline import ArcFaceEmbedder   # noqa: E402

# §16: below ~25px genuine and impostor scores overlap, so identification is
# guesswork. Above ~65px liveness can also run (§9). Same variable, two floors.
IDENT_PX = 25
LIVENESS_PX = 65


def grab_rtsp(url: str, n: int):
    import os
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit(f"could not open {url}")
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append((f"frame{len(frames)}", f))
        time.sleep(0.5)     # spread the sample over the scene, don't burst one moment
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="frames to measure")
    ap.add_argument("--rtsp", help="grab frames from a stream instead")
    ap.add_argument("--frames", type=int, default=5, help="frames to grab from --rtsp")
    ap.add_argument("--sizes", default="640,1280,1920")
    ap.add_argument("--repeat", type=int, default=3, help="timing runs per image")
    args = ap.parse_args()

    if args.rtsp:
        loaded = grab_rtsp(args.rtsp, args.frames)
    else:
        loaded = [(p.split("/")[-1], cv2.imread(p)) for p in args.images]
        loaded = [(n, im) for n, im in loaded if im is not None]
    if not loaded:
        sys.exit("no images to measure")

    print(f"{'image':22s} {'det':>5s} {'ms':>7s} {'total':>5s} | "
          f"{'<25 noise':>9s} {'>=25 usable':>11s} {'>=65 judgeable':>14s}")
    usable_by_size = {}
    for size in (int(s) for s in args.sizes.split(",")):
        emb = ArcFaceEmbedder(det_size=(size, size), det_thresh=0.4)
        totals = [0, 0.0]
        for name, im in loaded:
            emb.detect(im)                      # warm up; first call pays graph setup
            t = time.time()
            for _ in range(args.repeat):
                faces = emb.detect(im)
            ms = (time.time() - t) / args.repeat * 1000
            widths = [f.bbox[2] - f.bbox[0] for f in faces]
            usable = sum(w >= IDENT_PX for w in widths)
            judgeable = sum(w >= LIVENESS_PX for w in widths)
            totals[0] += usable
            totals[1] += ms
            print(f"{name:22s} {size:5d} {ms:7.1f} {len(widths):5d} | "
                  f"{len(widths) - usable:9d} {usable:11d} {judgeable:14d}")
        usable_by_size[size] = tuple(totals)

    base = min(usable_by_size)
    base_usable, base_ms = usable_by_size[base]
    print(f"\nagainst det_size={base} ({base_usable} usable faces, {base_ms:.0f}ms total):")
    for size, (usable, ms) in sorted(usable_by_size.items()):
        if size == base:
            continue
        d_faces = usable - base_usable
        d_time = (ms / base_ms - 1) * 100 if base_ms else 0
        verdict = "worth it" if d_faces > base_usable * 0.1 else "NOT worth it"
        print(f"  {size:5d}: {d_faces:+3d} usable faces for {d_time:+.0f}% detect time — {verdict}")
    print(f"\nRemember the deployment GPU is a GTX 1650, roughly 3-4x slower than\n"
          f"this one, against a {1000 / 3:.0f}ms budget per analysis frame at 3 fps.")


if __name__ == "__main__":
    main()
