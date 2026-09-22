"""
Build the gallery from a recording, because enrollment photos were refused.

    python tools/enroll_from_footage.py path/to/footage.mp4 --reset --phantoms 5

WHY NOT enroll.py
enroll_person() re-detects a face in a still and embeds that one view. At this
camera distance a face is ~30px and a single-view embedding of one scores about
0.28 against its own identity -- under the 0.32 threshold. A ByteTrack track
carries a rolling mean over many views instead, and averaging is the measured
difference between 0.2795 and 0.4638 on a 25px face. So identity here comes from
the track, not from a crop. Crops are saved only as thumbnails for the teacher.

WHAT A TRACK IS NOT
A track is one continuous sighting, not one person: a student whose body is
occluded by the row in front respawns under a new id. So tracks are merged by
cosine similarity afterwards, and the merge threshold is deliberately higher
than the recognition threshold -- a wrong merge welds two students into one
gallery entry and is invisible forever after.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config, ENROLLMENT_DIR, GALLERY_INDEX_PATH, GALLERY_META_PATH  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

# Higher than config.match_threshold (0.32) on purpose -- see module docstring.
MERGE_THRESHOLD = 0.55
# A track seen this few times has a noisy mean and is usually a misdetection or
# somebody walking through the back of frame.
MIN_OBSERVATIONS = 8
# Below this the embedding is in the band where genuine and impostor scores
# overlap, so enrolling it poisons the gallery rather than filling it.
MIN_FACE_PX = 28

# How many of a track's views feed the gallery vector, widest face first. Enough
# to keep the averaging gain, few enough that a close look at the door is not
# outvoted by an hour of the same person seated at 28px.
# ponytail: 30 mirrors config.track_embed_window rather than being fitted. Sweep
# it against a held-out clip if enrolment accuracy matters more than it does now.
TOP_N_VIEWS = 30

FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Reyansh", "Krishna", "Ishaan",
         "Rohan", "Kabir", "Ananya", "Diya", "Saanvi", "Aadhya", "Myra", "Anika",
         "Priya", "Riya", "Neha", "Kavya", "Rahul", "Karan", "Nikhil", "Varun",
         "Ayaan", "Dhruv", "Meera", "Tanvi", "Ishita", "Sneha", "Harsh", "Yash",
         "Pooja", "Shreya", "Aryan", "Manav", "Nandini", "Tara", "Vikram", "Zoya"]
LAST = ["Sharma", "Verma", "Gupta", "Mehta", "Reddy", "Nair", "Iyer", "Joshi",
        "Kapoor", "Malhotra", "Chopra", "Bose", "Rao", "Pillai", "Desai", "Shah",
        "Banerjee", "Chatterjee", "Kulkarni", "Patel", "Singh", "Kumar"]


def sample_frames(path: str, every_sec: float):
    """Yield (position_seconds, frame). Seeks rather than decoding everything."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps * every_sec)))
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        yield idx / fps, frame
    cap.release()


def collect_tracks(path: str, every_sec: float, det_size: int, limit_sec: float | None):
    """One pass over the footage. Returns track_id -> observations of that track."""
    cfg = Config()
    cfg.profile.det_size = (det_size, det_size)
    pipeline = Pipeline(cfg)
    tracks: dict[int, dict] = {}
    n_frames = 0
    t0 = time.time()
    # Why a person did not make it in. A missing student is not a mystery to be
    # guessed at afterwards -- it is one of three countable reasons, and they
    # have three different fixes.
    rej = {"no_face": 0, "too_small": 0, "smallest": 999, "faces": 0}

    for pos, frame in sample_frames(path, every_sec):
        if limit_sec is not None and pos > limit_sec:
            break
        n_frames += 1
        # Straight to process_frame: should_analyze() throttles to the profile's
        # fps, which is a live-feed concern. Sampling is already the throttle.
        result = pipeline.process_frame(frame, timestamp=pos)
        for d in result.detections:
            if d.embedding is None or d.face_bbox is None:
                rej["no_face"] += 1           # body only -- nothing to enrol from
                continue
            fx1, fy1, fx2, fy2 = (int(v) for v in d.face_bbox)
            width = fx2 - fx1
            rej["faces"] += 1
            rej["smallest"] = min(rej["smallest"], width)
            if width < MIN_FACE_PX:
                rej["too_small"] += 1
                continue
            t = tracks.setdefault(d.track_id, {"embs": [], "best": None, "best_px": 0})
            # Face width travels WITH the embedding so the mean can be taken over
            # the closest views later. A student walking in the door is 60-80px;
            # the same student seated at the back is 28px. Averaging both equally
            # throws away the good look the doorway gave us.
            t["embs"].append((width, d.embedding))
            if width > t["best_px"]:
                # Generous margin: the thumbnail is for a human to recognise, and
                # a tight 30px box of a face is unreadable to one.
                pad = width // 2
                h, w = frame.shape[:2]
                crop = frame[max(0, fy1 - pad):min(h, fy2 + pad),
                             max(0, fx1 - pad):min(w, fx2 + pad)]
                if crop.size:
                    t["best"], t["best_px"] = crop.copy(), width
        if n_frames % 50 == 0:
            print(f"  {pos/60:5.1f} min  {n_frames:4d} frames  "
                  f"{len(tracks):3d} tracks  {time.time()-t0:5.0f}s", flush=True)

    return tracks, rej


def _mean(pairs, top_n: int = None) -> np.ndarray:
    """
    Average the (width, embedding) observations, biggest face first.

    Not the single best frame -- one view of a 25px face scores 0.2795 against
    its own identity where the averaged track scores 0.4638, so averaging is
    what makes CCTV faces matchable at all. But not every view either: a person
    who walked past the door at 70px and then sat at the back at 28px should be
    remembered as they looked at the door. Taking the mean of the top_n widest
    keeps the averaging and drops the dilution.

    CEILING, so nobody reads more into this than it does: pipeline.py hands back
    track.embedding, which is ALREADY a rolling mean of the last 30 observations
    (pipeline.py:579). So these are not independent single views -- picking the
    widest 30 picks the moments whose *window* was widest, which lags the actual
    close-up by up to 30 frames and makes consecutive picks near-duplicates. It
    biases toward the doorway correctly, but with fewer effective samples than
    the count suggests.
    # ponytail: exposing the per-frame embedding on Detection would make this
    # exact. Only worth it if enrolment accuracy is measured to fall short.
    """
    if top_n:
        pairs = sorted(pairs, key=lambda p: -p[0])[:top_n]
    m = np.mean([e for _, e in pairs], axis=0)
    n = np.linalg.norm(m)
    return (m / n if n > 0 else m).astype(np.float32)


def merge(tracks: dict, threshold: float = MERGE_THRESHOLD,
          min_obs: int = MIN_OBSERVATIONS,
          top_n: int = TOP_N_VIEWS) -> tuple[list[dict], int]:
    """
    Fold tracks of the same person together. Greedy, largest track first, so a
    long confident sighting is the anchor a short one attaches to rather than
    the other way round.
    """
    people = []
    # Every track with a face gets to merge. The observation floor is applied to
    # the PERSON afterwards, not to each track on the way in -- a student broken
    # into three tracks of five sightings has fifteen sightings, and filtering
    # first binned all three before they could find each other. That penalised
    # exactly the people who move around, which is who walks in through a door.
    usable = [(tid, t) for tid, t in tracks.items() if t["best"] is not None]
    faceless = len(tracks) - len(usable)
    for tid, t in sorted(usable, key=lambda x: -len(x[1]["embs"])):
        emb = _mean(t["embs"], top_n)
        hit = None
        for p in people:
            if float(np.dot(emb, p["emb"])) >= threshold:
                hit = p
                break
        if hit is None:
            people.append({"emb": emb, "embs": list(t["embs"]), "tracks": [tid],
                           "best": t["best"], "best_px": t["best_px"]})
        else:
            hit["embs"].extend(t["embs"])
            hit["tracks"].append(tid)
            hit["emb"] = _mean(hit["embs"], top_n)
            if t["best_px"] > hit["best_px"]:
                hit["best"], hit["best_px"] = t["best"], t["best_px"]

    # Now the floor, against the whole person. Some floor is necessary: a face
    # seen twice has a mean built from two looks and will confidently attach
    # somebody else's name later. But a wrong gallery entry is the expensive
    # failure, not a missing one -- a missing student still tracks as a body,
    # comes up nameless, and lands in the teacher's queue to be resolved.
    kept = [p for p in people if len(p["embs"]) >= min_obs]
    return kept, {"faceless": faceless, "thin": len(people) - len(kept),
                  "merged_from": len(usable)}


def write_gallery(people: list[dict], phantoms: int, reset: bool, seed: int):
    rng = random.Random(seed)
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(names)

    if reset or not GALLERY_INDEX_PATH.exists():
        index, meta = faiss.IndexFlatIP(512), []
    else:
        index, meta = faiss.read_index(str(GALLERY_INDEX_PATH)), json.loads(
            GALLERY_META_PATH.read_text())

    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    base = 11825210000 + len(meta)
    for i, p in enumerate(people):
        roll = str(base + i + 1)
        thumb = ENROLLMENT_DIR / f"{roll}.jpg"
        cv2.imwrite(str(thumb), p["best"])
        index.add(p["emb"].reshape(1, -1))
        meta.append({"name": names[len(meta)], "roll": roll, "photo": str(thumb),
                     "embedding_idx": len(meta), "observations": len(p["embs"]),
                     "face_px": p["best_px"], "tracks": p["tracks"],
                     "source": "footage"})

    # Roster padding. A random unit vector is ~orthogonal to every real face
    # (cosine 0 +/- 0.04 in 512 dimensions, against a 0.32 threshold), so a
    # phantom is mathematically incapable of matching anybody -- it can only ever
    # be reported absent, which is the entire point of it.
    for i in range(phantoms):
        v = rng.gauss
        vec = np.array([v(0, 1) for _ in range(512)], np.float32)
        vec /= np.linalg.norm(vec)
        roll = str(base + len(people) + i + 1)
        index.add(vec.reshape(1, -1))
        meta.append({"name": names[len(meta)], "roll": roll, "photo": "",
                     "embedding_idx": len(meta), "observations": 0,
                     "source": "phantom"})

    faiss.write_index(index, str(GALLERY_INDEX_PATH))
    GALLERY_META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=2.0, help="sample interval, seconds")
    ap.add_argument("--det-size", type=int, default=1280)
    ap.add_argument("--phantoms", type=int, default=5)
    ap.add_argument("--limit-sec", type=float, default=None, help="stop after N seconds of footage")
    ap.add_argument("--reset", action="store_true", help="discard the existing gallery")
    ap.add_argument("--seed", type=int, default=7)
    # The two unmeasured dials. MIN_FACE_PX is deliberately NOT one: it comes
    # from the measured band where genuine and impostor scores overlap, and
    # lowering it does not recover the back row, it enrols guesses as students.
    ap.add_argument("--merge", type=float, default=MERGE_THRESHOLD,
                    help="lower = more tracks fold together = fewer people")
    ap.add_argument("--min-obs", type=int, default=MIN_OBSERVATIONS)
    ap.add_argument("--top-n", type=int, default=TOP_N_VIEWS,
                    help="views averaged per person, widest face first; 0 = all")
    args = ap.parse_args()

    tracks, rej = collect_tracks(args.video, args.every, args.det_size, args.limit_sec)
    people, drop = merge(tracks, args.merge, args.min_obs, args.top_n)

    # Read top to bottom: every person who did not reach the gallery dropped out
    # at exactly one of these lines, and each line has a different fix.
    print(f"\n  {rej['faces']:6d} face sightings, smallest {rej['smallest']}px")
    print(f"  {rej['too_small']:6d} rejected under {MIN_FACE_PX}px "
          f"({100*rej['too_small']//max(rej['faces'],1)}%) -- back rows, "
          f"needs a second camera, not a lower threshold")
    print(f"  {rej['no_face']:6d} body sightings with no face at all -- head down or turned away")
    print(f"  {len(tracks):6d} raw tracks")
    print(f"  {drop['faceless']:6d} with no usable face crop at all")
    print(f"  {drop['merged_from']:6d} merged at {args.merge} -> "
          f"{len(people) + drop['thin']} people")
    print(f"  {drop['thin']:6d} of those still under {args.min_obs} total "
          f"observations -- dropped")
    print(f"  {len(people):6d} people enrolled")

    meta = write_gallery(people, args.phantoms, args.reset, args.seed)
    print(f"\ngallery: {len(meta)} entries -> {GALLERY_META_PATH}")
    for m in meta:
        tag = "PHANTOM (never seen -> absent)" if m["source"] == "phantom" else \
              f"{m['observations']:4d} obs, {m.get('face_px',0):3d}px"
        print(f"  {m['roll']}  {m['name']:<22} {tag}")


if __name__ == "__main__":
    main()
