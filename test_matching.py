"""
Guards the two failure modes that made every stranger read as "Krish".

Run: .venv/bin/python test_matching.py
"""
import json

import faiss
import numpy as np

from src.config import Config
from src.pipeline import GalleryMatcher


def _fake_gallery(tmp="/tmp/_am_test"):
    """One aligned (web-enrolled) entry + two unaligned CASIA-style entries."""
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(3, 512)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(512)
    index.add(vecs)
    faiss.write_index(index, tmp + ".index")
    meta = [
        {"name": "WebPerson", "roll": "W1"},                          # aligned
        {"name": "Person_0001", "roll": "0001", "source_scene": "0001"},  # unaligned
        {"name": "Person_0002", "roll": "0002", "source_scene": "0002"},  # unaligned
    ]
    with open(tmp + ".json", "w") as f:
        json.dump(meta, f)
    return tmp + ".index", tmp + ".json", vecs


def test_live_only_drops_unaligned_entries():
    idx, meta, _ = _fake_gallery()
    assert GalleryMatcher(idx, meta).index.ntotal == 3, "eval path must keep CASIA entries"

    live = GalleryMatcher(idx, meta, live_only=True)
    assert live.index.ntotal == 1, "live path must drop the unaligned CASIA entries"
    assert [m["name"] for m in live.meta] == ["WebPerson"]
    assert len(live.meta) == live.index.ntotal, "meta and index must stay in lockstep"


def test_live_only_keeps_index_meta_aligned():
    """Filtering must renumber rows, or match() returns the wrong person."""
    idx, meta, vecs = _fake_gallery()
    live = GalleryMatcher(idx, meta, live_only=True)
    # Row 0 of the filtered index must still be WebPerson's original vector.
    name, roll, score = live.match(vecs[0], threshold=0.5)
    assert (name, roll) == ("WebPerson", "W1"), f"got {name}/{roll}"
    assert score > 0.99, score


def test_threshold_sits_above_the_impostor_floor():
    """
    Measured on this repo's data: impostors peak ~0.25, genuine faces >=40px
    land 0.36+. Guard the gap so nobody 'fixes' recall by reopening it.
    """
    c = Config()
    assert 0.25 < c.match_threshold < 0.36, (
        f"match_threshold={c.match_threshold} falls outside the measured "
        "impostor/genuine gap — lowering it relabels strangers, it does not "
        "improve recall. Aggregate more observations instead."
    )


def _only_track(tracker):
    """The one track we expect. Asserting the count catches ID churn."""
    assert len(tracker.tracks) == 1, f"expected 1 track, got {len(tracker.tracks)}"
    return next(iter(tracker.tracks.values()))


def _unit(rng):
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_track_averages_embeddings_over_window():
    """
    Small CCTV faces are matched by averaging a track's observations. If a track
    ever goes back to holding only the latest embedding, distant faces stop
    matching and the whole point of carrying track IDs is lost.
    """
    from src.pipeline import ByteTrackWrapper

    rng = np.random.default_rng(1)
    truth = _unit(rng)

    def noisy():
        v = truth + rng.normal(scale=1.4, size=512).astype(np.float32)
        return v / np.linalg.norm(v)

    tracker = ByteTrackWrapper(embed_window=30)
    box = np.array([0, 0, 25, 30], dtype=np.float32)  # a 25px-wide face, CCTV scale

    # ByteTrack withholds an ID until minimum_consecutive_frames is met, so the
    # first frame's embedding is dropped — there is no track to attach it to yet.
    # 30 updates therefore yield 29 observations. Not worth buffering pending
    # detections to recover one frame out of a 30-frame window.
    tracker.update([box], [noisy()])
    tracker.update([box], [noisy()])
    track = _only_track(tracker)
    few = float(track.embedding @ truth)

    for _ in range(28):
        tracker.update([box], [noisy()])
    averaged = float(track.embedding @ truth)

    assert track.observations == 29, track.observations
    assert averaged > few + 0.1, (
        f"averaging {track.observations} observations scored {averaged:.3f} vs "
        f"{few:.3f} for one frame — the window is not being aggregated"
    )
    assert abs(np.linalg.norm(track.embedding) - 1.0) < 1e-5, "must stay L2-normalized"


def test_track_window_rolls_off_old_identity():
    """Bounded window, so a ByteTrack ID switch doesn't poison the track forever."""
    from src.pipeline import ByteTrackWrapper

    rng = np.random.default_rng(2)
    a, b = _unit(rng), _unit(rng)

    tracker = ByteTrackWrapper(embed_window=10)
    box = np.array([0, 0, 25, 30], dtype=np.float32)
    for _ in range(11):
        tracker.update([box], [a])
    assert float(_only_track(tracker).embedding @ a) > 0.99

    for _ in range(10):  # ID switched to a different person
        tracker.update([box], [b])
    assert float(_only_track(tracker).embedding @ b) > 0.99, "old identity never rolled off"


def test_walking_person_survives_an_occlusion_gap():
    """
    The bug that started this: a walking face picked up a new ID every frame, so
    the embedding window never accumulated and the name flapped.

    Prediction is what carries the gap in the middle here: three frames with no
    detection at all, the way a person passing behind someone else looks.
    """
    from src.pipeline import ByteTrackWrapper

    rng = np.random.default_rng(3)
    emb = _unit(rng)

    tracker = ByteTrackWrapper(embed_window=30, frame_rate=20)

    def step(i):
        x = 10 + i * 13  # 13px/frame on a 30px box — 1.4 m/s at 20 fps
        return np.array([x, 40, x + 30, 80], dtype=np.float32)

    for i in range(8):
        tracker.update([step(i)], [emb], [0.9])
    before = _only_track(tracker).id

    for _ in range(3):      # occluded: detector returns nothing
        tracker.update([], [], [])
    # Resume at 11, not 8 — they keep walking while hidden. Reappearing where
    # they vanished is a teleport backwards, and prediction rightly rejects it.
    for i in range(11, 19):
        tracker.update([step(i)], [emb], [0.9])

    track = _only_track(tracker)
    assert track.id == before, f"ID changed {before} -> {track.id} across an occlusion"
    assert track.observations >= 14, (
        f"only {track.observations} observations survived — the ID is being dropped "
        "and re-created as the person moves"
    )


def test_three_fps_needs_body_boxes_not_face_boxes():
    """
    The geometric reason analysis_fps could drop 20 -> 3, asserted rather than
    claimed in a comment.

    Displacement measured in box widths is scale-invariant — it depends on the
    subject's real width, not on how far away the camera is. At 1.0 m/s and 3 fps
    a person moves 33cm per frame, which is 2.1 widths of a 16cm face (boxes do
    not overlap at all, IoU 0, no association is possible) but 0.67 widths of a
    50cm body, which still overlaps enough to link.

    That first link is the whole problem: velocity is unknown until two frames
    have already been associated, so Kalman prediction cannot rescue it.

    Measured ceiling, swept in this same harness: bodies hold to ~1.2 m/s and
    break at 1.4 m/s (a brisk outdoor stride). Faces hold at no speed at all.
    Someone jogging through the door will be a new track — acceptable, because
    dwell-based attendance does not care about people who do not stop. If that
    ever stops being acceptable, raise analysis_fps; the rate and the choice of
    box are one decision, not two.
    """
    from src.pipeline import ByteTrackWrapper

    rng = np.random.default_rng(7)
    emb = _unit(rng)
    px_per_m = 200
    step = 1.0 / 3 * px_per_m        # 1.0 m/s, normal indoor walking pace

    def run(width_m, height_m):
        tracker = ByteTrackWrapper(embed_window=30, frame_rate=3)
        for i in range(10):
            x = 10.0 + i * step
            tracker.update([np.array([x, 40, x + width_m * px_per_m,
                                      40 + height_m * px_per_m], dtype=np.float32)],
                           [emb], [0.9])
        return tracker

    faces = run(0.16, 0.20)
    assert not faces.tracks, (
        "a face box held a track across 3 fps walking motion — that should be "
        "geometrically impossible, so check the test before trusting the result"
    )

    bodies = run(0.50, 1.75)
    assert len(bodies.tracks) == 1, (
        f"{len(bodies.tracks)} body tracks for one walker at 3 fps — "
        "body tracking is not surviving the rate drop"
    )
    assert _only_track(bodies).observations >= 8, _only_track(bodies).observations


def test_lost_track_is_not_reported_as_present():
    """
    update() used to return every track in the dict, including ones not matched
    this frame. A person who walked out kept a ghost box at their last position
    for max_age frames — counted present, and overlapping their own new track on
    return, which raised a bogus multiple_overlapping alert on the live camera.
    """
    from src.pipeline import ByteTrackWrapper

    rng = np.random.default_rng(5)
    emb = _unit(rng)
    tracker = ByteTrackWrapper(embed_window=30, frame_rate=20)
    box = np.array([10, 40, 40, 80], dtype=np.float32)

    for _ in range(5):
        tracker.update([box], [emb], [0.9])
    assert len(tracker.update([box], [emb], [0.9])) == 1

    # They leave: detector returns nothing.
    assert tracker.update([], [], []) == [], "a vanished face was still reported"
    # ...and the embedding window is still retained for their return.
    assert tracker.tracks, "track was discarded instantly instead of buffered"


def test_one_frame_detection_is_not_a_person():
    """
    min_hits was set in three places and read in none, so a single spurious
    detection was reported as a confirmed attendee. Now it must earn the ID.
    """
    from src.pipeline import ByteTrackWrapper

    rng = np.random.default_rng(4)
    tracker = ByteTrackWrapper()
    tracker.update([np.array([0, 0, 25, 30], dtype=np.float32)], [_unit(rng)], [0.9])
    assert not tracker.tracks, "a one-frame blip became a tracked person"


class _FakeFace:
    def __init__(self, bbox, emb, score=0.9):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.normed_embedding = emb
        self.det_score = score


class _FakeEmbedder:
    """Stands in for ArcFace so this runs without loading 300MB of ONNX."""

    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def detect(self, _frame):
        faces = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return faces


def test_identity_is_sticky_once_matched():
    """
    The label used to be re-decided from scratch every frame, so anyone scoring
    near the threshold strobed between their name and Unknown — and the box
    flipped green/grey with it. Once a track has been identified it keeps the
    name for its lifetime.
    """
    from src.config import Config
    from src.pipeline import Pipeline

    idx, meta, vecs = _fake_gallery()
    known = vecs[0]                        # the one aligned entry
    stranger = vecs[1]                     # scores well below threshold vs known

    box = [0, 0, 25, 30]
    frames = [[_FakeFace(box, known)]] * 3 + [[_FakeFace(box, stranger)]] * 3

    # window=1 so the track holds only the latest embedding; otherwise averaging
    # would keep the score up and the dip under test could never happen.
    cfg = Config(gpu_profile="dev", track_embed_window=1)
    p = Pipeline(cfg, embedder=_FakeEmbedder(frames))
    p.load_gallery(idx, meta, live_only=True)

    names, scores = [], []
    for _ in range(6):
        for d in p.process_frame(np.zeros((120, 120, 3), dtype=np.uint8)).detections:
            names.append(d.name)
            scores.append(d.match_score)

    assert "WebPerson" in names, f"never matched at all: {names}"
    assert scores[-1] < cfg.match_threshold, (
        f"final score {scores[-1]:.3f} never dipped below {cfg.match_threshold} — "
        "the test is not exercising the sticky path"
    )
    assert set(names) == {"WebPerson"}, f"identity flapped: {names}"


class _FakePersonDetector:
    """Yields a scripted list of body boxes per frame, like PersonDetector."""

    def __init__(self, frames, score=0.9):
        self.frames = frames
        self.score = score
        self.i = 0

    def detect(self, _frame):
        from src.persons import PersonBox
        boxes = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return [PersonBox(bbox=np.asarray(b, dtype=np.float32), score=self.score)
                for b in boxes]


def test_face_is_not_lost_with_a_body_that_never_tracked():
    """
    Found on real photos: ByteTrack only spawns tracks from its high-confidence
    bucket, so a body below the spawn threshold is reported by nothing. The face
    inside it had already been assigned to that body and removed from the face
    path, so the person vanished from both — 3 of 11 people on pic1, 7 of 15 on
    pic3, silently absent from the register.

    A face must fall through to the face tracker whenever its body did not
    actually produce a track.
    """
    from src.config import Config
    from src.pipeline import BODY_SPAWN_THRESHOLD, Pipeline

    idx, meta, vecs = _fake_gallery()
    body = [100, 100, 200, 400]
    face = _FakeFace([130, 110, 170, 150], vecs[0])

    cfg = Config(gpu_profile="dev")
    p = Pipeline(cfg, embedder=_FakeEmbedder([[face]]),
                 # Detected, but too weak to start a body track.
                 person_detector=_FakePersonDetector([[body]],
                                                     score=BODY_SPAWN_THRESHOLD - 0.05))
    p.load_gallery(idx, meta, live_only=True)

    for _ in range(4):
        dets = p.process_frame(np.zeros((480, 640, 3), dtype=np.uint8)).detections

    assert len(dets) == 1, f"expected exactly one person, got {len(dets)}"
    assert dets[0].name == "WebPerson", dets[0].name


def test_identity_survives_the_face_disappearing():
    """
    The requirement this whole design exists for: once a face names someone,
    they keep that name while turned around or covered up.

    A face turned away scored 0.06 against its own enrollment — below the ~0.25
    impostor floor — so no threshold can re-derive identity per frame. It has to
    be carried by the body track.
    """
    from src.config import Config
    from src.pipeline import Pipeline

    idx, meta, vecs = _fake_gallery()
    known = vecs[0]

    body = [100, 100, 200, 400]
    face = [130, 110, 170, 150]           # inside the body box

    # Three frames with a visible face, then six with the body only.
    face_frames = [[_FakeFace(face, known)]] * 3 + [[]] * 6
    body_frames = [[body]] * 9

    cfg = Config(gpu_profile="dev")
    p = Pipeline(cfg, embedder=_FakeEmbedder(face_frames),
                 person_detector=_FakePersonDetector(body_frames))
    p.load_gallery(idx, meta, live_only=True)

    names, ids = [], []
    for _ in range(9):
        for d in p.process_frame(np.zeros((480, 640, 3), dtype=np.uint8)).detections:
            names.append(d.name)
            ids.append(d.track_id)

    assert names[-1] == "WebPerson", (
        f"lost the name once the face went away: {names}")
    assert len(set(ids)) == 1, f"the body track was not kept intact: {ids}"
    assert names.count("WebPerson") >= 6, (
        f"name did not persist across the covered frames: {names}")


def test_a_face_inside_a_body_is_one_person_not_two():
    """
    Bodies and faces are tracked by two separate trackers. A face contained in a
    body box must be folded into that body's track — reporting both would make
    one attendee two, and double-count the room.
    """
    from src.config import Config
    from src.pipeline import Pipeline

    idx, meta, vecs = _fake_gallery()

    body = [100, 100, 200, 400]
    inside = _FakeFace([130, 110, 170, 150], vecs[0])
    outside = _FakeFace([400, 100, 440, 140], vecs[0])   # back row, no body found

    cfg = Config(gpu_profile="dev")
    p = Pipeline(cfg, embedder=_FakeEmbedder([[inside, outside]]),
                 person_detector=_FakePersonDetector([[body]]))
    p.load_gallery(idx, meta, live_only=True)

    for _ in range(4):
        dets = p.process_frame(np.zeros((480, 640, 3), dtype=np.uint8)).detections

    assert len(dets) == 2, f"expected body + orphan face, got {len(dets)}"
    tracked_body = [d for d in dets if d.bbox[3] > 300]
    assert len(tracked_body) == 1, "the contained face was counted as its own person"
    assert tracked_body[0].face_bbox is not None, "body track lost its face box"
    # Distinct namespaces, or the two trackers' id 1s collide into one record.
    assert len({d.track_id for d in dets}) == 2, [d.track_id for d in dets]


def test_zone_membership_uses_feet_not_chin():
    """
    Zone containment reads the bottom-centre of the reported box. Now that the
    box is a body, that is the person's feet — the chin anchor put a tall person
    in the row in front of the one they were standing in.
    """
    from src.config import Config
    from src.pipeline import Pipeline
    from src.zones import Zone

    idx, meta, vecs = _fake_gallery()

    # Zone covers the lower half of the frame only.
    body = [100, 100, 200, 400]           # feet at y=400 (inside), head at y=100 (outside)
    face = [130, 110, 170, 150]

    cfg = Config(gpu_profile="dev")
    p = Pipeline(cfg, embedder=_FakeEmbedder([[_FakeFace(face, vecs[0])]]),
                 person_detector=_FakePersonDetector([[body]]))
    p.load_gallery(idx, meta, live_only=True)
    p.zones = [Zone("floor", [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]])]

    for _ in range(3):
        dets = p.process_frame(np.zeros((480, 640, 3), dtype=np.uint8)).detections

    assert dets and dets[0].zone == "floor", (
        f"body at y=100..400 in a 480px frame should stand in 'floor', got "
        f"{dets[0].zone if dets else 'no detections'}")


class _Det:
    """Minimal stand-in for pipeline.Detection."""

    def __init__(self, name="Unknown", roll="", track_id=1, score=0.9,
                 is_spoof=False, observations=30, zone="", face_visible=True):
        self.name, self.roll, self.track_id = name, roll, track_id
        self.match_score, self.is_spoof = score, is_spoof
        self.liveness_score, self.observations = 0.1 if is_spoof else 0.9, observations
        self.zone = zone
        self.bbox = np.array([0, 0, 40, 120], dtype=np.float32)       # body
        # None models someone turned around or covered up: the body is tracked,
        # the face contributes nothing this frame.
        self.face_bbox = np.array([5, 0, 35, 35], dtype=np.float32) if face_visible else None


def test_spoofed_face_does_not_mark_attendance():
    """
    is_spoof used to only colour the box red — attendance.py never read it, so
    holding up a printed photo registered that person as present.
    """
    from src.attendance import AttendanceLogger

    log = AttendanceLogger(session_name="_test_spoof")
    for i in range(10):
        log.process_detections([_Det(name="Krish", roll="K1", is_spoof=True)], i, 1000.0 + i)

    assert "K1" not in log.records, "a spoofed face was logged as present"
    spoof_alerts = [a for a in log.alerts if a.alert_type == "spoof"]
    assert len(spoof_alerts) == 1, f"{len(spoof_alerts)} spoof alerts for one spoofer"
    assert spoof_alerts[0].severity == "critical"
    assert spoof_alerts[0].count == 10, spoof_alerts[0].count


def test_alerts_collapse_instead_of_flooding():
    """
    A stranger standing in shot raised one alert per analysed frame — 2103 in a
    short test, nearly all identical. One row per subject, with a count.
    """
    from src.attendance import AttendanceLogger

    log = AttendanceLogger(session_name="_test_flood")
    for i in range(200):
        log.process_detections([_Det(track_id=7)], i, 1000.0 + i * 0.05)

    assert len(log.alerts) == 1, f"{len(log.alerts)} rows for one unknown face"
    assert log.alerts[0].count == 200, log.alerts[0].count

    # A second stranger is a separate subject, not a repeat of the first.
    for i in range(5):
        log.process_detections([_Det(track_id=9)], i, 1100.0 + i)
    assert len(log.alerts) == 2, f"two strangers should be two rows: {len(log.alerts)}"


def test_new_track_is_not_called_a_stranger_immediately():
    """
    A track is legitimately Unknown for the first few frames while its embedding
    window fills. Alerting then was most of the old noise.
    """
    from src.attendance import AttendanceLogger

    log = AttendanceLogger(session_name="_test_young")
    for obs in range(1, 5):
        log.process_detections([_Det(track_id=3, observations=obs)], obs, 1000.0 + obs)
    assert not log.alerts, f"alerted on a {obs}-observation track: {log.alerts}"

    log.process_detections([_Det(track_id=3, observations=9)], 9, 1009.0)
    assert len(log.alerts) == 1, "should alert once the track has settled"


def test_batched_embeddings_match_the_reference_path():
    """
    detect() batches ArcFace instead of running one forward pass per face (98
    faces: 911ms -> 192ms). The batched crops MUST come out identical to
    FaceAnalysis.get(), because the gallery was enrolled through that path — an
    embedding taken even slightly differently is not comparable to it, which is
    precisely how every stranger once matched "Krish".

    Skipped when the model or sample photos aren't present.
    """
    import os

    photo = "data/test_faces/pic1.jpeg"
    if not os.path.exists(photo):
        print("    (skipped: sample photo not available)")
        return

    import cv2

    from src.config import Config
    from src.pipeline import ArcFaceEmbedder

    img = cv2.imread(photo)
    cfg = Config(gpu_profile="dev")
    emb = ArcFaceEmbedder(det_size=cfg.profile.det_size, emb_batch=cfg.profile.emb_batch)

    new, old = emb.detect(img), emb.model.get(img)
    assert len(new) == len(old), f"face counts diverged: {len(new)} vs {len(old)}"

    def centre(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    for f in new:
        c = centre(f.bbox)
        nearest = min(old, key=lambda g: (centre(g.bbox)[0] - c[0]) ** 2
                      + (centre(g.bbox)[1] - c[1]) ** 2)
        sim = float(f.normed_embedding @ nearest.normed_embedding)
        assert sim > 0.9999, f"batched embedding diverged from reference: {sim:.6f}"
        assert abs(np.linalg.norm(f.normed_embedding) - 1.0) < 1e-5


def test_boxes_glide_between_analysis_frames():
    """
    Analysis runs slower than the camera on purpose. Drawing the last analysed box
    verbatim on the frames in between makes it freeze then jump, which reads as lag
    even when the video is smooth. Boxes must carry forward at their last measured
    velocity — and must not sail off when analysis stalls.
    """
    from src.webapp import BoxGlide

    class D:
        def __init__(self, tid, x):
            self.track_id = tid
            self.bbox = np.array([x, 40, x + 30, 80], dtype=np.float32)

    g = BoxGlide()
    g.observe([D(1, 10)], 0.0)
    g.observe([D(1, 40)], 0.3)          # 100 px/sec

    mid = g.bbox_at(1, 0.45)
    assert abs(mid[0] - 55.0) < 0.01, f"expected x=55 halfway, got {mid[0]}"

    stalled = g.bbox_at(1, 10.0)
    cap = 40 + 100 * g.max_extrapolate_sec
    assert abs(stalled[0] - cap) < 0.01, f"extrapolation ran away to {stalled[0]}"

    # One sighting is not a velocity.
    g2 = BoxGlide()
    g2.observe([D(2, 10)], 0.0)
    assert g2.bbox_at(2, 5.0)[0] == 10, "extrapolated from a single observation"

    g.observe([], 0.6)
    assert g.bbox_at(1, 0.6) is None, "a vanished track kept gliding"


def test_zone_gates_attendance():
    """
    A zone that doesn't actually filter is decoration. Someone outside every
    configured zone — the corridor visible through a doorway — must not be marked
    present, and must not raise a stranger alert either.
    """
    import tempfile
    from pathlib import Path

    from src.attendance import AttendanceLogger
    from src.zones import Zone, load_zones, save_zones, zone_for

    tmp = Path(tempfile.mkdtemp()) / "zones.json"
    save_zones([Zone("room", [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])], tmp)
    assert [z.name for z in load_zones(tmp)] == ["room"], "round-trip failed"

    # Centre face is inside; top-left face is outside.
    inside = np.array([300, 200, 340, 250], dtype=np.float32)
    outside = np.array([10, 5, 50, 40], dtype=np.float32)
    zs = load_zones(tmp)
    assert zone_for(inside, zs, 640, 480) == "room"
    assert zone_for(outside, zs, 640, 480) == ""

    log = AttendanceLogger(session_name="_test_zone")
    log.zones_active = True      # as if zones.json were configured

    log.process_detections([_Det(name="Krish", roll="K1", zone="room")], 1, 1000.0)
    log.process_detections([_Det(name="Piyush", roll="P1", zone="")], 2, 1001.0)
    log.process_detections([_Det(track_id=5, zone="")], 3, 1002.0)

    assert "K1" in log.records, "person inside the zone was not counted"
    assert "P1" not in log.records, "person outside the zone was counted"
    assert log.records["K1"].zone == "room"
    assert not log.alerts, f"a face outside every zone raised an alert: {log.alerts}"

    # With no zones configured the whole frame counts — the default must not
    # silently start filtering people out.
    open_log = AttendanceLogger(session_name="_test_nozone")
    open_log.zones_active = False
    open_log.process_detections([_Det(name="Piyush", roll="P1", zone="")], 1, 1000.0)
    assert "P1" in open_log.records, "no zones configured must mean no filtering"


def _walk_past(log, det, start, seconds, step=0.5):
    """Feed `seconds` of continuous detections, then one frame well after."""
    t = start
    while t < start + seconds:
        log.process_detections([det], 0, t)
        t += step
    return t


def test_walking_past_the_door_is_not_attendance():
    """
    The old rule marked you present after 3 detections — 300ms at 10fps. Anyone
    crossing the doorway got a full attendance record.
    """
    from src.attendance import AttendanceLogger

    log = AttendanceLogger(session_name="_test_dwell_brief")
    end = _walk_past(log, _Det(name="Krish", roll="K1"), 1000.0, seconds=4.0)

    rec = log.records["K1"]
    assert rec.duration_sec < 5, rec.duration_sec
    assert rec.status == "brief", f"4 seconds counted as {rec.status}"
    assert log.get_summary()["present_now"] == 0, "a passer-by was counted present"

    # Same person, properly seated this time.
    _walk_past(log, _Det(name="Krish", roll="K1"), end + 1, seconds=40.0)
    assert rec.status == "present", rec.status
    assert rec.duration_sec >= log.min_dwell_sec, rec.duration_sec
    assert log.get_summary()["present_now"] == 1


def test_dwell_survives_a_head_down_gap_then_closes_on_exit():
    """
    A student with their head down vanishes for a minute and is still sitting
    there; someone who leaves must close their visit and stop accruing time.
    """
    from src.attendance import AttendanceLogger

    log = AttendanceLogger(session_name="_test_dwell_gap")
    det = _Det(name="Krish", roll="K1")
    _walk_past(log, det, 1000.0, seconds=40.0)
    rec = log.records["K1"]

    # Gap shorter than the grace period: one visit, and the gap is credited.
    log.process_detections([det], 0, 1070.0)
    assert len(rec.visits) == 1, f"a {log.exit_grace_sec}s gap split the visit"
    assert rec.duration_sec >= 69, rec.duration_sec
    assert rec.is_present

    # Now they actually leave.
    log.process_detections([], 0, 1070.0 + log.exit_grace_sec + 1)
    assert not rec.is_present, "left the room but still marked present"
    assert rec.exit_time, "no exit time recorded"
    assert rec.status == "left"
    dwell_at_exit = rec.duration_sec

    # Time passing with them gone must not add dwell.
    log.process_detections([], 0, 3000.0)
    assert rec.duration_sec == dwell_at_exit, "dwell kept ticking after they left"

    # They come back: second visit, dwell accumulates across both.
    _walk_past(log, det, 3001.0, seconds=10.0)
    assert len(rec.visits) == 2, len(rec.visits)
    assert rec.duration_sec > dwell_at_exit
    assert rec.is_present


def test_unresolved_person_is_queued_and_resolves_retroactively():
    """
    An unidentified face must not be dropped. It sits in the queue as personN,
    and naming it credits the time they were already there — not from now.
    """
    from src.attendance import AttendanceLogger

    log = AttendanceLogger(session_name="_test_unresolved")
    _walk_past(log, _Det(track_id=4, observations=9), 1000.0, seconds=45.0)

    queue = log.get_summary()["unresolved"]
    assert len(queue) == 1, queue
    assert queue[0]["label"] == "person1"
    assert queue[0]["needs_action"], "45s in the room should need a decision"
    assert queue[0]["dwell_sec"] >= log.min_dwell_sec

    rec = log.resolve(4, "Rishit", "11825210004")
    assert rec is not None
    assert not log.get_summary()["unresolved"], "resolved person stayed in the queue"
    assert rec.duration_sec >= 44, f"back-dated dwell lost: {rec.duration_sec}"
    assert rec.status == "present", rec.status
    assert rec.resolved_from == ["person1"]

    assert log.resolve(4, "Rishit", "11825210004") is None, "resolved twice"


def test_session_close_finalises_everyone():
    """
    The last visit stays open until something closes it, so a saved report used
    to claim the whole class was still in the room.
    """
    from src.attendance import AttendanceLogger

    log = AttendanceLogger(session_name="_test_close")
    _walk_past(log, _Det(name="Krish", roll="K1"), 1000.0, seconds=40.0)
    _walk_past(log, _Det(track_id=8, observations=9), 1000.0, seconds=40.0)
    assert log.records["K1"].is_present

    log.close_session()
    rec = log.records["K1"]
    assert not rec.is_present and rec.exit_time, "open visit survived the session end"
    assert rec.status == "left", rec.status
    assert not log.unresolved[8].is_present
    dwell = rec.duration_sec
    log.close_session()
    assert rec.duration_sec == dwell, "closing twice changed the totals"


def test_recognised_face_clears_its_own_unresolved_entry():
    """
    Someone unrecognised for the first few seconds and then matched must not be
    left in the review queue as a phantom stranger.
    """
    from src.attendance import AttendanceLogger

    log = AttendanceLogger(session_name="_test_unresolved_clear")
    _walk_past(log, _Det(track_id=4, observations=9), 1000.0, seconds=5.0)
    assert log.unresolved

    log.process_detections([_Det(track_id=4, name="Krish", roll="K1")], 0, 1006.0)
    assert not log.unresolved, "matched face left behind an unresolved entry"


def test_covered_dwell_is_back_filled_when_the_face_finally_matches():
    """
    The payoff of tracking bodies. Someone sits down facing away, is person1 for
    40 seconds, then glances at the camera and matches. They must be credited
    from when they sat down — not from the moment they happened to look up.

    The old code popped the unresolved entry and threw its dwell away, so a
    student covered for most of the class read as 'brief'.
    """
    from src.attendance import AttendanceLogger

    log = AttendanceLogger(session_name="_test_backfill")
    covered = _Det(track_id=4, observations=9, face_visible=False)
    end = _walk_past(log, covered, 1000.0, seconds=40.0)
    assert log.unresolved[4].duration_sec >= 39, log.unresolved[4].duration_sec

    # Their face finally shows and matches.
    log.process_detections([_Det(track_id=4, name="Krish", roll="K1")], 0, end)

    rec = log.records["K1"]
    assert not log.unresolved, "resolved person left in the queue"
    assert rec.duration_sec >= 39, (
        f"only {rec.duration_sec:.1f}s credited — the covered portion was "
        "discarded instead of back-filled")
    assert rec.status == "present", rec.status
    assert rec.resolved_from == ["person1"], rec.resolved_from
    assert len(rec.visits) == 1, (
        f"{len(rec.visits)} visits — the merge split one continuous stay in two")


def test_every_label_a_churning_track_collects_is_recorded():
    """
    One person, two track IDs — which is what the first real-camera run actually
    produced: a seated man's body track died behind his chair and respawned, so
    he was person1 and later person2 before matching.

    resolved_from used to be assigned, not appended, so the earlier label was
    silently dropped and the report showed a single clean merge. That hid track
    churn in exactly the records where it had happened, which is how it went
    unnoticed until the dwell numbers were audited by hand.
    """
    from src.attendance import AttendanceLogger

    log = AttendanceLogger(session_name="_test_churn")

    # First track: unnamed, accrues time, then the track is lost.
    t = _walk_past(log, _Det(track_id=7, observations=9, face_visible=False),
                   1000.0, seconds=20.0)
    # Respawns with a fresh ID, still unnamed, then finally matches.
    t = _walk_past(log, _Det(track_id=8, observations=9, face_visible=False),
                   t, seconds=20.0)
    log.process_detections([_Det(track_id=8, name="Krish", roll="K1")], 0, t)
    # The first label is resolved by hand, the way a teacher would.
    log.resolve(7, "Krish", "K1")

    rec = log.records["K1"]
    assert rec.resolved_from == ["person2", "person1"], rec.resolved_from
    assert rec.duration_sec >= 39, (
        f"only {rec.duration_sec:.1f}s — dwell was lost across the respawn")


def test_liveness_head_is_read_as_a_three_class_softmax():
    """
    MiniFASNetV2's head is softmax over [live, print, replay]. The stub this
    replaced applied a sigmoid to index 1 and called it "real" — index 1 is the
    PRINT ATTACK logit. That combination runs without error and returns a
    confident number that means nothing, which is the worst way to be wrong.
    """
    from src.antispoof import SilentFaceLiveness as S

    assert S._liveness([10.0, 0.0, 0.0]) > 0.999      # live
    assert S._liveness([0.0, 10.0, 0.0]) < 0.001      # print attack
    assert S._liveness([0.0, 0.0, 10.0]) < 0.001      # replay attack
    assert abs(S._liveness([1.0, 1.0, 1.0]) - 1 / 3) < 1e-9
    # Shifted exponent, so a saturated head does not overflow to nan.
    assert abs(S._liveness([900.0, 900.0, 900.0]) - 1 / 3) < 1e-9


def test_liveness_crop_is_2_7x_and_stays_square_at_the_frame_edge():
    """
    The model was trained on boxes expanded 2.7x, and the tell for a print or
    replay attack is often outside the face — paper edge, phone bezel, flat
    background. Clamping matters: letting the box run off the frame and relying
    on numpy's silent truncation changes the effective scale, so a face near the
    edge would be fed to the model at the wrong zoom.
    """
    from src.antispoof import SilentFaceLiveness as S

    frame = np.zeros((480, 640, 3), np.uint8)
    middle = S._crop([300, 200, 340, 240], frame)     # 40px box -> 40*2.7 = 108
    assert middle.shape[:2] == (108, 108), middle.shape
    corner = S._crop([0, 0, 40, 40], frame)
    assert corner.shape[:2] == (108, 108), corner.shape


def test_missing_liveness_model_falls_back_instead_of_flagging():
    """
    No model file must mean 'use the heuristic', never 'everyone is a spoof'.
    A liveness check that fails closed would block the whole class.
    """
    from src.antispoof import SpoofChecker

    ck = SpoofChecker(model_path="data/models/definitely_not_here.onnx")
    assert ck.model.model is None
    assert ck.model.predict([0, 0, 40, 40], np.zeros((480, 640, 3), np.uint8)) == -1.0
    is_spoof, liveness = ck.check(np.array([100, 100, 140, 140]),
                                  np.zeros((480, 640, 3), np.uint8), track_id=1)
    assert is_spoof is False and liveness == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("all passed")
