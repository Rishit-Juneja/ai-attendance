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

    Geometry is what matters. A ~16cm face at 1.4 m/s is ~13px/frame at 20 fps on
    a 30px box (IoU ~0.4, associates fine) but ~2.9 face-widths/frame at 3 fps
    (IoU 0, associates never). That is why analysis_fps moved 3 -> 20; motion
    prediction cannot rescue the *first* association, since velocity is unknown
    until two frames have already linked.

    What prediction does buy is the gap in the middle here: three frames with no
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


class _Det:
    """Minimal stand-in for pipeline.Detection."""

    def __init__(self, name="Unknown", roll="", track_id=1, score=0.9,
                 is_spoof=False, observations=30, zone=""):
        self.name, self.roll, self.track_id = name, roll, track_id
        self.match_score, self.is_spoof = score, is_spoof
        self.liveness_score, self.observations = 0.1 if is_spoof else 0.9, observations
        self.zone = zone
        self.bbox = np.array([0, 0, 40, 50], dtype=np.float32)


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("all passed")
