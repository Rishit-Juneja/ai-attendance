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


def test_track_averages_embeddings_over_window():
    """
    Small CCTV faces are matched by averaging a track's observations. If a track
    ever goes back to holding only the latest embedding, distant faces stop
    matching and the whole point of carrying track IDs is lost.
    """
    from src.pipeline import ByteTrackWrapper

    rng = np.random.default_rng(1)
    truth = rng.normal(size=512).astype(np.float32)
    truth /= np.linalg.norm(truth)

    def noisy():
        v = truth + rng.normal(scale=1.4, size=512).astype(np.float32)
        return v / np.linalg.norm(v)

    tracker = ByteTrackWrapper(embed_window=30)
    box = np.array([0, 0, 25, 30])  # a 25px-wide face, CCTV scale

    first = noisy()
    tracker.update([box], [first])
    single = float(tracker.tracks[1].embedding @ truth)

    for _ in range(29):
        tracker.update([box], [noisy()])
    track = tracker.tracks[1]
    averaged = float(track.embedding @ truth)

    assert track.observations == 30, track.observations
    assert averaged > single + 0.1, (
        f"averaging {track.observations} observations scored {averaged:.3f} vs "
        f"{single:.3f} for one frame — the window is not being aggregated"
    )
    assert abs(np.linalg.norm(track.embedding) - 1.0) < 1e-5, "must stay L2-normalized"


def test_track_window_rolls_off_old_identity():
    """Bounded window, so a ByteTrack ID switch doesn't poison the track forever."""
    from src.pipeline import ByteTrackWrapper

    rng = np.random.default_rng(2)
    a, b = rng.normal(size=512), rng.normal(size=512)
    a, b = (a / np.linalg.norm(a)).astype(np.float32), (b / np.linalg.norm(b)).astype(np.float32)

    tracker = ByteTrackWrapper(embed_window=10)
    box = np.array([0, 0, 25, 30])
    for _ in range(10):
        tracker.update([box], [a])
    assert float(tracker.tracks[1].embedding @ a) > 0.99

    for _ in range(10):  # ID switched to a different person
        tracker.update([box], [b])
    assert float(tracker.tracks[1].embedding @ b) > 0.99, "old identity never rolled off"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("all passed")
