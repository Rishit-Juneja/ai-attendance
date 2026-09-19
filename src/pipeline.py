"""
Core pipeline: Detection (YOLO bodies + SCRFD faces) → Tracking (ByteTrack) →
Embedding (ArcFace) → Matching (FAISS).

Tracking runs on BODIES, and faces only supply the embedding that names one.
A face turned away scores ~0.06 against its own enrollment — below the impostor
floor — so identity can only be carried by a track, never re-derived per frame.
Bodies also survive a low analysis rate: at 3 fps a walking person moves ~3 face
widths (no IoU overlap, association impossible) but only ~0.3 body widths.

A second tracker still runs on faces that fall inside no body box. Person
detection under-counts a packed room badly — measured 8 bodies against 98 faces
on this project's own crowd photo — so back rows would disappear entirely if
bodies were the only path.

The pipeline is stateless per frame — tracking state lives in the Trackers.
"""
import collections
import time
from dataclasses import dataclass, field

import cv2
import faiss
import numpy as np
import supervision as sv
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from trackers import ByteTrackTracker

from .config import Config, GPUProfile
from .persons import PersonDetector, face_to_person
from .zones import load_zones, zone_for

# Two trackers both number their IDs from 1, and track_id is a dict key in the
# unresolved queue, the spoof checker and BoxGlide. Without a namespace, body 4
# and face 4 are two different people sharing one record.
FACE_TRACK_ID_OFFSET = 1_000_000

# Confidence a YOLO person box needs before it may start a new body track. Must
# stay above PersonDetector's own floor, or its low-confidence bucket is empty
# and ByteTrack's occlusion-rescue stage has nothing to work with.
BODY_SPAWN_THRESHOLD = 0.35


def _area(bbox) -> float:
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


@dataclass
class Detection:
    track_id: int
    bbox: np.ndarray        # [x1, y1, x2, y2] — the body box when one was found
    face_bbox: np.ndarray | None = None   # None while this person's face is hidden
    embedding: np.ndarray | None = None
    name: str = "Unknown"
    roll: str = ""
    match_score: float = 0.0
    is_spoof: bool = False
    liveness_score: float = 1.0
    observations: int = 0     # frames averaged into this track's embedding; higher = more confident
    zone: str = ""            # named zone containing this person, "" if none defined


@dataclass
class FrameResult:
    frame_idx: int
    timestamp: float
    detections: list[Detection]
    inference_ms: float = 0.0
    num_tracked: int = 0      # people being tracked, face visible or not


@dataclass
class _DetectedFace:
    """
    What detect() returns. Mirrors the attributes of insightface's Face that this
    codebase actually uses, so the batched path is a drop-in for FaceAnalysis.get().
    """
    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    embedding: np.ndarray
    normed_embedding: np.ndarray


class ArcFaceEmbedder:
    """Wraps InsightFace for detection + embedding."""

    def __init__(self, det_size=(640, 640), use_half=False, providers=None, det_thresh=0.4,
                 emb_batch=32):
        import onnxruntime as ort
        available = ort.get_available_providers()
        if providers:
            provider_list = [p for p in providers if p in available]
        elif "CUDAExecutionProvider" in available:
            provider_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            provider_list = ["CPUExecutionProvider"]
        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=provider_list,
        )
        # Below InsightFace's 0.5 default on purpose: ByteTrack's second association
        # stage needs low-confidence boxes to rescue occluded tracks, and at 0.5
        # SCRFD never emits any (measured floor: 0.505). Dropping to 0.4 left pic1
        # and pic3 unchanged at 12/35 faces and only added boxes in a dense crowd,
        # so this buys the low bucket without flooding normal scenes with junk.
        self.model.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
        self.det_model = self.model.det_model
        self.rec_model = self.model.models["recognition"]
        self.emb_batch = emb_batch

        # get_available_providers() lists CUDA even when its libs fail to load at
        # session creation, and ORT then falls back to CPU without raising. Report
        # what the session actually bound to, so "is this on the GPU?" is answerable.
        sessions = [m.session for m in self.model.models.values() if hasattr(m, "session")]
        self.providers = sessions[0].get_providers() if sessions else provider_list
        self.on_gpu = "CUDAExecutionProvider" in self.providers
        if not self.on_gpu and "CUDAExecutionProvider" in provider_list:
            print("[WARN] CUDA was requested but the session fell back to CPU. "
                  "Check cuDNN/CUDA runtime libs.")
        print(f"[ArcFace] running on {'GPU' if self.on_gpu else 'CPU'} ({self.providers[0]})")

    def detect(self, frame: np.ndarray) -> list:
        """
        Detect + embed every face, embedding in batches.

        FaceAnalysis.get() runs recognition one face at a time — measured 9.3ms
        each, so a 98-face frame cost 926ms while detection itself stayed flat at
        10ms. That serial loop, not the detector, is what puts a crowded room out
        of real-time reach. emb_batch has been in config.py all along, read by
        nothing.

        Alignment is identical to the path this replaces: face_align.norm_crop with
        the 5-point landmarks is exactly what rec_model.get() does internally.
        That matters — embeddings taken without it are not comparable to the
        gallery, which is the bug that made every stranger match "Krish".
        """
        bboxes, kpss = self.det_model.detect(frame, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return []

        crops = [face_align.norm_crop(frame, landmark=kpss[i], image_size=112)
                 for i in range(len(bboxes))]
        feats = [self.rec_model.get_feat(crops[i:i + self.emb_batch])
                 for i in range(0, len(crops), self.emb_batch)]
        feats = np.vstack(feats).astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        normed = feats / np.maximum(norms, 1e-9)

        return [_DetectedFace(bbox=bboxes[i, :4].astype(np.float32),
                              kps=kpss[i],
                              det_score=float(bboxes[i, 4]),
                              embedding=feats[i],
                              normed_embedding=normed[i])
                for i in range(len(bboxes))]

    def embed(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        face = self.model.get(frame, det_size=tuple(map(int, bbox)))
        if not face:
            return None
        emb = face[0].embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb


class GalleryMatcher:
    """FAISS-based nearest neighbor search against enrolled gallery."""

    def __init__(self, index_path: str, meta_path: str, live_only: bool = False):
        """
        live_only: drop auto-enrolled CASIA entries (those carry "source_scene").
        Those were embedded by DirectArcFaceEmbedder, which resizes the crop with
        NO landmark alignment, while every live/webcam query goes through
        FaceAnalysis.get(), which does align. The two sets therefore live in
        different regions of the embedding space, and the unaligned ones lose
        every comparison — which made the one aligned entry a magnet that
        captured ~28% of all stranger faces. Keep them for evaluate.py/run_demo.py,
        where queries are unaligned too and the comparison is apples-to-apples.
        """
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            import json
            self.meta = json.load(f)
        self.dim = 512

        if live_only:
            keep = [i for i, m in enumerate(self.meta) if "source_scene" not in m]
            if len(keep) < len(self.meta):
                vecs = self.index.reconstruct_n(0, self.index.ntotal)
                filtered = faiss.IndexFlatIP(self.dim)
                if keep:
                    filtered.add(vecs[keep])
                self.index = filtered
                self.meta = [self.meta[i] for i in keep]

    def match(self, embedding: np.ndarray, threshold: float = 0.45) -> tuple[str, str, float]:
        """Returns (name, roll, similarity). Higher similarity = better match (IndexFlatIP)."""
        if self.index.ntotal == 0:
            return "Unknown", "", 0.0

        query = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        similarities, indices = self.index.search(query, 1)
        score = float(similarities[0][0])
        idx = int(indices[0][0])

        # IndexFlatIP returns inner product = cosine similarity on L2-normalized vecs
        # Higher score = more similar. Match if score exceeds threshold.
        if score > threshold and 0 <= idx < len(self.meta):
            m = self.meta[idx]
            return m["name"], m["roll"], score
        return "Unknown", "", score


class ByteTrackWrapper:
    """
    Real ByteTrack (trackers.ByteTrackTracker) plus our per-track embedding window.

    The library owns association — Kalman motion prediction and the two-stage
    high/low-confidence matching that BYTE is actually named for. We own the
    rolling embedding average, which is what lets a small CCTV face clear the
    match threshold (measured 0.28 -> 0.46 on a 25px face).

    Thresholds sit below the library defaults deliberately. Measured on this
    project's own crowd photos, SCRFD scores a sub-25px face at a median 0.685
    with a floor of 0.505, so the stock track_activation_threshold of 0.7 refuses
    to track 26% of real faces — and they are precisely the distant ones this
    system exists to identify.

    spawn_threshold is a constructor argument because a confidence score is only
    comparable within one detector. Applying the SCRFD-calibrated 0.5 to YOLO
    person scores silently dropped 3 of 11 bodies on pic1 and 7 of 15 on pic3.

    Detections below it are not discarded — BYTE's second association stage still
    uses them to continue tracks that already exist, which is how an occluded
    person keeps their ID. That is the whole reason both detectors are run with
    a floor well below the spawn threshold.
    """

    def __init__(self, max_age=30, min_hits=2, embed_window=30, frame_rate=15.0,
                 spawn_threshold=0.5):
        self.max_age = max_age
        self.embed_window = embed_window
        self.tracks: dict[int, ByteTrackWrapper._Track] = {}
        self.frame_count = 0
        self._tracker = ByteTrackTracker(
            lost_track_buffer=max_age,
            frame_rate=frame_rate,
            minimum_consecutive_frames=min_hits,
            # One number, set twice, because the library gates spawning on both:
            # a detection must land in the high-confidence bucket AND clear the
            # activation threshold. Setting only one of them changes nothing,
            # which cost an afternoon to discover.
            track_activation_threshold=spawn_threshold,
            high_conf_det_threshold=spawn_threshold,
            minimum_iou_threshold=0.1,
        )

    class _Track:
        """
        Holds a rolling window of this person's embeddings rather than just the
        latest one. A single small/blurry CCTV face gives a noisy embedding, but
        the noise is largely independent frame to frame, so averaging the window
        cancels it: measured 0.28 -> 0.46 genuine on a 25px face, while impostor
        scores stay put. This is the whole point of carrying track IDs around.

        Bounded window (not a cumulative mean) so that if ByteTrack hands this ID
        to a different person, the old identity rolls off instead of poisoning the
        average forever.
        """

        __slots__ = ("id", "bbox", "_embeddings", "age", "hits",
                     "time_since_update", "name", "roll", "last_embed_frame", "src_idx")

        def __init__(self, tid, bbox, embedding=None, window=30):
            self.id = tid
            self.bbox = bbox
            self._embeddings = collections.deque(maxlen=window)
            self.age = 0
            self.hits = 1
            self.time_since_update = 0
            self.name = "Unknown"
            self.roll = ""
            self.last_embed_frame = 0
            # Which row of THIS frame's input list matched. Only meaningful on a
            # track that was updated this frame, which is all update() returns.
            self.src_idx = -1
            if embedding is not None:
                self.add_embedding(embedding)

        def add_embedding(self, emb: np.ndarray):
            self._embeddings.append(emb)

        @property
        def embedding(self) -> np.ndarray | None:
            """Mean of the window, re-normalized. None until we have one."""
            if not self._embeddings:
                return None
            mean = np.mean(self._embeddings, axis=0)
            norm = np.linalg.norm(mean)
            return mean / norm if norm > 0 else mean

        @property
        def observations(self) -> int:
            """How many frames back this identity. More = more trustworthy."""
            return len(self._embeddings)

    def update(
        self,
        detections: list[np.ndarray],
        embeddings: list[np.ndarray | None] | None = None,
        scores: list[float] | None = None,
    ) -> list[_Track]:
        """
        detections: list of [x1,y1,x2,y2] arrays
        embeddings: parallel list of embeddings (or None if not computed yet)
        scores:     parallel list of SCRFD confidences. ByteTrack splits its two
                    association stages on these; without them the low-confidence
                    rescue that handles occlusion cannot happen at all.
        Returns list of active tracks.
        """
        self.frame_count += 1
        n = len(detections)
        if embeddings is None:
            embeddings = [None] * n
        if scores is None:
            scores = [1.0] * n

        for track in self.tracks.values():
            track.age += 1
            track.time_since_update += 1

        if n:
            dets = sv.Detections(
                xyxy=np.asarray(detections, dtype=np.float32).reshape(-1, 4),
                confidence=np.asarray(scores, dtype=np.float32),
                # The tracker reorders and drops rows, so carry each detection's
                # input position through as data rather than zipping by index on
                # the way back out. Verified: it returned idx=[1,0] on frame 0.
                data={"idx": np.arange(n)},
            )
        else:
            dets = sv.Detections.empty()

        tracked = self._tracker.update(dets)

        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            # -1 until minimum_consecutive_frames is satisfied. A detection that
            # has not yet earned an ID is a maybe, not a person — emitting it is
            # how the old tracker turned single-frame false positives into
            # attendees.
            if tid < 0:
                continue
            src = int(tracked.data["idx"][i])
            track = self.tracks.get(tid)
            if track is None:
                track = self._Track(tid, tracked.xyxy[i], window=self.embed_window)
                self.tracks[tid] = track
            track.bbox = tracked.xyxy[i]
            track.hits += 1
            track.time_since_update = 0
            track.src_idx = src
            if embeddings[src] is not None:
                track.add_embedding(embeddings[src])
                track.last_embed_frame = self.frame_count

        self._remove_dead()
        # Only tracks matched THIS frame. self.tracks keeps lost ones alive for
        # max_age so their embedding window survives a brief occlusion, but
        # emitting them as live detections meant a person who left the frame kept
        # a ghost box at their last position for ~2s — counted as present, and
        # overlapping the new track when they came back, which is what raised
        # "multiple_overlapping IoU=0.67" for a single person on the live camera.
        return [t for t in self.tracks.values() if t.time_since_update == 0]

    def _remove_dead(self):
        dead = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in dead:
            del self.tracks[tid]


class Pipeline:
    """End-to-end: frame → FrameResult with matched identities."""

    def __init__(self, config: Config, embedder: "ArcFaceEmbedder | None" = None,
                 person_detector: "PersonDetector | None" = None):
        self.config = config
        profile = config.profile

        self.embedder = embedder or ArcFaceEmbedder(
            det_size=profile.det_size,
            use_half=profile.use_half_precision,
            emb_batch=profile.emb_batch,
        )
        # Optional: without it the pipeline degrades to the old face-only
        # behaviour rather than refusing to start. That matters because the
        # .onnx is a build artefact, not something in the repo.
        if person_detector is None:
            try:
                person_detector = PersonDetector()
            except Exception as exc:  # noqa: BLE001 - missing model or no ORT
                print(f"[WARN] person detection off ({exc}). Faces only — "
                      "identity will not survive someone turning around.")
        self.person_detector = person_detector

        self.tracker = None
        self.body_tracker = None
        self.reset_tracker()
        self.matcher = None  # lazy-loaded
        self.zones = load_zones()  # empty list = whole frame counts

        self._frame_interval = 1.0 / profile.analysis_fps
        self._last_analysis_time = 0.0
        self._frame_idx = 0

    def reset_tracker(self):
        """
        Fresh tracker for a new session — IDs and embedding windows must not carry
        across sessions. Every caller goes through here so the settings live in one
        place; three separate call sites used to repeat them, and one of them
        quietly omitted embed_window.
        """
        profile = self.config.profile

        def build(spawn_threshold=0.5):
            return ByteTrackWrapper(
                spawn_threshold=spawn_threshold,
                # Frames, so it has to scale with the analysis rate to stay a fixed
                # number of seconds of tolerance. Measured on the live camera: at 2s
                # (40 frames) turning away for ~9s destroyed the track, and the person
                # came back nameless for 5.4s until a clean frontal view rebuilt the
                # embedding window. Identity is sticky per track, so a longer buffer
                # is what actually carries a name through a face being hidden.
                max_age=max(1, int(profile.analysis_fps * self.config.track_lost_sec)),
                min_hits=2,
                embed_window=self.config.track_embed_window,
                frame_rate=profile.analysis_fps,
            )

        # 0.35 for bodies, measured: YOLO person scores in a crowded room run
        # 0.37-0.91, so the face-calibrated 0.5 sat in the middle of the real
        # distribution and refused to start a track for a third of the people in
        # the room. PersonDetector emits from 0.25, leaving 0.25-0.35 as a
        # genuine low-confidence bucket for continuing occluded tracks.
        self.body_tracker = build(BODY_SPAWN_THRESHOLD)   # carries identity
        self.tracker = build()                 # faces belonging to no visible body

    def load_gallery(self, index_path: str, meta_path: str, live_only: bool = False):
        self.matcher = GalleryMatcher(index_path, meta_path, live_only=live_only)

    def should_analyze(self) -> bool:
        now = time.monotonic()
        if now - self._last_analysis_time >= self._frame_interval:
            self._last_analysis_time = now
            return True
        return False

    def process_frame(self, frame: np.ndarray, spoof_checker=None) -> FrameResult:
        t0 = time.time()
        self._frame_idx += 1
        profile = self.config.profile

        faces = self.embedder.detect(frame)

        # Cap faces to avoid OOM. Keep the most confident ones: the old key was
        # the embedding's L2 norm, which is not a quality measure of anything.
        if len(faces) > profile.max_faces_per_frame:
            faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
            faces = faces[:profile.max_faces_per_frame]

        persons = self.person_detector.detect(frame) if self.person_detector else []

        # Assign each face to the body containing it. A face inside a body box IS
        # that body — counting both would make one person two attendees.
        # ponytail: O(faces x bodies) in Python; ~3k comparisons at the measured
        # worst case (98 faces, 33 bodies), well under a millisecond. Spatial
        # binning only if a much wider camera changes those numbers.
        owners = [face_to_person(f.bbox, persons) for f in faces]

        # One body box often contains two faces in a crowd — the person it was
        # drawn around, and someone standing behind their shoulder whose own body
        # the detector missed. The larger face wins the body; the loser is NOT
        # discarded, it falls through to the face tracker below. Dropping it
        # deleted 17 of 35 people from this project's own crowd photo.
        claimed: list = [None] * len(persons)      # (face index, face) per body
        for i, (face, owner) in enumerate(zip(faces, owners)):
            if owner is None:
                continue
            cur = claimed[owner]
            if cur is None or _area(face.bbox) > _area(cur[1].bbox):
                claimed[owner] = (i, face)

        body_faces = [None if c is None else c[1] for c in claimed]

        # Float, not int: the Kalman filter works in continuous coordinates and
        # rounding every box to whole pixels feeds it quantisation noise.
        # insightface already L2-normalizes the embeddings for us.
        body_tracks = self.body_tracker.update(
            [p.bbox.astype(np.float32) for p in persons],
            [None if f is None else f.normed_embedding.astype(np.float32) for f in body_faces],
            [p.score for p in persons],
        )

        # A face is only represented by its body if that body actually became a
        # track. Not every detection does — ByteTrack spawns tracks solely from
        # its high-confidence bucket, so a body below the split is reported by
        # nothing. Checking rather than assuming means the person still gets
        # counted, by the face path, however the detector is tuned.
        tracked_bodies = {t.src_idx for t in body_tracks}
        covered = {c[0] for j, c in enumerate(claimed)
                   if c is not None and j in tracked_bodies}
        orphans = [f for i, f in enumerate(faces) if i not in covered]
        face_tracks = self.tracker.update(
            [f.bbox.astype(np.float32) for f in orphans],
            [f.normed_embedding.astype(np.float32) for f in orphans],
            [float(f.det_score) for f in orphans],
        )

        # (track, its face this frame, id namespace). Body tracks first so a
        # person with a visible face is reported as a body, not twice.
        rows = [(t, body_faces[t.src_idx], 0) for t in body_tracks]
        rows += [(t, orphans[t.src_idx], FACE_TRACK_ID_OFFSET) for t in face_tracks]
        if not rows:
            return FrameResult(
                frame_idx=self._frame_idx,
                timestamp=time.time(),
                detections=[],
                inference_ms=(time.time() - t0) * 1000,
            )

        active_ids = {t.id + off for t, _, off in rows}

        # Match against gallery + selective anti-spoof
        detections = []
        for track, face, id_offset in rows:
            det = Detection(
                track_id=track.id + id_offset,
                bbox=track.bbox.copy(),
                face_bbox=None if face is None else face.bbox.astype(np.float32),
                embedding=track.embedding,
            )

            # Gallery match against the track's averaged embedding, so a person
            # seen across many small frames is identified as well as one seen
            # close up. No face-size filter: small faces are the CCTV workload.
            det.observations = track.observations
            # Bottom-centre of a body box is where the person is standing. This
            # retires the chin anchor, which sat at head height and put people in
            # the wrong zone depending on how they leaned.
            det.zone = zone_for(track.bbox, self.zones, frame.shape[1], frame.shape[0])
            if track.embedding is not None and self.matcher:
                name, roll, score = self.matcher.match(track.embedding, self.config.match_threshold)
                if name != "Unknown":
                    track.name, track.roll = name, roll
                # Identity is sticky for the life of the track. The averaged
                # embedding only improves as observations accumulate, so a frame
                # that dips back under the threshold is noise, not a different
                # person — and re-deciding every frame made the label strobe
                # between the name and Unknown for anyone scoring near 0.32.
                # The window still rolls off on an ID switch, so a genuinely new
                # person in this slot re-matches rather than inheriting the name.
                det.name = track.name
                det.roll = track.roll
                det.match_score = score

            # Anti-spoof: only check every N-th analysis frame per track, and
            # only on the face box — the heuristic is micro-motion in a face
            # crop, and running it over a whole body measures walking.
            if (spoof_checker
                    and det.face_bbox is not None
                    and self._frame_idx % profile.spoof_check_every_n == 0):
                is_spoof, liveness = spoof_checker.check(det.face_bbox, frame,
                                                         track_id=det.track_id)
                det.is_spoof = is_spoof
                det.liveness_score = liveness

            detections.append(det)

        # Periodic cleanup of spoof state for dead tracks
        if spoof_checker and self._frame_idx % 60 == 0:
            spoof_checker.cleanup(active_ids)

        return FrameResult(
            frame_idx=self._frame_idx,
            timestamp=time.time(),
            detections=detections,
            inference_ms=(time.time() - t0) * 1000,
            num_tracked=len(detections),
        )
