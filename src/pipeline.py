"""
Core pipeline: Detection (SCRFD) → Tracking (ByteTrack) → Embedding (ArcFace) → Matching (FAISS).
The pipeline is stateless per frame — tracking state lives in the Tracker.
"""
import collections
import time
from dataclasses import dataclass, field

import cv2
import faiss
import numpy as np
import supervision as sv
from insightface.app import FaceAnalysis
from trackers import ByteTrackTracker

from .config import Config, GPUProfile
from .zones import load_zones, zone_for


@dataclass
class Detection:
    track_id: int
    bbox: np.ndarray        # [x1, y1, x2, y2]
    embedding: np.ndarray | None = None
    name: str = "Unknown"
    roll: str = ""
    match_score: float = 0.0
    is_spoof: bool = False
    liveness_score: float = 1.0
    observations: int = 0     # frames averaged into this track's embedding; higher = more confident
    zone: str = ""            # named zone containing this face, "" if none defined


@dataclass
class FrameResult:
    frame_idx: int
    timestamp: float
    detections: list[Detection]
    inference_ms: float = 0.0
    num_faces: int = 0


class ArcFaceEmbedder:
    """Wraps InsightFace for detection + embedding."""

    def __init__(self, det_size=(640, 640), use_half=False, providers=None, det_thresh=0.4):
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
        return self.model.get(frame)

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
    """

    def __init__(self, max_age=30, min_hits=2, embed_window=30, frame_rate=15.0):
        self.max_age = max_age
        self.embed_window = embed_window
        self.tracks: dict[int, ByteTrackWrapper._Track] = {}
        self.frame_count = 0
        self._tracker = ByteTrackTracker(
            lost_track_buffer=max_age,
            frame_rate=frame_rate,
            minimum_consecutive_frames=min_hits,
            track_activation_threshold=0.5,
            high_conf_det_threshold=0.5,
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
                     "time_since_update", "name", "roll", "last_embed_frame")

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
            if embeddings[src] is not None:
                track.add_embedding(embeddings[src])
                track.last_embed_frame = self.frame_count

        self._remove_dead()
        return list(self.tracks.values())

    def _remove_dead(self):
        dead = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in dead:
            del self.tracks[tid]


class Pipeline:
    """End-to-end: frame → FrameResult with matched identities."""

    def __init__(self, config: Config, embedder: "ArcFaceEmbedder | None" = None):
        self.config = config
        profile = config.profile

        self.embedder = embedder or ArcFaceEmbedder(
            det_size=profile.det_size,
            use_half=profile.use_half_precision,
        )
        self.tracker = None
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
        self.tracker = ByteTrackWrapper(
            # Frames, so it has to scale with the analysis rate to stay a fixed
            # ~2s of tolerance for someone walking behind an obstruction.
            max_age=profile.analysis_fps * 2,
            min_hits=2,
            embed_window=self.config.track_embed_window,
            frame_rate=profile.analysis_fps,
        )

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

        # Detect faces
        faces = self.embedder.detect(frame)
        if not faces:
            return FrameResult(
                frame_idx=self._frame_idx,
                timestamp=time.time(),
                detections=[],
                inference_ms=(time.time() - t0) * 1000,
            )

        # Cap faces to avoid OOM. Keep the most confident ones: the old key was
        # the embedding's L2 norm, which is not a quality measure of anything.
        if len(faces) > profile.max_faces_per_frame:
            faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
            faces = faces[:profile.max_faces_per_frame]

        # Extract embeddings for all detected faces
        bboxes = []
        embeddings = []
        scores = []
        for face in faces:
            # Float, not int: the Kalman filter works in continuous coordinates
            # and rounding every box to whole pixels feeds it quantisation noise.
            bboxes.append(face.bbox.astype(np.float32))
            # insightface already L2-normalizes this for us
            embeddings.append(face.normed_embedding.astype(np.float32))
            scores.append(float(face.det_score))

        # Track
        tracks = self.tracker.update(bboxes, embeddings, scores)
        active_ids = {t.id for t in tracks}

        # Match against gallery + selective anti-spoof
        detections = []
        for track in tracks:
            det = Detection(
                track_id=track.id,
                bbox=track.bbox.copy(),
                embedding=track.embedding,
            )

            # Gallery match against the track's averaged embedding, so a person
            # seen across many small frames is identified as well as one seen
            # close up. No face-size filter: small faces are the CCTV workload.
            det.observations = track.observations
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

            # Anti-spoof: only check every N-th analysis frame per track,
            # and only for active tracks (not stale/drifted)
            if (spoof_checker
                    and track.embedding is not None
                    and track.time_since_update <= 1
                    and self._frame_idx % profile.spoof_check_every_n == 0):
                is_spoof, liveness = spoof_checker.check(track.bbox, frame, track_id=track.id)
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
            num_faces=len(detections),
        )
