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
from insightface.app import FaceAnalysis

from .config import Config, GPUProfile


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


@dataclass
class FrameResult:
    frame_idx: int
    timestamp: float
    detections: list[Detection]
    inference_ms: float = 0.0
    num_faces: int = 0


class ArcFaceEmbedder:
    """Wraps InsightFace for detection + embedding."""

    def __init__(self, det_size=(640, 640), use_half=False, providers=None):
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
        self.model.prepare(ctx_id=0, det_size=det_size)
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
    Lightweight ByteTrack implementation using scipy's linear_sum_assignment.
    Maintains persistent IDs across frames.
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, embed_window=30):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.embed_window = embed_window
        self.tracks: dict[int, _Track] = {}
        self._next_id = 1
        self.frame_count = 0

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

    def update(self, detections: list[np.ndarray], embeddings: list[np.ndarray | None] | None = None) -> list[_Track]:
        """
        detections: list of [x1,y1,x2,y2] arrays
        embeddings: parallel list of embeddings (or None if not computed yet)
        Returns list of active tracks.
        """
        self.frame_count += 1
        if embeddings is None:
            embeddings = [None] * len(detections)

        # Predict new positions (simple constant-velocity)
        for track in self.tracks.values():
            track.age += 1
            track.time_since_update += 1

        if not detections:
            self._remove_dead()
            return list(self.tracks.values())

        # Compute IoU matrix
        det_array = np.array(detections)
        track_list = list(self.tracks.values())
        if track_list:
            track_array = np.array([t.bbox for t in track_list])
            iou_matrix = self._compute_iou(det_array, track_array)
        else:
            iou_matrix = np.zeros((len(detections), 0))

        # Hungarian assignment
        from scipy.optimize import linear_sum_assignment
        if iou_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        else:
            row_ind, col_ind = np.array([]), np.array([])

        matched_dets = set()
        matched_trks = set()

        for r, c in zip(row_ind, col_ind):
            r, c = int(r), int(c)
            if iou_matrix[r, c] < self.iou_threshold:
                continue
            track = track_list[c]
            track.bbox = det_array[r]
            track.hits += 1
            track.time_since_update = 0
            if embeddings[r] is not None:
                track.add_embedding(embeddings[r])
                track.last_embed_frame = self.frame_count
            matched_dets.add(r)
            matched_trks.add(c)

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                tid = self._next_id
                self._next_id += 1
                emb = embeddings[i] if i < len(embeddings) else None
                self.tracks[tid] = self._Track(tid, det, emb, window=self.embed_window)

        self._remove_dead()
        return list(self.tracks.values())

    def _remove_dead(self):
        dead = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in dead:
            del self.tracks[tid]

    def get_needing_embed(self, reembed_interval: int) -> list[int]:
        """Return track IDs that need re-embedding."""
        need = []
        for tid, t in self.tracks.items():
            if t.time_since_update >= reembed_interval or t.last_embed_frame == 0:
                need.append(tid)
        return need

    @staticmethod
    def _compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of boxes. boxes_a: (N,4), boxes_b: (M,4)."""
        N = len(boxes_a)
        M = len(boxes_b)
        iou = np.zeros((N, M))
        for i in range(N):
            # np.maximum/np.minimum, NOT the builtins: these broadcast one box
            # against the whole (M,4) array. The builtins raise on any M > 1,
            # which meant tracking crashed the moment a second person appeared.
            xa = np.maximum(boxes_a[i, 0], boxes_b[:, 0])
            ya = np.maximum(boxes_a[i, 1], boxes_b[:, 1])
            xb = np.minimum(boxes_a[i, 2], boxes_b[:, 2])
            yb = np.minimum(boxes_a[i, 3], boxes_b[:, 3])
            inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
            area_a = (boxes_a[i, 2] - boxes_a[i, 0]) * (boxes_a[i, 3] - boxes_a[i, 1])
            area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
            union = area_a + area_b - inter
            iou[i] = inter / np.maximum(union, 1e-6)
        return iou


class Pipeline:
    """End-to-end: frame → FrameResult with matched identities."""

    def __init__(self, config: Config, embedder: "ArcFaceEmbedder | None" = None):
        self.config = config
        profile = config.profile

        self.embedder = embedder or ArcFaceEmbedder(
            det_size=profile.det_size,
            use_half=profile.use_half_precision,
        )
        self.tracker = ByteTrackWrapper(max_age=30, min_hits=2, embed_window=config.track_embed_window)
        self.matcher = None  # lazy-loaded

        self._frame_interval = 1.0 / profile.analysis_fps
        self._last_analysis_time = 0.0
        self._frame_idx = 0

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

        # Cap faces to avoid OOM
        if len(faces) > profile.max_faces_per_frame:
            faces = sorted(faces, key=lambda f: np.linalg.norm(f.embedding) if hasattr(f, 'embedding') else 0, reverse=True)
            faces = faces[:profile.max_faces_per_frame]

        # Extract embeddings for all detected faces
        bboxes = []
        embeddings = []
        for face in faces:
            bboxes.append(face.bbox.astype(int))
            # insightface already L2-normalizes this for us
            embeddings.append(face.normed_embedding.astype(np.float32))

        # Track
        tracks = self.tracker.update(bboxes, embeddings)

        # Identify which tracks need (re-)embedding — only embed stale or new tracks
        need_embed_ids = set(self.tracker.get_needing_embed(profile.track_reembed_interval))
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
            if track.embedding is not None and self.matcher:
                name, roll, score = self.matcher.match(track.embedding, self.config.match_threshold)
                det.name = name
                det.roll = roll
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
