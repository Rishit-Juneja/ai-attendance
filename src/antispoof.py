"""
Anti-spoofing / liveness detection.

Strategy:
1. Primary: motion heuristic — compute pixel intensity variance within the face bbox
   across N consecutive frames. Real faces have micro-movements (breathing, blinking,
   slight head motion). Printed photos/screens have near-zero variance.
2. Optional: Silent-Face-Anti-Spoofing model (if installed). Falls back to heuristic.

The spoof check runs on a rolling window per track, not per frame — only when the
track's face has been stable (low motion) for enough frames do we flag it.
"""
import collections
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SpoofState:
    """Per-track spoof detection state."""
    frame_buffer: collections.deque  # recent face crops (grayscale)
    motion_scores: collections.deque  # motion per frame pair
    flagged: bool = False
    check_count: int = 0


class SpoofChecker:
    """
    Motion-based liveness detection.

    For each face bbox, we:
    1. Crop the face region (with padding)
    2. Compute the average absolute pixel difference between consecutive crops
    3. If the average motion is below threshold for N frames → flag as spoof
    """

    def __init__(
        self,
        buffer_size: int = 15,
        movement_threshold: float = 1.5,
        flag_after_n: int = 10,
        face_padding: float = 0.2,
        model_path: str = None,
    ):
        self.buffer_size = buffer_size
        self.movement_threshold = movement_threshold
        self.flag_after_n = flag_after_n
        self.face_padding = face_padding
        self._states: dict[int, SpoofState] = {}
        # When the model is present it replaces the heuristic rather than voting
        # with it. Measured on this camera, the heuristic scores a photo on a
        # screen the same as a live face, so any combination that lets it veto
        # would just reintroduce the hole the model exists to close.
        self.model = SilentFaceLiveness(model_path)

    def check(self, bbox: np.ndarray, frame: np.ndarray, track_id: int = 0) -> tuple[bool, float]:
        """
        Returns (is_spoof, liveness_score).
        liveness_score: 0.0 = definitely spoof, 1.0 = definitely real.
        """
        if self.model.model is not None:
            return self._check_model(bbox, frame, track_id)
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = frame.shape[:2]

        # Pad the bbox
        pad_x = int((x2 - x1) * self.face_padding)
        pad_y = int((y2 - y1) * self.face_padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return False, 1.0

        # Extract and resize face crop
        face_crop = frame[y1:y2, x1:x2]
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))

        # Get or create state for this track
        if track_id not in self._states:
            self._states[track_id] = SpoofState(
                frame_buffer=collections.deque(maxlen=self.buffer_size),
                motion_scores=collections.deque(maxlen=self.buffer_size - 1),
            )

        state = self._states[track_id]
        state.check_count += 1

        if len(state.frame_buffer) > 0:
            prev = state.frame_buffer[-1]
            diff = cv2.absdiff(face_resized, prev)
            motion = float(np.mean(diff))
            state.motion_scores.append(motion)

        state.frame_buffer.append(face_resized)

        # Need enough frames to decide
        if len(state.motion_scores) < self.flag_after_n:
            return False, 1.0

        # Compute average motion over the window
        recent_scores = list(state.motion_scores)[-self.flag_after_n:]
        avg_motion = np.mean(recent_scores)

        # Score: map motion to 0-1 liveness score
        # Below threshold → low liveness → spoof
        # Above threshold → high liveness → real
        liveness = min(1.0, avg_motion / (self.movement_threshold * 3))
        is_spoof = avg_motion < self.movement_threshold

        if is_spoof:
            state.flagged = True
        elif avg_motion > self.movement_threshold * 2:
            # Person is moving normally again — unflag (was probably just standing still)
            state.flagged = False

        return state.flagged, liveness

    def _check_model(self, bbox, frame, track_id: int) -> tuple[bool, float]:
        """
        Same per-track smoothing the heuristic gets, for the same reason: a
        single frame is not evidence. One blurred or half-turned face scoring
        fake would otherwise block a real student's attendance outright.
        """
        score = self.model.predict(bbox, frame)
        if score < 0:
            return False, 1.0

        state = self._states.get(track_id)
        if state is None:
            state = self._states[track_id] = SpoofState(
                frame_buffer=collections.deque(maxlen=self.buffer_size),
                motion_scores=collections.deque(maxlen=self.buffer_size),
            )
        state.check_count += 1
        state.motion_scores.append(score)   # liveness probabilities here

        if len(state.motion_scores) < self.flag_after_n:
            return False, score
        # Median, not mean: one confident wrong frame should not swing a verdict
        # that stops someone being marked present.
        state.flagged = float(np.median(list(state.motion_scores)[-self.flag_after_n:])) < 0.5
        return state.flagged, score

    def reset_track(self, track_id: int):
        self._states.pop(track_id, None)

    def cleanup(self, active_track_ids: set[int]):
        """Remove state for dead tracks."""
        dead = [tid for tid in self._states if tid not in active_track_ids]
        for tid in dead:
            del self._states[tid]


CROP_SCALE = 2.7   # the "2.7" in 2.7_80x80_MiniFASNetV2 — see _crop below
INPUT_SIZE = 80


class SilentFaceLiveness:
    """
    MiniFASNetV2 (minivision-ai Silent-Face-Anti-Spoofing, Apache 2.0), run on
    onnxruntime. Classifies a face patch as live / print / replay from texture,
    which is the part that matters here: the motion heuristic cannot see a photo
    displayed on a screen, because the monitor's refresh beats against the camera
    shutter and the resulting banding reads as life. Texture does not care.

    Returns -1.0 with no model file, which is the caller's signal to fall back.

    The three things the earlier version of this class got wrong, all of which
    produce confident nonsense rather than an error:
      * ImageNet mean/std normalization. The model wants a plain /255.
      * A sigmoid over one logit. The head is a 3-class softmax.
      * Reading index 1 as "real". Index 1 is PRINT ATTACK; live is index 0.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self._input = None
        if model_path and Path(model_path).exists():
            import onnxruntime as ort
            self.model = ort.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self._input = self.model.get_inputs()[0].name

    @staticmethod
    def _crop(bbox, frame: np.ndarray) -> np.ndarray:
        """
        Expand the face box 2.7x about its centre before cropping.

        Not cosmetic padding: the model was trained on boxes at this scale, and
        the give-away for a print or replay attack is often OUTSIDE the face —
        a paper edge, a phone bezel, the flat background moving with the face.
        A tight crop throws exactly that evidence away.
        """
        x1, y1, x2, y2 = (float(v) for v in bbox)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * CROP_SCALE / 2
        h, w = frame.shape[:2]
        # Clamp to the frame; letting the box run off the edge and relying on
        # numpy's silent truncation would change the effective scale.
        cx = min(max(cx, half), max(half, w - half))
        cy = min(max(cy, half), max(half, h - half))
        return frame[max(0, int(cy - half)):min(h, int(cy + half)),
                     max(0, int(cx - half)):min(w, int(cx + half))]

    @staticmethod
    def _liveness(logits: np.ndarray) -> float:
        """Softmax over [live, print, replay]; live is index 0."""
        z = np.asarray(logits, dtype=np.float64).ravel()
        e = np.exp(z - z.max())      # shift for numerical stability
        return float((e / e.sum())[0])

    def predict(self, bbox, frame: np.ndarray) -> float:
        """Probability the face is a real one (0=spoof, 1=live), -1.0 if no model."""
        if self.model is None:
            return -1.0
        patch = self._crop(bbox, frame)
        if patch.size == 0:
            return -1.0
        patch = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE))   # BGR, as captured
        patch = (patch.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        return self._liveness(self.model.run(None, {self._input: patch})[0])
