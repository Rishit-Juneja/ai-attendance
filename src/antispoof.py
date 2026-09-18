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
    ):
        self.buffer_size = buffer_size
        self.movement_threshold = movement_threshold
        self.flag_after_n = flag_after_n
        self.face_padding = face_padding
        self._states: dict[int, SpoofState] = {}

    def check(self, bbox: np.ndarray, frame: np.ndarray, track_id: int = 0) -> tuple[bool, float]:
        """
        Returns (is_spoof, liveness_score).
        liveness_score: 0.0 = definitely spoof, 1.0 = definitely real.
        """
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

    def reset_track(self, track_id: int):
        self._states.pop(track_id, None)

    def cleanup(self, active_track_ids: set[int]):
        """Remove state for dead tracks."""
        dead = [tid for tid in self._states if tid not in active_track_ids]
        for tid in dead:
            del self._states[tid]


class SilentFaceLiveness:
    """
    Optional: use the Silent-Face-Anti-Spoofing ONNX model if available.
    This is a lightweight CNN that classifies face patches as real/fake.
    Falls back to motion heuristic if model not found.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        if model_path:
            try:
                import onnxruntime as ort
                self.model = ort.InferenceSession(model_path)
            except Exception:
                pass

    def predict(self, face_patch: np.ndarray) -> float:
        """Returns probability of being real (0=fake, 1=real)."""
        if self.model is None:
            return -1.0  # signal fallback

        # Preprocess: resize to 80x80, normalize
        patch = cv2.resize(face_patch, (80, 80))
        patch = patch.astype(np.float32) / 255.0
        patch = (patch - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        patch = patch.transpose(2, 0, 1)[np.newaxis]

        input_name = self.model.get_inputs()[0].name
        output = self.model.run(None, {input_name: patch.astype(np.float32)})
        prob = 1.0 / (1.0 + np.exp(-output[0][0][1]))  # sigmoid for real class
        return float(prob)
