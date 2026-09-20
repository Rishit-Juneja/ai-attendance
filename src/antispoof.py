"""
Anti-spoofing / liveness detection.

Strategy:
1. Primary: MiniFASNetV2 texture model, when the weights are present. Texture is
   the axis that actually separates paper and screens from skin.
2. Fallback: motion heuristic — mean absolute pixel difference within the face
   bbox across consecutive frames. Catches print and NOTHING ELSE; a photo on an
   LCD produced motion in 69% of samples here, because the monitor's refresh
   beats against the camera shutter and the banding reads as life.

Either way the verdict is smoothed over a rolling window per track, and either
way a check that could not run returns -1.0 rather than a clean score. "Judged
alive" and "never judged" are different facts and a report that collapses them
is claiming a safety property this system does not have.
"""
import collections
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CROP_SCALE = 2.7   # the "2.7" in 2.7_80x80_MiniFASNetV2 — see SilentFaceLiveness._crop
INPUT_SIZE = 80

# MEASURED, because the model card is wrong on both counts. It documents a /255
# input and a [live, print, replay] head with live at index 0. Probed with
# tools/probe_spoof_preproc.py against this export:
#
#   /255 : random noise, pure black, pure white and real faces ALL return
#          p=[0.000 0.007 0.992]. The model is returning a constant — the
#          normalization is already in the graph, so dividing again flattens
#          the input to a range the first conv cannot separate.
#   raw  : real faces  p=[0.010 0.250 0.740]
#          random noise p=[0.003 0.983 0.014]
#
# So the input is raw 0-255 and real faces land on class 2, not class 0. Reading
# index 0 would have marked every living person a spoof with total confidence.
LIVE_CLASS = 2

# Below this face width the model is not given enough to judge and its answer is
# not used. Texture is the entire basis for the verdict, blur reads like the
# flatness of print, and a false spoof flag COSTS A REAL STUDENT THEIR
# ATTENDANCE — so abstaining is the safe failure.
#
# 65, not the 30 first guessed from the 80x80 input size. Measured against a real
# person on the CCTV feed, sorted by face width:
#     47px 0.011   48px 0.035   55px 0.290   61px 0.010   62px 0.024
#     66px 0.885   72px 0.883   74px 0.970   76px 0.913   77px 0.986
# A living person falls off a cliff below ~65px. PROVISIONAL: one 20s capture,
# and the subject was looking down for most of it, so face width is confounded
# with pose — leaning in makes the face both bigger and more frontal. Re-measure
# with tools/verify_spoof_model.py --rtsp before relying on the exact value.
#
# Consequence worth stating plainly: a real classroom puts faces at 20-30px, so
# this check will almost never run there. That is not a tuning problem. A model
# cannot read skin texture that the sensor did not capture.
LIVENESS_MIN_FACE_PX = 65


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
        movement_threshold: float = 1.0,   # measured; see config.spoof_pixel_movement_thresh
        flag_after_n: int = 10,
        face_padding: float = 0.2,
        model_path: str = None,
        min_face_px: int = LIVENESS_MIN_FACE_PX,
    ):
        self.buffer_size = buffer_size
        self.movement_threshold = movement_threshold
        self.flag_after_n = flag_after_n
        self.face_padding = face_padding
        self.min_face_px = min_face_px
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
            return False, -1.0      # too small to diff; abstain, don't approve

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

        # Need enough frames to decide. Abstain until then — reporting 1.0 here
        # credited a clean liveness check to every track in its first 10 frames,
        # which for a face that appears briefly is every frame it ever gets.
        if len(state.motion_scores) < self.flag_after_n:
            return False, -1.0

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
        # -1.0, not 1.0. "Too small to judge" and "judged, looks alive" are
        # different facts, and collapsing them is how a system ends up reporting
        # zero spoofs for a room it never checked.
        if float(bbox[2]) - float(bbox[0]) < self.min_face_px:
            return False, -1.0
        score = self.model.predict(bbox, frame)
        if score < 0:
            return False, -1.0

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


class SilentFaceLiveness:
    """
    MiniFASNetV2 (minivision-ai Silent-Face-Anti-Spoofing, Apache 2.0), run on
    onnxruntime. Classifies a face patch as live / print / replay from texture,
    which is the part that matters here: the motion heuristic cannot see a photo
    displayed on a screen, because the monitor's refresh beats against the camera
    shutter and the resulting banding reads as life. Texture does not care.

    Returns -1.0 with no model file, which is the caller's signal to fall back.

    Preprocessing and the output head are MEASURED against this export, not read
    off the model card — the card is wrong on both and both failures are silent.
    See the note above LIVE_CLASS before changing anything here, and re-run
    tools/probe_spoof_preproc.py if you swap the weights.
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
        """Softmax over the 3-class head; see LIVE_CLASS for why index 2."""
        z = np.asarray(logits, dtype=np.float64).ravel()
        e = np.exp(z - z.max())      # shift for numerical stability
        return float((e / e.sum())[LIVE_CLASS])

    def predict(self, bbox, frame: np.ndarray) -> float:
        """Probability the face is a real one (0=spoof, 1=live), -1.0 if no model."""
        if self.model is None:
            return -1.0
        patch = self._crop(bbox, frame)
        if patch.size == 0:
            return -1.0
        # Raw 0-255, BGR, NCHW. NOT /255 — see the note on LIVE_CLASS above; the
        # normalization is baked into the graph and dividing again flattens it.
        patch = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE))   # BGR, as captured
        patch = patch.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        return self._liveness(self.model.run(None, {self._input: patch})[0])
