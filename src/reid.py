"""
Session-scoped re-identification: stitching a person's new track to the name
their old one had.

Tracking already survives a head turned away or a head down — ByteTrack holds
the body. What it cannot survive is the person leaving the camera's view
entirely: the track dies, and the one that appears when they walk back in starts
nameless and has to earn its identity from the face again.

The cascade, strictly in this order:

    1. FACE        — decided in pipeline.py, not here. Always wins. This module
                     is only consulted when the face gave no answer.
    2. STATURE     — body box height and aspect. The build of the person.
    3. CLOTHING    — torso colour histogram. Tertiary: in A607 most students
                     wear white shirts, so this separates almost nobody on its
                     own and is only ever a tie-breaker.
    4. ABSTAIN     — no confident answer goes to the teacher queue rather than
                     to a guess. A wrong merge is worse than no merge: it does
                     not merely lose one student their attendance, it awards it
                     to somebody else.

WHY THIS ONLY RUNS AT THE DOOR
Pixel height is a function of distance, so across a deep room it measures where
someone is standing, not how tall they are — a student at the back of A607 is
roughly half the pixel height of one at the front, and matching on raw height
would confidently pair strangers who happen to share a row.

Re-entry happens at the door, which is one fixed spot, so door sighting against
door sighting holds scale constant and height becomes directly comparable. That
is the entire reason this module is cheap.
# ponytail: door-region only. Matching anywhere in the room needs a ground-plane
# fit (body height against floor y, one pass over a recording) to normalise
# stature; add that only if door-scoped matching is measured to fall short.
"""
from dataclasses import dataclass

import cv2
import numpy as np


# Weights across the three cues. Stature outranks colour deliberately — see the
# white-shirt note above.
# ponytail: UNMEASURED starting point, unlike the thresholds in antispoof.py
# which were probed against real footage. Re-fit with a labelled pair set from
# the recordings before trusting the exact numbers.
W_HEIGHT, W_ASPECT, W_COLOUR = 0.45, 0.25, 0.30

# A match must beat the runner-up by this much. This, not the absolute score, is
# what keeps a room of identical white shirts safe: when several people look
# alike their scores bunch together, the margin collapses, and the match is
# refused automatically — per person, at runtime, with nobody tuning a number
# for "how distinctive is a shirt today".
DEFAULT_MARGIN = 0.12
DEFAULT_MIN_SCORE = 0.55

# Heights closer than this (as a ratio) or it is not the same person, full stop.
#
# A hard gate rather than another weighted term, because a weighted sum does not
# actually express "clothing is tertiary": colour is worth 0.30, so two people of
# visibly different build in the same white shirt still cleared the line together.
# The self-check caught exactly that — a 300px body matched a 180px one at 0.72.
# Gating means no amount of matching shirt can rescue a stature mismatch, which
# is what tertiary has to mean.
#
# 0.85 allows the ~15% a real person varies by at a fixed distance: stride,
# posture, a rucksack, the box clipping at the frame edge. Only meaningful
# because matching is door-scoped and therefore scale-constant.
HEIGHT_GATE = 0.85


@dataclass
class Signature:
    height: float           # body box height, px — only comparable at fixed scale
    aspect: float           # height/width; scale-free, so it survives distance
    colour: np.ndarray      # HSV histogram over the torso band


def signature(frame: np.ndarray, bbox) -> Signature | None:
    """
    Appearance of one body box. None if the box is too small to describe.

    The colour band runs shoulders-to-waist rather than over the whole box: the
    head brings in hair and skin (which are similar across a cohort and would
    wash out the little variation the shirt gives), and the legs bring in the
    floor, which is the same polished marble for everybody in the corridor.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = y2 - y1, x2 - x1
    if h < 40 or w < 15:
        return None         # too few pixels to describe; abstaining beats guessing

    fh, fw = frame.shape[:2]
    top = max(0, y1 + int(0.18 * h))        # below the head
    bot = min(fh, y1 + int(0.55 * h))       # above the legs
    crop = frame[top:bot, max(0, x1):min(fw, x2)]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Hue x saturation. Value is left out on purpose: it tracks how brightly that
    # part of the corridor happens to be lit, which changes between the doorway
    # and the window end and has nothing to do with the garment.
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return Signature(height=float(h), aspect=float(h) / max(w, 1), colour=hist.flatten())


def _ratio(a: float, b: float) -> float:
    """1.0 for identical, falling to 0 as they diverge. Scale-free."""
    hi = max(abs(a), abs(b))
    return 1.0 - abs(a - b) / hi if hi > 0 else 0.0


def compare(a: Signature, b: Signature) -> float:
    """Similarity in 0..1. Face is NOT part of this — see the module docstring."""
    if a is None or b is None:
        return 0.0
    height = _ratio(a.height, b.height)
    if height < HEIGHT_GATE:
        return 0.0         # different build; nothing else gets a vote
    colour = float(cv2.compareHist(a.colour, b.colour, cv2.HISTCMP_CORREL))
    return (W_HEIGHT * height
            + W_ASPECT * _ratio(a.aspect, b.aspect)
            # Correlation is -1..1 and a negative one means "actively unlike",
            # which is not evidence of anything here; floor it rather than let it
            # drag a good stature match below the line.
            + W_COLOUR * max(0.0, colour))


def best_match(query: Signature, gallery: dict, exclude=(),
               margin: float = DEFAULT_MARGIN,
               min_score: float = DEFAULT_MIN_SCORE) -> tuple | None:
    """
    Best entry in `gallery` for `query`, or None when the answer is not clear.

    `gallery` maps key -> Signature; `exclude` is everyone currently visible in
    some other track. Excluding them is the cheapest guard there is: a person
    already on screen cannot also be the one who just walked back in, and that
    single rule removes most of the ways this can go wrong.

    Returns None — meaning "ask a human" — when the gallery is empty, when
    nothing clears `min_score`, or when the top two are within `margin`.
    """
    scored = sorted(((compare(query, sig), key)
                     for key, sig in gallery.items() if key not in exclude),
                    reverse=True)
    if not scored or scored[0][0] < min_score:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
        return None         # two plausible people: refuse, do not pick one
    return scored[0][1], round(scored[0][0], 3)


if __name__ == "__main__":
    # Self-check: the guards, not the weights. Each of these is a way a wrong
    # merge could silently award one student's attendance to another.
    rng = np.random.default_rng(0)

    def body(h, w, hue, sat=200):
        """A synthetic body box: solid shirt colour over the torso band."""
        frame = np.zeros((900, 600, 3), np.uint8)
        img = np.full((h, w, 3), 40, np.uint8)
        img[int(0.18 * h):int(0.55 * h)] = cv2.cvtColor(
            np.full((1, 1, 3), (hue, sat, 220), np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        frame[100:100 + h, 100:100 + w] = img
        return signature(frame, (100, 100, 100 + w, 100 + h))

    tall_red = body(300, 100, 0)
    tall_red2 = body(304, 101, 2)        # same person, a step later
    short_red = body(180, 100, 0)        # same shirt, clearly shorter build
    tall_blue = body(300, 100, 110)

    assert signature(np.zeros((900, 600, 3), np.uint8), (0, 0, 10, 10)) is None, \
        "a box too small to describe must abstain, not return a weak signature"

    assert compare(tall_red, tall_red2) > compare(tall_red, short_red), \
        "stature must outrank an identical shirt"
    assert compare(tall_red, short_red) == 0.0, \
        "a matching shirt must not rescue a mismatched build — clothing is tertiary"
    assert compare(tall_red, tall_red2) > compare(tall_red, tall_blue), \
        "same build and same shirt must beat same build alone"
    assert compare(tall_red, None) == 0.0

    # The white-shirt case: three people of near-identical build and dress. The
    # margin must collapse and the match must be refused.
    clones = {f"p{i}": body(300 + i, 100, 0) for i in range(3)}
    assert best_match(tall_red, clones) is None, \
        "indistinguishable candidates must go to the teacher queue"

    # One clearly different build among them resolves cleanly.
    assert best_match(short_red, {"short": short_red, "tall": tall_red})[0] == "short"

    # Mutual exclusion: someone already on screen cannot be the returner, so the
    # answer must change rather than quietly stay the same.
    two = {"a": tall_red, "b": short_red}
    assert best_match(tall_red, two)[0] == "a"
    assert best_match(tall_red, two, exclude=("a",)) is None, \
        "excluding the true match must abstain, not fall through to the next body"

    assert best_match(tall_red, {}) is None, "empty gallery must abstain"
    assert best_match(tall_red, {"only": tall_blue}, min_score=0.99) is None, \
        "a lone candidate still has to clear min_score"

    print("reid.py self-check passed")
