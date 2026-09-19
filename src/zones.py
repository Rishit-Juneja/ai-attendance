"""
Named polygon zones — "count people in this area, ignore the rest of the frame".

Why an area and not a tripwire: attendance asks "who is in the room", which is a
containment question. A tripwire answers "who crossed this line, in which
direction" and needs per-track crossing history. If directional entry/exit turns
out to be what's wanted, that is a different feature on top of this one, not a
reinterpretation of it.

Coordinates are normalized to 0..1 of frame width/height rather than pixels. The
camera is a 640x480 stream that is expected to be reconfigured to 1080p, and
pixel polygons would silently end up pointing at the wrong part of the room the
moment that happens.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import DATA_DIR

ZONES_PATH = DATA_DIR / "zones.json"


@dataclass
class Zone:
    name: str
    points: list[list[float]] = field(default_factory=list)   # normalized [[x,y], ...]

    def polygon(self, width: int, height: int) -> np.ndarray:
        return np.array([[p[0] * width, p[1] * height] for p in self.points], dtype=np.int32)

    def contains(self, x: float, y: float, width: int, height: int) -> bool:
        if len(self.points) < 3:      # not a polygon yet
            return False
        poly = self.polygon(width, height)
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def anchor_point(bbox) -> tuple[float, float]:
    """
    Where a detection "stands" for containment purposes: bottom-centre of the box.

    The pipeline passes a BODY box, so this is the feet — the honest answer to
    "which part of the room is this person in". It degrades to the chin for a
    face in the crowd fallback, where no body was detected; that reads as in or
    out depending on head tilt, so draw zone edges with a little margin.
    """
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[3]))


def load_zones(path: Path = ZONES_PATH) -> list[Zone]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [Zone(name=z["name"], points=z.get("points", [])) for z in raw]


def save_zones(zones: list[Zone], path: Path = ZONES_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        [{"name": z.name, "points": z.points} for z in zones], indent=2))
    return path


def zone_for(bbox, zones: list[Zone], width: int, height: int) -> str:
    """Name of the first zone containing this detection, or "" if none/no zones."""
    if not zones:
        return ""
    x, y = anchor_point(bbox)
    for z in zones:
        if z.contains(x, y, width, height):
            return z.name
    return ""


def draw_zones(frame: np.ndarray, zones: list[Zone]) -> np.ndarray:
    """Outline each zone on the frame, in place."""
    if not zones:
        return frame
    h, w = frame.shape[:2]
    for z in zones:
        if len(z.points) < 3:
            continue
        poly = z.polygon(w, h)
        cv2.polylines(frame, [poly], True, (0, 170, 255), 2)
        cv2.putText(frame, z.name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 170, 255), 2)
    return frame


if __name__ == "__main__":
    # Self-check: containment, normalization, and the empty-zones default.
    z = Zone("room", [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])

    # Same normalized zone must behave identically at both resolutions — this is
    # the whole reason coordinates are not stored in pixels.
    for w, h in [(640, 480), (1920, 1080)]:
        inside = [0.5 * w - 20, 0, 0.5 * w + 20, 0.5 * h]      # chin at centre
        outside = [0.05 * w, 0, 0.05 * w + 40, 0.1 * h]        # chin top-left
        assert zone_for(inside, [z], w, h) == "room", (w, h)
        assert zone_for(outside, [z], w, h) == "", (w, h)

    assert zone_for([0, 0, 10, 10], [], 640, 480) == "", "no zones must mean no filtering"
    assert not Zone("line", [[0.1, 0.1], [0.9, 0.9]]).contains(100, 100, 640, 480), \
        "two points is not a polygon"
    print("zones.py self-check passed")
