"""
Attendance logger: tracks entry/exit timestamps, computes duration,
logs anomalies (spoof, unknown faces, hiding face).
"""
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import LOGS_DIR


@dataclass
class PersonRecord:
    name: str
    roll: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    duration_sec: float = 0.0
    is_present: bool = True
    detection_count: int = 0
    spoof_flags: int = 0
    anomaly_flags: list = field(default_factory=list)


@dataclass
class Alert:
    timestamp: str
    alert_type: str  # "spoof", "unknown_face", "face_hiding", "multiple_overlapping"
    track_id: int
    details: str


class AttendanceLogger:
    def __init__(self, session_name: str = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{ts}"
        self.log_dir = LOGS_DIR / self.session_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.records: dict[str, PersonRecord] = {}  # keyed by roll
        self.alerts: list[Alert] = []
        self.frame_log: list[dict] = []
        self._entry_cooldown: dict[str, float] = {}
        self.cooldown_sec = 30.0

        # Face hiding detection: track consecutive frames where known person is absent
        self._recently_present: dict[str, float] = {}  # roll → last timestamp seen
        self._face_hiding_alerted: set[str] = set()  # avoid duplicate alerts per session

    def process_detections(self, detections: list, frame_idx: int, timestamp: float):
        """Process pipeline detections into attendance records."""
        present_rolls = set()
        active_bboxes = []

        for det in detections:
            # Collect bboxes for overlap detection
            active_bboxes.append(det.bbox)

            if det.name == "Unknown":
                self._log_alert("unknown_face", det.track_id,
                                f"Unrecognized face (score={det.match_score:.3f})")
                continue

            roll = det.roll
            present_rolls.add(roll)
            self._recently_present[roll] = timestamp
            now_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

            if roll not in self.records:
                self.records[roll] = PersonRecord(
                    name=det.name,
                    roll=roll,
                    first_seen=timestamp,
                    entry_time=now_str,
                )

            rec = self.records[roll]
            rec.last_seen = timestamp
            rec.is_present = True
            rec.detection_count += 1

            # Entry logging with cooldown
            if roll not in self._entry_cooldown or \
               (timestamp - self._entry_cooldown[roll]) > self.cooldown_sec:
                if rec.detection_count <= 3:  # likely an entry
                    rec.entry_time = now_str
                self._entry_cooldown[roll] = timestamp

            # Spoof detection
            if det.is_spoof:
                rec.spoof_flags += 1
                self._log_alert("spoof", det.track_id,
                                f"{det.name} ({roll}): spoof detected, liveness={det.liveness_score:.3f}")
                rec.anomaly_flags.append(f"spoof@{now_str}")

        # Face hiding detection: known person was recently present but face disappeared
        # This suggests they're covering their face while still in the room
        for roll, rec in self.records.items():
            if roll not in present_rolls and rec.is_present:
                last_seen = self._recently_present.get(roll, 0)
                elapsed = timestamp - last_seen
                # If absent for 10-60 seconds (not long enough to be "left the room" at 300s),
                # but face disappeared suddenly, flag potential face hiding
                if 10 < elapsed < 300 and roll not in self._face_hiding_alerted:
                    self._log_alert("face_hiding", 0,
                                    f"{rec.name} ({roll}): face disappeared after being present, possible hiding")
                    rec.anomaly_flags.append(f"face_hiding@{datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
                    self._face_hiding_alerted.add(roll)
                elif elapsed > 300:  # 5 minutes absent → mark absent
                    rec.is_present = False
                    rec.exit_time = datetime.fromtimestamp(rec.last_seen).strftime("%H:%M:%S")
                    rec.duration_sec = rec.last_seen - rec.first_seen

        # Multiple overlapping faces detection:
        # Two or more bboxes with high IoU but different track IDs = possible photo/screen spoof
        if len(active_bboxes) >= 2:
            self._check_overlapping_faces(active_bboxes, detections, timestamp)

    def _check_overlapping_faces(self, bboxes, detections, timestamp):
        """Flag when multiple face bboxes significantly overlap (potential spoof)."""
        import numpy as np
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                b1, b2 = bboxes[i], bboxes[j]
                # Compute IoU
                xa = max(b1[0], b2[0])
                ya = max(b1[1], b2[1])
                xb = min(b1[2], b2[2])
                yb = min(b1[3], b2[3])
                inter = max(0, xb - xa) * max(0, yb - ya)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = area1 + area2 - inter
                iou = inter / max(union, 1e-6)

                # If IoU > 0.5 and different tracks, flag it
                if iou > 0.5:
                    d1 = detections[i] if i < len(detections) else None
                    d2 = detections[j] if j < len(detections) else None
                    if d1 and d2 and d1.track_id != d2.track_id:
                        names = f"track {d1.track_id} and track {d2.track_id}"
                        self._log_alert("multiple_overlapping", d1.track_id,
                                        f"Overlapping faces (IoU={iou:.2f}): {names} — possible photo spoof")
                        return  # one alert per frame is enough

    def _log_alert(self, alert_type: str, track_id: int, details: str):
        alert = Alert(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            alert_type=alert_type,
            track_id=track_id,
            details=details,
        )
        self.alerts.append(alert)

    def get_summary(self) -> dict:
        present = [r for r in self.records.values() if r.is_present]
        absent = [r for r in self.records.values() if not r.is_present]
        spoofs = [r for r in self.records.values() if r.spoof_flags > 0]
        return {
            "session": self.session_name,
            "total_enrolled": len(self.records),
            "present_now": len(present),
            "absent": len(absent),
            "spoof_detected": len(spoofs),
            "alerts": len(self.alerts),
            "persons": [
                {
                    "name": r.name,
                    "roll": r.roll,
                    "entry": r.entry_time,
                    "exit": r.exit_time,
                    "duration_min": round(r.duration_sec / 60, 1),
                    "present": r.is_present,
                    "detections": r.detection_count,
                    "spoofs": r.spoof_flags,
                }
                for r in self.records.values()
            ],
            "alerts_log": [
                {"time": a.timestamp, "type": a.alert_type, "details": a.details}
                for a in self.alerts
            ],
        }

    def save_log(self):
        summary = self.get_summary()
        log_path = self.log_dir / "attendance.json"
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[LOG] Saved attendance to {log_path}")
        return log_path

    def export_csv(self, path: str = None):
        import csv
        if path is None:
            path = self.log_dir / "attendance.csv"

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Roll", "Entry", "Exit", "Duration(min)", "Present", "Detections", "Spoofs"])
            for r in self.records.values():
                writer.writerow([
                    r.name, r.roll, r.entry_time, r.exit_time,
                    round(r.duration_sec / 60, 1), r.is_present,
                    r.detection_count, r.spoof_flags,
                ])
        print(f"[CSV] Saved to {path}")
        return path
