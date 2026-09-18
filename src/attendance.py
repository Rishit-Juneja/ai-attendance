"""
Attendance logger.

Presence is DWELL-based, not sighting-based. Being seen does not make you
present; accumulating `min_dwell_sec` inside a zone does. A visit stays open
across gaps shorter than `exit_grace_sec`, so a student whose face is hidden for
a minute keeps accruing time instead of being logged out and back in.

Faces that never match the gallery do not vanish — they become `person1..N` in
the unresolved queue with the same dwell accounting, and a teacher assigns the
real identity afterwards via resolve(), which back-dates the attendance.
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import LOGS_DIR


def _hms(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


@dataclass
class Visit:
    """One continuous stay. `last_seen` is the last frame the person was in shot."""
    entered: float
    last_seen: float
    zone: str = ""
    closed: bool = False

    @property
    def seconds(self) -> float:
        return max(0.0, self.last_seen - self.entered)


@dataclass
class _Dwelling:
    """
    Everything derived from a list of visits. Shared because a known person and
    an unresolved stranger are counted by exactly the same rules — the only
    difference is whether we have a name for them yet.
    """
    zone: str = ""          # zone they were last counted in
    detection_count: int = 0
    visits: list[Visit] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        """Time in zone summed across visits. Gaps between visits don't count."""
        return sum(v.seconds for v in self.visits)

    @property
    def is_present(self) -> bool:
        """In the room right now (a visit is open)."""
        return bool(self.visits) and not self.visits[-1].closed

    @property
    def first_seen(self) -> float:
        return self.visits[0].entered if self.visits else 0.0

    @property
    def last_seen(self) -> float:
        return self.visits[-1].last_seen if self.visits else 0.0

    @property
    def entry_time(self) -> str:
        return _hms(self.first_seen) if self.visits else ""

    @property
    def exit_time(self) -> str:
        closed = [v for v in self.visits if v.closed]
        return _hms(closed[-1].last_seen) if closed else ""


@dataclass
class PersonRecord(_Dwelling):
    name: str = ""
    roll: str = ""
    spoof_flags: int = 0
    resolved_from: str = "" # set when a teacher named this person from the queue
    anomaly_flags: list = field(default_factory=list)
    min_dwell_sec: float = 30.0

    @property
    def status(self) -> str:
        if self.duration_sec >= self.min_dwell_sec:
            return "present" if self.is_present else "left"
        # Seen, but not for long enough to count. Distinct from absent: absent
        # means never seen at all, which this class cannot observe.
        return "brief"


@dataclass
class Unresolved(_Dwelling):
    """A tracked person who never matched the gallery."""
    label: str = ""         # person1, person2, ...
    track_id: int = 0
    crop_path: str = ""
    crop_area: int = 0      # biggest face seen so far, so the teacher gets the best shot


@dataclass
class Alert:
    timestamp: str      # first occurrence
    alert_type: str     # "spoof", "unknown_face", "face_hiding", "multiple_overlapping"
    track_id: int
    details: str
    severity: str = "warn"   # "info" | "warn" | "critical"
    count: int = 1           # times this same subject re-triggered
    last_timestamp: str = ""

    @property
    def is_repeating(self) -> bool:
        return self.count > 1


# A track needs this many averaged observations before "Unknown" means anything.
# A face that just appeared is legitimately unidentified for the first few frames
# while its embedding window fills; alerting on that produced most of the 2103
# alerts in a short test run.
MIN_OBS_BEFORE_UNKNOWN_ALERT = 5


class AttendanceLogger:
    def __init__(self, session_name: str = None, config=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{ts}"
        self.log_dir = LOGS_DIR / self.session_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if config is None:
            from .config import DEFAULT_CONFIG as config
        self.min_dwell_sec = config.min_dwell_sec
        self.exit_grace_sec = config.exit_grace_sec
        self.hiding_alert_after_sec = config.hiding_alert_after_sec

        self.records: dict[str, PersonRecord] = {}      # keyed by roll
        self.unresolved: dict[int, Unresolved] = {}     # keyed by track_id
        self._person_seq = 0
        self.alerts: list[Alert] = []
        self.frame_log: list[dict] = []

        # One row per (alert_type, subject) per session. Without this, an
        # unrecognised face standing in shot raised an alert on every analysed
        # frame — 20/second once the analysis rate went up.
        self._alert_index: dict[tuple[str, str], Alert] = {}

        # Read once per session. A session is started after zones are edited, so
        # this stays consistent for the run rather than shifting mid-session.
        from .zones import load_zones
        self.zones_active = bool(load_zones())

    # ---------------- ingestion ----------------

    def process_detections(self, detections: list, frame_idx: int, timestamp: float,
                           frame=None):
        """
        Fold one analysed frame into the records. `frame` is optional and only
        used to save a face crop for the unresolved queue.
        """
        active_bboxes = []

        for det in detections:
            # Outside every configured zone = not in the room. Skipped before the
            # alert checks too, so a passer-by in the corridor behind the door
            # neither marks attendance nor raises a stranger alert.
            # No zones configured at all means the whole frame counts.
            if self.zones_active and not det.zone:
                continue

            active_bboxes.append(det.bbox)

            # A spoofed face must not accrue dwell for anybody, named or not.
            if det.is_spoof:
                who = f"{det.name} ({det.roll})" if det.name != "Unknown" else f"track {det.track_id}"
                self._log_alert("spoof", det.track_id,
                                f"{who}: spoof detected, liveness={det.liveness_score:.3f}"
                                f" — attendance NOT logged",
                                key=det.roll or str(det.track_id), severity="critical")
                rec = self.records.get(det.roll)
                if rec is not None:
                    rec.spoof_flags += 1
                    rec.anomaly_flags.append(f"spoof@{_hms(timestamp)}")
                continue

            if det.name == "Unknown":
                self._touch_unresolved(det, timestamp, frame)
                continue

            rec = self._record_for(det.name, det.roll)
            self._touch(rec, det.zone, timestamp)

            # Someone identified here cancels their own unresolved entry: the
            # track was a stranger only until the face finally matched.
            self.unresolved.pop(det.track_id, None)

        self._expire(timestamp)

        # Multiple overlapping faces: two bboxes with high IoU but different
        # track IDs = possible photo/screen spoof.
        if len(active_bboxes) >= 2:
            self._check_overlapping_faces(active_bboxes, detections, timestamp)

    def _record_for(self, name: str, roll: str) -> PersonRecord:
        rec = self.records.get(roll)
        if rec is None:
            rec = PersonRecord(name=name, roll=roll, min_dwell_sec=self.min_dwell_sec)
            self.records[roll] = rec
        return rec

    def _touch(self, rec: _Dwelling, zone: str, timestamp: float):
        """Extend the open visit, or start a new one."""
        rec.zone = zone
        rec.detection_count += 1
        if rec.visits and not rec.visits[-1].closed:
            rec.visits[-1].last_seen = timestamp
            rec.visits[-1].zone = zone
        else:
            rec.visits.append(Visit(entered=timestamp, last_seen=timestamp, zone=zone))

    def _touch_unresolved(self, det, timestamp: float, frame=None):
        entry = self.unresolved.get(det.track_id)
        if entry is None:
            # Numbered by how many have ever appeared, not by dict size, so
            # labels never get reused after one is resolved away.
            self._person_seq += 1
            entry = Unresolved(label=f"person{self._person_seq}", track_id=det.track_id)
            self.unresolved[det.track_id] = entry
        self._touch(entry, det.zone, timestamp)
        self._save_crop(entry, det, frame)

        # Wait for the embedding window to fill before calling someone a
        # stranger — see MIN_OBS_BEFORE_UNKNOWN_ALERT.
        if getattr(det, "observations", 0) >= MIN_OBS_BEFORE_UNKNOWN_ALERT:
            self._log_alert("unknown_face", det.track_id,
                            f"{entry.label}: unrecognised face "
                            f"(score={det.match_score:.3f}) — awaiting review",
                            key=str(det.track_id), severity="warn")

    def _save_crop(self, entry: Unresolved, det, frame):
        """Keep the largest face crop seen for this track — the teacher's evidence."""
        if frame is None:
            return
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area <= entry.crop_area:
            return
        import cv2
        h, w = frame.shape[:2]
        pad = int(0.35 * max(x2 - x1, y2 - y1))  # context helps a human far more than a tight crop
        crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size == 0:
            return
        out_dir = self.log_dir / "unresolved"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{entry.label}.jpg"
        if cv2.imwrite(str(path), crop):
            entry.crop_path = str(path)
            entry.crop_area = area

    def _expire(self, now: float):
        """Close visits whose gap has outlived the grace period, and flag hiding."""
        for rec in list(self.records.values()) + list(self.unresolved.values()):
            if not rec.visits or rec.visits[-1].closed:
                continue
            visit = rec.visits[-1]
            elapsed = now - visit.last_seen
            if elapsed > self.exit_grace_sec:
                visit.closed = True
            elif elapsed > self.hiding_alert_after_sec and isinstance(rec, PersonRecord):
                # Still inside the visit, so dwell keeps counting — this is a
                # heads-up, not an exit.
                self._log_alert("face_hiding", 0,
                                f"{rec.name} ({rec.roll}): face gone {int(elapsed)}s "
                                f"while still counted present",
                                key=rec.roll, severity="warn")

    def close_session(self):
        """
        Class is over. Close every open visit so exit times and totals are final
        — without this the last visit stays open forever and the saved report
        shows everybody still in the room.
        """
        for rec in list(self.records.values()) + list(self.unresolved.values()):
            if rec.visits and not rec.visits[-1].closed:
                rec.visits[-1].closed = True

    # ---------------- teacher review ----------------

    def resolve(self, track_id: int, name: str, roll: str) -> PersonRecord | None:
        """
        Assign a real identity to an unresolved person. Their dwell is merged into
        the named record — so someone who sat through the class with their face
        covered gets credited for the whole time, not from the moment of naming.
        """
        entry = self.unresolved.pop(track_id, None)
        if entry is None:
            return None
        rec = self._record_for(name, roll)
        rec.visits.extend(entry.visits)
        rec.visits.sort(key=lambda v: v.entered)
        rec.detection_count += entry.detection_count
        rec.zone = rec.zone or entry.zone
        rec.resolved_from = entry.label
        return rec

    # ---------------- alerts ----------------

    def _check_overlapping_faces(self, bboxes, detections, timestamp):
        """Flag when multiple face bboxes significantly overlap (potential spoof)."""
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                b1, b2 = bboxes[i], bboxes[j]
                xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
                xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0, xb - xa) * max(0, yb - ya)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / max(area1 + area2 - inter, 1e-6)

                if iou > 0.5:
                    d1 = detections[i] if i < len(detections) else None
                    d2 = detections[j] if j < len(detections) else None
                    if d1 and d2 and d1.track_id != d2.track_id:
                        names = f"track {d1.track_id} and track {d2.track_id}"
                        # Keyed on the track pair so two people standing in line
                        # raise one alert, not one per frame for as long as they
                        # stand there. Ordered so (3,7) and (7,3) are one subject.
                        pair = tuple(sorted((d1.track_id, d2.track_id)))
                        self._log_alert("multiple_overlapping", d1.track_id,
                                        f"Overlapping faces (IoU={iou:.2f}): {names} — possible photo spoof",
                                        key=f"{pair[0]}-{pair[1]}", severity="info")
                        return  # one alert per frame is enough

    def _log_alert(self, alert_type: str, track_id: int, details: str,
                   key: str = "", severity: str = "warn"):
        """
        Collapse repeats of the same (alert_type, subject) into one row with a
        count, instead of appending another every analysed frame. `key` is the
        subject — a roll number for a person, a track id for a stranger.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        idx = (alert_type, key or str(track_id))

        existing = self._alert_index.get(idx)
        if existing is not None:
            existing.count += 1
            existing.last_timestamp = now
            existing.details = details      # keep the freshest score/liveness
            return existing

        alert = Alert(
            timestamp=now,
            alert_type=alert_type,
            track_id=track_id,
            details=details,
            severity=severity,
            last_timestamp=now,
        )
        self.alerts.append(alert)
        self._alert_index[idx] = alert
        return alert

    # ---------------- output ----------------

    def get_summary(self) -> dict:
        confirmed = [r for r in self.records.values() if r.duration_sec >= self.min_dwell_sec]
        return {
            "session": self.session_name,
            "min_dwell_sec": self.min_dwell_sec,
            "total_enrolled": len(self.records),
            # "Present" now means dwell-confirmed AND in the room, which is what a
            # register means. Seen-but-brief is reported separately rather than
            # being silently counted as attendance.
            "present_now": sum(1 for r in confirmed if r.is_present),
            "confirmed": len(confirmed),
            "brief": len(self.records) - len(confirmed),
            "unresolved_count": len(self.unresolved),
            "spoof_detected": sum(1 for r in self.records.values() if r.spoof_flags > 0),
            "alerts": len(self.alerts),
            "persons": [
                {
                    "name": r.name,
                    "roll": r.roll,
                    "entry": r.entry_time,
                    "exit": r.exit_time,
                    "duration_min": round(r.duration_sec / 60, 1),
                    "dwell_sec": round(r.duration_sec, 1),
                    "status": r.status,
                    "present": r.is_present,
                    "visits": len(r.visits),
                    "detections": r.detection_count,
                    "spoofs": r.spoof_flags,
                    "zone": r.zone,
                    "resolved_from": r.resolved_from,
                }
                for r in self.records.values()
            ],
            "unresolved": [
                {
                    "track_id": u.track_id,
                    "label": u.label,
                    "dwell_sec": round(u.duration_sec, 1),
                    "duration_min": round(u.duration_sec / 60, 1),
                    "entry": _hms(u.visits[0].entered) if u.visits else "",
                    "exit": _hms(u.visits[-1].last_seen) if u.visits and u.visits[-1].closed else "",
                    "present": u.is_present,
                    "zone": u.zone,
                    "detections": u.detection_count,
                    "crop": f"/api/live/unresolved/{u.track_id}/crop" if u.crop_path else "",
                    # Only worth a teacher's time if they were actually here.
                    "needs_action": u.duration_sec >= self.min_dwell_sec,
                }
                for u in sorted(self.unresolved.values(),
                                key=lambda x: -x.duration_sec)
            ],
            "alerts_log": [
                {
                    "time": a.timestamp,
                    "last_time": a.last_timestamp,
                    "type": a.alert_type,
                    "details": a.details,
                    "severity": a.severity,
                    "count": a.count,
                }
                # Critical first, then whatever fired most recently.
                for a in sorted(
                    self.alerts,
                    key=lambda x: ({"critical": 0, "warn": 1, "info": 2}.get(x.severity, 3),
                                   x.last_timestamp),
                )
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
            writer.writerow(["Name", "Roll", "Entry", "Exit", "Dwell(min)", "Status",
                             "Visits", "Detections", "Spoofs", "Zone"])
            for r in self.records.values():
                writer.writerow([
                    r.name, r.roll, r.entry_time, r.exit_time,
                    round(r.duration_sec / 60, 1), r.status,
                    len(r.visits), r.detection_count, r.spoof_flags, r.zone,
                ])
            for u in self.unresolved.values():
                writer.writerow([
                    u.label, "UNRESOLVED", _hms(u.visits[0].entered) if u.visits else "",
                    "", round(u.duration_sec / 60, 1), "unresolved",
                    len(u.visits), u.detection_count, 0, u.zone,
                ])
        print(f"[CSV] Saved to {path}")
        return path
