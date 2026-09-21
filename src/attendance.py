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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import LOGS_DIR


def _hms(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# NVR exports carry the capture time in the filename. Dahua/CP-Plus write
#   10.20.30.40_ch5_20260921090256_20260921120411.asf
# i.e. <start><end> as 14-digit stamps. Anchored, not greedy, so the dotted IP
# in front cannot contribute digits.
_FILENAME_STAMP = re.compile(r"(?<!\d)(\d{14})(?!\d)")


def recording_start_time(path: str, fallback: float | None = None) -> float:
    """
    Wall-clock epoch at which this footage was FILMED.

    Why it matters: dwell is measured against a timetable ("was this student in
    the 09:00 lecture"), and a recording analysed at 22:30 would otherwise stamp
    a 09:00 class with tonight's hour and match no lecture at all.

    The old note here said nothing in the file records when it was filmed. That
    was true of the camera's own clock -- which reads 2000-01-01 -- but not of
    the NVR's filename, which carries the start stamp exactly.

    Falls back to mtime, which for an exported file is usually the moment the
    export FINISHED: wrong by the length of the recording, but the right day and
    roughly the right hour, which still beats the wall clock at analysis time.
    """
    m = _FILENAME_STAMP.search(Path(path).name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            pass        # 14 digits that are not a date; fall through
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return time.time() if fallback is None else fallback


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


def dwell_between(visits: list[Visit], start: float, end: float) -> float:
    """
    Seconds of these visits falling inside [start, end).

    Visits are CLIPPED to the window, never counted whole: a student seated from
    09:50 to 10:40 is credited 10 minutes to a lecture ending at 10:00 and 40 to
    the next one, not 50 to both.

    Gaps BETWEEN visits are simply absent from the sum, which is why a washroom
    trip needs no special case anywhere in this file. Leaving closes one visit,
    returning opens another, and only the two in-room stretches are added up --
    so a 10-minute absence from a 60-minute lecture leaves 50 minutes of dwell
    and the student still clears a 30-minute bar.
    """
    return sum(max(0.0, min(v.last_seen, end) - max(v.entered, start))
               for v in visits)


@dataclass
class Lecture:
    name: str
    start: float        # epoch
    end: float

    @property
    def minutes(self) -> float:
        return (self.end - self.start) / 60.0


def parse_lectures(spans, day: datetime = None) -> list[Lecture]:
    """
    Turn ("09:00-10:30", "10:45-12:15") into epoch windows on `day` (default today).

    Wall-clock strings rather than offsets from session start, because a
    timetable belongs to the institution and a session does not: the same string
    has to name the same lecture whether it is applied to a live feed now or to
    a recording analysed this evening. That only holds if recording timestamps
    are anchored to capture time -- see recording_start_time().

    Spans are taken as given, not assumed hourly. A607's own footage shows the
    room full at 09:59:56 and empty at 10:30:55, so a boundary generated on the
    hour would cut straight through a lecture that was still running.
    """
    base = (day or datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    for span in spans:
        try:
            lo, hi = (s.strip() for s in span.split("-", 1))
            h1, m1 = (int(v) for v in lo.split(":"))
            h2, m2 = (int(v) for v in hi.split(":"))
        except ValueError:
            continue    # malformed entry must not take the whole timetable down
        start = base.timestamp() + h1 * 3600 + m1 * 60
        end = base.timestamp() + h2 * 3600 + m2 * 60
        if end > start:
            out.append(Lecture(name=span, start=start, end=end))
    return out


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
    # Every queue label that merged into this person, in merge order. A list, not
    # a string: track churn gives one person several labels over a session.
    resolved_from: list = field(default_factory=list)
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
    crop_area: int = 0      # biggest crop seen so far, so the teacher gets the best shot
    crop_is_face: bool = False   # False = we only ever saw their body


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
        # Liveness coverage, by roll (or track id for the unnamed).
        self._liveness_checked: set[str] = set()
        self._liveness_unjudged: set[str] = set()

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
        # (detection, face box) for everyone whose face is actually visible.
        # Body boxes are useless for the overlap check — two people standing side
        # by side overlap heavily and are not a photo spoof.
        active_faces = []

        for det in detections:
            # Outside every configured zone = not in the room. Skipped before the
            # alert checks too, so a passer-by in the corridor behind the door
            # neither marks attendance nor raises a stranger alert.
            # No zones configured at all means the whole frame counts.
            if self.zones_active and not det.zone:
                continue

            face_bbox = getattr(det, "face_bbox", None)
            if face_bbox is not None:
                active_faces.append((det, face_bbox))

            # Whether this person was ever actually liveness-checked. A negative
            # score means the check abstained — face too small to judge, or no
            # model loaded. Tracked because "0 spoofs found" and "0 people
            # checked" look identical in a report, and only one of them is
            # reassuring. At classroom distances the second is the normal case.
            if getattr(det, "liveness_score", -1.0) >= 0:
                self._liveness_checked.add(det.roll or f"t{det.track_id}")
            else:
                self._liveness_unjudged.add(det.roll or f"t{det.track_id}")

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

            # A track that was person3 until its face finally matched: merge the
            # dwell it accrued while unnamed instead of discarding it. This is
            # the point of tracking bodies — someone who sat through the class
            # facing away is credited from when they sat down, not from the
            # moment they happened to glance at the camera. Merge BEFORE the
            # touch so the still-open visit is the one that gets extended.
            rec = (self.resolve(det.track_id, det.name, det.roll)
                   or self._record_for(det.name, det.roll))
            self._touch(rec, det.zone, timestamp)

        self._expire(timestamp)

        # Multiple overlapping faces: two bboxes with high IoU but different
        # track IDs = possible photo/screen spoof.
        if len(active_faces) >= 2:
            self._check_overlapping_faces(active_faces, timestamp)

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
        """
        Keep the largest face crop seen for this track — the teacher's evidence.

        Falls back to the body box when the face is hidden, which is exactly the
        case that put this person in the queue. Clothing and posture are what the
        teacher has to go on then, so a torso beats no crop at all.
        """
        if frame is None:
            return
        face_bbox = getattr(det, "face_bbox", None)
        is_face = face_bbox is not None
        x1, y1, x2, y2 = (int(v) for v in (face_bbox if is_face else det.bbox))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        # A face always beats a body crop; within the same kind, bigger wins.
        # Comparing on area alone would let one torso — many times the area of
        # any face — permanently block every later face from being saved.
        if (is_face, area) <= (entry.crop_is_face, entry.crop_area):
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
            entry.crop_is_face = is_face

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
        # Append rather than assign. One person routinely collects several labels
        # when their track dies and respawns, and overwriting kept only the last
        # — which hid the churn precisely in the records where it happened. On
        # the first real-camera run one man absorbed person2 and then person3,
        # and the report showed only person3.
        rec.resolved_from.append(entry.label)
        return rec

    # ---------------- alerts ----------------

    def _check_overlapping_faces(self, active_faces, timestamp):
        """
        Flag when two face bboxes significantly overlap (potential photo spoof).

        Takes (detection, face_bbox) pairs so the two stay aligned. They used to
        be separate lists indexed in parallel, which they were not: the zone
        filter skips detections without appending a box, so every alert after
        the first skip named the wrong tracks.
        """
        for i in range(len(active_faces)):
            for j in range(i + 1, len(active_faces)):
                d1, b1 = active_faces[i]
                d2, b2 = active_faces[j]
                if d1.track_id == d2.track_id:
                    continue
                xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
                xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0, xb - xa) * max(0, yb - ya)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / max(area1 + area2 - inter, 1e-6)

                if iou > 0.5:
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
            # Read these two together or not at all: "0 spoofs" is only good news
            # if liveness_checked is non-zero.
            "liveness_checked": len(self._liveness_checked),
            "liveness_unjudged": len(self._liveness_unjudged - self._liveness_checked),
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
                    "resolved_from": ", ".join(r.resolved_from),
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

    def per_lecture(self, lectures: list[Lecture], min_dwell_sec: float = None) -> list[dict]:
        """
        Attendance sliced per lecture, from the visits already recorded.

        Deliberately a pure read over `self.records` rather than new state in the
        tracker: a visit already carries absolute entry and exit times, so which
        lecture it belongs to is arithmetic, not something to observe. Re-running
        this with a corrected timetable re-scores a finished session for free.

        `min_dwell_sec` is the per-lecture bar (default config.lecture_min_dwell_sec,
        1800s) and is a different quantity from the logger's own min_dwell_sec,
        which is a 30-second noise floor separating "was here" from "walked past".

        Absence is only meaningful RELATIVE TO THE PEOPLE SEEN THAT DAY. Someone
        who never appeared on any camera has no record here and cannot be listed
        absent -- that needs the enrolled roster, which this class does not hold.
        """
        if min_dwell_sec is None:
            from .config import DEFAULT_CONFIG
            min_dwell_sec = getattr(DEFAULT_CONFIG, "lecture_min_dwell_sec", 1800.0)

        out = []
        for lec in lectures:
            rows = []
            for r in self.records.values():
                dwell = dwell_between(r.visits, lec.start, lec.end)
                # Visits touching this window at all, for the in/out trail.
                spans = [v for v in r.visits
                         if v.last_seen > lec.start and v.entered < lec.end]
                rows.append({
                    "name": r.name,
                    "roll": r.roll,
                    "dwell_sec": round(dwell, 1),
                    "dwell_min": round(dwell / 60, 1),
                    # Share of the lecture actually attended -- the number a
                    # 75%-attendance rule would be applied to.
                    "share": round(dwell / max(lec.end - lec.start, 1e-6), 3),
                    "status": ("present" if dwell >= min_dwell_sec
                               else "partial" if dwell > 0 else "absent"),
                    "in_out": [[_hms(max(v.entered, lec.start)),
                                _hms(min(v.last_seen, lec.end))] for v in spans],
                    "spoofs": r.spoof_flags,
                })
            rows.sort(key=lambda x: -x["dwell_sec"])
            out.append({
                "lecture": lec.name,
                "start": _hms(lec.start),
                "end": _hms(lec.end),
                "minutes": round(lec.minutes, 1),
                "min_dwell_sec": min_dwell_sec,
                "present": sum(1 for x in rows if x["status"] == "present"),
                "partial": sum(1 for x in rows if x["status"] == "partial"),
                "absent": sum(1 for x in rows if x["status"] == "absent"),
                "students": rows,
            })
        return out

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


if __name__ == "__main__":
    # Self-check: the window arithmetic and the capture-time anchor. Both fail
    # silently in production -- a wrong boundary just moves attendance to the
    # neighbouring lecture, and a wrong anchor scores every lecture as empty.
    DAY = datetime(2026, 9, 21)
    NINE = DAY.replace(hour=9).timestamp()

    def v(h1, m1, h2, m2, closed=True):
        return Visit(entered=DAY.replace(hour=h1, minute=m1).timestamp(),
                     last_seen=DAY.replace(hour=h2, minute=m2).timestamp(),
                     closed=closed)

    lec1, lec2 = parse_lectures(("09:00-10:00", "10:00-11:00"), DAY)
    assert lec1.start == NINE, "09:00 must anchor to 09:00 on the given day"
    assert lec1.minutes == 60.0

    # Clipping: one visit straddling the boundary splits, never double-counts.
    straddle = [v(9, 50, 10, 40)]
    assert dwell_between(straddle, lec1.start, lec1.end) == 600.0
    assert dwell_between(straddle, lec2.start, lec2.end) == 2400.0

    # Disjoint in both directions scores zero, not a negative.
    assert dwell_between([v(11, 0, 11, 30)], lec1.start, lec1.end) == 0.0
    assert dwell_between([v(7, 0, 8, 0)], lec1.start, lec1.end) == 0.0

    # THE washroom case: out at 09:20, back at 09:30. Two visits, 50 minutes
    # total, still present against a 30-minute bar -- with no special casing.
    loo = [v(9, 0, 9, 20), v(9, 30, 10, 0)]
    assert dwell_between(loo, lec1.start, lec1.end) == 3000.0

    log = AttendanceLogger(session_name="_selfcheck")
    log.records["0001"] = PersonRecord(name="Present", roll="0001", visits=loo)
    log.records["0002"] = PersonRecord(name="Brief", roll="0002", visits=[v(9, 5, 9, 9)])
    log.records["0003"] = PersonRecord(name="Elsewhere", roll="0003", visits=[v(11, 0, 11, 5)])
    [rep] = log.per_lecture([lec1], min_dwell_sec=1800.0)
    got = {s["name"]: s["status"] for s in rep["students"]}
    assert got == {"Present": "present", "Brief": "partial", "Elsewhere": "absent"}, got
    assert (rep["present"], rep["partial"], rep["absent"]) == (1, 1, 1)
    assert rep["students"][0]["share"] == round(3000 / 3600, 3)
    # Both stretches shown, so a teacher can see the gap rather than infer it.
    assert len(rep["students"][0]["in_out"]) == 2

    # Capture time comes off the NVR filename, not the clock or the mtime.
    real = "10.20.30.40_ch5_20260921090256_20260921120411.asf"
    assert recording_start_time(real) == datetime(2026, 9, 21, 9, 2, 56).timestamp()
    # The dotted IP must not be mined for digits, and a bad stamp must not raise.
    assert recording_start_time("cam_99999999999999_x.mp4", fallback=1.0) == 1.0
    assert recording_start_time("no_stamp_here.mp4", fallback=1.0) == 1.0

    assert parse_lectures(("bogus", "09:00-10:00"), DAY) == [lec1], "bad span must be skipped"
    assert parse_lectures(("10:00-09:00",), DAY) == [], "backwards span must be dropped"

    import shutil
    shutil.rmtree(log.log_dir, ignore_errors=True)
    print("attendance.py self-check passed")
