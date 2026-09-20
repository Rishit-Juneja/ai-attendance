# ai-attendance — Handover

Face-recognition classroom attendance off a live CCTV stream. This document is
the whole system: what exists, how each piece works, and why the non-obvious
decisions were made that way.

---

## 1. Run it

```fish
cd ~/projects/ai-attendance
source .venv/bin/activate.fish
python -m src.webapp          # http://127.0.0.1:5000
```

Camera stream URL to paste into the Live page:

```
rtsp://CAMERA-IP:554/stream1
```

That path is not cosmetic. `/onvif1`, `/onvif2`, `/11`, `/12`, `/live/ch00_0`,
`/main` and bare `/` all connect fine and all hand back the **640x480
sub-stream**. Only `/stream1` gives 1920x1080. Verify with `ffprobe`, never by
"it connected so it must be right".

Reaching the camera needs the secondary NIC address — the camera sits on
192.168.1.0/24 (factory subnet), the LAN is 192.168.0.0/24:

```fish
sudo nmcli connection modify "Wired connection 1" +ipv4.addresses HOST-IP/24
```

This is already applied permanently. Check with `ip -4 addr` — you want both
HOST-IP and the DHCP 192.168.0.x on `enp6s0`.

Camera quirks: answers **no** discovery protocol (no ONVIF WS-Discovery, no
SADP, no DHIP, no XiongMai) — subnet scan is the only way to find it. Its
on-screen clock reads 2000-01-01 and is irrelevant; attendance timestamps come
from server time.

---

## 2. Architecture in one pass

```
RTSP frame
   │
   ├─ every frame ───────────────► overlay draw + MJPEG to browser
   │                                (boxes carried by BoxGlide)
   │
   └─ 3 times a second ──► Pipeline.process_frame()
          │
          ├─ SCRFD detect faces  ──► batched ArcFace embed (512-d, L2-normed)
          ├─ YOLO11n detect bodies
          ├─ face→body ownership (largest face inside a body claims it)
          │
          ├─ body_tracker (ByteTrack)  ← bodies + the face embedding that names them
          └─ face_tracker  (ByteTrack) ← faces belonging to no tracked body
                 │
                 ├─ per-track rolling embedding average (window 30)
                 ├─ FAISS IndexFlatIP match vs gallery (cosine, thresh 0.32)
                 ├─ zone_for(body box bottom-centre)
                 └─ anti-spoof on the face crop
                         │
                         ▼
                 AttendanceLogger.process_detections()
                         ├─ dwell accounting per person
                         ├─ unresolved queue for unmatched people
                         └─ deduplicated alerts
```

Files, by responsibility:

| File | Job |
|---|---|
| `src/config.py` | All tunables + the two GPU profiles (`dev`, `demo`) |
| `src/pipeline.py` | Detect → track → embed → match. The core. |
| `src/persons.py` | YOLO11n body detection on ONNX Runtime |
| `src/antispoof.py` | Motion-based liveness heuristic |
| `src/zones.py` | Named polygon zones, normalized coordinates |
| `src/attendance.py` | Dwell accounting, unresolved queue, alerts |
| `src/enroll.py` | Add/remove people, rebuild the FAISS index |
| `src/auto_enroll.py` | Bulk enrol from a CASIA-style folder tree |
| `src/webapp.py` | Flask operator console + all HTTP APIs |
| `src/reports.py` | CSV + PDF (ReportLab) export |
| `src/dashboard.py` | Standalone websocket dashboard (pre-Flask, still works) |
| `src/main.py`, `src/run_demo.py` | CLI entry points for video files |
| `src/evaluate.py` | Accuracy measurement against a labelled set |
| `test_matching.py` | 27 asserts covering every non-trivial rule below |

---

## 3. Tracking: bodies carry identity, faces only stamp it

**The rule: a track owns a name. A frame never re-derives one.**

Two measurements forced this:

1. A person facing away from the camera scored **0.06** against his own
   enrollment. That is below the impostor floor (~0.25). No threshold rescues
   it — if identity has to be re-established every frame, anyone who turns
   around ceases to exist.
2. At 3 fps, a walking face box clears **~3 of its own widths** between
   analysed frames. ByteTrack associates by IoU; zero overlap means zero
   association, so every frame spawned a fresh track ID. A body box is ~7×
   wider, so the same walk is **~0.3 body widths** and overlap stays high.

So: YOLO detects bodies, ByteTrack tracks the bodies, and a face — when visible
— supplies the embedding that names the body track it sits inside. Face hidden?
The track keeps the name.

**Two trackers, not one.** Person detection badly under-counts a packed room —
measured **8 bodies against 98 faces** on this project's own crowd photo. A
face that falls inside no tracked body gets its own track in a second
ByteTrack instance. Without that fallback the back rows vanish entirely.

Track IDs from the two trackers are namespaced by
`FACE_TRACK_ID_OFFSET = 1_000_000`. Both libraries number from 1, and
`track_id` is a dict key in the unresolved queue, the spoof checker and
BoxGlide — without the offset, body 4 and face 4 are one record shared by two
people.

**Face→body ownership.** Each face is assigned to the body box containing it
(a face inside a body *is* that body; counting both makes one person two
attendees). One body box often contains two faces in a crowd — the person it
was drawn around, plus someone behind their shoulder. The **larger face wins**
the body; the loser is *not* discarded, it falls through to the face tracker.
Discarding it deleted 17 of 35 people from the crowd photo.

A face is only considered "covered by its body" if that body actually became a
track. ByteTrack spawns only from its high-confidence bucket, so a low-scoring
body is reported by nothing — checking rather than assuming means the person
still gets counted via the face path.

**Spawn thresholds are per-detector.** A confidence score is only comparable
within one model. `BODY_SPAWN_THRESHOLD = 0.35` for YOLO (person scores run
0.37–0.91 in a crowded room; the face-calibrated 0.5 sat mid-distribution and
refused to track a third of the room). Faces spawn at 0.5. Both detectors emit
well below their spawn threshold on purpose — those low boxes are BYTE's second
association stage, which is how an occluded person keeps their ID. They can
never start a track, so a false positive there is inert.

ByteTrack gotcha that cost an afternoon: the library gates spawning on *both*
`track_activation_threshold` and `high_conf_det_threshold`. Setting one does
nothing.

**Rolling embedding window (30).** A single small CCTV face gives a noisy
embedding, but the noise is independent frame to frame, so averaging cancels
it — measured **0.28 → 0.46** genuine on a 25px face while impostor scores
stayed put. This is the entire reason track IDs are carried around. The window
is bounded (not a cumulative mean) so if ByteTrack hands an ID to a different
person, the old identity rolls off instead of poisoning the average forever.

**Identity is sticky.** Once a track matches, the name holds for the life of
the track. The averaged embedding only improves as observations accumulate, so
a frame that dips back under 0.32 is noise, not a different person —
re-deciding every frame made the label strobe for anyone scoring near threshold.

Only tracks matched **this** frame are emitted. Lost tracks stay in memory for
`max_age` so their embedding survives an occlusion, but emitting them left a
ghost box at the last position for ~2s — counted as present, and overlapping
the real track on return, which is what raised bogus "multiple_overlapping
IoU=0.67" alerts for a single person.

---

## 4. Recognition and the gallery

- **Embeddings:** ArcFace `buffalo_l`, 512-d, L2-normalized.
- **Index:** FAISS `IndexFlatIP` — inner product on normed vectors = cosine.
- **Threshold:** `match_threshold = 0.32`. Measured here: impostors top out at
  0.25–0.26 even against an averaged reference; genuine pairs clear 0.33+ once
  aggregated.
- **No face-size filter.** Small faces *are* the CCTV workload. They are
  handled by averaging across the track, not by being thrown away.

**Batched embedding.** `FaceAnalysis.get()` runs recognition one face at a
time — 9.3 ms each, so a 98-face frame cost 926 ms while detection itself
stayed flat at 10 ms. That serial loop, not the detector, is what put a crowded
room out of real-time reach. `ArcFaceEmbedder.detect()` does one detector pass
then embeds in batches of `emb_batch`.

**Alignment is mandatory.** `face_align.norm_crop` with the 5-point landmarks
is exactly what `rec_model.get()` does internally. Embeddings taken without it
are not comparable to the gallery — that is the bug that made every stranger
match "Krish".

**`live_only` gallery filter.** Auto-enrolled CASIA entries (identified by a
`source_scene` key in the metadata) were embedded *without* landmark alignment,
while every live query is aligned. The two sets live in different regions of
the embedding space, the unaligned ones lose every comparison, and the one
aligned entry became a magnet that captured ~28% of all stranger faces. The
live path drops them; `evaluate.py` / `run_demo.py` keep them, because there
the queries are unaligned too and the comparison is fair.

**SCRFD detection threshold is 0.4**, below InsightFace's 0.5 default. At 0.5
SCRFD never emits a low-confidence box (measured floor: 0.505), so BYTE's
occlusion-rescue stage has nothing to work with. Dropping to 0.4 left the test
photos unchanged and only added boxes in dense crowds.

Gallery lives at `data/gallery/gallery.index` + `gallery_meta.json`.
`src/enroll.py` adds and removes people and rebuilds the index;
`remove_person()` rebuilds rather than tombstones, so indices stay aligned with
metadata rows.

---

## 5. Attendance is dwell time, not sightings

`src/attendance.py`. Being seen does not make you present. Accumulating
`min_dwell_sec` (30s) inside a zone does. Walking past the open door no longer
marks you present — the old "seen in 3 frames" rule did exactly that.

**Visit model.** A `Visit` is one continuous stay (`entered`, `last_seen`,
`zone`, `closed`). A person's dwell is the sum of their visits; gaps between
visits do not count.

**`exit_grace_sec = 60`.** A visit stays open across gaps shorter than this, so
the gap itself is credited as dwell. Deliberately generous: a student with
their head down or turned away loses their face for a minute at a stretch and
is still sitting there. Too short and their dwell is shredded into
uncreditable slivers; too long and someone who walked out keeps accruing time.

**Status:**
- `present` — dwell ≥ 30s and a visit is currently open
- `left` — dwell ≥ 30s, all visits closed
- `brief` — seen, but never accumulated 30s. Explicitly *not* "absent";
  absent means never seen at all, which this class cannot observe.

`close_session()` closes every open visit when class ends — without it the last
visit stays open forever and the report shows everyone still in the room.

**Zone gating.** A detection outside every configured zone is skipped before
*any* accounting, including alerts — so a passer-by in the corridor behind the
door neither marks attendance nor raises a stranger alert. No zones configured
at all means the whole frame counts.

**Spoofed detections accrue dwell for nobody**, named or not.

---

## 6. Zones

`src/zones.py`. Named polygons in **normalized 0..1** coordinates, stored in
`data/zones.json`, drawn in the browser on the Live page.

Normalized, not pixels, because the camera is currently serving a stream that
is expected to be reconfigured — pixel polygons would silently point at the
wrong part of the room the moment resolution changes.

**Anchor point = bottom-centre of the box.** With a body box that is the feet,
which is the honest answer to "which part of the room is this person in". It
degrades to the chin when only a face was detected; that reads as in-or-out
depending on head tilt, so draw zone edges with a little margin.

Area, not tripwire, on purpose: attendance asks "who is in the room", which is
a containment question. "Who crossed this line, in which direction" needs
per-track crossing history and is a separate feature on top of this one.

`python -m src.zones` runs a self-check covering containment, resolution
independence, and the empty-zones default.

---

## 7. Unresolved-person review queue

Faces that never match the gallery do not vanish. They become `person1..N` in
the unresolved queue with **identical dwell accounting** — a known person and
an unresolved stranger share the `_Dwelling` base class, because the only
difference between them is whether we have a name yet.

**Evidence crop.** The largest face crop seen for that track is saved to
`data/logs/<session>/unresolved/<label>.jpg`, padded 35% for context (a human
reviewer needs surroundings far more than a tight crop). If the face was never
visible — which is exactly the case that put this person in the queue — it
falls back to the body box; clothing and posture are what the teacher has to go
on, and a torso beats no crop. Ranking is `(is_face, area)`, so a face always
beats a body crop and comparing on area alone can't let one torso permanently
block every later face.

**Labels never get reused.** Numbering comes from a monotonic counter, not
`len(dict)`, so resolving person3 away doesn't create a second person3.

**Resolving back-dates the dwell.** `resolve(track_id, name, roll)` merges the
unresolved visits into the named record and re-sorts. Someone who sat through
the whole class facing away is credited **from when they sat down**, not from
the moment they happened to glance at the camera. This is the entire payoff of
tracking bodies.

The merge also happens automatically mid-session: when a track that was
`person3` finally matches the gallery, `process_detections` merges first and
*then* extends the visit, so the still-open visit is the one that grows.

**Review survives session stop.** Reviewing happens *after* class. Stopping the
session used to drop the logger — taking the whole queue with it at exactly the
moment it was needed. Now the logger moves to `_review`, and resolving a person
afterwards rewrites the CSV/PDF reports.

`needs_action` in the API is `dwell ≥ min_dwell_sec` — only people who were
actually there are worth a teacher's time.

---

## 8. Alerts

Four types: `spoof` (critical), `unknown_face` (warn), `face_hiding` (warn),
`multiple_overlapping` (info).

**Deduplicated by (type, subject).** One row per subject per session with a
`count`, not one row per frame. Without this, an unrecognised face standing in
shot raised an alert on *every analysed frame* — 20/second at the old rate.
Subject key is a roll number for a known person, a track id for a stranger, and
a sorted track *pair* for overlaps so two people standing in line raise one
alert instead of one per frame.

**`MIN_OBS_BEFORE_UNKNOWN_ALERT = 5`.** A face that just appeared is
legitimately unidentified while its embedding window fills. Alerting on that
produced most of the 2103 alerts in one short test run.

**`face_hiding`** fires when a face has been gone `hiding_alert_after_sec` (15s)
*inside an open visit*. It is a heads-up, not an exit — dwell keeps counting.

**`multiple_overlapping`** compares **face** boxes only. Body boxes are useless
for this: two people standing side by side overlap heavily and are not a photo
spoof. It carries `(detection, bbox)` pairs rather than two parallel lists —
they weren't actually parallel, because the zone filter skips detections
without appending a box, so every alert after the first skip named the wrong
tracks.

---

## 9. Anti-spoof ⚠️ NEEDS TUNING

`src/antispoof.py`. Motion heuristic: crop the face (20% padding), take the
mean absolute pixel difference between consecutive crops, and flag a track
whose motion stays below `spoof_pixel_movement_thresh` for `flag_after_n`
frames. Real faces micro-move (breathing, blinking); printed photos and screens
don't.

Runs only on the **face box** — the heuristic is micro-motion in a face crop,
and running it over a whole body measures walking.

> **Outstanding, and it needs you and a printed photo — but it is no longer
> guesswork.** `spoof_pixel_movement_thresh = 1.5` was calibrated when
> consecutive samples were 50 ms apart. They are now **333 ms** apart (3 fps),
> so both a real face and a hand-held photo move considerably more between
> samples. Micro-motion doesn't scale linearly with the gap, so the number
> can't be rescaled on paper. Measure it:
>
> ```fish
> python tools/calibrate_spoof.py --label live  --seconds 40   # just sit there
> python tools/calibrate_spoof.py --label spoof --seconds 40   # hold up a photo
> python tools/calibrate_spoof.py --report
> ```
>
> It samples through the real detector at the configured analysis rate using the
> same crop-and-diff maths as `SpoofChecker`, then compares the **live 5th
> percentile against the spoof 95th percentile** — the tails, because a
> threshold set from the means flags every live person who briefly holds still.
> If those two overlap it says so: that means no threshold separates them on
> this camera and the heuristic needs replacing with a real liveness model
> (`SilentFaceLiveness`), not retuning.

`spoof_check_every_n` is 1 (every analysed frame). It skipped 4 in 5 when
analysis ran at 20 fps, which was the point; at 3 fps the same setting would
take 25 seconds to flag a held-up photo.

---

## 10. Frame rate: why 3

`analysis_fps = 3` (dev profile), 2 for demo. Display stays at full camera rate.

Three constraints, all pointing the same way:

1. **Geometric (the binding one).** Association is IoU-based. What matters is
   *box widths travelled per frame*, not fps in the abstract. 3 fps works for
   bodies (~0.3 body widths per frame at walking pace) and does not work for
   faces (~3 face widths). Measured movement ceiling for reliable tracking:
   **~1.2 m/s**. Someone running through the frame will break association —
   that's the known limit, not a bug.
2. **Network.** The inference GPU is 10–15 km away with unpredictable latency.
   A full-rate 640x480 feed is ~5 Mbps per camera and there is no budget to
   ship it.
3. **Hardware.** Deployment target is a GTX 1650, not the RTX 5060 this was
   developed on.

3 is the target rate, not a degraded fallback. Everything downstream is
expressed in seconds and converted using `analysis_fps` — `max_age` is
`analysis_fps × track_lost_sec` — so changing the rate doesn't silently change
how long a track survives.

**BoxGlide** (`src/webapp.py`) is what makes 3 fps look like 30. It advances
each box along its measured velocity between analyses and publishes an
annotated frame *every* frame. Drawing the last analysed box verbatim made it
sit still and jump, which reads as lag even when the video is smooth.
Extrapolation is capped at 0.4s so a box doesn't sail across the room when
someone stops walking or analysis stalls.

Related: 640x480 caps recognition *range* — it's a pixels-per-face problem, and
it's the reason `/stream1` (1080p) matters.

---

## 11. Web app

Flask, `src/webapp.py`, port 5000. Dark red/black theme,
`static/css/app.css` + Jinja2 templates.

**Pages:** `/` overview · `/identify` (dual dropzone: reference photo +
target image/video) · `/live` (stream, zone editor, unresolved queue).

**APIs:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/identify/image` | POST | Find a reference face in an uploaded image |
| `/api/identify/video` | POST | Same, across a video file |
| `/api/gallery` | GET/POST | List / enrol a person |
| `/api/gallery/<roll>` | DELETE | Remove, rebuild index |
| `/api/zones` | GET/POST | Read / save polygon zones |
| `/api/live/stream_start` | POST | Open RTSP, start the worker thread |
| `/api/live/snapshot` | GET | Latest annotated frame (drives the feed) |
| `/api/live/stream_status` | GET | Running / error state |
| `/api/live/stream_stop` | POST | Stop the stream, keep the session |
| `/api/live/start` | POST | Begin an attendance session |
| `/api/live/frame` | POST | Push a browser-webcam frame (no RTSP path) |
| `/api/live/unresolved` | GET | The review queue |
| `/api/live/unresolved/<id>/crop` | GET | Evidence JPEG |
| `/api/live/resolve` | POST | Teacher names a person; back-dates dwell |
| `/api/live/stop` | POST | Close session, write CSV/JSON/PDF, keep for review |

**RTSP over TCP.** `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` is set
before opening — OpenCV/FFmpeg default to UDP, which drops packets and smears
macroblocks across faces on a shared LAN. `CAP_PROP_BUFFERSIZE=1` keeps latency
at one frame instead of a backlog; open/read timeouts are 5s so a dead camera
can't wedge the worker.

Session output lands in `data/logs/<session>/`: `attendance.json`,
`attendance.csv`, unresolved crops, plus the daily CSV/PDF in `data/reports/`.

---

## 12. Config quick reference

`src/config.py` — everything tunable, with the measurement behind each value in
a comment at the setting itself.

| Setting | Value | Why |
|---|---|---|
| `analysis_fps` | 3 (dev) / 2 (demo) | §10 |
| `match_threshold` | 0.32 | Impostor ceiling 0.26, genuine floor 0.33 |
| `track_embed_window` | 30 | 0.28→0.46 on a 25px face |
| `track_lost_sec` | 8.0 | Covers a body fully occluded. Raise to 30–60 for a seated classroom — association is near-trivial there (IoU ~1.0) so the buffer is cheap. Caveat: identity is sticky, so a long buffer also means a misassigned ID keeps the wrong name longer. |
| `min_dwell_sec` | 30.0 | Presence threshold |
| `exit_grace_sec` | 60.0 | Head-down tolerance |
| `hiding_alert_after_sec` | 15.0 | Face-covered heads-up |
| `spoof_pixel_movement_thresh` | 1.5 | **Stale — see §9** |
| `det_size` | 640×640 | SCRFD input |
| `emb_batch` | 32 | Batched recognition |
| `BODY_SPAWN_THRESHOLD` | 0.35 | In `pipeline.py`; YOLO-calibrated |

Profile switch: `GPU_PROFILE=demo python -m src.webapp`.

---

## 13. Tests

```fish
python test_matching.py     # 27 asserts, all green
python -m src.zones         # zones self-check
```

Coverage: dwell accumulation across visits, grace-period merging, back-dated
resolution, label non-reuse, crop ranking, alert dedup, track-ID namespacing,
face→body ownership with the shoulder-surfer case, and zone containment at two
resolutions.

**A warning worth keeping.** Two silent people-loss bugs in the hybrid
body/face path — faces dropped when their body wasn't tracked, and the
second-face-in-one-body case — passed every synthetic test and were only caught
by running real crowd photos through the pipeline and counting heads. Synthetic
fixtures do not have shoulder-surfers. When you change the detection or
ownership logic, re-run it against `data/test_faces/` and count.

---

## 14. Known limits

- **~1.2 m/s** movement ceiling at 3 fps. Faster than that breaks association.
- Person detection under-counts dense rooms (8/98 on the crowd photo). The face
  tracker is the fallback, and it's why both exist.
- Anti-spoof threshold is stale (§9).
- Sub-stream (640x480) caps recognition range. Use `/stream1`.
- ONNX Runtime binds to CUDA on this machine (verified: both `[ArcFace]` and
  `[YOLO person]` log `running on GPU (CUDAExecutionProvider)` with
  `onnxruntime-gpu` 1.30 + `nvidia-cudnn-cu13` 9.24). Keep trusting those boot
  lines rather than `get_available_providers()` — that lists CUDA even when the
  libs fail to load and ORT silently falls back to CPU. A CPU-only install still
  runs; it just leaves the 5060 idle.
- The Flask dev server is the dev server. Put a real WSGI server in front of it
  before this sits in a school.
