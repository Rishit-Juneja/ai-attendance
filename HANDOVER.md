# ai-attendance — Handover

Face-recognition classroom attendance off a live CCTV stream. This document is
the whole system: what exists, how each piece works, and why the non-obvious
decisions were made that way.

**New to the project? Read §15 (what it's trying to do) and §16 (the room it
has to work in) first.** §1–§14 are the machine; §15–§16 are the point.

---

## 1. Run it

```fish
cd ~/projects/ai-attendance
source .venv/bin/activate.fish
python -m src.webapp          # http://127.0.0.1:5000
```

Camera stream URL to paste into the Live page:

```
rtsp://<camera-ip>:554/stream1
```

That path is not cosmetic. `/onvif1`, `/onvif2`, `/11`, `/12`, `/live/ch00_0`,
`/main` and bare `/` all connect fine and all hand back the **640x480
sub-stream**. Only `/stream1` gives 1920x1080. Verify with `ffprobe`, never by
"it connected so it must be right".

These cameras often ship on a **factory subnet** that is not the LAN's, so the
host needs a second address on the camera's subnet before it can be reached at
all:

```fish
sudo nmcli connection modify "<connection>" +ipv4.addresses <host-ip>/24
```

Make it permanent, then check with `ip -4 addr` — you want both that static
address and the DHCP one on the same interface.

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
| `src/antispoof.py` | MiniFASNetV2 liveness (texture), motion heuristic fallback |
| `src/zones.py` | Named polygon zones, normalized coordinates |
| `src/attendance.py` | Dwell accounting, unresolved queue, alerts |
| `src/enroll.py` | Add/remove people, rebuild the FAISS index |
| `src/auto_enroll.py` | Bulk enrol from a CASIA-style folder tree |
| `src/webapp.py` | Flask operator console + all HTTP APIs |
| `src/reports.py` | CSV + PDF (ReportLab) export |
| `src/dashboard.py` | Standalone websocket dashboard (pre-Flask, still works) |
| `src/main.py`, `src/run_demo.py` | CLI entry points for video files |
| `src/evaluate.py` | Accuracy measurement against a labelled set |
| `tools/calibrate_spoof.py` | Measures the motion threshold on this camera |
| `tools/probe_spoof_preproc.py` | Sweeps model preprocessing; proves it responds at all |
| `tools/verify_spoof_model.py` | Scores real images or live frames through the model |
| `tools/export_person_model.py` | Produces `yolo11n.onnx` (needs torch; one-off) |
| `tools/measure_det_size.py` | Usable faces vs detect time per `det_size`. Run on real footage |
| `test_matching.py` | 35 tests covering every non-trivial rule below |

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

## 9. Anti-spoof — two modes, and a hard pixel floor

`src/antispoof.py`. Everything here was measured on the real 1080p feed at
3 fps on 2026-09-20, not read off a datasheet. Which mode is active is printed
at boot.

### Mode A — MiniFASNetV2 (primary, if the weights are present)

Judges liveness from **texture**, which is the axis that actually separates
paper and screens from skin. 1.74 MB, Apache-2.0, not committed (`data/` is
gitignored):

```fish
curl -L -o data/models/minifasnet_v2.onnx \
  https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx
```

**The model card is wrong twice, and both failures are silent.** It documents a
`/255` input and a `[live, print, replay]` head with live at index 0.

- Under `/255`, random noise, pure black, pure white and real faces all return
  `p=[0.000 0.007 0.992]` — a **constant**. The normalization is already baked
  into the graph, so dividing again flattens the input past what the first conv
  can separate.
- Fed raw 0–255 it responds: real faces `[0.010 0.250 0.740]`, noise
  `[0.003 0.983 0.014]`. Real faces land on **class 2**, not class 0.

Reading index 0 as documented marks every living person a spoof with total
confidence and nothing raises. `tools/probe_spoof_preproc.py` is what found
this; run it before trusting any exported model.

The **2.7× crop** the model name refers to is not padding. The tell for a print
or replay attack is often *outside* the face — paper edge, phone bezel, flat
background moving with the head — so a tight crop throws away the evidence.
Clamped to the frame, because truncating at the edge silently changes the zoom
the model sees. This is also why `tools/verify_spoof_model.py` scores live
frames rather than saved crops: saved crops are already tight, so the 2.7×
expansion has nothing to expand into.

Verdicts are smoothed over the per-track window on a **median** — one blurred
frame should not cost a real student their attendance.

### Mode B — motion heuristic (fallback, no weights)

Crop the face (20% padding), take the mean absolute pixel difference between
consecutive crops, flag a track whose motion stays below
`spoof_pixel_movement_thresh` for `flag_after_n` frames. Runs on the **face
box** only — over a whole body it measures walking.

`spoof_pixel_movement_thresh = 1.0`, measured: 95 live samples floored at
**1.40** (p5 5.75, median 13.85), a static printed image sat at **0.00–0.06**.
Set below the live floor rather than midway between the distributions, because
a false flag costs a real student their attendance while a missed spoof costs
one detection.

> **It catches print and nothing else.** A photo displayed on an LCD produced
> motion in **69% of samples** — the monitor's refresh beats against the camera
> shutter and the banding reads as life. Replayed through the flagging logic,
> the longest run of sub-threshold samples was 4 against the 15 needed. A phone
> held up by a judge walks straight through, by construction. No threshold
> fixes this; the flicker is larger than the signal.

Re-measure if `analysis_fps` changes — the number is only valid at the sampling
interval it was taken at, and micro-motion does not scale linearly with the gap:

```fish
python tools/calibrate_spoof.py --label live  --seconds 40   # just sit there
python tools/calibrate_spoof.py --label spoof --seconds 40   # hold up a photo
python tools/calibrate_spoof.py --report
```

Keep **exactly one face in shot**. With a person and a photo both visible the
detector finds one or the other between frames, and diffing a face against a
photograph poisons the result — the first run alternated 0.04 and 21.94 between
333 ms samples, which no human body does, and the tool confidently reported
OVERLAP off it. It now requires box overlap with the previous sample and
refuses to write a file when subject switches exceed 20% of usable samples.

### The pixel floor — the part that decides whether any of this matters

Liveness needs **~65px of face width**, not the 30px the 80×80 input implies.
Measured against a real person on the CCTV feed:

| 47px | 48px | 55px | 61px | 62px | **66px** | 72px | 74px | 76px | 77px |
|---|---|---|---|---|---|---|---|---|---|
| 0.011 | 0.035 | 0.290 | 0.010 | 0.024 | **0.885** | 0.883 | 0.970 | 0.913 | 0.986 |

A living person falls off a cliff below ~65px. Provisional — one 20s capture,
and the subject was looking down for much of it, so width is confounded with
pose. Below the floor the checker **abstains**.

The target room gives 34–64px faces (§16). **This check will almost never run
there**, and no model fixes that — it cannot read skin texture the sensor never
captured. It is largely self-cancelling: below ~30px nobody is identified
*including a spoofer*, so they land in the unresolved queue like anyone else.
The genuinely exposed band is roughly **30–65px — identifiable but not
judgeable.**

### Abstention is not approval

The checker returns **`-1.0`** when it cannot judge, and the summary carries
`liveness_checked` beside `liveness_unjudged`. "0 spoofs detected" and "0
people checked" render identically in a report and mean opposite things; only
one is reassuring, and at classroom distance the other is the normal case. A
report that cannot express that difference is claiming a safety property the
system does not have.

**`Detection.liveness_score` defaults to `-1.0`, and that default is the whole
mechanism.** The spoof check only runs when `face_bbox is not None`, so the
head-down person the body tracker exists for never reaches it at all. While the
default was `1.0` every one of them arrived at the logger carrying a clean
score nobody produced, and `liveness_checked` counted them — the accounting
layer was honest and the producer was not. The heuristic's warm-up frames and
its too-small-to-diff crop had the same hole. Anything that returns a liveness
score must return a negative one when no check ran.

Note what hid this: `test_matching.py` builds its own detection stub and sets
`liveness_score` by hand, so 32 tests exercised the field and none of them ever
saw its default. One test even asserted the wrong value outright. When a
dataclass default encodes a safety claim, test the real class.

A missing model file falls back to the heuristic and never flags. Failing
closed would block the whole class.

`spoof_check_every_n` is 1 (every analysed frame). It skipped 4 in 5 when
analysis ran at 20 fps, which was the point; at 3 fps the same setting would
take 25 seconds to flag a held-up photo.

**Still unmeasured:** what a spoof actually scores *through the model*. A real
person got a median of 0.665, which is uninterpretable until a printed photo is
run down the same path.

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
| `/api/live/stream_start` | POST | Open an RTSP/HTTP camera **or a video file path**, start the worker |
| `/api/live/snapshot` | GET | Latest annotated frame (drives the feed) |
| `/api/live/stream_status` | GET | Running / error state, plus `done` + `pos_sec` for a recording |
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

### Analysing a recording

Paste a **video file path** into the same box as a camera URL. This is the path
that matters for deployment: the classroom footage arrives as files, and the
gallery has to be built from them because student photos were refused (§16).

`_is_recording()` decides by extension, and a file is handled as the opposite of
a camera on nearly every axis:

| | Live camera | Recording |
|---|---|---|
| Clock | wall clock | the video's own `CAP_PROP_POS_MSEC` |
| Buffering | `BUFFERSIZE=1`, 5s timeouts | none — dropping a frame would skip footage |
| End of input | error after 30 misses | success; closes the session itself |
| Speed | camera rate | as fast as the GPU manages (2.8× on 1080p/35 faces) |
| Overlay | browser feed | `annotated.mp4`, every frame, for scrubbing |

**The clock is the whole feature.** Both `should_analyze()` and
`process_frame()` took the wall clock unconditionally, and both are wrong for a
file. The throttle would admit 3 frames per *real* second while minutes of
footage streamed past unexamined; and every duration downstream — `min_dwell_sec`,
`exit_grace_sec`, `hiding_alert_after_sec` — would be measured against how long
the crunch took rather than the class. A 40-minute class analysed in 4 minutes
credited everyone 4 minutes of dwell and the entire room came out `brief`. Both
now take an optional clock and default to wall time, so the live path is
untouched.

**Recording timestamps are offsets, not real times.** Video position starts at
0.0, which rendered as `05:30:01` in the queue — the 1970 epoch in local time.
They are anchored to the wall clock when analysis *started*, so durations are
unaffected (they are differences) and times read sensibly and sort correctly.
Nothing in the file says when it was filmed and the camera's own clock reads
2000-01-01 (§1), so do not read a recording's timestamps as class times.

A file running out closes the session through `_finish_session()` — the same
path as the Stop button, because the footage ending *is* the end of class and
has to settle the books identically. It matters that this is automatic: the
analysis finishes faster than real time and usually unattended.

**You cannot judge tracking from the browser feed, and the annotated video is
why.** The feed polls single JPEGs at ~8fps of *wall* time while a recording is
analysed at 2–3× real speed — measured **2.8×** on 1080p/30fps with 35 faces,
so a viewer sees about **1 frame in 13**. That is enough to confirm boxes exist
and nothing like enough to tell a held track from one that died and respawned.

So every annotated frame is written to `data/logs/<session>/annotated.mp4`
instead, which you scrub, pause and step through — and which is what §13's
"count heads by hand" actually needs. Labels carry the **track id on matched
people too**, not just strangers: a name hopping to a different track is the
failure you are looking for, and a name-only label hides it.

Cost, measured on 1080p/30fps: throughput drops **2.8× → 1.8×** real time and
the file grows at **~0.84 MB per second of footage** — a 40-minute class is
about 2 GB. `log_video_detections = False` turns it off for long unattended
runs.

Session output lands in `data/logs/<session>/`: `attendance.json`,
`attendance.csv`, unresolved crops, plus the daily CSV/PDF in `data/reports/`.
Recording sessions are named `rec_<timestamp>`, camera ones `cam_<timestamp>`.

---

## 12. Config quick reference

`src/config.py` — everything tunable, with the measurement behind each value in
a comment at the setting itself.

| Setting | Value | Why |
|---|---|---|
| `analysis_fps` | 3 (dev) / 2 (demo) | §10 |
| `match_threshold` | 0.32 | Impostor ceiling 0.26, genuine floor 0.33 |
| `track_embed_window` | 30 | 0.28→0.46 on a 25px face |
| `track_lost_sec` | 45.0 | Raised from 8 after a real run: a seated man's body track died behind his chair and respawned with a new ID, producing 3 queue labels for 2 people. Each respawn re-matches the gallery from scratch. Cheap because `update()` only returns tracks matched *this* frame, so a longer buffer buys ID continuity without crediting dwell to anyone invisible. **Lower it again for a doorway view**, where a stale box can be claimed by a stranger. |
| `min_dwell_sec` | 30.0 | Presence threshold |
| `exit_grace_sec` | 60.0 | Head-down tolerance |
| `hiding_alert_after_sec` | 15.0 | Face-covered heads-up |
| `spoof_pixel_movement_thresh` | 1.0 | Measured: live floor 1.40, print 0.00–0.06. §9 |
| `spoof_model_path` | `data/models/minifasnet_v2.onnx` | Absent = heuristic fallback. §9 |
| `LIVENESS_MIN_FACE_PX` | 65 | In `antispoof.py`. Below this the checker abstains (`-1.0`). §9 |
| `det_size` | 640×640 | SCRFD input. Raising it is measured **not** to pay — §16 |
| `emb_batch` | 32 | Batched recognition |
| `BODY_SPAWN_THRESHOLD` | 0.35 | In `pipeline.py`; YOLO-calibrated |

Profile switch: `GPU_PROFILE=demo python -m src.webapp`.

---

## 13. Tests

```fish
python test_matching.py     # 35 tests, all green
python -m src.zones         # zones self-check
```

Coverage: dwell accumulation across visits, grace-period merging, back-dated
resolution, label non-reuse and churn (`resolved_from` is a list), crop
ranking, alert dedup, track-ID namespacing, face→body ownership with the
shoulder-surfer case, zone containment at two resolutions, liveness
preprocessing, and abstention being reported separately from a clean pass.

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
- **Liveness abstains at classroom distance.** It needs ~65px of face; the
  target room gives 34–64px. Exposed band is 30–65px: identifiable but not
  judgeable (§9).
- **The motion fallback does not catch screens.** Print only. If the weights
  aren't downloaded, a phone held up defeats the check by construction (§9).
- Sub-stream (640x480) caps recognition range. Use `/stream1`.
- `det_size=640` is **not** the hidden range limit an earlier version of this
  document claimed. Measured: 1280 buys 2 usable faces for +43% detect time,
  and most of what it adds is sub-25px guesswork (§16). Unverified on real
  1920 classroom footage — re-run `tools/measure_det_size.py` there.
- ONNX Runtime binds to CUDA on this machine (verified: both `[ArcFace]` and
  `[YOLO person]` log `running on GPU (CUDAExecutionProvider)` with
  `onnxruntime-gpu` 1.30 + `nvidia-cudnn-cu13` 9.24). Keep trusting those boot
  lines rather than `get_available_providers()` — that lists CUDA even when the
  libs fail to load and ORT silently falls back to CPU. A CPU-only install still
  runs; it just leaves the 5060 idle.
- The Flask dev server is the dev server. Put a real WSGI server in front of it
  before this sits in a school.
- Ultralytics YOLO is **AGPL-3.0**. Fine for a competition; it is network
  copyleft if this is ever served publicly. Apache-2.0 swaps: YOLOX, RTMDet.

---

## 15. What the system is actually for

Five targets. They are not independent — every one of them bottoms out in
pixels-per-face, which is why §16 exists.

### T1 — Real-time tracking on a remote GPU

**Goal:** follow every person in the room continuously, with an overlay that
looks live, while inference runs on a GPU **10–15 km away** with latency nobody
can predict in advance. Deployment hardware is a **GTX 1650 (4GB)**, up to ~100
people in a room.

**Where it stands: done, and the architecture is the answer.** The
display/analysis split is mandatory, not an optimisation — at 640x480 a
full-rate feed is ~5 Mbps per camera and there is no budget to ship it. Display
runs locally at camera rate; analysis runs at 3 fps; `BoxGlide` carries boxes
forward at measured velocity in between so the overlay stays smooth (§10).

The cost measurements that made 100 people feasible, on the RTX 5060 (a 1650 is
~3–4× slower):

| faces | detect | detect+embed (serial) | batched |
|---|---|---|---|
| 12 | 7.3 ms | 118.6 ms | 27.0 ms |
| 35 | 9.5 ms | 327.2 ms | 86.4 ms |
| 98 | 10.2 ms | 926.5 ms | 192.0 ms |

**Detection is flat with crowd size. Per-face embedding was the entire
problem** — batching gave 4.7× with bit-identical embeddings. YOLO person
detection adds ~10 ms and is also flat. The detector is never the bottleneck.

100 people stays feasible only because of **track-then-identify**: embed a
person once, then track them. Not re-embed 100 faces every frame.

**Left to do:** when the real RTT is known, extrapolate `BoxGlide` by it so
boxes compensate for network delay instead of just for the analysis gap.

### T2 — Foolproofing: presence must mean presence

**Goal:** a name on the register should mean that person was in the room, for a
real length of time, and not because the system guessed.

**Where it stands: done, and this is what dwell accounting is** (§5). Being
seen is not being present — 30s of accumulated dwell inside a zone is. Walking
past the open door doesn't mark you present; the old "seen in 3 frames" rule
did exactly that. `brief` is a real status, distinct from absent, because
"seen but not long enough" and "never here" are different claims.

The supporting rules all exist for the same reason: identity is sticky per
track so labels don't strobe; a spoofed detection accrues dwell for nobody; the
60s grace means a head-down student keeps accruing instead of having their
dwell shredded into uncreditable slivers.

**The honest part:** the system reports what it could not judge rather than
implying a clean result. `liveness_unjudged` beside `liveness_checked` (§9) is
the clearest example — a report that renders "0 spoofs" and "0 checked"
identically is claiming a safety property the system does not have.

### T3 — Evasion detection

**Goal:** notice people actively working around the system, and notice the
system failing, rather than silently logging a clean-looking register.

**Where it stands: partly done. This is the weakest target.** What exists:

| Signal | Catches | Status |
|---|---|---|
| `face_hiding` | Face gone 15s+ while still counted present | Works |
| `unknown_face` | Someone the gallery doesn't know | Works, deduped |
| `multiple_overlapping` | Two face boxes at IoU > 0.5 — a held-up photo | Works, face boxes only |
| `spoof` | Print (heuristic) or texture (model) | **Abstains under 65px** |
| Unresolved queue | Anyone never matched, with an evidence crop | Works |

**Proxy attendance is the attack that matters and is not directly addressed.**
Someone sitting in for an absent student is, to this system, simply an
unrecognised person in the unresolved queue — which is the correct behaviour,
but it makes the teacher the detector, not the system.

The **body-carries-identity** design is itself an evasion countermeasure and
the most effective one here: covering your face no longer removes you. You stay
tracked, you stay in the queue with a torso crop, and your dwell is still
counted — it just needs a name attached afterwards. Under a face-only pipeline
you would simply cease to exist.

**Not built, deliberately:** directional entry/exit. Attendance asks "who is in
the room" (containment). "Who crossed this line, in which direction" needs
per-track crossing history and is a separate feature on top of zones (§6), not
a reinterpretation of them.

### T4 — Body structure carries the identity

**Goal:** a person who turns around, covers their face, or sits with their head
down must not disappear from the register.

**Where it stands: done, and it is the spine of the whole system** (§3).

The measurement that forced it: a person facing away scored **0.06** against
his own enrollment — below the impostor floor of ~0.25. **No threshold rescues
that.** If identity has to be re-derived per frame, anyone who turns around
ceases to exist. So the track owns the name, and a face only has to establish
it once.

Body structure buys four separate things, and it's worth keeping them distinct:

1. **Continuity of identity** through occlusion, turning away, head-down.
2. **Trackability at 3 fps.** What matters is box-widths-travelled-per-frame,
   not fps. A walking face clears ~3 of its own widths between analyses (zero
   IoU, association impossible); a body clears ~0.3. Measured ceiling for
   reliable association: **~1.2 m/s**. A person running through frame will
   break it — known limit, not a bug.
3. **Honest zone membership.** Bottom-centre of a body box is the feet. The
   face-only version used the chin, which sits at head height and put people in
   the wrong zone depending on how they leaned.
4. **A fallback crop for the queue.** No face ever visible? The teacher gets a
   torso — clothing and posture are what they have to go on, and that beats no
   crop at all.

**The counter-measurement that keeps the face tracker alive:** person detection
under-counts a packed room badly — **8 bodies against 98 faces** on this
project's own crowd photo. Bodies alone would delete the back rows. Hence two
trackers, namespaced by `FACE_TRACK_ID_OFFSET`.

### T5 — Identification at classroom distance

**Goal:** recognise 60 students from CCTV, where a face is a few dozen pixels.

**Where it stands: the open problem.** See §16 — it gets its own section
because every other target depends on it.

---

## 16. The real deployment: 60 students, very small faces

Everything above was built and measured against close-range stills and a single
desk-distance test camera. **The actual room is different and harder**, and the
single number that decides success is **face width in pixels.**

### The geometry

**60 students, 15 per row, 4 rows.** Face width follows directly from
students-per-row. A seated person is ~60 cm wide, a face ~16 cm, so a face is
roughly **27% of the per-person width**:

| Setup | Across one frame | px/person | **Face width** | Verdict |
|---|---|---|---|---|
| **1 camera**, 1920px | 15 | 128 px | **~34 px** | Above the ~30px floor, but only just |
| **2 cameras**, 1920px each | ~8 | 240 px | **~64 px** | Comfortable |

**Two cameras were available. Getting that second feed is worth more than any
model change** — it roughly doubles pixels-per-face, which is the only lever
that moves identification *and* liveness at the same time.

### Why pixels-per-face is the master variable

Measured on this project's photos, ArcFace genuine scores track face width
almost linearly:

| 67px | 75px | 47px | 26px | 25px |
|---|---|---|---|---|
| 0.4662 | 0.3916 | 0.4430 | 0.2796 | 0.2543 |

Against a 0.32 threshold and an impostor floor of ~0.25–0.26. **Below ~25px,
genuine and impostor scores overlap and identification is guesswork.**

Rule of thumb: roughly **`frame_width / 45` identifiable people per row.**

The same variable governs liveness at a *higher* floor (~65px, §9). So there
are three regimes in the target room:

- **> 65px** — identifiable and judgeable. Only reachable with two cameras.
- **30–65px** — identifiable, **not** judgeable. The exposed band.
- **< 30px** — neither. Self-cancelling for spoofing (a spoofer isn't
  identified either) but everyone in this band lands in the unresolved queue.

### `det_size`: the obvious fix that measurement does not support

This section used to say SCRFD's 640×640 input scales a 1920-wide frame 3×, so
a 34px face arrives as 11px and is "simply not found", and that raising
`det_size` to (1280,1280) was the first thing to try. **That was reasoning, not
measuring, and it is wrong.** Measured on this project's own photos with
`tools/measure_det_size.py`:

| det_size | usable faces (≥25px) | detect time |
|---|---|---|
| 640 | 65 | baseline |
| 1280 | 67 (+2) | +43% |
| 1920 | 68 (+3) | +94% |

Two things the raw detection count hides:

- **Most of what 1280 adds is unusable.** On the 98-face crowd photo it finds
  149 boxes instead of 98 — and 130 of them are under 25px, where this
  project's own measurements put genuine and impostor scores on top of each
  other. Those aren't recoveries, they're guesses that cost embedding time and
  put phantoms in front of a teacher.
- **The faces that matter were never missing.** The 1600px photo's 37–47px
  faces are all found at 640. SCRFD's feature pyramid goes down to stride 8, so
  a face downscaled to ~15px is still within reach — the plain
  frame_width ÷ det_size arithmetic overstates the loss badly.

Meanwhile the cost lands where there is no headroom: 3 fps allows **333 ms** per
analysis frame, detection at 1280 on the crowd photo already takes 247 ms on
the RTX 5060, and the deployment target is a GTX 1650 at roughly 3–4× slower —
before any embedding.

So `det_size` stays at 640. **The caveat that keeps this open:** these are
1280–1600px photos, not 1920 classroom footage at the real seating geometry.
Re-run the tool on the first real frames — it prints usable faces against
detect time, which is the trade the arithmetic above missed. If the target room
turns out to sit right on the stride-8 edge the answer could flip, but flip it
on a measurement this time.

### Enrollment: photos were denied, and that turned out to be better

Individual student photos were refused on privacy grounds partway through the
project. The gallery has to be built by **extracting faces from a clear
recording** and labelling them `Student_01..60`, with a seating-chart pass
later if the output ever needs to map to real people.

**This is an improvement, not a setback.** A phone selfie enrolled against CCTV
footage is a domain mismatch, and that mismatch is exactly why krish scored
**0.2663 against his own enrollment**. Enrolling from the same camera, angle
and lighting removes it.

The clustering is nearly free, because **a ByteTrack track is a person** — it
already carries a 30-observation averaged embedding. The job is merging a
handful of tracks per student, not clustering raw crops. And averaging is what
lifts small faces over threshold in the first place: **0.2663 → 0.3044** on a
24px face, and **0.2795 → 0.4638** on a 25px face while impostors moved only
0.2527 → 0.2629.

That is why **face-size filters were deliberately removed from the pipeline.**
Small faces are not noise to be discarded — they are the entire workload.

The gallery will hold real biometric templates of real students. It lives in
`data/`, which is gitignored. Keep it that way.

### Checklist for the first real-footage session

1. **Two cameras** if at all possible. Nothing else doubles pixels-per-face,
   and it is the only lever that moves identification and liveness together.
2. `python tools/measure_det_size.py --rtsp <url>` on real frames. Do not raise
   `det_size` because the arithmetic says to — on the photos here it cost 43%
   of the detect budget for two faces. Let the tool answer it for the room.
3. **Enroll from the footage**, not from photos. Merge tracks into
   `Student_01..60`.
4. **Re-measure the spoof threshold** at whatever `analysis_fps` ends up — it
   is only valid at the interval it was taken at.
5. **Count heads by hand** on a crowd frame and compare. Two silent
   people-loss bugs passed every synthetic test and were only caught this way
   (§13).
6. Expect liveness to **abstain** on most of the room. That is the honest
   result, not a failure — check `liveness_unjudged`, not just `spoofs`.
