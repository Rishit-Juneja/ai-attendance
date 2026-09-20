# AI Video-Based Attendance System

SRM Inter-College AI Innovation Challenge — Facial recognition attendance from CCTV/video.

## Architecture

```
                          ┌→ [YOLO11 Person Detect] ─→ [ByteTrack on bodies] ─┐
Video/Camera → [Sampler] ─┤                                                   ├→ [FAISS Match]
                          └→ [SCRFD Face Detect] ──→ [ArcFace Embedding] ─────┘        ↓
                                     ↓                                          [Attendance Logger]
                              [Spoof Checker] → [Anomaly Alerts]                       ↓
                                                                              [Dashboard / Reports]
```

**Tracking is on bodies; faces only supply identity.** A face turned away scores
0.06 against its own enrollment — below the impostor floor — so identity cannot
be re-derived per frame, only carried by a track. A face inside a body box is
folded into that body's track and never counted twice. Faces belonging to no
detected body (back rows, where bodies are occluded) get their own tracks, so a
packed room is not reduced to the handful of people YOLO can separate.

## Quick Start

### 1. Install Dependencies

```bash
cd projects/ai-attendance
pip install -r requirements.txt
```

### 2. Enroll People

```bash
# Single person
python -m src.enroll --name "John Doe" --roll "CS2024001" --photo john.jpg

# Batch from CSV (columns: name, roll, photo_path)
python -m src.enroll --csv enrollments.csv
```

### 3. Run the System

```bash
# With video file (dev GPU)
python -m src.main --source path/to/video.mp4 --gpu-profile dev

# With camera (demo GPU)
python -m src.main --source 0 --gpu-profile demo

# Open dashboard
# WebSocket: ws://localhost:8765
# Web UI: http://localhost:5000
```

### 3b. Web Console (Overview / Identify / Live + Database)

```bash
python -m src.webapp --gpu-profile dev --port 5000
# Open http://localhost:5000
```

Three pages: an overview dashboard, an "Identify" tool (upload reference photos +
a target image/video to find that person), and a "Live" page (camera recognition
+ attendance logging, plus an enrollment panel for the face database).

The Live page takes either a browser webcam or a **network camera** — paste an
MJPEG URL from a phone app (`http://<phone-ip>:8080`, `/video` is appended) or a
CCTV `rtsp://` URL. Network cameras are captured and processed server-side, so
they need no `/dev/video` device and no browser permission. `main.py` remains the
headless batch runner.

**Enroll people from several photos.** Embeddings are averaged, and that averaging
is what lets a distant face clear the threshold — measured on a 24px face, a
1-photo entry scored 0.2663 (miss) against 0.3044 for a 2-photo entry.

**Zones.** Draw a polygon on the live feed ("Attendance zone" on the Live page)
and only people standing inside it count — useful when a corridor or doorway is
visible in shot. Membership is decided by the bottom-centre of the body box, so
it is the person's feet that have to be inside, not their chin. With no zone
defined the whole frame counts, so this is opt-in.
Points are stored normalized 0..1 in `data/zones.json`, so a zone stays correct
if the camera resolution changes. A zone is an *area*, not a tripwire: it answers
"who is in the room", not "who crossed this line, which way".

**Attendance is earned by dwell time, not by being seen.** Being detected marks
nothing; accumulating `min_dwell_sec` (default 30s) inside a zone does. A visit
survives gaps shorter than `exit_grace_sec` (default 60s) and keeps counting
through them — that is deliberate, because a student with their head down or
turned away loses their face for a minute at a time and is still sitting there.
Longer gaps close the visit and record an exit; coming back opens a new one and
the totals add up across all of them. Someone who only walks past the door ends
up as `brief` and is reported separately rather than counted present.

**Unresolved queue.** A tracked person who never matches the gallery is not
thrown away — they become `person1..N` with their own dwell record, their best
crop, and their entry time, and appear under "Needs review" on the Live page. A
teacher types a name and roll, and the time that person was already in the room
is credited to them retroactively.

The same back-fill happens automatically: someone who sits facing away for forty
seconds and then glances at the camera is credited from when they sat down, not
from the moment they looked up. The crop prefers a face and falls back to the
body, which is the only thing available for the people most likely to need
review. This queue outlives the session on purpose:
the review happens after class, and resolving someone rewrites the saved report.
Expect it to be the normal path rather than an edge case — identification needs
roughly a 30px face where tracking only needs a 25px body, so in a large room
the back rows are tracked long before they can be named.

### 4. Generate Reports

Reports are auto-generated on shutdown. Manual:
```bash
python -m src.reports  # see module for API
```

## GPU Profiles

| Setting | dev (RTX 5060 8GB) | demo (GTX 1650 4GB) |
|---------|-------------------|---------------------|
| Analysis FPS | 3 | 2 |
| Detection size | 640×640 | 320×320 |
| Emb batch size | 32 | 8 |
| FP16 | No | Yes |
| Max faces/frame | 200 | 100 |

Switch via `--gpu-profile` or `GPU_PROFILE=dev` env var.

## Key Design Decisions

1. **ByteTrack over DeepSORT**: via the `trackers` package — Kalman motion
   prediction plus the two-stage high/low-confidence association BYTE is named
   for. `ByteTrackWrapper` in pipeline.py adds a rolling embedding average per
   track on top, which is what lets a small CCTV face clear the match threshold
   (0.28 → 0.46 measured on a 25px face). Two instances run: one on bodies, one
   on the leftover faces.

   Spawn thresholds are per-tracker, because a confidence score only means
   something within one detector. SCRFD scores a sub-25px face at a median 0.685,
   so the library's stock 0.7 would refuse to track 26% of real faces; YOLO
   person scores in a crowded room run 0.37–0.91, so the face-calibrated 0.5 sat
   in the middle of the real distribution and refused a track to a third of the
   people in the room. Both detectors emit *below* their tracker's spawn
   threshold on purpose — those boxes cannot start a track, but BYTE's second
   stage uses them to keep an occluded person's existing track alive.

2. **3 fps analysis, and why it needs bodies**: displacement measured in box
   widths is scale-invariant. At 1 m/s a person moves 2.1 face-widths per frame
   at 3 fps (boxes do not overlap, so no association is possible) but only 0.67
   body-widths (they do). Bodies hold to ~1.2 m/s; beyond that a track breaks,
   which dwell-based attendance does not care about because people who do not
   stop are not present anyway. `test_three_fps_needs_body_boxes_not_face_boxes`
   asserts this rather than trusting the comment.

3. **FAISS in-memory**: 500 faces × 512 dimensions = ~1MB. No vector DB needed.

4. **Liveness by texture, with motion as the fallback**: the original design
   checked pixel change in the face crop across frames, on the theory that real
   faces micro-move and photos don't. Half right. Measured on this camera, a
   live face floors at 1.40 and a static image sits at 0.00–0.06 — clean
   separation, so print is caught. But a photo *on a screen* moved in 69% of
   samples: LCD refresh beating against the camera shutter produces banding that
   is indistinguishable from life by this metric, and no threshold fixes it
   because the flicker is larger than the signal. So MiniFASNetV2 is the primary
   when its weights are present, judging texture rather than movement, and the
   heuristic is what runs when they aren't.

5. **Frame throttling**: Analysis runs at 3 fps regardless of source framerate;
   display/recording runs at native FPS, with `BoxGlide` carrying boxes forward
   at their measured velocity in between so the overlay stays smooth. This is not
   only a GPU saving — the competition GPU is 10-15km away, and a full-rate
   640x480 feed is ~5 Mbps per camera.

## Project Structure

```
ai-attendance/
├── src/
│   ├── config.py         # GPU profiles, thresholds, paths
│   ├── pipeline.py       # Detect → Track (bodies) → Embed (faces) → Match
│   ├── persons.py        # YOLO11 person detection + face→body association
│   ├── antispoof.py      # MiniFASNetV2 liveness, motion heuristic fallback
│   ├── attendance.py     # Dwell-based presence, unresolved queue, alerts
│   ├── dashboard.py      # WebSocket + Flask dashboard
│   ├── reports.py        # CSV + PDF report generation
│   ├── enroll.py         # Enrollment script
│   └── main.py           # Orchestrator
├── templates/
│   └── dashboard.html    # Live dashboard UI
├── data/
│   ├── enrollments/      # Reference photos
│   ├── gallery/          # FAISS index + metadata
│   ├── logs/             # Session logs
│   └── reports/          # Generated reports
└── requirements.txt
```

## Tuning for 150 People

- SCRFD detects ~150 faces in ~30ms on RTX 5060
- ArcFace embedding: ~5ms per face, but batched (32 at a time)
- ByteTrack: only embed tracks that need re-identification
- Expected throughput: ~2-3 fps on RTX 5060, ~1 fps on GTX 1650
- This is sufficient for attendance (no frame-perfect recognition needed)

## Limitations (Be Blunt)

- A single enrollment photo is *not* enough for distant faces. ArcFace is
  pose-robust but resolution-sensitive: on a 24px face a 1-photo entry scored
  0.2663 against a 0.32 threshold, a 2-photo average 0.3044. Use 3-4 photos.
- Reference/enrollment photo quality dominates everything else. A 54px reference
  face scored 0.2663 where a 75px one scored 0.3289 on the same target.
- Recognition quality tracks pixels-per-face, not GPU. Roughly
  `frame_width / 45` identifiable people per row — ~35 at 1600px wide. Beyond
  that faces fall under ~25px and scores collapse into the impostor range.
- Anti-spoofing has two modes and the good one needs a 1.74MB download:

  ```bash
  curl -L -o data/models/minifasnet_v2.onnx \
    https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx
  ```

  With that file present, MiniFASNetV2 judges liveness from texture and the
  motion heuristic is not used at all. Without it, the heuristic runs and
  **anything on a screen defeats it.** That is measured, not assumed: a photo on
  an LCD produced motion in 69% of samples, because the monitor's refresh beats
  against the camera shutter and the banding reads as life. The threshold
  (`spoof_pixel_movement_thresh = 1.0`) was measured the same way — 95 live
  samples floored at 1.40, a static image sat at 0.00–0.06 — so the heuristic
  does reliably catch a *printed* photo, and nothing else. Boot logs which mode
  is active. Either way the flag blocks attendance; a spoofed face is never
  marked present.
- Re-measure the spoof threshold if `analysis_fps` changes — it is only valid
  for the sampling interval it was taken at. `tools/calibrate_spoof.py` does it
  in two 40s runs. Keep exactly one face in shot: with a person and a photo both
  visible the detector finds one or the other between frames, and diffing a face
  against a photograph poisons the result. The tool now rejects such a run
  instead of reporting it.
- Alert *categories* are still the original guesses. The flooding is fixed —
  repeats collapse into one row per subject with a count — but whether
  `face_hiding` (fires when a known person leaves frame for 10s+) is worth having
  at all has not been decided.
- The 29 auto-enrolled CASIA entries are embedded without landmark alignment
  (`DirectArcFaceEmbedder`), so they are excluded from live matching via
  `GalleryMatcher(live_only=True)`. Mixing them in makes aligned queries collapse
  onto whichever aligned entry exists. They remain valid for evaluate.py/run_demo.py,
  where queries are unaligned too.
- No persistent database — everything is file-based (JSON + FAISS index)
