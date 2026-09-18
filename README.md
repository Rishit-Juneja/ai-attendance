# AI Video-Based Attendance System

SRM Inter-College AI Innovation Challenge — Facial recognition attendance from CCTV/video.

## Architecture

```
Video/Camera → [Frame Sampler] → [SCRFD Detection] → [ByteTrack] → [ArcFace Embedding] → [FAISS Match]
                                         ↓                                      ↓
                                  [Spoof Checker]                        [Attendance Logger]
                                         ↓                                      ↓
                                  [Anomaly Alerts]                     [Dashboard / Reports]
```

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
and only faces standing inside it count — useful when a corridor or doorway is
visible in shot. With no zone defined the whole frame counts, so this is opt-in.
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

**Unresolved queue.** A tracked face that never matches the gallery is not
thrown away — it becomes `person1..N` with its own dwell record, its best crop,
and its entry time, and appears under "Needs review" on the Live page. A teacher
types a name and roll, and the time that person was already in the room is
credited to them retroactively. This queue outlives the session on purpose:
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
   (0.28 → 0.46 measured on a 25px face). Activation thresholds are set below the
   library defaults on purpose: SCRFD scores a sub-25px face at a median 0.685,
   so the stock 0.7 would refuse to track 26% of real faces.

2. **FAISS in-memory**: 500 faces × 512 dimensions = ~1MB. No vector DB needed.

3. **Motion-based spoof detection**: Instead of requiring a separate ONNX model, we check pixel variance in the face region across frames. Real faces have micro-movements (breathing, blinking). Printed photos/screens have near-zero variance.

4. **Frame throttling**: Analysis runs at 2-3 fps regardless of source framerate. Display/recording runs at native FPS. This keeps GPU usage manageable on weak hardware.

## Project Structure

```
ai-attendance/
├── src/
│   ├── config.py         # GPU profiles, thresholds, paths
│   ├── pipeline.py       # Detect → Track → Embed → Match
│   ├── antispoof.py      # Liveness/spoof detection
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
- Anti-spoofing via motion heuristic is basic — it catches a still printed photo
  and nothing else; a video on a phone screen passes. `SilentFaceLiveness` in
  antispoof.py is a stub waiting for the Silent-Face ONNX model. The flag now
  does block attendance (a spoofed face is never marked present), so the weak
  part is the detection, not the consequence.
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
