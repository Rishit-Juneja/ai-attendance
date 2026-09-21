"""
Central configuration. Two GPU profiles: 'dev' (RTX 5060) and 'demo' (GTX 1650).
Switch via environment variable GPU_PROFILE or pass to Pipeline.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ENROLLMENT_DIR = DATA_DIR / "enrollments"
GALLERY_DIR = DATA_DIR / "gallery"
GALLERY_INDEX_PATH = GALLERY_DIR / "gallery.index"
GALLERY_META_PATH = GALLERY_DIR / "gallery_meta.json"
LOGS_DIR = DATA_DIR / "logs"
REPORTS_DIR = DATA_DIR / "reports"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

for d in [ENROLLMENT_DIR, GALLERY_DIR, LOGS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class GPUProfile:
    name: str
    analysis_fps: int            # frames/sec sent to the analysis pipeline
    det_size: tuple[int, int]    # input size for SCRFD detector
    det_batch: int               # batch size for detector
    emb_batch: int               # batch size for ArcFace embedder
    max_faces_per_frame: int     # skip frame if too many faces (avoid OOM)
    track_reembed_interval: int  # re-embed track every N frames of inactivity
    spoof_check_every_n: int     # run liveness check every N embeddings
    use_half_precision: bool     # FP16 for lower VRAM


PROFILES = {
    "dev": GPUProfile(
        name="dev",
        # 20 until tracking moved onto bodies. The old ceiling was geometric, not
        # computational: ByteTrack associates by IoU, and at 3 fps a walking face
        # box clears ~3 of its own widths between frames, so overlap hits zero and
        # every frame spawned a new track ID. A body box is ~7x wider, so the same
        # walk is ~0.3 body widths and overlap stays high.
        #
        # 3 is the target rate, not a fallback. The GPU is 10-15km away with
        # unpredictable latency and the deployment hardware is a GTX 1650 — a
        # full-rate 640x480 feed is ~5 Mbps per camera and there is no budget to
        # ship it. Display stays at camera rate locally; BoxGlide carries the
        # overlay between analyses.
        analysis_fps=3,
        det_size=(640, 640),
        det_batch=4,
        emb_batch=32,
        max_faces_per_frame=200,
        track_reembed_interval=30,
        # Every analysed frame. This skipped 4 in 5 when analysis ran at 20fps,
        # which was the point; at 3fps the same setting would take 25 seconds to
        # flag a held-up photo (spoof_frame_count samples, 1.67s apart).
        spoof_check_every_n=1,
        use_half_precision=False,
    ),
    "demo": GPUProfile(
        name="demo",
        analysis_fps=2,
        det_size=(320, 320),
        det_batch=1,
        emb_batch=8,
        max_faces_per_frame=100,
        track_reembed_interval=60,
        spoof_check_every_n=1,
        use_half_precision=True,
    ),
}


@dataclass
class Config:
    gpu_profile: str = os.environ.get("GPU_PROFILE", "dev")
    source_type: str = "video"          # "video" or "camera"
    source_path: str = ""               # video file path or camera index
    camera_index: int = 0
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8765
    flask_port: int = 5000
    # Cosine similarity threshold (higher = stricter, IndexFlatIP).
    # Measured here: impostors top out ~0.25-0.26 even against an averaged
    # reference, genuine pairs clear 0.33+ once embeddings are aggregated.
    # Small CCTV faces are handled by averaging observations across a ByteTrack
    # track (see track_embed_window), NOT by filtering them out — a 25px face
    # scores 0.46 aggregated vs 0.28 from a single frame.
    match_threshold: float = 0.32
    track_embed_window: int = 30        # observations averaged per track; rolls off on ID switch
    # Seconds a track survives with no detection before it is destroyed and the
    # person returns as a stranger. Now that tracking is on bodies this only has
    # to cover a body being fully occluded — a hidden FACE no longer costs the
    # name, which is what forced this up from 2s originally.
    # CAUTION: identity is sticky per track, so a long buffer also means a track
    # whose ID gets handed to someone else keeps the old name.
    #
    # Raised 8 -> 45 on 2026-09-20 after the first real-camera run: two people in
    # two minutes produced THREE unresolved labels, because a seated man's body
    # track died behind his chair and respawned with a new ID. Each respawn
    # re-matches the gallery from scratch; that one got lucky and landed on the
    # same name, but a miss would have split him into two half-length records.
    # 45s is cheap here because a lost track is never emitted — update() returns
    # only tracks matched to a detection this frame — so a longer buffer buys ID
    # continuity without crediting dwell to anyone who is not visible.
    # Safe for a seated room, where association is near-trivial (IoU ~1.0 on a
    # stationary person). Lower it toward 10-15s for a doorway or corridor view,
    # where people genuinely leave and a stale box can be claimed by a stranger.
    track_lost_sec: float = 45.0
    # Attendance is dwell-based: you are present once you have accumulated this
    # much time inside a zone, across however many visits. Walking past the door
    # no longer marks you present, which the old "seen in 3 frames" rule did.
    min_dwell_sec: float = 30.0
    # Timetable, as wall-clock spans. Empty = no slicing, and a session is scored
    # as one block exactly as before.
    #
    # Hourly, but off the HALF hour — and the footage confirms it: A607 is full at
    # 09:59:56 and empty at 10:30:55, which is the 09:30 period ending, not a
    # 10:00 one. Boundaries generated on :00 would have cut every period in half.
    #
    # Lunch, free periods and labs are not marked, and do not need to be. Labs
    # run in another room where the students are marked by that room's camera, so
    # an idle A607 during one is NOT sixty absences — per_lecture() reports such
    # a period as no_session and records no verdict against anybody. A full day's
    # run is therefore also how you discover which periods these are.
    lectures: tuple = (
        "09:30-10:30", "10:30-11:30", "11:30-12:30", "12:30-13:30",
        "13:30-14:30", "14:30-15:30", "15:30-16:30",
    )
    # Per-lecture bar for being marked present. Distinct from min_dwell_sec below,
    # which is a noise floor ("was in the room" vs "walked past the door"); this
    # one is policy ("attended enough of the lecture to count").
    lecture_min_dwell_sec: float = 1800.0       # 30 minutes
    # A visit stays open across gaps shorter than this, so the gap is credited as
    # dwell. Deliberately generous: in a classroom a student with their head down
    # or turned away loses their face for a minute at a time and is still sitting
    # there. Too short and their dwell is shredded into uncreditable slivers; too
    # long and someone who walked out keeps accruing time.
    exit_grace_sec: float = 60.0
    # No face for this long inside an open visit = possibly covering it.
    hiding_alert_after_sec: float = 15.0
    # Motion heuristic threshold, MEASURED 2026-09-20 on the 1080p CCTV feed at
    # 3 fps (333ms between samples) via tools/calibrate_spoof.py. 95 live samples:
    # min 1.40, p5 5.75, median 13.85 — a live face never once dropped below 1.4.
    # A static image sat at 0.00-0.06. So 1.0 clears the live floor with margin
    # while still catching a motionless photo.
    #
    # Deliberately set BELOW the live minimum rather than midway between the two
    # distributions: a false spoof flag blocks a real student's attendance, which
    # is a worse failure than missing a spoof. flag_after_n adds a second margin
    # by requiring 15 consecutive samples (5s) under this before flagging.
    #
    # KNOWN HOLE: this catches print, not screens. In the same measurement a photo
    # displayed on an LCD produced motion in 69% of samples (median 9.72) because
    # the monitor's refresh beats against the camera shutter — banding shifts
    # between frames and reads as life. No threshold fixes that; the flicker is
    # genuinely larger than the signal. A phone held up by a judge defeats this.
    # SilentFaceLiveness in antispoof.py is the answer there, not a number here.
    #
    # Re-measure if analysis_fps changes — the value is only valid for the
    # interval it was measured at.
    spoof_pixel_movement_thresh: float = 1.0
    spoof_frame_count: int = 15         # frames to check for motion
    # MiniFASNetV2 (Apache 2.0). Present = it replaces the motion heuristic
    # entirely; absent = heuristic, and screens go undetected. Fetch with:
    #   curl -L -o data/models/minifasnet_v2.onnx \
    #     https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx
    spoof_model_path: str = str(DATA_DIR / "models" / "minifasnet_v2.onnx")
    unknown_face_alert: bool = True
    # Write data/logs/<session>/annotated.mp4 when analysing a recording. On by
    # default because judging tracking is the main reason to feed a recording in
    # at all, and the browser feed cannot do it — it polls ~8fps of wall time
    # while a file runs at 2-3x real speed, so ~1 frame in 13 is ever seen.
    # Measured cost on 1080p/30fps: throughput 2.8x -> 1.8x real time, and
    # ~0.84 MB per second of footage (a 40-minute class is ~2 GB). Turn off for
    # long runs nobody is going to watch.
    log_video_detections: bool = True

    @property
    def profile(self) -> GPUProfile:
        return PROFILES[self.gpu_profile]


DEFAULT_CONFIG = Config()
