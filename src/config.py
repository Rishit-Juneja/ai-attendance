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
    # whose ID gets handed to someone else keeps the old name. Raise toward
    # 30-60s for a seated classroom, where association is near-trivial (IoU ~1.0
    # on a stationary person) and the buffer is therefore cheap.
    track_lost_sec: float = 8.0
    # Attendance is dwell-based: you are present once you have accumulated this
    # much time inside a zone, across however many visits. Walking past the door
    # no longer marks you present, which the old "seen in 3 frames" rule did.
    min_dwell_sec: float = 30.0
    # A visit stays open across gaps shorter than this, so the gap is credited as
    # dwell. Deliberately generous: in a classroom a student with their head down
    # or turned away loses their face for a minute at a time and is still sitting
    # there. Too short and their dwell is shredded into uncreditable slivers; too
    # long and someone who walked out keeps accruing time.
    exit_grace_sec: float = 60.0
    # No face for this long inside an open visit = possibly covering it.
    hiding_alert_after_sec: float = 15.0
    # Motion heuristic threshold. NEEDS RE-TUNING against a real printed photo:
    # it was set when consecutive samples were 50ms apart and they are now 333ms,
    # so both a real face and a hand-held photo move considerably more between
    # them. Erring high costs detections of real spoofs; erring low flags live
    # people. No substitute for holding a photo up to the camera and reading the
    # liveness scores off the live overlay.
    spoof_pixel_movement_thresh: float = 1.5
    spoof_frame_count: int = 15         # frames to check for motion
    unknown_face_alert: bool = True
    log_video_detections: bool = True   # draw boxes on saved video

    @property
    def profile(self) -> GPUProfile:
        return PROFILES[self.gpu_profile]


DEFAULT_CONFIG = Config()
