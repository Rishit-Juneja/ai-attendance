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
        # Measured on an RTX 5060: detect+embed is 7.4ms/frame (135 fps ceiling),
        # so match the camera instead of throttling. This is a tracking setting as
        # much as a speed one — ByteTrack associates by IoU, and at 3 fps a walking
        # person's box clears its own width between frames, so overlap hits zero
        # and every frame spawns a new track ID.
        # Set above the 15fps camera on purpose: should_analyze() gates on a
        # wall-clock interval, so an exactly-matched rate drops every other frame
        # on timing jitter. Overshooting means the gate never falsely skips.
        analysis_fps=20,
        det_size=(640, 640),
        det_batch=4,
        emb_batch=32,
        max_faces_per_frame=200,
        track_reembed_interval=30,
        spoof_check_every_n=5,
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
        spoof_check_every_n=10,
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
    # person returns as a stranger. Raised from 2s after live testing: a face
    # hidden for longer than this loses its name entirely.
    # CAUTION: identity is sticky per track, so a long buffer also means a track
    # whose ID gets handed to someone else keeps the old name. Body tracking is
    # the real fix; this is the knob until then.
    track_lost_sec: float = 8.0
    entry_cooldown_sec: float = 30.0    # min seconds between repeated entry logs
    absent_after_sec: float = 300.0     # mark absent after 5 min of no detection
    spoof_pixel_movement_thresh: float = 1.5  # motion heuristic threshold
    spoof_frame_count: int = 15         # frames to check for motion
    unknown_face_alert: bool = True
    log_video_detections: bool = True   # draw boxes on saved video

    @property
    def profile(self) -> GPUProfile:
        return PROFILES[self.gpu_profile]


DEFAULT_CONFIG = Config()
