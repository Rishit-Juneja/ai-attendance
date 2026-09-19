"""
Interactive web console: overview dashboard, ad-hoc person identification
(image/video), and a live webcam page with attendance logging + enrollment
database management.

This is separate from dashboard.py (the aiohttp+Flask live-run dashboard driven
by main.py) — this app is the operator console used to test/enroll/demo before
real CCTV ingestion exists.

Run:
    python -m src.webapp --gpu-profile dev
    Open http://localhost:5000
"""
import argparse
import base64
import json
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory

from . import enroll as enroll_mod
from .antispoof import SpoofChecker
from .attendance import AttendanceLogger
from .config import (
    Config,
    ENROLLMENT_DIR,
    GALLERY_INDEX_PATH,
    GALLERY_META_PATH,
    LOGS_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
)
from .pipeline import ArcFaceEmbedder, Pipeline
from .reports import generate_daily_csv, generate_daily_pdf
from .zones import Zone, draw_zones, load_zones, save_zones

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB upload cap


@app.url_defaults
def _stamp_static(endpoint, values):
    """
    Append the file's mtime to every static URL. Without this the browser keeps
    serving a cached app.js/app.css after an edit, so the page silently runs old
    code against a new API — which looks like a backend bug and isn't one.
    """
    if endpoint == "static" and "filename" in values:
        path = STATIC_DIR / values["filename"]
        if path.exists():
            values["v"] = int(path.stat().st_mtime)

_config = Config()
_embedder: ArcFaceEmbedder | None = None
_pipeline: Pipeline | None = None
_spoof_checker: SpoofChecker | None = None
_logger: AttendanceLogger | None = None
_review: AttendanceLogger | None = None   # last finished session, still reviewable

_PLACEHOLDER_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='40' height='40'>"
    "<rect width='40' height='40' rx='8' fill='%231c1c1f'/>"
    "<circle cx='20' cy='16' r='7' fill='%233a3a3f'/>"
    "<path d='M6 34a14 14 0 0 1 28 0Z' fill='%233a3a3f'/></svg>"
)


def get_embedder() -> ArcFaceEmbedder:
    global _embedder
    if _embedder is None:
        profile = _config.profile
        _embedder = ArcFaceEmbedder(det_size=profile.det_size, use_half=profile.use_half_precision)
    return _embedder


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(_config, embedder=get_embedder())
        if GALLERY_INDEX_PATH.exists() and GALLERY_META_PATH.exists():
            _pipeline.load_gallery(str(GALLERY_INDEX_PATH), str(GALLERY_META_PATH), live_only=True)
    return _pipeline


def get_spoof_checker() -> SpoofChecker:
    global _spoof_checker
    if _spoof_checker is None:
        _spoof_checker = SpoofChecker(
            movement_threshold=_config.spoof_pixel_movement_thresh,
            flag_after_n=_config.spoof_frame_count,
        )
    return _spoof_checker


def _reload_gallery_into_pipeline():
    if _pipeline is not None and GALLERY_INDEX_PATH.exists() and GALLERY_META_PATH.exists():
        _pipeline.load_gallery(str(GALLERY_INDEX_PATH), str(GALLERY_META_PATH), live_only=True)


# ---------------- small image helpers ----------------

def read_upload_image(file_storage) -> np.ndarray | None:
    data = file_storage.read()
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def decode_data_url(data_url: str) -> np.ndarray | None:
    b64data = data_url.split(",", 1)[1] if "," in data_url else data_url
    arr = np.frombuffer(base64.b64decode(b64data), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg_b64(img: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii")


# Advisory only — nothing is filtered on this. A reference face below this width
# makes a noisy embedding that drags every score down, so we say so rather than
# letting the user conclude the person isn't in the photo. Small faces in the
# *target* are fine and expected; it's the reference that needs to be good.
WEAK_REF_PX = 80

# Two good reference shots of the same person pair at ~0.5-0.6 (measured: two
# Krish photos = 0.574); different people land near 0. 0.35 splits them.
SAME_PERSON_REF_SIM = 0.35


def embed_reference(img: np.ndarray) -> tuple[np.ndarray, int] | None:
    """
    Largest-face embedding from a single reference photo, plus that face's pixel
    width. The width matters: a reference cropped from a small/distant face makes
    a weak embedding that drags down every score computed against it, which reads
    as "the system can't find this person" when the real problem is the reference.
    """
    faces = get_embedder().model.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    bbox = face.bbox.astype(int)
    return face.normed_embedding.astype(np.float32), int(bbox[2] - bbox[0])


def embed_references(images: list[np.ndarray]) -> tuple[np.ndarray, int, int] | None:
    """
    Average several reference photos of one person into one vector.

    A single image search can't use track aggregation (there's no track), so this
    is the only lever that lifts a distant target face out of the impostor noise.
    Measured against a 25px target: 1 photo scored 0.2795, 4 photos scored 0.3449,
    while the best impostor moved only 0.2527 -> 0.2629.

    Returns (mean_embedding, widest_face_px, count, indices_of_odd_ones_out).
    """
    got = [r for r in (embed_reference(im) for im in images) if r is not None]
    if not got:
        return None
    embs = [e for e, _ in got]
    mean = np.mean(embs, axis=0)
    norm = np.linalg.norm(mean)
    mean = (mean / norm if norm > 0 else mean).astype(np.float32)

    # Each reference contributes its *largest* face. Hand this a group photo and
    # it silently averages in a stranger, dragging the reference away from the
    # person you're looking for. Catch that instead of quietly degrading.
    #
    # Compare references pairwise, NOT against their own mean: with two vectors
    # the mean sits exactly between them, so even two unrelated faces score 0.707
    # against it — geometry, not similarity. Pairwise is count-independent.
    odd = []
    if len(embs) > 1:
        sims = np.array(embs) @ np.array(embs).T
        np.fill_diagonal(sims, -1.0)
        # Clean frontal shots of one person pair well above the match threshold;
        # anything that resembles no other reference is the odd one out.
        odd = [i for i in range(len(embs)) if sims[i].max() < SAME_PERSON_REF_SIM]
    return mean, max(w for _, w in got), len(got), odd


def gallery_lookup(emb: np.ndarray) -> dict | None:
    """Who is this, according to the enrolled database? None if nobody clears the bar."""
    matcher = get_pipeline().matcher
    if matcher is None or matcher.index.ntotal == 0:
        return None
    name, roll, score = matcher.match(emb, _config.match_threshold)
    if name == "Unknown":
        return None
    return {"name": name, "roll": roll, "score": round(score, 4)}


def detect_and_score(img: np.ndarray, ref_emb: np.ndarray, threshold: float | None = None) -> list[dict]:
    """
    Detect every face, score it against the reference, and separately ask the
    enrolled gallery who it is. The two answers are independent: the reference
    says "is this the person you uploaded", the gallery says "is this someone we
    already know". Showing both means an enrolled person gets named even when the
    reference photo is too weak to clear the threshold on its own.
    """
    threshold = _config.match_threshold if threshold is None else threshold
    faces = get_embedder().model.get(img)
    results = []
    for face in faces:
        emb = face.normed_embedding.astype(np.float32)
        bbox = face.bbox.astype(int)
        results.append({
            "bbox": bbox.tolist(),
            "width": int(bbox[2] - bbox[0]),
            "score": float(np.dot(emb, ref_emb)),
            "matched": float(np.dot(emb, ref_emb)) > threshold,
            "identity": gallery_lookup(emb),
        })
    return results


def draw_matches(img: np.ndarray, results: list[dict]) -> np.ndarray:
    out = img.copy()
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        who = r.get("identity")
        if r["matched"]:
            color, thickness = (0, 200, 0), 3
            label = f"{who['name']} {r['score']:.2f}" if who else f"MATCH {r['score']:.2f}"
        elif who:
            # Not the person being searched for, but someone the database knows.
            color, thickness, label = (200, 140, 0), 2, f"{who['name']} {r['score']:.2f}"
        else:
            color, thickness, label = (140, 140, 140), 1, f"{r['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, ly), (x1 + tw + 8, ly + th + 8), color, -1)
        cv2.putText(out, label, (x1 + 4, ly + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


# ---------------- overview data ----------------

def _overview_data():
    enrolled = 0
    if GALLERY_META_PATH.exists():
        with open(GALLERY_META_PATH) as f:
            enrolled = len(json.load(f))

    session_dirs = sorted(
        (d for d in LOGS_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    ) if LOGS_DIR.exists() else []

    sessions = []
    total_alerts = 0
    last_present = 0
    last_present_set = False

    for d in session_dirs:
        log_file = d / "attendance.json"
        if not log_file.exists():
            continue
        with open(log_file) as f:
            data = json.load(f)
        total_alerts += data.get("alerts", 0)
        if not last_present_set:
            last_present = data.get("present_now", 0)
            last_present_set = True
        if len(sessions) < 8:
            sessions.append({
                "session": data.get("session", d.name),
                "present_now": data.get("present_now", 0),
                "absent": data.get("absent", 0),
                "spoof_detected": data.get("spoof_detected", 0),
                "alerts": data.get("alerts", 0),
                "saved_at": datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })

    stats = {
        "enrolled": enrolled,
        "sessions": len(session_dirs),
        "total_alerts": total_alerts,
        "last_present": last_present,
    }
    return stats, sessions


def _photo_url_for(m: dict) -> str:
    candidates = []
    if m.get("photo"):
        candidates.append(Path(m["photo"]).name)
    candidates.append(f"{m['roll']}_best.jpg")
    for name in candidates:
        if name and (ENROLLMENT_DIR / name).exists():
            return f"/enrollment_photo/{name}"
    return _PLACEHOLDER_AVATAR


def _gallery_list() -> list[dict]:
    if not GALLERY_META_PATH.exists():
        return []
    with open(GALLERY_META_PATH) as f:
        meta = json.load(f)
    return [{"name": m["name"], "roll": m["roll"], "photo_url": _photo_url_for(m)} for m in meta]


# ---------------- pages ----------------

@app.route("/")
def home():
    stats, sessions = _overview_data()
    return render_template(
        "home.html", active_page="home", gallery_count=stats["enrolled"],
        gpu_profile=_config.gpu_profile, stats=stats, sessions=sessions,
    )


@app.route("/identify")
def identify_page():
    stats, _ = _overview_data()
    return render_template(
        "identify.html", active_page="identify", gallery_count=stats["enrolled"],
        gpu_profile=_config.gpu_profile, match_threshold=_config.match_threshold,
    )


@app.route("/live")
def live_page():
    stats, _ = _overview_data()
    return render_template(
        "live.html", active_page="live", gallery_count=stats["enrolled"],
        gpu_profile=_config.gpu_profile, gallery=_gallery_list(),
    )


@app.route("/enrollment_photo/<path:filename>")
def enrollment_photo(filename):
    return send_from_directory(str(ENROLLMENT_DIR), filename)


# ---------------- identify API ----------------

@app.route("/api/identify/image", methods=["POST"])
def api_identify_image():
    ref_files = request.files.getlist("reference")
    target_file = request.files.get("target")
    if not (ref_files and target_file):
        return jsonify({"ok": False, "error": "Both a reference photo and a target image are required."}), 400

    ref_imgs = [im for im in (read_upload_image(f) for f in ref_files) if im is not None]
    target_img = read_upload_image(target_file)
    if not ref_imgs or target_img is None:
        return jsonify({"ok": False, "error": "Could not read one of the uploaded images."}), 400

    ref = embed_references(ref_imgs)
    if ref is None:
        return jsonify({"ok": False, "error": "No face detected in the reference photo(s)."}), 400
    ref_emb, ref_width, ref_count, ref_odd = ref

    results = detect_and_score(target_img, ref_emb)
    annotated = draw_matches(target_img, results)

    return jsonify({
        "ok": True,
        "matches": results,
        "best_score": max((r["score"] for r in results), default=0.0),
        "ref_width": ref_width,
        "ref_count": ref_count,
        "ref_odd": ref_odd,
        "ref_identity": gallery_lookup(ref_emb),
        "known_count": sum(1 for r in results if r.get("identity")),
        # One reference photo is the single biggest limit on a distant match, so
        # nudge for more before the user concludes the person isn't in the photo.
        "ref_weak": ref_width < WEAK_REF_PX or ref_count < 2 or bool(ref_odd),
        "weak_ref_px": WEAK_REF_PX,
        "threshold": _config.match_threshold,
        "annotated_image": encode_jpeg_b64(annotated),
    })


# ponytail: synchronous request capped at MAX_SAMPLED_FRAMES so a big video can't
# hang the server; move to a background job + progress polling if longer videos
# or higher sample rates are needed.
MAX_SAMPLED_FRAMES = 90


@app.route("/api/identify/video", methods=["POST"])
def api_identify_video():
    ref_files = request.files.getlist("reference")
    target_file = request.files.get("target")
    if not (ref_files and target_file):
        return jsonify({"ok": False, "error": "Both a reference photo and a target video are required."}), 400

    ref_imgs = [im for im in (read_upload_image(f) for f in ref_files) if im is not None]
    if not ref_imgs:
        return jsonify({"ok": False, "error": "Could not read the reference image."}), 400
    ref = embed_references(ref_imgs)
    if ref is None:
        return jsonify({"ok": False, "error": "No face detected in the reference photo(s)."}), 400
    ref_emb, ref_width, ref_count, ref_odd = ref

    try:
        target_fps = float(request.form.get("fps", 3))
    except ValueError:
        target_fps = 3.0
    target_fps = max(1.0, min(5.0, target_fps))

    suffix = Path(target_file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        target_file.save(tmp.name)
        tmp_path = tmp.name

    frames_out = []
    best_score = 0.0
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"ok": False, "error": "Could not open that video file."}), 400
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(source_fps / target_fps))

        frame_idx = 0
        while len(frames_out) < MAX_SAMPLED_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                results = detect_and_score(frame, ref_emb)
                matched = any(r["matched"] for r in results)
                score = max((r["score"] for r in results), default=0.0)
                best_score = max(best_score, score)
                annotated = draw_matches(frame, results) if results else frame
                thumb = cv2.resize(annotated, (240, 180))
                frames_out.append({
                    "t": frame_idx / source_fps,
                    "matched": matched,
                    "score": score,
                    "thumb": encode_jpeg_b64(thumb, quality=70),
                })
            frame_idx += 1
        cap.release()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return jsonify({
        "ok": True,
        "frames": frames_out,
        "best_score": best_score,
        "ref_width": ref_width,
        "ref_count": ref_count,
        "ref_odd": ref_odd,
        "ref_identity": gallery_lookup(ref_emb),
        "ref_weak": ref_width < WEAK_REF_PX or ref_count < 2 or bool(ref_odd),
        "weak_ref_px": WEAK_REF_PX,
        "threshold": _config.match_threshold,
    })


# ---------------- gallery (database) API ----------------

@app.route("/api/gallery", methods=["GET"])
def api_gallery_list():
    return jsonify({"ok": True, "gallery": _gallery_list()})


@app.route("/api/gallery", methods=["POST"])
def api_gallery_add():
    name = request.form.get("name", "").strip()
    roll = request.form.get("roll", "").strip()
    photos = [p for p in request.files.getlist("photo") if p.filename]
    if not (name and roll and photos):
        return jsonify({"ok": False, "error": "Name, roll and at least one photo are required."}), 400

    # Several photos get averaged into one gallery vector — that averaging is what
    # lets a distant CCTV face clear the threshold. Measured on a 24px face: a
    # 1-photo entry scored 0.2663, a 2-photo average of the same person 0.3044.
    tmp_paths = []
    try:
        for photo in photos:
            suffix = Path(photo.filename or "photo.jpg").suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                photo.save(tmp.name)
                tmp_paths.append(tmp.name)
        ok = enroll_mod.enroll_person(name, roll, tmp_paths, get_embedder())
    finally:
        for p in tmp_paths:
            Path(p).unlink(missing_ok=True)

    if not ok:
        return jsonify({"ok": False, "error": "No face detected, or that roll number is already enrolled."}), 400

    _reload_gallery_into_pipeline()
    return jsonify({"ok": True, "photos": len(tmp_paths)})


@app.route("/api/gallery/<roll>", methods=["DELETE"])
def api_gallery_remove(roll):
    ok = enroll_mod.remove_person(roll)
    if ok:
        _reload_gallery_into_pipeline()
    return jsonify({"ok": ok})


# ---------------- live webcam API ----------------

@app.route("/api/live/start", methods=["POST"])
def api_live_start():
    global _logger, _spoof_checker
    _logger = AttendanceLogger(session_name=f"live_{datetime.now():%Y%m%d_%H%M%S}", config=_config)
    pipeline = get_pipeline()
    pipeline.reset_tracker()
    _spoof_checker = SpoofChecker(
        movement_threshold=_config.spoof_pixel_movement_thresh,
        flag_after_n=_config.spoof_frame_count,
    )
    return jsonify({"ok": True, "session": _logger.session_name})


@app.route("/api/live/frame", methods=["POST"])
def api_live_frame():
    global _logger
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image", "")
    if not data_url:
        return jsonify({"ok": False, "error": "No image received."}), 400
    frame = decode_data_url(data_url)
    if frame is None:
        return jsonify({"ok": False, "error": "Could not decode frame."}), 400

    if _logger is None:
        _logger = AttendanceLogger(session_name=f"live_{datetime.now():%Y%m%d_%H%M%S}", config=_config)

    pipeline = get_pipeline()
    result = pipeline.process_frame(frame, spoof_checker=get_spoof_checker())
    _logger.process_detections(result.detections, result.frame_idx, result.timestamp,
                               frame=frame)
    summary = _logger.get_summary()

    detections = [{
        "track_id": d.track_id,
        # The body box when one was found, so it stays drawn while the face is
        # hidden. face_bbox is what actually named them, drawn thinner inside.
        "bbox": d.bbox.tolist(),
        "face_bbox": None if d.face_bbox is None else d.face_bbox.tolist(),
        "name": d.name,
        "roll": d.roll,
        "score": round(d.match_score, 3),
        "is_spoof": d.is_spoof,
    } for d in result.detections]

    return jsonify({
        "ok": True,
        "detections": detections,
        "summary": summary,
        "inference_ms": result.inference_ms,
    })


# ---------------- network camera (phone / CCTV) ----------------
#
# The browser-webcam path needs a real /dev/video device. A phone running an
# MJPEG server app, or any CCTV camera speaking RTSP, is read straight by
# cv2.VideoCapture instead — no v4l2loopback, no kernel module, no root. This is
# the same ingestion path a real camera will use.

_stream = {"cap": None, "thread": None, "run": False, "frame": None, "err": None, "url": ""}
_stream_lock = threading.Lock()


class BoxGlide:
    """
    Advances detection boxes along their measured velocity between analysis frames.

    Analysis is meant to run slower than the camera — re-detecting every frame
    just to keep the overlay alive is the GPU doing display work. But drawing the
    last analysed box verbatim on the frames in between makes it sit still and
    then jump, which reads as lag even though the video itself is smooth. This
    carries each box forward at the speed it was last seen moving, so the overlay
    tracks the person continuously while detection stays cheap.

    Extrapolation is capped at one step so a box does not sail off across the room
    when someone stops walking or the analysis stalls.
    """

    def __init__(self, max_extrapolate_sec: float = 0.4):
        self._last: dict[int, tuple[np.ndarray, float]] = {}
        self._prev: dict[int, tuple[np.ndarray, float]] = {}
        self.max_extrapolate_sec = max_extrapolate_sec

    def observe(self, detections, now: float):
        for d in detections:
            tid = d.track_id
            if tid in self._last:
                self._prev[tid] = self._last[tid]
            self._last[tid] = (np.asarray(d.bbox, dtype=np.float32).copy(), now)
        live = {d.track_id for d in detections}
        for tid in [t for t in self._last if t not in live]:
            self._last.pop(tid, None)
            self._prev.pop(tid, None)

    def bbox_at(self, track_id: int, now: float) -> np.ndarray | None:
        cur = self._last.get(track_id)
        if cur is None:
            return None
        bbox, t1 = cur
        prev = self._prev.get(track_id)
        if prev is None:
            return bbox                      # only one sighting; nothing to extrapolate from
        pbox, t0 = prev
        dt = t1 - t0
        if dt <= 1e-3:
            return bbox
        velocity = (bbox - pbox) / dt
        ahead = min(max(now - t1, 0.0), self.max_extrapolate_sec)
        return bbox + velocity * ahead


def _stream_worker(url: str):
    """Pull frames, run the pipeline, keep the latest annotated frame for the feed."""
    global _logger
    if url.startswith("rtsp"):
        # OpenCV/FFmpeg default RTSP to UDP, which drops packets and smears
        # macroblocks across faces on a shared LAN. TCP costs nothing here.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency at one frame, not a backlog
    # Don't let a dead camera wedge the worker forever.
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    if not cap.isOpened():
        _stream["err"] = f"Could not open stream: {url}"
        _stream["run"] = False
        return

    _stream["cap"] = cap
    pipeline = get_pipeline()
    misses = 0
    last_dets = []
    glide = BoxGlide()
    while _stream["run"]:
        ok, frame = cap.read()
        if not ok:
            misses += 1
            if misses > 30:
                _stream["err"] = "Stream ended or camera unreachable."
                break
            continue
        misses = 0

        # Throttle analysis to the profile's fps; the feed still shows every frame.
        now = time.time()
        if pipeline.should_analyze():
            result = pipeline.process_frame(frame, spoof_checker=get_spoof_checker())
            last_dets = result.detections
            glide.observe(last_dets, now)
            if _logger is not None:
                _logger.process_detections(result.detections, result.frame_idx,
                                           result.timestamp, frame=frame)

        # Drawn on EVERY frame, not just analysed ones — publishing un-annotated
        # frames in between made the box strobe. Positions are carried forward by
        # BoxGlide rather than frozen, so the overlay stays smooth while analysis
        # runs at a fraction of the camera's frame rate.
        draw_zones(frame, pipeline.zones)
        for d in last_dets:
            glided = glide.bbox_at(d.track_id, now)
            x1, y1, x2, y2 = (d.bbox if glided is None else glided).astype(int)
            known = d.name != "Unknown"
            color = (0, 0, 255) if d.is_spoof else ((0, 200, 0) if known else (140, 140, 140))
            label = f"{d.name} {d.match_score:.2f}" if known else f"#{d.track_id} {d.match_score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        with _stream_lock:
            _stream["frame"] = frame
    cap.release()
    _stream["cap"] = None
    _stream["run"] = False


@app.route("/api/zones", methods=["GET", "POST"])
def api_zones():
    """
    Polygon zones in normalized 0..1 coordinates. Attendance counts only faces
    standing inside one; no zones at all means the whole frame counts.
    """
    if request.method == "GET":
        return jsonify({"ok": True, "zones": [
            {"name": z.name, "points": z.points} for z in load_zones()]})

    payload = (request.get_json(silent=True) or {}).get("zones", [])
    zones = []
    for z in payload:
        name = str(z.get("name", "")).strip()
        pts = z.get("points", [])
        if not name:
            return jsonify({"ok": False, "error": "Every zone needs a name."}), 400
        if len(pts) < 3:
            return jsonify({"ok": False,
                            "error": f"'{name}' has {len(pts)} points; a zone needs at least 3."}), 400
        # Reject out-of-range coords rather than storing a zone that silently
        # sits off-frame and matches nobody.
        for x, y in pts:
            if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
                return jsonify({"ok": False,
                                "error": f"'{name}' has a point outside the frame."}), 400
        zones.append(Zone(name=name, points=[[float(x), float(y)] for x, y in pts]))

    save_zones(zones)
    # The pipeline is a singleton built once, so it would keep the old polygons
    # until a restart otherwise.
    if _pipeline is not None:
        _pipeline.zones = zones
    return jsonify({"ok": True, "saved": len(zones)})


@app.route("/api/live/stream_start", methods=["POST"])
def api_live_stream_start():
    global _logger, _spoof_checker
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not url:
        return jsonify({"ok": False, "error": "A camera URL is required."}), 400

    # IP Webcam-style apps serve the stream at /video; accept the bare host too.
    # Normalize before the running-check so the same camera compares equal.
    if url.startswith("http") and not any(url.rstrip("/").endswith(s) for s in ("/video", ".mjpg", ".mjpeg", "/videofeed")):
        url = url.rstrip("/") + "/video"

    if _stream["run"]:
        # Reconnecting to the camera already playing is what someone means when
        # they hit Connect after a refresh. Refusing left them wedged until the
        # server restarted, with a live feed they couldn't attach to.
        if _stream["url"] == url:
            return jsonify({"ok": True, "url": url, "reattached": True,
                            "session": _logger.session_name if _logger else ""})
        return jsonify({"ok": False,
                        "error": f"Already streaming {_stream['url']}. Stop it first."}), 400

    _logger = AttendanceLogger(session_name=f"cam_{datetime.now():%Y%m%d_%H%M%S}", config=_config)
    pipeline = get_pipeline()
    pipeline.reset_tracker()
    _spoof_checker = SpoofChecker(
        movement_threshold=_config.spoof_pixel_movement_thresh,
        flag_after_n=_config.spoof_frame_count,
    )
    _stream.update(run=True, err=None, frame=None, url=url)
    _stream["thread"] = threading.Thread(target=_stream_worker, args=(url,), daemon=True)
    _stream["thread"].start()
    return jsonify({"ok": True, "url": url, "session": _logger.session_name})


@app.route("/api/live/snapshot")
def api_live_snapshot():
    """
    Latest annotated frame as a plain JPEG, polled by the client.

    Deliberately not multipart/x-mixed-replace: that holds a connection open for
    the whole session, which starves Flask's dev server of workers, and some
    embedded browsers refuse to render it at all. Polling single JPEGs is a few
    more requests and works everywhere.
    """
    with _stream_lock:
        frame = _stream["frame"]
    if frame is None:
        return ("", 204)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return ("", 204)
    return Response(buf.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/api/live/stream_status")
def api_live_stream_status():
    return jsonify({
        "ok": True,
        "running": _stream["run"],
        "url": _stream["url"],
        "error": _stream["err"],
        "summary": _logger.get_summary() if _logger else None,
    })


@app.route("/api/live/stream_stop", methods=["POST"])
def api_live_stream_stop():
    _stream["run"] = False
    if _stream["thread"]:
        _stream["thread"].join(timeout=3)
    return api_live_stop()


# ---------------- unresolved review queue ----------------

def _reviewable():
    """
    The session a teacher can still act on. Reviewing happens AFTER class, and
    stopping the session used to drop the logger — taking the whole unresolved
    queue with it at exactly the moment it was needed.
    """
    return _logger if _logger is not None else _review


@app.route("/api/live/unresolved")
def api_unresolved():
    log = _reviewable()
    if log is None:
        return jsonify({"ok": True, "unresolved": []})
    return jsonify({"ok": True, "session": log.session_name,
                    "unresolved": log.get_summary()["unresolved"]})


@app.route("/api/live/unresolved/<int:track_id>/crop")
def api_unresolved_crop(track_id):
    log = _reviewable()
    entry = log.unresolved.get(track_id) if log else None
    if entry is None or not entry.crop_path or not Path(entry.crop_path).exists():
        return ("", 404)
    return send_file(entry.crop_path, mimetype="image/jpeg")


@app.route("/api/live/resolve", methods=["POST"])
def api_resolve():
    """Teacher names an unresolved person; their dwell is back-dated to that name."""
    log = _reviewable()
    if log is None:
        return jsonify({"ok": False, "error": "No session to review."}), 400
    payload = request.get_json(silent=True) or {}
    track_id = payload.get("track_id")
    roll = (payload.get("roll") or "").strip()
    name = (payload.get("name") or "").strip()
    if track_id is None or not roll or not name:
        return jsonify({"ok": False, "error": "Name and roll are both required."}), 400

    rec = log.resolve(int(track_id), name, roll)
    if rec is None:
        return jsonify({"ok": False, "error": "That person is no longer in the queue."}), 404
    if log is _review:
        _write_session_files(log)   # the reports were already written; redo them
    return jsonify({"ok": True, "roll": rec.roll, "name": rec.name,
                    "dwell_sec": round(rec.duration_sec, 1), "status": rec.status})


def _write_session_files(log):
    log.save_log()
    log.export_csv()
    generate_daily_csv(log.records, log.session_name)
    generate_daily_pdf(log.records, log.alerts, log.session_name)


@app.route("/api/live/stop", methods=["POST"])
def api_live_stop():
    global _logger, _review
    if _logger is None:
        return jsonify({"ok": False, "error": "No active session."}), 400

    _logger.close_session()
    _write_session_files(_logger)
    # Kept for review, not discarded: naming the unresolved people is the work
    # that happens after the class, and it rewrites these same files.
    _review = _logger
    session_name = _logger.session_name
    _logger = None
    return jsonify({"ok": True, "session": session_name,
                    "unresolved": len(_review.unresolved)})


def main():
    parser = argparse.ArgumentParser(description="AI Attendance web console")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--gpu-profile", default=_config.gpu_profile, choices=["dev", "demo"])
    args = parser.parse_args()
    _config.gpu_profile = args.gpu_profile
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
