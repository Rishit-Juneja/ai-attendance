"""
Live demo: processes CASIA dataset frames, sends face images + labels to dashboard.
"""
import argparse
import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import Config, GALLERY_INDEX_PATH, GALLERY_META_PATH
from .data_loader import CASIALoader, DirectArcFaceEmbedder
from .dashboard import WebDashboard
from .attendance import AttendanceLogger


def encode_face_b64(img_bgr: np.ndarray, size=(128, 128)) -> str:
    """Resize face crop and encode as base64 JPEG for WebSocket transmission."""
    resized = cv2.resize(img_bgr, size, interpolation=cv2.INTER_CUBIC)
    _, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('ascii')


def build_scene_composite(face_frames: list, canvas_w=640, canvas_h=480) -> str:
    """
    Build a composite "CCTV view" image with all detected faces placed on a grid.
    Returns base64-encoded JPEG.
    """
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30

    if not face_frames:
        _, buf = cv2.imencode('.jpg', canvas)
        return base64.b64encode(buf).decode('ascii')

    n = len(face_frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cell_w = canvas_w // cols
    cell_h = canvas_h // rows
    face_size = min(cell_w, cell_h) - 8

    for i, ff in enumerate(face_frames):
        row = i // cols
        col = i % cols
        x = col * cell_w + 4
        y = row * cell_h + 4

        # Resize face to fit cell
        face_resized = cv2.resize(ff['image'], (face_size, face_size), interpolation=cv2.INTER_CUBIC)

        # Color border based on match
        if ff['is_spoof']:
            border_color = (0, 0, 255)  # red
        elif ff['name'] != 'Unknown':
            border_color = (0, 200, 0)  # green
        else:
            border_color = (0, 165, 255)  # orange

        cv2.rectangle(canvas, (x-1, y-1), (x+face_size+1, y+face_size+1), border_color, 2)
        canvas[y:y+face_size, x:x+face_size] = face_resized

        # Draw name label
        label = ff['name'] if ff['name'] != 'Unknown' else '?'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(canvas, (x, y+face_size-th-4), (x+tw+4, y+face_size), border_color, -1)
        cv2.putText(canvas, label, (x+2, y+face_size-2), font, font_scale, (255,255,255), thickness)

    _, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode('ascii')


class DemoRunner:
    def __init__(self, config: Config, scene_name: str, fps: int = 3):
        self.config = config
        self.scene_name = scene_name
        self.fps = fps

        self.loader = CASIALoader(data_root="/tmp")
        self.loader.discover()

        self.embedder = DirectArcFaceEmbedder()

        import faiss
        if GALLERY_INDEX_PATH.exists() and GALLERY_META_PATH.exists():
            self.index = faiss.read_index(str(GALLERY_INDEX_PATH))
            with open(GALLERY_META_PATH) as f:
                self.gallery_meta = json.load(f)
            print(f"[DEMO] Gallery: {self.index.ntotal} faces")
        else:
            print("[ERROR] No gallery. Run auto_enroll first.")
            sys.exit(1)

        self.logger = AttendanceLogger(session_name=f"demo_{scene_name}")
        self.dashboard = WebDashboard(ws_port=8765, http_port=5000)

        self._frame_count = 0
        self._fps_timer = time.monotonic()
        self._current_fps = 0.0
        self._recent_faces = []  # rolling window for composite view

    def _match(self, embedding: np.ndarray) -> tuple[str, str, float]:
        query = embedding.reshape(1, -1).astype(np.float32)
        similarities, indices = self.index.search(query, 1)
        score = float(similarities[0][0])
        idx = int(indices[0][0])
        if score > self.config.match_threshold and 0 <= idx < len(self.gallery_meta):
            m = self.gallery_meta[idx]
            return m["name"], m["roll"], score
        return "Unknown", "", score

    def run(self):
        scene = self.loader.get_scene(self.scene_name)
        if not scene:
            print(f"[ERROR] Scene {self.scene_name} not found")
            return

        print(f"[DEMO] Scene: {self.scene_name} ({len(scene.frames)} faces)")
        print(f"[DEMO] Open http://localhost:5000 in your browser\n")

        self.dashboard.start()
        time.sleep(1)

        interval = 1.0 / self.fps
        from .pipeline import Detection

        try:
            for face_frame, timestamp, idx in self.loader.iter_frames_at_fps(self.scene_name, self.fps):
                t0 = time.time()

                emb = self.embedder.embed(face_frame.image)
                if emb is None:
                    continue

                name, roll, score = self._match(emb)

                det = Detection(
                    track_id=idx,
                    bbox=np.array([0, 0, 96, 96]),
                    embedding=emb,
                    name=name,
                    roll=roll,
                    match_score=score,
                )

                self.logger.process_detections([det], idx, time.time())

                # Keep recent faces for composite view (last 20)
                face_entry = {
                    'image': face_frame.image.copy(),
                    'name': name,
                    'roll': roll,
                    'score': score,
                    'person_id': face_frame.person_id,
                    'is_spoof': False,
                }
                self._recent_faces.append(face_entry)
                if len(self._recent_faces) > 20:
                    self._recent_faces.pop(0)

                inference_ms = (time.time() - t0) * 1000

                self._frame_count += 1
                now = time.monotonic()
                elapsed = now - self._fps_timer
                if elapsed >= 1.0:
                    self._current_fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_timer = now

                # Build message with face images
                summary = self.logger.get_summary()
                scene_composite = build_scene_composite(self._recent_faces)

                msg = {
                    "type": "frame_update",
                    "frame_idx": idx,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "fps": round(self._current_fps, 1),
                    "inference_ms": round(inference_ms, 1),
                    "scene_composite": scene_composite,
                    "detections": [
                        {
                            "track_id": idx,
                            "name": name,
                            "roll": roll,
                            "score": round(score, 3),
                            "face_image": encode_face_b64(face_frame.image),
                            "is_spoof": False,
                            "liveness": 1.0,
                        }
                    ],
                    "summary": summary,
                    "alerts": [
                        {"time": a.timestamp, "type": a.alert_type, "details": a.details}
                        for a in self.logger.alerts[-20:]
                    ],
                }
                self.dashboard.push_frame(msg)

                # Progress bar in terminal
                pct = (idx + 1) / len(scene.frames) * 100
                bar = '█' * int(pct // 2) + '░' * (50 - int(pct // 2))
                print(f"\r  [{bar}] {pct:5.1f}% | {name}({roll}) score={score:.3f} | {self._current_fps:.0f}fps", end="", flush=True)

                processing_time = time.time() - t0
                sleep_time = max(0, interval - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"\n\n[DEMO] Done. {len(scene.frames)} faces processed.")
            print("[DEMO] Dashboard still running. Press Ctrl+C to stop.")
            # Keep servers alive so user can view the dashboard
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

        except KeyboardInterrupt:
            print("\n[DEMO] Interrupted")
        finally:
            self.logger.save_log()
            self.logger.export_csv()
            self.dashboard.stop()


def main():
    parser = argparse.ArgumentParser(description="Live demo with face visualization")
    parser.add_argument("--scene", default="P1E_S1_C1", help="Comma-separated scenes, e.g. P1E_S1_C1,P1L_S1_C1")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--gpu-profile", default="dev")
    parser.add_argument("--match-threshold", type=float, default=0.25)
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scene.split(",") if s.strip()]
    config = Config(gpu_profile=args.gpu_profile, match_threshold=args.match_threshold)

    for i, scene_name in enumerate(scenes):
        if i > 0:
            print(f"\n[DEMO] === Switching to scene: {scene_name} ===\n")
            time.sleep(2)
        runner = DemoRunner(config, scene_name, args.fps)
        runner.run()


if __name__ == "__main__":
    main()
