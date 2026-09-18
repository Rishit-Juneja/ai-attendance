"""
Main orchestrator: ties pipeline, anti-spoof, attendance logger, and dashboard together.

Usage:
    python -m src.main --source video.mp4 --gpu-profile dev
    python -m src.main --source 0 --gpu-profile demo    # camera index
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .config import Config, GALLERY_INDEX_PATH, GALLERY_META_PATH
from .pipeline import Pipeline
from .antispoof import SpoofChecker
from .attendance import AttendanceLogger
from .dashboard import WebDashboard, build_frame_message
from .reports import generate_daily_csv, generate_daily_pdf


class AttendanceSystem:
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = Pipeline(config)
        self.spoof_checker = SpoofChecker(
            movement_threshold=config.spoof_pixel_movement_thresh,
            flag_after_n=config.spoof_frame_count,
        )
        self.logger = AttendanceLogger()
        self.dashboard = WebDashboard(
            ws_port=config.dashboard_port,
            http_port=config.flask_port,
        )

        # Load gallery
        if GALLERY_INDEX_PATH.exists() and GALLERY_META_PATH.exists():
            self.pipeline.load_gallery(str(GALLERY_INDEX_PATH), str(GALLERY_META_PATH))
            print(f"[SYSTEM] Loaded gallery: {self.pipeline.matcher.index.ntotal} faces")
        else:
            print("[WARN] No gallery found. Run enrollment first.")

        self._running = False
        self._frame_count = 0
        self._analysis_count = 0
        self._fps_timer = time.monotonic()
        self._current_fps = 0.0

    def run(self, source):
        """
        Main loop. source can be a video file path or camera index (int).
        """
        if isinstance(source, str) and source.isdigit():
            cap = cv2.VideoCapture(int(source))
        elif isinstance(source, int):
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {source}")
            sys.exit(1)

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[SYSTEM] Source FPS: {source_fps}, Total frames: {total_frames}")

        self.dashboard.start()
        self._running = True

        # Video writer for annotated output
        writer = None
        if self.config.log_video_detections:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = self.logger.log_dir / "annotated_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, source_fps, (w, h))

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    if total_frames > 0:  # video file ended
                        break
                    continue  # camera: keep trying

                self._frame_count += 1

                # Throttle analysis
                if self.pipeline.should_analyze():
                    self._analysis_count += 1
                    result = self.pipeline.process_frame(
                        frame,
                        spoof_checker=self.spoof_checker,
                    )

                    # Update attendance
                    self.logger.process_detections(
                        result.detections,
                        result.frame_idx,
                        result.timestamp,
                    )

                    # Update FPS counter
                    now = time.monotonic()
                    elapsed = now - self._fps_timer
                    if elapsed >= 1.0:
                        self._current_fps = self._analysis_count / elapsed
                        self._analysis_count = 0
                        self._fps_timer = now

                    # Push to dashboard
                    summary = self.logger.get_summary()
                    msg = build_frame_message(
                        frame_idx=result.frame_idx,
                        detections=result.detections,
                        summary=summary,
                        alerts=self.logger.alerts,
                        inference_ms=result.inference_ms,
                        fps=self._current_fps,
                    )
                    self.dashboard.push_frame(msg)

                    # Annotate frame
                    if writer:
                        self._draw_detections(frame, result.detections)
                        writer.write(frame)

                    # Progress
                    if self._frame_count % 100 == 0:
                        print(
                            f"[FRAME {self._frame_count}] "
                            f"faces={result.num_faces} "
                            f"inference={result.inference_ms:.0f}ms "
                            f"fps={self._current_fps:.1f} "
                            f"present={summary.get('present_now', 0)}"
                        )

                # Slow down to source framerate (don't burn CPU on blank frames)
                time.sleep(1.0 / source_fps)

        except KeyboardInterrupt:
            print("\n[SYSTEM] Interrupted")
        finally:
            self._cleanup(cap, writer)

    def _draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            if det.is_spoof:
                color = (0, 0, 255)  # red for spoof
            elif det.name != "Unknown":
                color = (0, 255, 0)  # green for known
            else:
                color = (0, 165, 255)  # orange for unknown

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.name} ({det.roll})" if det.roll else det.name
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if det.is_spoof:
                cv2.putText(frame, "SPOOF", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _cleanup(self, cap, writer):
        self._running = False
        cap.release()
        if writer:
            writer.release()
        self.dashboard.stop()

        # Save logs
        self.logger.save_log()
        self.logger.export_csv()
        generate_daily_csv(self.logger.records, self.logger.session_name)
        generate_daily_pdf(self.logger.records, self.logger.alerts, self.logger.session_name)

        print("[SYSTEM] Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="AI Video Attendance System")
    parser.add_argument("--source", required=True, help="Video file path or camera index (0, 1, ...)")
    parser.add_argument("--gpu-profile", default="dev", choices=["dev", "demo"])
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--http-port", type=int, default=5000)
    parser.add_argument("--match-threshold", type=float, default=0.45)
    parser.add_argument("--session-name", default=None)
    args = parser.parse_args()

    config = Config(
        gpu_profile=args.gpu_profile,
        dashboard_port=args.dashboard_port,
        flask_port=args.http_port,
        match_threshold=args.match_threshold,
    )

    system = AttendanceSystem(config)
    if args.session_name:
        system.logger = AttendanceLogger(session_name=args.session_name)

    # Parse source
    source = args.source
    if source.isdigit():
        source = int(source)

    system.run(source)


if __name__ == "__main__":
    main()
