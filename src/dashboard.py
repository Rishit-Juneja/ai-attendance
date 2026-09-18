"""
Live dashboard: aiohttp WebSocket + Flask web UI.
"""
import asyncio
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from aiohttp import web
from flask import Flask, render_template

from .config import STATIC_DIR, TEMPLATES_DIR


class Dashboard:
    """aiohttp WebSocket server that broadcasts frame results."""

    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients: set = set()
        self.latest_data: dict = {}
        self._thread: threading.Thread | None = None
        self._app = web.Application()
        self._app.router.add_get("/ws", self._ws_handler)
        self._runner: web.AppRunner | None = None

    async def _ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)
        try:
            if self.latest_data:
                await ws.send_json(self.latest_data)
            async for msg in ws:
                pass  # we only push
        finally:
            self.clients.discard(ws)
        return ws

    async def _broadcast(self, data: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        self._runner = web.AppRunner(self._app)
        loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, self.host, self.port)
        loop.run_until_complete(site.start())
        print(f"[DASHBOARD] WebSocket server on ws://{self.host}:{self.port}")
        loop.run_forever()

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        time.sleep(0.5)

    def push_frame(self, frame_data: dict):
        self.latest_data = frame_data
        if hasattr(self, '_loop') and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(frame_data), self._loop)

    def stop(self):
        if hasattr(self, '_loop'):
            self._loop.call_soon_threadsafe(self._loop.stop)


class WebDashboard:
    """Flask web app serving the dashboard UI."""

    def __init__(self, ws_host="0.0.0.0", ws_port=8765, http_port=5000):
        self.ws_dashboard = Dashboard(host=ws_host, port=ws_port)
        self.http_port = http_port
        self.app = Flask(
            "attendance_dashboard",
            template_folder=str(TEMPLATES_DIR),
            static_folder=str(STATIC_DIR),
        )
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template("dashboard.html", ws_port=self.ws_dashboard.port)

    def start(self):
        self.ws_dashboard.start()
        print(f"[DASHBOARD] HTTP server on port {self.http_port}")
        threading.Thread(
            target=lambda: self.app.run(host="0.0.0.0", port=self.http_port, debug=False, use_reloader=False),
            daemon=True,
        ).start()

    def push_frame(self, frame_data: dict):
        self.ws_dashboard.push_frame(frame_data)

    def stop(self):
        self.ws_dashboard.stop()


def build_frame_message(frame_idx, detections, summary, alerts, inference_ms, fps):
    return {
        "type": "frame_update",
        "frame_idx": frame_idx,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "fps": round(fps, 1),
        "inference_ms": round(inference_ms, 1),
        "detections": [
            {
                "track_id": d.track_id,
                "name": d.name,
                "roll": d.roll,
                "score": round(d.match_score, 3),
                "bbox": d.bbox.tolist(),
                "is_spoof": d.is_spoof,
                "liveness": round(d.liveness_score, 3),
            }
            for d in detections
        ],
        "summary": summary,
        "alerts": [
            {"time": a.timestamp, "type": a.alert_type, "details": a.details}
            for a in alerts[-20:]
        ],
    }
