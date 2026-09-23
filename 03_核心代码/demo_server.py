#!/usr/bin/env python3
"""Zero-dependency browser demo for the travel Function Calling project."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from demo_agent import TravelDemoAgent
from runtime_config import get_env


WEB_ROOT = Path(__file__).with_name("web")





class DemoHandler(BaseHTTPRequestHandler):
    agent = TravelDemoAgent()
    agent_lock = threading.Lock()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[demo] {self.address_string()} - {fmt % args}")

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            body = (WEB_ROOT / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        assets = {
            "/assets/style.css": ("style.css", "text/css; charset=utf-8"),
            "/assets/app.js": ("app.js", "application/javascript; charset=utf-8"),
            "/api/evaluation": ("evaluation.json", "application/json; charset=utf-8"),
        }
        if path in assets:
            filename, content_type = assets[path]
            body = (WEB_ROOT / filename).read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/recording":
            legacy = Path(__file__).with_name("recording_page.html")
            if not legacy.exists():
                self._send_json({"error": "legacy recording page not included"}, HTTPStatus.NOT_FOUND)
                return
            body = legacy.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/health":
            self._send_json({"status": "healthy", "mode": "offline", "guides": len(self.agent.guide_index)})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/reset":
            with self.agent_lock:
                self.agent.reset()
            self._send_json({"status": "ok"})
            return
        if path != "/api/chat":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 16000:
                raise ValueError("message payload exceeds limit")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict) or not isinstance(payload.get("message", ""), str):
                raise ValueError("message must be a string")
            message = str(payload.get("message", ""))
            with self.agent_lock:
                response = self.agent.chat(message).to_dict()
            self._send_json(response)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": f"invalid request: {exc}"}, HTTPStatus.BAD_REQUEST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the offline travel Agent browser demo")
    parser.add_argument("--host", default=get_env("DEMO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(get_env("DEMO_PORT", "8090")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Travel Agent Demo: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
