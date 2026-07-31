"""Memory Vault Webhook — HTTP endpoint for triggering exports.

Run with:
    python -m memory_vault.webhook [--port 9876] [--host 127.0.0.1]

Endpoints:
    GET  /webhook/status         → health check + metrics
    POST /webhook/export         → export a Hermes session
    POST /webhook/daily          → build daily pack
    POST /webhook/project        → snapshot a project dir

All POST endpoints accept JSON body and return JSON responses.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── API functions (import memory-vault at module level) ────────────

try:
    from memory_vault.core.builder import ContextBuilder

    _HAVE_VAULT = True
except ImportError:
    _HAVE_VAULT = False

_AUTO_DIR = Path(
    os.environ.get("MEMORY_VAULT_AUTO_PACK_DIR",
                   str(Path.home() / ".hermes" / "memory-vault-packs"))
)
_AUTO_DIR.mkdir(parents=True, exist_ok=True)

_REQUEST_LOG: list[dict] = []


def _get_builder() -> ContextBuilder:
    return ContextBuilder(hermes_home=os.environ.get("HERMES_HOME"))


def _json_response(data: dict, status: int = 200) -> tuple[str, int, dict]:
    body = json.dumps(data, default=str, indent=2) + "\n"
    return body, status, {"Content-Type": "application/json"}


# ── Request handler ────────────────────────────────────────────────


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP handler for Memory Vault webhook endpoints."""

    def log_message(self, fmt, *args):
        """Suppress default logging; we track requests ourselves."""
        now = datetime.now().isoformat()
        _REQUEST_LOG.append({"time": now, "method": self.command, "path": self.path})
        if self.command == "GET":
            pass  # quiet
        else:
            print(f"  [{now}] {self.command} {self.path}")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _handle(self):
        try:
            if not _HAVE_VAULT:
                self._send(*_json_response(
                    {"error": True, "message": "memory-vault not installed"}, 500
                ))
                return

            path = self.path.rstrip("/")

            if self.command == "GET" and path == "/webhook/status":
                self._handle_status()
            elif self.command == "POST" and path == "/webhook/export":
                self._handle_export()
            elif self.command == "POST" and path == "/webhook/daily":
                self._handle_daily()
            elif self.command == "POST" and path == "/webhook/project":
                self._handle_project()
            else:
                self._send(*_json_response(
                    {"error": True, "message": f"Not found: {self.command} {path}"}, 404
                ))
        except Exception as e:
            self._send(*_json_response(
                {"error": True, "message": str(e)}, 500
            ))

    def _send(self, body: str, status: int, headers: dict):
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    # ── Endpoint handlers ─────────────────────────────────────────

    def _handle_status(self):
        packs = list(_AUTO_DIR.glob("*.hermes-memory"))
        self._send(*_json_response({
            "ok": True,
            "memory_vault_installed": _HAVE_VAULT,
            "auto_pack_dir": str(_AUTO_DIR),
            "packs_in_dir": len(packs),
            "requests_served": len(_REQUEST_LOG),
        }))

    def _handle_export(self):
        body = self._read_body()
        session_id = body.get("session_id", "")
        if not session_id:
            self._send(*_json_response(
                {"error": True, "message": "session_id is required"}, 400
            ))
            return

        output = body.get("output", "")
        narrate = body.get("narrate", False)

        builder = _get_builder()
        pack = builder.build_from_session(
            session_id=session_id,
            include_artifacts=True,
            narrate=narrate,
        )

        if output:
            out = Path(output)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = _AUTO_DIR / f"webhook_{session_id[:12]}_{ts}.hermes-memory"

        pack.write(out)
        s = pack.summary()
        self._send(*_json_response({
            "ok": True,
            "path": str(out),
            "title": s["title"],
            "messages": s["message_count"],
            "artifacts": s["artifact_count"],
        }))

    def _handle_daily(self):
        body = self._read_body()
        date_str = body.get("date_str", "")
        limit = body.get("limit", 50)

        builder = _get_builder()
        pack = builder.build_daily_pack(date_str=date_str or None, limit=limit)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = _AUTO_DIR / f"daily-webhook_{ts}.hermes-memory"
        pack.write(out)
        s = pack.summary()
        self._send(*_json_response({
            "ok": True,
            "path": str(out),
            "title": s["title"],
            "sessions": s["message_count"],
        }))

    def _handle_project(self):
        body = self._read_body()
        project_dir = body.get("project_dir", "")
        if not project_dir or not Path(project_dir).is_dir():
            self._send(*_json_response(
                {"error": True, "message": "Invalid or missing project_dir"}, 400
            ))
            return

        title = body.get("title", "")
        tree_depth = body.get("tree_depth", 2)

        builder = _get_builder()
        pack = builder.build_project_pack(
            project_dir=project_dir,
            title=title or None,
            tree_depth=tree_depth,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = _AUTO_DIR / f"project-webhook_{ts}.hermes-memory"
        pack.write(out)
        s = pack.summary()
        self._send(*_json_response({
            "ok": True,
            "path": str(out),
            "title": s["title"],
            "project": s.get("project_name", ""),
            "artifacts": s["artifact_count"],
        }))

    # ── HTTP methods ──────────────────────────────────────────────

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()


# ── Entry point ────────────────────────────────────────────────────


def serve(host: str = "127.0.0.1", port: int = 9876):
    print("🧠 Memory Vault Webhook")
    print(f"   Endpoint: http://{host}:{port}/webhook/status")
    print(f"   Auto-pack dir: {_AUTO_DIR}")
    print(f"   memory-vault installed: {_HAVE_VAULT}")
    print("   Listening...")

    server = HTTPServer((host, port), WebhookHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n   Shutting down.")
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory Vault Webhook")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9876, help="Port (default: 9876)")
    args = parser.parse_args()

    serve(host=args.host, port=args.port)
