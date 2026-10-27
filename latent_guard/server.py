"""Zero-dependency HTTP sidecar.

Routes (JSON in, JSON out):
  POST /score   {prompt, answer, context} -> verdict, risk, sentences
  GET  /queue   -> flagged answers awaiting review
  POST /review {request_id, decision: confirmed|released, note}
  GET  /metrics -> counts, flag rate, precision on reviewed items
  GET  /export  -> labeled examples as JSON lines
  GET  /health  -> ok

Run: python3 -m latent_guard.server (default port 8765, data in ./data)
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_guard.queue import ReviewQueue  # noqa: E402
from latent_guard.scorer import score_answer  # noqa: E402

DATA_DIR = os.environ.get("LATENT_GUARD_DATA", os.path.join(os.getcwd(), "data"))
os.makedirs(DATA_DIR, exist_ok=True)
STORE_TEXT = os.environ.get("LATENT_GUARD_STORE_TEXT", "1") != "0"
QUEUE = ReviewQueue(os.path.join(DATA_DIR, "queue.json"), store_text=STORE_TEXT)
STARTED = time.time()
LATENCIES: list = []


def _send(handler, status: int, payload) -> None:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output clean
        pass

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode() or "{}")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except Exception:
            return _send(self, 400, {"error": "invalid JSON"})
        if not isinstance(body, dict):
            return _send(self, 400, {"error": "JSON object expected"})
        if path == "/score":
            # Accept "output" as an alias for "answer", matching the raw
            # /score body field name. This is still the local shim, not
            # Latent's reader: see README and latent_guard/reader.py.
            answer = body.get("answer") or body.get("output")
            if not answer:
                return _send(self, 400, {"error": "answer is required"})
            result = score_answer(body.get("prompt", ""), answer, body.get("context", ""))
            # Add sentence offsets, like the reader's detail block: the
            # local checker reports text and risk only, so offsets are
            # computed by walking the answer in order.
            offset = 0
            for sentence in result["sentences"]:
                start = answer.find(sentence["text"], offset)
                sentence["start"] = start if start >= 0 else None
                sentence["end"] = (start + len(sentence["text"])) if start >= 0 else None
                offset = (start + len(sentence["text"])) if start >= 0 else offset
            result["scored"] = True
            result["source"] = "local"
            LATENCIES.append(result["latency_ms"])
            entry = QUEUE.add(result, body.get("prompt", ""), answer)
            return _send(self, 200, {**result, "queue_status": entry["status"]})
        if path == "/review":
            try:
                entry = QUEUE.review(body["request_id"], body["decision"], body.get("note", ""))
            except (KeyError, ValueError, TypeError) as exc:
                return _send(self, 400, {"error": str(exc)})
            return _send(self, 200, entry)
        return _send(self, 404, {"error": "unknown route"})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return _send(self, 200, {"ok": True, "uptime_s": round(time.time() - STARTED, 1)})
        if path == "/queue":
            return _send(self, 200, QUEUE.pending())
        if path == "/metrics":
            metrics = QUEUE.metrics()
            metrics["mean_score_latency_ms"] = (
                round(sum(LATENCIES) / len(LATENCIES), 3) if LATENCIES else None
            )
            return _send(self, 200, metrics)
        if path == "/export":
            lines = "\n".join(json.dumps(e) for e in QUEUE.labeled_examples())
            return _send(self, 200, lines)
        return _send(self, 404, {"error": "unknown route"})


def main() -> None:
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"latent_guard listening on 127.0.0.1:{port} (data: {DATA_DIR})")
    server.serve_forever()


if __name__ == "__main__":
    main()
