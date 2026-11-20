"""Static server for the demo with cross-origin isolation (needed for WASM threads).

    python web/serve.py [--port 8000] [--host 127.0.0.1]

Phone (Android Chrome): connect USB, `adb reverse tcp:8000 tcp:8000`, open
http://localhost:8000 on the phone (localhost is a secure context, so threads work).
Over plain LAN http the page is not cross-origin isolated and falls back to 1 thread.
"""
import argparse
import functools
import http.server
from pathlib import Path


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".mjs": "text/javascript", ".js": "text/javascript"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="127.0.0.1")
a = ap.parse_args()
root = Path(__file__).resolve().parent
srv = http.server.ThreadingHTTPServer((a.host, a.port), functools.partial(Handler, directory=str(root)))
print(f"serving {root} on http://{a.host}:{a.port} (COOP/COEP on)", flush=True)
srv.serve_forever()
