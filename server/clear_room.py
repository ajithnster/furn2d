"""Room-clearing API for the showroom page: photo of a furnished room in, empty room out.

    POST /clear-room   body: JPEG/PNG/WebP bytes (the page downsizes to ≤1536 px first)
                       200 → PNG bytes of the same room with the furniture removed
                       413 too large · 415 not an image · 429 limit reached · 502 model failed

Standard library only (like the other services on this server). Photos are processed in
memory and never written to disk; only per-IP request counts are kept, for rate limiting.

Environment:
    GEMINI_API_KEY   required
    PORT             default 8092 (bind 127.0.0.1; nginx proxies /showroom/api/ here)
    PER_IP_HOUR      default 5     requests per visitor per hour
    PER_DAY          default 100   requests per day for everyone combined (cost cap)
    STATE_FILE       default /var/lib/showroom-api/usage.json
    GEMINI_MODEL     default gemini-2.5-flash-image
"""
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_KEY = os.environ.get("GEMINI_API_KEY", "")
PORT = int(os.environ.get("PORT", "8092"))
PER_IP_HOUR = int(os.environ.get("PER_IP_HOUR", "5"))
PER_DAY = int(os.environ.get("PER_DAY", "100"))
STATE_FILE = os.environ.get("STATE_FILE", "/var/lib/showroom-api/usage.json")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-image")
MAX_BYTES = 5 * 1024 * 1024

PROMPT = """\
This is a photograph of a room. Remove all furniture and movable objects from it: sofas, \
chairs, tables, beds, cabinets, shelving units, rugs, lamps, plants, cushions, wall art, \
TVs, decorations and clutter. Fill in the floor and walls that were hidden behind them so \
they continue naturally, with the same materials, pattern, colour and lighting.
This is an edit of the given photo, not a new picture. Every pixel that was not covered by \
furniture must stay identical: same camera position, framing and perspective, same walls, \
ceiling, floor material, curtains, radiators, built-in fixtures, light switches and the view \
outside. Windows and doors must keep exactly the same shape, size, position and number of \
panes; where furniture hid part of a window, continue that window as it was. Do not add \
anything, do not restyle or re-light the room, and do not crop or zoom. Return the full \
photograph of the empty room."""

ASPECTS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]

lock = threading.Lock()


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"day": "", "day_count": 0, "ips": {}}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_FILE)


def take_slot(ip: str) -> str | None:
    """Reserve one request for this IP. Returns an error message if a limit is reached."""
    now = time.time()
    today = time.strftime("%Y-%m-%d", time.gmtime(now))
    with lock:
        st = load_state()
        if st.get("day") != today:
            st["day"], st["day_count"] = today, 0
        ips = {k: [t for t in v if now - t < 3600] for k, v in st.get("ips", {}).items()}
        ips = {k: v for k, v in ips.items() if v}
        if st["day_count"] >= PER_DAY:
            return "The daily limit for room clearing has been reached. Please try again tomorrow."
        if len(ips.get(ip, [])) >= PER_IP_HOUR:
            return f"You can clear up to {PER_IP_HOUR} rooms per hour. Please try again later."
        ips.setdefault(ip, []).append(now)
        st["ips"], st["day_count"] = ips, st["day_count"] + 1
        save_state(st)
    return None


def image_type(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def nearest_aspect(w: int, h: int) -> str:
    r = w / h
    return min(ASPECTS, key=lambda a: abs(int(a.split(":")[0]) / int(a.split(":")[1]) - r))


def clear_room(data: bytes, mime: str, w: int, h: int) -> bytes:
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}},
        ]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": nearest_aspect(w, h)} if w and h else {},
        },
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": API_KEY},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise RuntimeError("model returned no image: " + json.dumps(resp)[:300])


class Handler(BaseHTTPRequestHandler):
    server_version = "showroom-api"

    def send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def error(self, code: int, msg: str) -> None:
        self.send(code, json.dumps({"error": msg}).encode())

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            return self.send(200, b'{"ok":true}')
        self.error(404, "not found")

    def do_POST(self):
        if self.path.split("?")[0] != "/clear-room":
            return self.error(404, "not found")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return self.error(400, "No image received.")
        if n > MAX_BYTES:
            return self.error(413, "That image is too large.")
        data = self.rfile.read(n)
        mime = image_type(data)
        if not mime:
            return self.error(415, "Please upload a JPEG, PNG or WebP photo.")
        ip = self.headers.get("X-Real-IP") or self.client_address[0]
        msg = take_slot(ip)
        if msg:
            return self.error(429, msg)
        try:
            w = int(self.headers.get("X-Image-Width") or 0)
            h = int(self.headers.get("X-Image-Height") or 0)
            out = clear_room(data, mime, w, h)
        except urllib.error.HTTPError as e:
            self.log_error("gemini HTTP %s: %s", e.code, e.read()[:300])
            return self.error(502, "The image service is unavailable right now. Please try again.")
        except Exception as e:  # noqa: BLE001 - report any model failure as a 502
            self.log_error("clear-room failed: %s", e)
            return self.error(502, "Couldn't clear this room. Please try another photo.")
        self.send(200, out, image_type(out) or "image/png")

    def log_message(self, fmt, *args):  # no per-request access log (no client IPs in the journal)
        pass

    def log_error(self, fmt, *args):     # errors still go to the journal (stderr)
        import sys
        print("showroom-api: " + (fmt % args), file=sys.stderr, flush=True)


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("GEMINI_API_KEY is not set")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
