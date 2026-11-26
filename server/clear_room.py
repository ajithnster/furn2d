"""Image API for the showroom page (Gemini image editing, server-side key).

    POST /clear-room   body: JPEG/PNG/WebP bytes (the page downsizes to ≤1536 px first)
                       200 → PNG bytes of the same room with the furniture removed
    POST /render?style=…&lighting=…&mood=…
                       body: PNG/JPEG of a House plan 3D view
                       200 → photorealistic image of the same view. The prompt is built only from
                       the fixed option lists below (no free text), so this is not a general
                       image generator.
    POST /plan-import  body: JPEG/PNG/WebP image of a floor plan
                       200 → {"project": OpenPlan3D project JSON, "labels": [...], "scaleFrom": "..."}
                       (walls/doors/windows by image processing, labels and scale via one Gemini call;
                       runs plan_import_worker.py in its own venv, PLAN_IMPORT_PYTHON)
    Errors: 400 bad option · 413 too large · 415 not an image · 422 not a readable plan · 429 limit
    reached · 502 model failed. These image endpoints share the per-visitor and daily limits.

    POST /quote        JSON {items: [{productId, variantId, qty}], contact: {name, email, phone, notes}, source}
                       200 → the saved quote (number, lines with server-side prices, total). Prices and ids
                       are checked against PRODUCTS_FILE, never taken from the request. Own limit:
                       QUOTES_PER_IP_HOUR per visitor. Saved as QUOTES_DIR/<number>.json.
    GET  /quotes       list of saved quotes (newest first)  } nginx protects these with a password
    GET  /quotes/<n>   one saved quote                       }

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
PLAN_PYTHON = os.environ.get("PLAN_IMPORT_PYTHON", "/opt/showroom-api/venv/bin/python")
PRODUCTS_FILE = os.environ.get("PRODUCTS_FILE", "/var/www/showroom/products.json")
QUOTES_DIR = os.environ.get("QUOTES_DIR", "/var/lib/showroom-api/quotes")
QUOTES_PER_IP_HOUR = int(os.environ.get("QUOTES_PER_IP_HOUR", "10"))
EMAIL_RE = __import__("re").compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")
PLAN_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan_import_worker.py")

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

# Photorealistic render of a House plan 3D view (same options as openPlan3D's AI Render).
RENDER_STYLES = ['photorealistic', 'architectural visualization', 'interior design magazine', 'minimalist',
                 'scandinavian', 'industrial', 'mid-century modern', 'luxury']
RENDER_LIGHTING = ['natural daylight', 'warm afternoon', 'golden hour', 'soft ambient', 'dramatic shadows',
                   'bright and airy', 'moody evening', 'studio lighting']
RENDER_MOODS = ['warm and inviting', 'clean and modern', 'cozy', 'elegant', 'rustic charm', 'sophisticated',
                'relaxed', 'vibrant']


def render_prompt(style: str, lighting: str, mood: str) -> str:
    return (f"Transform this interior 3D floor plan render into a {style} image. "
            f"Lighting: {lighting}. Mood: {mood}. "
            "Keep the exact same room geometry, furniture placement, and camera angle. "
            "Add realistic materials, textures, shadows, and reflections. "
            "Make walls, floors, and furniture look like real materials (wood, fabric, metal, etc). "
            "Keep each piece of furniture's exact design, shape, proportions and colour. "
            "Do NOT change the room layout, furniture positions, or camera perspective.")


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


def gemini_edit(prompt: str, data: bytes, mime: str, w: int, h: int) -> bytes:
    body = {
        "contents": [{"parts": [
            {"text": prompt},
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


quote_lock = threading.Lock()
quote_hits: dict = {}


def quote_slot(ip: str) -> bool:
    now = time.time()
    with quote_lock:
        hits = [t for t in quote_hits.get(ip, []) if now - t < 3600]
        if len(hits) >= QUOTES_PER_IP_HOUR:
            quote_hits[ip] = hits
            return False
        quote_hits[ip] = hits + [now]
        return True


def make_quote(body: dict) -> dict:
    """Validate a quote request against the product file and price it server-side."""
    import secrets
    with open(PRODUCTS_FILE) as f:
        catalog = json.load(f)
    by_id = {p["productId"]: p for p in catalog["products"]}
    items = body.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= 50:
        raise ValueError("Add between 1 and 50 items.")
    lines = []
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("Invalid item.")
        p = by_id.get(str(it.get("productId", "")))
        v = next((v for v in (p or {}).get("variants", []) if v["id"] == str(it.get("variantId", ""))), None)
        if p is None or v is None:
            raise ValueError("One of the items is no longer available. Please reload the page.")
        qty = it.get("qty")
        if not isinstance(qty, int) or not 1 <= qty <= 50:
            raise ValueError("Quantities must be between 1 and 50.")
        lines.append({"productId": p["productId"], "code": p["code"], "name": p["name"], "url": p["url"],
                      "variantId": v["id"], "variant": v["title"] if v["title"] != "Default Title" else "",
                      "sku": v.get("sku"), "unitPrice": v["price"], "qty": qty,
                      "lineTotal": round(v["price"] * qty, 2)})
    c = body.get("contact") or {}
    name = str(c.get("name", "")).strip()[:100]
    email = str(c.get("email", "")).strip()[:254]
    phone = str(c.get("phone", "")).strip()[:30]
    notes = str(c.get("notes", "")).strip()[:1000]
    if not name:
        raise ValueError("Please enter your name.")
    if not EMAIL_RE.match(email):
        raise ValueError("Please enter a valid email address.")
    now = time.gmtime()
    number = "Q-" + time.strftime("%y%m%d", now) + "-" + secrets.token_hex(2).upper()
    quote = {
        "number": number, "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", now),
        "validUntil": time.strftime("%Y-%m-%d", time.gmtime(time.time() + 30 * 86400)),
        "currency": catalog.get("currency", "USD"), "pricesAsOf": catalog.get("pricesAsOf"),
        "source": str(body.get("source", ""))[:20], "status": "new",
        "contact": {"name": name, "email": email, "phone": phone, "notes": notes},
        "lines": lines, "subtotal": round(sum(l["lineTotal"] for l in lines), 2),
    }
    os.makedirs(QUOTES_DIR, exist_ok=True)
    with open(os.path.join(QUOTES_DIR, number + ".json"), "x") as f:
        json.dump(quote, f, indent=1)
    return quote


def list_quotes() -> list:
    out = []
    if os.path.isdir(QUOTES_DIR):
        for n in os.listdir(QUOTES_DIR):
            if n.endswith(".json"):
                try:
                    with open(os.path.join(QUOTES_DIR, n)) as f:
                        out.append(json.load(f))
                except (OSError, ValueError):
                    pass
    return sorted(out, key=lambda q: q.get("createdAt", ""), reverse=True)


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
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            return self.send(200, b'{"ok":true}')
        # Quote list for the store. nginx requires a password on /showroom/api/quotes.
        if path == "/quotes":
            return self.send(200, json.dumps(list_quotes()).encode())
        if path.startswith("/quotes/"):
            n = path.rsplit("/", 1)[1]
            if __import__("re").fullmatch(r"Q-\d{6}-[0-9A-F]{4}", n) and os.path.exists(os.path.join(QUOTES_DIR, n + ".json")):
                with open(os.path.join(QUOTES_DIR, n + ".json"), "rb") as f:
                    return self.send(200, f.read())
            return self.error(404, "Quote not found.")
        self.error(404, "not found")

    def plan_import(self, data: bytes, mime: str):
        import subprocess, tempfile
        ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[mime]
        with tempfile.TemporaryDirectory(prefix="plan-") as work:   # deleted on exit, success or not
            src = os.path.join(work, "in" + ext)
            with open(src, "wb") as f:
                f.write(data)
            try:
                done = subprocess.run([PLAN_PYTHON, PLAN_WORKER, src, work], capture_output=True, timeout=120,
                                      env={**os.environ, "GEMINI_API_KEY": API_KEY})
            except subprocess.TimeoutExpired:
                return self.error(502, "Reading the plan took too long. Please try again.")
            if done.returncode != 0:
                err = done.stderr.decode(errors="replace")[-400:]
                self.log_error("plan-import failed: %s", err)
                if "cannot set the scale" in err or "No room" in err:
                    return self.error(422, "Couldn't read walls and rooms from this image. Use a clear plan with dark walls on a light background.")
                return self.error(502, "Couldn't convert this plan. Please try another image.")
            return self.send(200, done.stdout, "application/json")

    def do_POST(self):
        path, _, query = self.path.partition("?")
        if path == "/quote":
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= 64 * 1024:
                return self.error(413, "Quote request too large.")
            try:
                body = json.loads(self.rfile.read(n))
            except ValueError:
                return self.error(400, "Invalid quote request.")
            ip = self.headers.get("X-Real-IP") or self.client_address[0]
            if not quote_slot(ip):
                return self.error(429, f"You can request up to {QUOTES_PER_IP_HOUR} quotes per hour. Please try again later.")
            try:
                quote = make_quote(body if isinstance(body, dict) else {})
            except ValueError as e:
                return self.error(400, str(e))
            except OSError as e:
                self.log_error("quote save failed: %s", e)
                return self.error(502, "The quote could not be saved. Please try again.")
            return self.send(200, json.dumps(quote).encode())
        if path == "/plan-import":
            prompt = None
        elif path == "/clear-room":
            prompt = PROMPT
        elif path == "/render":
            from urllib.parse import parse_qs
            q = {k: v[0] for k, v in parse_qs(query).items()}
            style, lighting, mood = q.get("style", "photorealistic"), q.get("lighting", "natural daylight"), q.get("mood", "warm and inviting")
            if style not in RENDER_STYLES or lighting not in RENDER_LIGHTING or mood not in RENDER_MOODS:
                return self.error(400, "Unknown style, lighting or mood.")
            prompt = render_prompt(style, lighting, mood)
        else:
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
        if path == "/plan-import":
            return self.plan_import(data, mime)
        try:
            w = int(self.headers.get("X-Image-Width") or 0)
            h = int(self.headers.get("X-Image-Height") or 0)
            out = gemini_edit(prompt, data, mime, w, h)
        except urllib.error.HTTPError as e:
            self.log_error("gemini HTTP %s: %s", e.code, e.read()[:300])
            return self.error(502, "The image service is unavailable right now. Please try again.")
        except Exception as e:  # noqa: BLE001 - report any model failure as a 502
            self.log_error("clear-room failed: %s", e)
            return self.error(502, "Couldn't process this image. Please try again.")
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
