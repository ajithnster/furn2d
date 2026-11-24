"""Stage 2: ask an image-edit model to stage the product inside the empty room.

Two passes per product, so every colour of a product keeps the same geometry
(the showroom swaps layers in place, so the sofa must not jump between colours):

  base   = model(empty room + product photo)          -> sofa as photographed
  colour = model(base render [+ swatch photo])        -> same sofa, new upholstery
"""
import base64
import io
import os
from pathlib import Path

import requests
from PIL import Image

from .config import USER_AGENT, Room, Spot

PLACE_PROMPT = """\
Image 1 is a photograph of an empty living room. Image 2 is a catalogue photo of a sofa.

Edit image 1 so the room contains exactly the sofa from image 2, {hint}.
- Reproduce the sofa faithfully: same design, silhouette, arms, cushions, legs, stitching, \
material and colour as image 2. Do not add pillows, throws or any other objects.
- Give it a realistic size for a full-size sofa in this room and match the room's camera \
perspective and vanishing lines.
- Light it with the soft daylight coming from the windows and add natural soft contact \
shadows on the floor under and behind it.
- Everything else must stay exactly as in image 1: same framing, camera, walls, windows, \
trees, floorboards, colours and exposure. Do not crop, zoom or re-light the room.
Return the full room photograph."""

RECOLOUR_PROMPT = """\
Change only the upholstery of the sofa in image 1 to {fabric}.{swatch_note}
Keep the sofa's shape, size, position, legs, seams and cushion folds identical, keep realistic \
fabric texture and the same lighting and shadows. Every pixel outside the sofa's upholstery \
must stay unchanged. Return the full image."""


def load_image(src: str | Path) -> Image.Image:
    """Open a local image or URL. URLs are downloaded once into cache/downloads/ so re-runs
    don't hit the store again (repeated requests trigger its bot protection)."""
    s = str(src)
    if s.startswith("http"):
        import hashlib
        from .config import CACHE
        path = CACHE / "downloads" / (hashlib.sha1(s.encode()).hexdigest()[:16] + ".img")
        if not path.exists():
            r = requests.get(s, headers={"User-Agent": USER_AGENT}, timeout=60)
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
        return Image.open(path).convert("RGB")
    return Image.open(s).convert("RGB")


def downscale(img: Image.Image, max_side: int = 1600) -> Image.Image:
    img.thumbnail((max_side, max_side))
    return img


GEMINI_ASPECTS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


def nearest_aspect(size: tuple[int, int]) -> str:
    r = size[0] / size[1]
    def ratio(a: str) -> float:
        w, h = map(int, a.split(":"))
        return w / h
    return min(GEMINI_ASPECTS, key=lambda a: abs(ratio(a) - r))


class Generator:
    def edit(self, prompt: str, images: list[Image.Image], size: tuple[int, int]) -> Image.Image:
        raise NotImplementedError


class GeminiGenerator(Generator):
    """Google Gemini image editing ("Nano Banana"). Needs GEMINI_API_KEY."""

    def __init__(self, model: str):
        from google import genai
        self.genai = genai
        self.client = genai.Client()
        self.model = model

    def edit(self, prompt, images, size):
        from google.genai import types
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, *images],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=nearest_aspect(size)),
            ),
        )
        for part in resp.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB").resize(size, Image.LANCZOS)
        raise RuntimeError(f"model returned no image: {resp.text if hasattr(resp, 'text') else resp}")


class OpenAIGenerator(Generator):
    """OpenAI gpt-image edits. Needs OPENAI_API_KEY."""

    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def edit(self, prompt, images, size):
        files = []
        for i, im in enumerate(images):
            buf = io.BytesIO()
            im.save(buf, "PNG")
            files.append((f"image{i}.png", buf.getvalue(), "image/png"))
        resp = self.client.images.edit(
            model=self.model, image=files, prompt=prompt,
            size=("1536x1024" if size[0] > size[1] * 1.2 else "1024x1536" if size[1] > size[0] * 1.2 else "1024x1024"),
            input_fidelity="high", quality="high",
        )
        data = base64.b64decode(resp.data[0].b64_json)
        return Image.open(io.BytesIO(data)).convert("RGB").resize(size, Image.LANCZOS)


def make_generator(backend: str, model: str | None) -> Generator:
    if backend == "gemini":
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise SystemExit("Set GEMINI_API_KEY to use the gemini backend.")
        return GeminiGenerator(model or "gemini-2.5-flash-image")
    if backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY to use the openai backend.")
        return OpenAIGenerator(model or "gpt-image-1.5")
    raise SystemExit(f"unknown backend {backend!r}")


def render_base(gen: Generator, room: Room, spot: Spot, product_photo: Image.Image) -> Image.Image:
    empty = Image.open(room.image).convert("RGB")
    return gen.edit(PLACE_PROMPT.format(hint=spot.hint), [empty, downscale(product_photo)], room.size)


def render_variant(gen: Generator, room: Room, base: Image.Image, fabric: str,
                   swatch: Image.Image | None) -> Image.Image:
    note = " Image 2 is a close-up photo of that fabric; match its colour, weave and pattern." if swatch else ""
    images = [base] + ([swatch] if swatch else [])
    return gen.edit(RECOLOUR_PROMPT.format(fabric=fabric, swatch_note=note), images, room.size)
