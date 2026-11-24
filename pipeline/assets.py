"""Background-free, recolourable sofa assets.

For each product this writes, on one shared canvas size:

  original.webp   sofa as photographed, transparent background, soft floor shadow
  neutral.webp    same sofa with the upholstery turned neutral grey (AI edit), same format
  shading.webp    grey = upholstery lighting (folds, seams, highlights), alpha = upholstery mask
  fabric-<id>.webp  optional AI-recoloured named fabrics (patterns, textures)
  asset.json      sizes, files, and where the floor contact line is

Any background: every file is RGBA with nothing but the sofa and a black,
semi-transparent shadow, so it can be dropped on any room photo or colour.

Any colour: draw neutral.webp, then over it the upholstery pixels computed as
colour × shading (see recolor.js). Legs and frames stay untouched.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .config import ROOT

MODEL = ROOT / "models/birefnet_lite.onnx"
NEUTRAL_GREY = "#9A9A9A"  # mid grey: contrasts with white photo backgrounds, room for highlights

NEUTRAL_PROMPT = """\
Recolour only the upholstery (all fabric or leather: seat, cushions, back, arms, skirt, pillows) of the \
{thing} in this photo to a plain, untextured, neutral mid grey ({grey}), like an unpainted \
clay model. Keep every fold, seam, crease, highlight and shadow exactly as they are. Do not change \
the legs, wooden or metal parts, the {thing}'s shape, the camera angle, the framing or the plain \
background. Return the full image at the same composition."""

FABRIC_PROMPT = """\
Change only the {part} of the {thing} in image 1 to {fabric}.{swatch_note} Keep the {thing}'s shape, \
legs, seams, lighting, camera angle, framing and the plain background exactly as they are. \
Return the full image at the same composition."""

# What each category is called in prompts, which part a named finish changes, and whether
# it gets the any-colour (neutral + shading) treatment.
CATEGORIES = {
    "sofa": {"thing": "sofa", "part": "upholstery (cushions, back, arms, pillows)", "recolour": True},
    "chair": {"thing": "chair", "part": "upholstery (seat, back, arms)", "recolour": True},
    "coffee-table": {"thing": "coffee table", "part": "finish of the table top and body", "recolour": False},
    "dining-table": {"thing": "dining table", "part": "finish of the table top and base", "recolour": False},
    "dining-chair": {"thing": "dining chair", "part": "upholstery (seat and back)", "recolour": True},
    # Bed photos include bedding, which the upholstery mask cannot tell apart from a grey
    # headboard, so beds use their photographed finish (+ AI-rendered named finishes).
    "bed": {"thing": "bed", "part": "upholstered headboard and bed frame (keep the bedding unchanged)", "recolour": False},
}


# ---------- background removal ----------

@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort
    if not MODEL.exists():
        raise SystemExit(
            f"Missing {MODEL}. Download it with:\n  curl -L -o {MODEL} "
            "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx")
    return ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])


def matte(img: Image.Image) -> np.ndarray:
    """Alpha matte (0..1) of the main object, via BiRefNet."""
    x = np.asarray(img.convert("RGB").resize((1024, 1024), Image.BILINEAR), np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    logits = _session().run(None, {"input_image": x})[0][0, 0]
    prob = 1 / (1 + np.exp(-logits))
    a = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR), np.float32) / 255
    # Keep only the main object and remove faint haze (e.g. the photo's own floor shadow).
    a[a < 0.06] = 0
    lab, n = ndi.label(a > 0.5)
    if n > 1:
        sizes = ndi.sum(np.ones_like(a), lab, range(1, n + 1))
        keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 0.01 * sizes.max()])
        a *= ndi.binary_dilation(keep, iterations=6)
    return a


def decontaminate(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remove the old background colour from soft edges (no white halo on dark backgrounds)."""
    border = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3),
                             rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)])
    bg = np.median(border, 0)
    a = alpha[..., None]
    fg = (rgb - (1 - a) * bg) / np.maximum(a, 0.05)
    # Use the unmixed colour only on edges; fully opaque pixels keep their own colour.
    w = np.clip((1 - alpha) * 4, 0, 1)[..., None]
    return np.clip(rgb * (1 - w) + fg * w, 0, 1)


# ---------- shadow ----------

def _lower_hull(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """y of the lowest convex outline under each column (image y grows downward)."""
    pts = sorted(zip(xs.tolist(), ys.tolist()))
    hull = []
    for p in pts:  # monotone chain, keeping the hull that bounds the points from below
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) >= 0:
            hull.pop()
        hull.append(p)
    hx, hy = zip(*hull)
    return np.interp(np.arange(xs.min(), xs.max() + 1), hx, hy)


def floor_shadow(alpha: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Soft floor shadow (alpha of a black layer) that follows the line the legs stand on.

    The line through the lowest points of the silhouette (lower convex hull) is where the
    sofa meets the floor, even in a three-quarter view where front and back legs differ in
    height. Shade a band just above that line (mostly hidden under the sofa, visible between
    the legs) plus a wider, fainter ambient shadow.
    """
    solid = alpha > 0.5
    cols = np.nonzero(solid.any(0))[0]
    rows = np.nonzero(solid.any(1))[0]
    obj_h = rows[-1] - rows[0]
    bottoms = np.array([np.nonzero(solid[:, x])[0][-1] for x in cols])
    line = _lower_hull(cols, bottoms)
    h, w = alpha.shape
    yy = np.arange(h)[:, None]
    base = np.zeros((h, w), np.float32)
    lo = np.round(line).astype(int)
    obj_w0 = cols[-1] - cols[0]
    band_h = min(0.12 * obj_h, 0.12 * obj_w0)   # short band: open bases (tables, chairs) show the floor
    band = (yy >= lo - band_h) & (yy <= lo + 0.015 * obj_h)
    base[:, cols[0]:cols[-1] + 1] = band
    obj_w = cols[-1] - cols[0]
    contact = ndi.gaussian_filter(base, (0.018 * obj_h + 1.5, 0.012 * obj_w + 1.5))
    ambient = ndi.gaussian_filter(base, (0.05 * obj_h, 0.03 * obj_w))
    shadow = np.clip(0.75 * contact + 0.45 * ambient, 0, 1) * strength
    # Fade out before the image border so the shadow never ends in a straight cut.
    fade = lambda n, k: np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / k, 0, 1)
    return shadow * fade(h, 0.04 * h)[:, None] * fade(w, 0.04 * w)[None, :]


def over(fg_rgb, fg_a, bg_rgb, bg_a):
    out_a = fg_a + bg_a * (1 - fg_a)
    out = (fg_rgb * fg_a[..., None] + bg_rgb * (bg_a * (1 - fg_a))[..., None]) / np.maximum(out_a, 1e-6)[..., None]
    return out, out_a


# ---------- cut-out on a shared canvas ----------

def cutout(img: Image.Image, canvas: tuple[int, int], margin: float = 0.06,
           alpha: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return RGB, alpha (sofa+shadow composited) on a canvas of size `canvas`, object fit to it.

    Objects are scaled so their width fills the canvas minus margins and bottom-aligned on
    the same floor line, so all looks of one product line up when swapped.
    """
    rgb = np.asarray(img.convert("RGB"), np.float32) / 255
    a = matte(img) if alpha is None else alpha
    rgb = decontaminate(rgb, a)
    ys, xs = np.nonzero(a > 0.02)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb, a = rgb[y0:y1, x0:x1], a[y0:y1, x0:x1]

    W, H = canvas
    floor_y = int(H * 0.86)
    # Fit the width; tall pieces (or edits framed a little differently) fit the height instead.
    scale = min((W * (1 - 2 * margin)) / rgb.shape[1], (floor_y - H * 0.02) / rgb.shape[0])
    nw, nh = round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)
    ox, oy = (W - nw) // 2, floor_y - nh
    to_img = lambda arr: Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    rgb_s = np.asarray(to_img(rgb).resize((nw, nh), Image.LANCZOS), np.float32) / 255
    a_s = np.asarray(to_img(a).resize((nw, nh), Image.LANCZOS), np.float32) / 255

    C_rgb = np.zeros((H, W, 3), np.float32)
    C_a = np.zeros((H, W), np.float32)
    C_rgb[oy:oy + nh, ox:ox + nw] = rgb_s
    C_a[oy:oy + nh, ox:ox + nw] = a_s
    return C_rgb, C_a, {"x": ox, "y": oy, "width": nw, "height": nh, "floor_y": floor_y}


def with_shadow(rgb, a):
    s = floor_shadow(a)
    return over(rgb, a, np.zeros_like(rgb), s)


def upholstery(neutral_rgb: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upholstery mask + normalised shading from the neutral-grey render.

    After the neutral edit the upholstery is one dominant, low-saturation tint (the model may
    make it a warm or cool grey rather than a pure one). Upholstery = pixels whose chromaticity
    is close to that dominant tint; wood/brass legs differ in hue, and black legs are very dark.
    """
    lum = neutral_rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    chroma = neutral_rgb / np.maximum(neutral_rgb.sum(-1, keepdims=True), 1e-6)  # r,g,b shares
    solid = (a > 0.5) & (lum > 0.08)
    ref = np.median(chroma[solid], 0) if solid.any() else np.full(3, 1 / 3)
    dist = np.abs(chroma - ref).max(-1)
    fabric_lum = float(np.median(lum[solid & (dist < 0.035)])) if (solid & (dist < 0.035)).any() else 0.6
    # Dark metal/wood legs are neutral too, so also require brightness near the fabric's.
    mask = solid & (dist < 0.035) & (lum > max(0.16, 0.45 * fabric_lum))
    mask = ndi.binary_opening(mask, iterations=2)
    lab, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 0.02 * sizes.max()])
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=3)) & (a > 0.5)
    soft = ndi.gaussian_filter(mask.astype(np.float32), 0.8) * a
    med = float(np.median(lum[mask])) if mask.any() else 0.75
    shading = np.clip(lum / med * 0.5, 0, 1)   # 0.5 == the fabric's average brightness
    return shading, soft


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    return float((a & b).sum() / max((a | b).sum(), 1))


def edit_matte(edit: Image.Image, photo_a: np.ndarray) -> tuple[Image.Image, np.ndarray, bool]:
    """Matte for an AI-edited copy of the product photo.

    The edit normally keeps the photo's framing, so the photo's own matte is reused: it is
    cleaner when the new colour is close to the background (e.g. grey on a grey floor shadow).
    If the model mirrored the sofa, flip it back. If the framing drifted, matte it separately.
    """
    pa = np.asarray(Image.fromarray((photo_a * 255).astype(np.uint8)).resize(edit.size, Image.BILINEAR),
                    np.float32) / 255
    a = matte(edit)
    flipped = _iou(a[:, ::-1], pa) > _iou(a, pa) + 0.05
    if flipped:
        edit, a = edit.transpose(Image.FLIP_LEFT_RIGHT), a[:, ::-1].copy()
    # The model renders at its own aspect ratio, so the sofa can be shifted/scaled by a few
    # pixels. Fit the photo's matte onto the edit's bounding box before reusing it.
    def bbox(m):
        ys, xs = np.nonzero(m > 0.5)
        return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    px0, py0, px1, py1 = bbox(photo_a)
    ex0, ey0, ex1, ey1 = bbox(a)
    crop = Image.fromarray((photo_a[py0:py1, px0:px1] * 255).astype(np.uint8))
    fitted = np.zeros_like(a)
    fitted[ey0:ey1, ex0:ex1] = np.asarray(crop.resize((ex1 - ex0, ey1 - ey0), Image.BILINEAR), np.float32) / 255
    if _iou(a, fitted) <= 0.9:
        return edit, a, flipped
    # Where the edit itself shows plain background (e.g. a leg drawn a few px elsewhere),
    # trust the edit's own matte; elsewhere take the union of both.
    rgb = np.asarray(edit.convert("RGB"), np.float32) / 255
    border = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3)])
    bgish = np.abs(rgb - np.median(border, 0)).max(-1) < 0.08
    combined = np.maximum(fitted, a)
    sel = bgish & (a < 0.5)
    combined[sel] = a[sel]
    return edit, combined, flipped


def to_rgba(rgb, a) -> Image.Image:
    arr = np.dstack([rgb, a])
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")


def save(img: Image.Image, path: Path, lossless: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "WEBP", lossless=lossless, quality=90, method=6)


def canvas_for(photo: Image.Image, width: int = 1400) -> tuple[int, int]:
    a = matte(photo)
    ys, xs = np.nonzero(a > 0.5)
    obj_ratio = (ys.max() - ys.min()) / (xs.max() - xs.min())
    # Room for the object at 88% width plus floor/shadow below it.
    return width, int(width * (0.88 * obj_ratio + 0.26))


def build_product(pid: str, photo: Image.Image, looks: dict[str, Image.Image],
                  neutral: Image.Image | None, out_dir: Path) -> dict:
    canvas = canvas_for(photo)
    meta = {"id": pid, "canvas": {"width": canvas[0], "height": canvas[1]}, "looks": {}}

    photo_a = matte(photo)
    rgb, a, box = cutout(photo, canvas, alpha=photo_a)
    save(to_rgba(*with_shadow(rgb, a)), out_dir / "original.webp")
    meta["looks"]["original"] = {"image": "original.webp", "box": box}

    for vid, img in looks.items():
        img, a_img, flipped = edit_matte(img, photo_a)
        rgb_v, a_v, box_v = cutout(img, canvas, alpha=a_img)
        save(to_rgba(*with_shadow(rgb_v, a_v)), out_dir / f"fabric-{vid}.webp")
        meta["looks"][vid] = {"image": f"fabric-{vid}.webp", "box": box_v, "was_mirrored": flipped}

    if neutral is not None:
        neutral, a_img, flipped = edit_matte(neutral, photo_a)
        rgb_n, a_n, box_n = cutout(neutral, canvas, alpha=a_img)
        shading, mask = upholstery(rgb_n, a_n)
        save(to_rgba(*with_shadow(rgb_n, a_n)), out_dir / "neutral.webp")
        save(to_rgba(np.repeat(shading[..., None], 3, -1), mask), out_dir / "shading.webp", lossless=True)
        meta["recolor"] = {"base": "neutral.webp", "shading": "shading.webp", "box": box_n, "was_mirrored": flipped,
                           "upholstery_frac": round(float((mask > 0.5).sum() / max((a_n > 0.5).sum(), 1)), 3)}

    (out_dir / "asset.json").write_text(json.dumps(meta, indent=2))
    return meta
