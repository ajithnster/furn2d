"""Stage 3: cut the staged product out of the full-room render.

The showroom draws each variant as a transparent layer over empty-room.webp at
{x, y, width, height}. The layer is the product plus its shadow plus a band of
the render's own floor, with a soft feathered edge, so seams never show.

We find it by diffing the render against the empty room:
  1. undo small global drift the model introduced (shift + exposure),
  2. strong differences inside the spot ROI = the object; weaker connected
     differences = its shadow,
  3. fill holes, pad, feather -> alpha; crop to the alpha's bounding box.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


@dataclass
class Layer:
    rgba: Image.Image
    x: int
    y: int
    core: np.ndarray          # full-room bool mask of the object itself
    render: np.ndarray        # aligned render, float RGB 0..1
    qa: dict

    @property
    def box(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.rgba.width, "height": self.rgba.height}


def _f(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _gray(a: np.ndarray) -> np.ndarray:
    return a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def estimate_shift(a: np.ndarray, b: np.ndarray, outside: np.ndarray, max_shift: int = 24) -> tuple[int, int]:
    """Integer translation that best aligns b onto a (phase correlation on edges outside the ROI)."""
    def edges(x):
        g = _gray(x)
        e = np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1))
        return (e - e[outside].mean()) * outside
    fa, fb = np.fft.fft2(edges(a)), np.fft.fft2(edges(b))
    r = fa * np.conj(fb)
    corr = np.fft.ifft2(r / (np.abs(r) + 1e-6)).real
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape
    dy, dx = (dy - h if dy > h // 2 else dy), (dx - w if dx > w // 2 else dx)
    if abs(dy) > max_shift or abs(dx) > max_shift:
        return 0, 0
    return int(dy), int(dx)


def match_exposure(render: np.ndarray, empty: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel linear fit render->empty using pixels where nothing was added."""
    out = render.copy()
    for c in range(3):
        x, y = render[..., c][mask], empty[..., c][mask]
        A = np.stack([x, np.ones_like(x)], 1)
        gain, off = np.linalg.lstsq(A, y, rcond=None)[0]
        out[..., c] = render[..., c] * gain + off
    return np.clip(out, 0, 1)


def extract_layer(
    render_img: Image.Image,
    empty_img: Image.Image,
    roi: tuple[int, int, int, int],
    object_thresh: float = 0.14,
    shadow_thresh: float = 0.03,
    shadow_reach: float = 0.08,     # max shadow distance, as a fraction of object height
    pad: int = 12,
    feather: float = 9.0,
) -> Layer:
    empty = _f(empty_img)
    render = _f(render_img.resize(empty_img.size, Image.LANCZOS))
    h, w = empty.shape[:2]

    in_roi = np.zeros((h, w), bool)
    x0, y0, x1, y1 = roi
    in_roi[y0:y1, x0:x1] = True
    outside = ~ndi.binary_dilation(in_roi, iterations=20)

    # 1. global alignment
    dy, dx = estimate_shift(empty, render, outside)
    if dy or dx:
        render = np.roll(render, (dy, dx), axis=(0, 1))
    diff0 = np.abs(render - empty).max(-1)
    calm = outside & (diff0 < 0.15)
    render = match_exposure(render, empty, calm)

    # 2. difference map; colour + luminance (shadows are mostly luminance)
    diff = ndi.gaussian_filter(np.abs(render - empty).max(-1), 1.5)
    diff[~in_roi] = 0

    strong = ndi.binary_opening(diff > object_thresh, iterations=2)
    lab, n = ndi.label(strong)
    if n == 0:
        raise ValueError("no object found in ROI: the model probably did not place the product")
    sizes = ndi.sum(strong, lab, range(1, n + 1))
    biggest = sizes.max()
    # Keep secondary blobs (legs, headrests) only when they are near the main object.
    near_main = ndi.binary_dilation(lab == int(np.argmax(sizes)) + 1, iterations=40)
    keep = [i + 1 for i, s in enumerate(sizes) if s >= 0.02 * biggest and (near_main & (lab == i + 1)).any()]
    core = np.isin(lab, keep)
    core = ndi.binary_closing(core, iterations=6)
    core = ndi.binary_fill_holes(core)

    # Shadow: floor that got darker, close to the object, below its top edge.
    # (Models re-texture the floor slightly, so "any difference" would spread everywhere.)
    darker = ndi.gaussian_filter(_gray(empty) - _gray(render), 3) > shadow_thresh
    rows = np.nonzero(core.any(1))[0]
    reach = shadow_reach * (rows[-1] - rows[0])
    near = ndi.distance_transform_edt(~core) < reach
    lab_w, _ = ndi.label(darker & near & in_roi)
    touching = np.unique(lab_w[ndi.binary_dilation(core, iterations=3)])
    shadow = np.isin(lab_w, touching[touching > 0])
    shadow[: rows[0]] = False

    # 3. pad + feather
    region = ndi.binary_fill_holes(core | shadow)
    region = ndi.binary_dilation(region, iterations=pad)
    alpha = ndi.gaussian_filter(region.astype(np.float32), feather)
    alpha = np.maximum(alpha, ndi.binary_dilation(core, iterations=2).astype(np.float32))
    alpha[alpha < 0.02] = 0

    ys, xs = np.nonzero(alpha)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgba = np.dstack([render, alpha])[by0:by1, bx0:bx1]
    layer_img = Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), "RGBA")

    # QA numbers: how much of the room the model repainted, object size, ROI contact.
    drift = float(np.abs(render - empty).max(-1)[outside].mean())
    qa = {
        "shift": [dy, dx],
        "outside_drift": round(drift, 4),          # mean diff outside ROI after alignment
        "object_frac": round(float(core.mean()), 4),
        "touches_roi_edge": bool(core[y0, x0:x1].any() or core[y1 - 1, x0:x1].any()
                                 or core[y0:y1, x0].any() or core[y0:y1, x1 - 1].any()),
    }
    return Layer(layer_img, int(bx0), int(by0), core, render, qa)


def qa_problems(layer: Layer, max_drift: float = 0.06, min_frac: float = 0.02, max_frac: float = 0.35) -> list[str]:
    q, out = layer.qa, []
    if q["outside_drift"] > max_drift:
        out.append(f"model repainted the room (drift {q['outside_drift']})")
    if not (min_frac <= q["object_frac"] <= max_frac):
        out.append(f"object size {q['object_frac']:.1%} of room is implausible")
    if q["touches_roi_edge"]:
        out.append("object touches the ROI edge (wrong placement or room change)")
    return out


def make_swatch(layer: Layer, size: int = 96) -> Image.Image:
    """Crop the flattest fully-upholstered patch of the object (a fabric close-up)."""
    k = size
    inside = ndi.uniform_filter(layer.core.astype(np.float32), k) > 0.999
    g = _gray(layer.render)
    var = ndi.uniform_filter(g * g, k) - ndi.uniform_filter(g, k) ** 2
    var[~inside] = np.inf
    if not np.isfinite(var).any():
        cy, cx = map(int, ndi.center_of_mass(layer.core))
    else:
        cy, cx = np.unravel_index(np.argmin(var), var.shape)
    h, w = g.shape
    y0 = int(np.clip(cy - k // 2, 0, h - k))
    x0 = int(np.clip(cx - k // 2, 0, w - k))
    patch = layer.render[y0:y0 + k, x0:x0 + k]
    return Image.fromarray((patch * 255).astype(np.uint8))


def make_thumb(layer: Layer, width: int = 520) -> Image.Image:
    im = layer.rgba.copy()
    im.thumbnail((width, width))
    return im


def composite(empty_img: Image.Image, layer: Layer) -> Image.Image:
    out = empty_img.convert("RGBA")
    out.alpha_composite(layer.rgba, (layer.x, layer.y))
    return out.convert("RGB")
