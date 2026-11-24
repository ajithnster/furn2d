"""Room styling: recolourable walls, retintable wood floor, and AI floor-material variants.

Writes to out/rooms/<room>/:
  room-<floor>.webp     the empty room with each floor style (walls/windows pixel-identical)
  wall-shading.webp     grey = wall lighting (0.5 = average), alpha = wall mask
  floor-shading.webp    same for the original wood floor (for wood-tone tints)
  room.json             list of floor styles and files

Walls and wood tones are recoloured in the browser with the same maths as the furniture
(recolor.js). Floor materials that need a new pattern (tiles, herringbone, carpet) are
redrawn once by the image model, then only the floor region is pasted back onto the
original photo, so walls, windows and masks always line up.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .config import CACHE, ROOT, Room
from .extract import estimate_shift

FLOOR_STYLES = {
    "herringbone-oak": "natural light oak parquet laid in a herringbone pattern",
    "grey-tiles": "large-format matte light grey porcelain tiles (60x120 cm) with thin grout lines",
    "concrete": "smooth polished light grey concrete",
    "carpet": "soft wall-to-wall carpet in a warm off-white wool",
    "walnut": "dark walnut wide-plank hardwood with a satin finish",
}
FLOOR_NAMES = {"original": "Light oak", "herringbone-oak": "Herringbone oak",
               "grey-tiles": "Grey tiles", "concrete": "Polished concrete", "carpet": "Wool carpet",
               "walnut": "Dark walnut"}

FLOOR_PROMPT = """\
Replace only the floor of this empty room with {style}. Keep the floor's perspective, the \
camera, framing, walls, skirting boards, windows, ceiling and the view outside exactly the same. \
Keep the natural daylight and soft reflections of the windows on the floor where that material \
would show them. Do not add furniture or objects. Return the full image."""


def _f(path_or_img) -> np.ndarray:
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    return np.asarray(img.convert("RGB"), np.float32) / 255


def wall_mask(room: Room) -> np.ndarray:
    m = Image.open(room.image.parent / "wall-mask.png")
    return np.asarray(m.getchannel("A"), np.float32) / 255


def floor_mask(rgb: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Wood floor = warm, saturated region connected to the bottom edge, filled per column
    from its top edge down (so bright window reflections on the floor are included)."""
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    warm = (rgb[..., 0] >= rgb[..., 1]) & (rgb[..., 1] >= rgb[..., 2] * 0.98) & (sat > 0.12) & (mx > 0.25)
    warm = ndi.binary_opening(warm, iterations=2)
    lab, _ = ndi.label(warm)
    b = np.unique(lab[-1])
    floor = np.isin(lab, b[b > 0])
    h, w = floor.shape
    top = np.array([np.argmax(floor[:, x]) if floor[:, x].any() else h for x in range(w)])
    # Narrow smoothing follows sloped edges; wide smoothing bridges reflection gaps. Take the higher.
    top = np.minimum(ndi.median_filter(top, 41, mode="nearest"), ndi.median_filter(top, 161, mode="nearest"))
    m = (np.arange(h)[:, None] >= top[None, :]) & (wall < 0.5)
    return ndi.gaussian_filter(m.astype(np.float32), 1.2)


def shading(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    med = float(np.median(lum[mask > 0.5]))
    s = np.clip(lum / med * 0.5, 0, 1)
    arr = np.dstack([s, s, s, mask])
    return Image.fromarray((arr * 255 + 0.5).astype(np.uint8), "RGBA")


def paste_floor(base: np.ndarray, edit: np.ndarray, floor: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Align the model's redraw to the original (it can shift a few px) and paste only its floor."""
    dy, dx = estimate_shift(base, edit, wall > 0.5)
    if dy or dx:
        edit = np.roll(edit, (dy, dx), axis=(0, 1))
    soft = ndi.gaussian_filter(ndi.binary_erosion(floor > 0.5, iterations=1).astype(np.float32), 1.5)[..., None]
    return base * (1 - soft) + edit * soft


CREATE_PROMPT = """\
Image 1 is a photograph of an empty living room in a modern apartment. Create a photograph of an \
EMPTY {kind} in the same apartment: same photographic style, camera height, wide-angle lens, eye \
level and soft natural daylight, the same light oak plank floor, plain white walls, white skirting \
boards and slim black-framed windows with a view of trees. {layout} The room must be completely \
empty: no furniture, no rug, no curtains, no lamps, no art, no plants, no objects. Landscape 3:2."""

ROOM_LAYOUTS = {
    "bedroom": ("bedroom", "A plain, uninterrupted wall faces the camera where a bed would stand "
                "against it, with a window on one side wall and open floor in front."),
    "dining": ("dining room", "Large windows on one side, a plain wall facing the camera, and open "
               "floor in the centre where a dining table would stand."),
}

KEY_PROMPTS = {
    "wall": """Paint every wall surface in this room (all walls, pillars and wall sections between and \
above windows) a flat, uniform pure magenta (#FF00FF). The ceiling must stay white. Do not paint the \
floor, skirting boards, windows or window frames. Do not add, remove or move anything: no new doors, \
windows or outlets; the room's layout, camera and framing must stay exactly the same. Return the full image.""",
    "floor": """Paint the entire floor of this room a flat, uniform pure cyan (#00FFFF) with no shading, \
wood grain, reflections or shadows. Do not change the walls, skirting boards, windows, ceiling or \
framing. Return the full image.""",
}


def create_room(room: Room, gen, reference: Path, force: bool = False) -> None:
    if room.image.exists() and not force:
        return
    kind, layout = ROOM_LAYOUTS[room.id]
    print(f"    generating empty {kind} ...", flush=True)
    img = gen.edit(CREATE_PROMPT.format(kind=kind, layout=layout), [Image.open(reference).convert("RGB")], room.size)
    room.image.parent.mkdir(parents=True, exist_ok=True)
    img.save(room.image, quality=92)


def _key_sample(room: Room, gen, part: str, path: Path, force: bool = False, retries: int = 2) -> np.ndarray:
    """Mask of the walls: the model repaints them in a key colour and we pick that out.

    The model sometimes redraws the room (adds a door, moves a window); such outputs are
    rejected by comparing everything outside the key colour with the original photo.
    """
    base_img = Image.open(room.image).convert("RGB")
    base = _f(base_img)
    attempt = 0
    while True:
        if force or not path.exists():
            if gen is None:
                raise SystemExit(f"{room.id}: no {part} mask yet; run without --no-generate")
            attempt += 1
            print(f"    generating {part} mask ({attempt}/{retries + 1}) ...", flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            gen.edit(KEY_PROMPTS[part], [base_img], room.size).save(path)
        e = _f(Image.open(path).resize(room.size, Image.LANCZOS))
        dy, dx = estimate_shift(base, e, np.ones(base.shape[:2], bool))
        e = np.roll(e, (dy, dx), axis=(0, 1))
        r, g, b = e[..., 0], e[..., 1], e[..., 2]
        if part == "wall":   # magenta, possibly shaded
            m = (r - g > 0.2) & (b - g > 0.12)
        else:                # cyan
            m = (g - r > 0.2) & (b - r > 0.2)
        m = ndi.binary_closing(ndi.binary_opening(m, iterations=2), iterations=2)
        rest = ~ndi.binary_dilation(m, iterations=6)
        drift = float(np.abs(e - base).max(-1)[rest].mean()) if rest.any() else 1.0
        ok = m.mean() > 0.05 and drift < 0.08
        if ok or gen is None or attempt > retries:
            if not ok:
                print(f"    warning: {part} mask unreliable (coverage {m.mean():.0%}, drift {drift:.3f})")
            break
        print(f"    rejected: room changed (drift {drift:.3f}) or no key colour; retrying")
        force = True
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, [i + 1 for i, sz in enumerate(sizes) if sz >= 0.01 * sizes.max()])
    if part == "wall":
        # Snap to the real walls: bright, near-neutral pixels of the original photo.
        mx, mn = base.max(-1), base.min(-1)
        white = (mx > 0.55) & ((mx - mn) / np.maximum(mx, 1e-6) < 0.12)
        m = m & ndi.binary_dilation(white, iterations=1)
        m = ndi.binary_closing(m, iterations=2)
    else:
        m = ndi.binary_fill_holes(m)
    return ndi.gaussian_filter(m.astype(np.float32), 1.2)


def key_mask(room: Room, gen, part: str, force: bool = False, samples: int = 2) -> np.ndarray:
    """Union of several key-colour edits: each one tends to miss different small patches."""
    paths = [CACHE / "rooms" / room.id / (f"key-{part}.png" if i == 0 else f"key-{part}-{i + 1}.png")
             for i in range(samples)]
    return np.maximum.reduce([_key_sample(room, gen, part, p, force) for p in paths])


def room_masks(room: Room, gen, base: np.ndarray, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Wall + floor masks, saved next to the room photo so they are computed once."""
    d = room.image.parent
    wp, fp = d / "wall-mask.png", d / "floor-mask.png"
    if wp.exists() and not force:
        wall = np.asarray(Image.open(wp).getchannel("A"), np.float32) / 255
    else:
        wall = key_mask(room, gen, "wall", force)
        Image.fromarray(np.zeros((*wall.shape, 3), np.uint8)).convert("RGBA").save(wp)
        im = Image.open(wp); im.putalpha(Image.fromarray((wall * 255).astype(np.uint8))); im.save(wp)
    if fp.exists() and not force:
        floor = np.asarray(Image.open(fp).convert("L"), np.float32) / 255
    else:
        floor = floor_mask(base, wall)   # all rooms share the light oak floor
        floor = np.minimum(floor, 1 - wall)
        Image.fromarray((floor * 255).astype(np.uint8)).save(fp)
    return wall, floor


def build(room: Room, gen, out_dir: Path, styles: list[str], force: bool = False) -> dict:
    base = _f(room.image)
    wall, floor = room_masks(room, gen, base, force)
    out_dir.mkdir(parents=True, exist_ok=True)

    shading(base, wall).save(out_dir / "wall-shading.webp", lossless=True)
    shading(base, floor).save(out_dir / "floor-shading.webp", lossless=True)
    Image.fromarray((base * 255).astype(np.uint8)).save(out_dir / "room-original.webp", quality=90)
    floors = [{"id": "original", "name": FLOOR_NAMES["original"], "image": "room-original.webp", "tintable": True}]

    cache = CACHE / "rooms" / room.id
    for sid in styles:
        path = cache / f"floor-{sid}.png"
        if force or not path.exists():
            if gen is None:
                continue
            print(f"    generating floor {sid} ...", flush=True)
            img = gen.edit(FLOOR_PROMPT.format(style=FLOOR_STYLES[sid]), [Image.open(room.image).convert("RGB")], room.size)
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path)
        edit = _f(Image.open(path).resize(room.size, Image.LANCZOS))
        out = paste_floor(base, edit, floor, wall)
        Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(out_dir / f"room-{sid}.webp", quality=90)
        floors.append({"id": sid, "name": FLOOR_NAMES.get(sid, sid), "image": f"room-{sid}.webp", "tintable": False})

    meta = {"id": room.id, "size": list(room.size), "wallShading": "wall-shading.webp",
            "floorShading": "floor-shading.webp", "floors": floors}
    (out_dir / "room.json").write_text(json.dumps(meta, indent=2))
    return meta
