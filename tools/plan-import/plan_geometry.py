"""Floor-plan image -> wall geometry (pixels). Step 1 of plan import.

Clean architectural plans draw walls as thick solid black bands; furniture, door arcs and window
glazing are thin lines. So: threshold dark pixels, keep only thick strokes (morphological opening),
split them into horizontal and vertical bands, turn each band into a centreline segment with a
thickness, join segments at corners/T-junctions, and bridge gaps between collinear segments
(doors, windows, passages) so rooms close. Returns plain data; no AI involved.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


@dataclass
class Seg:
    horizontal: bool
    c: float          # centreline coordinate (y for horizontal, x for vertical)
    a: float          # start along the segment (x for horizontal, y for vertical), a < b
    b: float
    t: float          # thickness in px
    gaps: list = field(default_factory=list)   # (a, b) spans bridged across openings


WORK_SIDE = 1024   # every threshold below is in pixels at this size


def load_gray(path: str):
    """Grey image normalised so its longer side is WORK_SIDE px (thresholds are tuned for that)."""
    img = Image.open(path).convert('L')
    k = WORK_SIDE / max(img.size)
    if abs(k - 1) > 0.02:
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    return np.array(img).astype(np.float32)


def wall_mask(path: str, dark=90, min_thick=7):
    im = load_gray(path)
    dark_px = im < dark
    walls = ndi.binary_opening(dark_px, structure=np.ones((min_thick, min_thick)))
    return im, dark_px, walls


def bands(walls: np.ndarray, horizontal: bool, min_len=24):
    k = np.ones((1, min_len)) if horizontal else np.ones((min_len, 1))
    m = ndi.binary_opening(walls, structure=k)
    lab, n = ndi.label(m)
    segs = []
    for sl in ndi.find_objects(lab):
        ys, xs = sl
        if horizontal:
            t = ys.stop - ys.start
            segs.append(Seg(True, (ys.start + ys.stop - 1) / 2, xs.start, xs.stop - 1, t))
        else:
            t = xs.stop - xs.start
            segs.append(Seg(False, (xs.start + xs.stop - 1) / 2, ys.start, ys.stop - 1, t))
    # a band whose thickness exceeds its length is a corner block of the other orientation
    return [s for s in segs if (s.b - s.a) > s.t * 0.9]


def merge_collinear(segs, max_gap_px):
    """Join collinear segments; gaps up to max_gap_px are bridged and remembered as openings."""
    segs = sorted(segs, key=lambda s: (s.horizontal, s.a))
    out = []
    for s in segs:
        for o in out:
            if o.horizontal != s.horizontal or abs(o.c - s.c) > max(o.t, s.t) * 0.5 + 2:
                continue
            if s.a <= o.b + 2 and s.b >= o.a - 2:        # overlapping / touching
                pass
            else:                                        # an opening between them
                gap = (o.b, s.a) if s.a > o.b else (s.b, o.a)
                if gap[1] - gap[0] > max_gap_px:
                    continue
                o.gaps.append(gap)
            # keep the thicker piece's centreline; take the union of spans and openings
            if s.t > o.t: o.c, o.t = s.c, s.t
            o.a, o.b = min(o.a, s.a), max(o.b, s.b)
            o.gaps.extend(s.gaps)
            break
        else:
            out.append(Seg(s.horizontal, s.c, s.a, s.b, s.t, list(s.gaps)))
    return out


def snap_ends(segs, tol_px):
    """Move segment ends onto perpendicular centrelines (corners and T-junctions)."""
    for s in segs:
        for end in ('a', 'b'):
            v = getattr(s, end)
            best = None
            for o in segs:
                if o.horizontal == s.horizontal:
                    continue
                # o's centreline must be within reach of this end, and s's centreline within o's span
                d = abs(o.c - v)
                if d <= o.t / 2 + tol_px and o.a - o.t / 2 - tol_px <= s.c <= o.b + o.t / 2 + tol_px:
                    if best is None or d < best[0]:
                        best = (d, o)
            if best:
                o = best[1]
                setattr(s, end, o.c)
                # extend the perpendicular wall too if s lands just past its end (L-corner)
                if s.c < o.a: o.a = s.c
                if s.c > o.b: o.b = s.c
    return segs


def _contains(o: Seg, along: float, across: float, horizontal: bool, pad=1.0):
    """Is the plan point (along, across) - in s's frame - inside wall o's rectangle?"""
    x, y = (along, across) if horizontal else (across, along)
    h = o.t / 2 + pad
    if o.horizontal:
        return o.a - h <= x <= o.b + h and o.c - h <= y <= o.c + h
    return o.a - h <= y <= o.b + h and o.c - h <= x <= o.c + h


def _connected(s: Seg, end: str, segs, tol=3.0):
    v = getattr(s, end)
    for o in segs:
        if o is s: continue
        if _contains(o, v + (tol if end == 'b' else -tol), s.c, s.horizontal, pad=0):
            return o
    return None


def bridge_dangling(segs, max_gap_px):
    """Doorways usually sit between a wall END and another wall: march from each free end along
    its centreline; if a wall is hit within max_gap_px, extend to it and record the opening.
    If the hit is the end of a short stub (a door jamb), keep marching past it."""
    for _ in range(3):
        changed = False
        for s in segs:
            for end in ('a', 'b'):
                if _connected(s, end, segs):
                    continue
                step = 1 if end == 'b' else -1
                start = getattr(s, end)
                d = 3
                while d < max_gap_px:
                    p = start + step * d
                    hit = next((o for o in segs if o is not s and _contains(o, p, s.c, s.horizontal, pad=0)), None)
                    if hit is None:
                        d += 1; continue
                    new_end = hit.c if hit.horizontal != s.horizontal else (hit.a if end == 'b' else hit.b)
                    gap = (start, p) if end == 'b' else (p, start)
                    if gap[1] - gap[0] > 2: s.gaps.append((min(gap), max(gap)))
                    setattr(s, end, new_end)
                    changed = True
                    # Hit the END of a short perpendicular stub (a door jamb)? The wall line carries on
                    # past it to the next wall: keep marching from the far face of the jamb.
                    if hit.horizontal != s.horizontal and (abs(s.c - hit.a) <= hit.t or abs(s.c - hit.b) <= hit.t):
                        start = hit.c + step * (hit.t / 2 + 1); d = 1
                        rest = next((o for o in segs if o is not s and o is not hit and any(
                            _contains(o, start + step * k, s.c, s.horizontal, pad=0) for k in range(1, int(max_gap_px)))), None)
                        if rest is not None:
                            continue
                    break
        if not changed:
            break
    return segs


def geometry(path: str, max_gap_px=200, tol_px=8, door_gap_px=120, min_opening_px=15):
    im, dark_px, walls = wall_mask(path)
    segs = bands(walls, True) + bands(walls, False)
    segs = merge_collinear(segs, max_gap_px)
    segs = snap_ends(segs, tol_px)
    segs = merge_collinear(segs, 0)            # ends that now touch
    segs = bridge_dangling(segs, door_gap_px)
    segs = merge_collinear(segs, 0)
    segs = snap_ends(segs, tol_px)             # walls extended by bridging may now reach a corner
    segs = merge_collinear(segs, 0)
    # Slivers where a partition meets a wall are not openings: real doorways are much wider.
    for sg in segs:
        sg.gaps = sorted({(round(g0, 1), round(g1, 1)) for g0, g1 in sg.gaps if g1 - g0 >= max(min_opening_px, 2.5 * sg.t)})
    return im, dark_px, walls, segs


def raster(shape, segs, bridge=True):
    """Wall mask from segments (gaps optionally closed), for finding rooms."""
    m = np.zeros(shape, bool)
    for s in segs:
        h = max(2, int(round(s.t / 2)))
        a, b = int(round(s.a)) - h, int(round(s.b)) + h
        c0, c1 = int(round(s.c)) - h, int(round(s.c)) + h
        if s.horizontal: m[max(0, c0):c1 + 1, max(0, a):b + 1] = True
        else: m[max(0, a):b + 1, max(0, c0):c1 + 1] = True
        if not bridge:
            for g0, g1 in s.gaps:
                g0, g1 = int(round(g0)) + 1, int(round(g1)) - 1
                if s.horizontal: m[max(0, c0):c1 + 1, g0:g1 + 1] = False
                else: m[g0:g1 + 1, max(0, c0):c1 + 1] = False
    return m


def regions(shape, segs, min_area_px=600):
    """Enclosed free-space regions inside the bridged walls (outside excluded)."""
    m = raster(shape, segs, bridge=True)
    lab, n = ndi.label(~m)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if i in border or sl is None:
            continue
        area = int((lab[sl] == i).sum())
        if area < min_area_px:
            continue
        ys, xs = np.nonzero(lab == i)
        out.append({'label': i, 'area_px': area, 'cx': float(xs.mean()), 'cy': float(ys.mean()),
                    'x0': int(xs.min()), 'x1': int(xs.max()), 'y0': int(ys.min()), 'y1': int(ys.max())})
    return lab, out


def classify_openings(im: np.ndarray, segs, arc_ink=235, line_ink=190):
    """Read each bridged gap from the drawing itself (standard symbols):
    door = a thin quarter-circle arc of radius ~ gap width, centred on one jamb (gives hinge + swing side);
    double door = two half-width arcs centred on both jambs, same side;
    glazed (window / sliding door) = thin lines running along the wall inside the gap;
    passage = none of these. Returns {(seg_index, gap_index): {...}}."""
    H, W = im.shape
    # Thresholds follow the plan's own background (white paper ~255; textured floors darker).
    bg = float(np.percentile(im, 75))
    arc_ink, line_ink = min(arc_ink, bg - 20), min(line_ink, bg - 45)
    dark = im < line_ink                                   # glazing lines: mid grey
    near = ndi.binary_dilation(im < arc_ink, iterations=1)  # door arcs: faint 1 px lines
    def at(x, y):
        xi, yi = int(round(x)), int(round(y))
        return 0 <= xi < W and 0 <= yi < H and bool(near[yi, xi])
    def arc_score(cx, cy, r, horizontal, side, toward):
        # quarter circle from the wall direction (toward the other jamb) to the wall normal (side)
        hits = n = 0
        for k in range(4, 87, 3):
            t = np.radians(k)
            u, v = np.cos(t) * r * toward, np.sin(t) * r * side        # u along wall, v across
            x, y = (cx + u, cy + v) if horizontal else (cx + v, cy + u)
            n += 1; hits += at(x, y)
        return hits / n
    out = {}
    for si, s in enumerate(segs):
        for gi, (g0, g1) in enumerate(s.gaps):
            w = g1 - g0
            best = {'kind': 'passage', 'score': 0.0}
            arc = {'score': 0.0, 'hinge': 'a', 'side': 1}
            for hinge, cpos, toward in (('a', g0, 1), ('b', g1, -1)):
                for side in (1, -1):
                    cx, cy = (cpos, s.c) if s.horizontal else (s.c, cpos)
                    sc = max(arc_score(cx, cy, w * f, s.horizontal, side, toward) for f in (0.85, 0.92, 1.0, 1.08))
                    sc -= arc_score(cx, cy, w * 0.55, s.horizontal, side, toward)   # background, not an arc
                    if sc > arc['score']:
                        arc = {'score': sc, 'hinge': hinge, 'side': side}
                    if sc > best['score']:
                        best = {'kind': 'door', 'score': sc, 'hinge': hinge, 'side': side}
            for side in (1, -1):        # double door: two arcs of half width from both jambs
                ca = (g0, s.c) if s.horizontal else (s.c, g0)
                cb = (g1, s.c) if s.horizontal else (s.c, g1)
                sc = max((arc_score(*ca, w / 2 * f, s.horizontal, side, 1) + arc_score(*cb, w / 2 * f, s.horizontal, side, -1)) / 2
                         for f in (0.85, 0.92, 1.0, 1.08))
                sc -= (arc_score(*ca, w / 4, s.horizontal, side, 1) + arc_score(*cb, w / 4, s.horizontal, side, -1)) / 2
                if sc > best['score'] + 0.05:
                    best = {'kind': 'double_door', 'score': sc, 'side': side}
            # glazing: fraction of positions in the gap with >= 2 separate thin lines across the wall band
            glazed = 0
            for p in np.linspace(g0 + 2, g1 - 2, max(3, int(w // 3))):
                pts = [(int(round(s.c + o)), int(round(p))) if s.horizontal else (int(round(p)), int(round(s.c + o)))
                       for o in np.arange(-s.t / 2, s.t / 2 + 1)]
                band = [bool(dark[y, x]) for y, x in pts if 0 <= y < H and 0 <= x < W]
                runs = sum(1 for i in range(1, len(band)) if band[i] and not band[i - 1]) + (1 if band and band[0] else 0)
                glazed += runs >= 2
            glazed /= max(3, int(w // 3))
            if best['score'] < 0.55:
                best = {'kind': 'glazed' if glazed > 0.6 else 'passage', 'score': max(best['score'], glazed)}
            best['glazed'] = round(glazed, 2)
            best['arc'] = arc                     # best single-leaf arc, for swing even when weak
            best['confident'] = best['kind'] in ('door', 'double_door') and best['score'] >= 0.55 or best['kind'] == 'glazed'
            out[(si, gi)] = best
    return out
