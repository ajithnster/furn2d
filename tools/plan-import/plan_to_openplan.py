"""Floor-plan image -> OpenPlan3D web project JSON.

    .venv/bin/python tools/plan-import/plan_to_openplan.py plan.webp out.openplan.json [--debug DIR]

1. plan_geometry: walls, openings (bridged gaps) and enclosed regions, in pixels (image processing).
2. Gemini (one call, image in, JSON out): names each numbered region, reads its written dimensions,
   and says what each numbered opening is (door / double door / sliding door / window / passage).
3. Scale = median of (region size in px / written size in cm) over rooms with dimensions.
4. Write walls, doors, windows (cm); rooms are named by run_rooms.mjs using the editor's own
   room detection, so each name lands on the right detected room.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from plan_geometry import classify_openings, geometry, regions  # noqa: E402

PROMPT = """You get two images of the same architectural floor plan.
Image 1 is the clean plan. Image 2 is the same plan with my marks:
- a RED NUMBER in a white circle, placed in empty space inside each enclosed region (an area bounded by the
  thick black walls; door gaps count as closed);
- a BLUE LETTER next to every gap I found in the walls (a doorway, a window, or an open passage).

Return JSON only, in this shape:
{"rooms": [{"region": 2, "labels": [{"text": "Living", "type": "livingRoom", "dimensions": "12'-2\\" x 12'-8\\""},
                                     {"text": "Kitchen", "type": "kitchen", "dimensions": "14'-4\\" x 16'-8\\""}]}],
 "openings": [{"id": "A", "kind": "door"}]}

rooms: one entry per red number. "labels" lists EVERY ROOM NAME label (Bedroom, Kitchen, Living...; never
furniture names such as Bed, Sofa, Table, Toilet, and never dimension lines) printed inside the SAME enclosed region as
the number (read the text on image 1) - often one, sometimes two (an open-plan living room and kitchen),
and an empty list [] if no label is printed there (do not guess a name). For each label: "text" in normal
case, "type" one of livingRoom, bedroom, kitchen, bathroom, diningRoom, laundryRoom, office, hallway,
closet, pantry, entryway, balcony, other, and "dimensions" = the size text printed under it exactly as
written, or "".

openings: one entry per blue letter. "kind" is one of:
 door (single swing door: a quarter-circle arc), double_door (two arcs or two leaves, e.g. closet doors),
 sliding_door (a glazed sliding door, e.g. onto a balcony), window (thin parallel lines in an outside wall,
 no arc), passage (an opening with no door at all)."""

TYPES_TEXTURE = {'bathroom': 'ceramic-white', 'kitchen': 'porcelain', 'laundryRoom': 'ceramic-gray',
                 'bedroom': 'carpet-beige', 'closet': 'light-oak'}


def parse_ft_in(text):
    """"11'-4\"" -> cm. Also accepts 11' 4", 11'4, 3.4 m, 340 cm."""
    text = text.replace('’', "'").replace('”', '"').replace('″', '"').replace('′', "'")
    m = re.match(r"\s*(\d+)\s*'\s*-?\s*(\d+(?:\.\d+)?)?\s*\"?\s*$", text)
    if m:
        return (int(m.group(1)) * 12 + float(m.group(2) or 0)) * 2.54
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(m|cm)\s*$", text)
    if m:
        return float(m.group(1)) * (100 if m.group(2) == 'm' else 1)
    return None


def annotate(im, segs, regs, out_path, lab=None, dark=None):
    from scipy import ndimage as ndi
    img = Image.fromarray(im.astype(np.uint8)).convert('RGB')
    d = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 26)
        small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 18)
    except OSError:
        big = small = ImageFont.load_default()
    if lab is not None:
        ink = ndi.binary_dilation(dark, iterations=3)
        for i, r in enumerate(regs, 1):
            free = (lab == r['label']) & ~ink
            dt = ndi.distance_transform_edt(free)
            y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
            r['nx'], r['ny'] = float(x), float(y)       # a point surely inside the region
            rad = max(9, min(15, int(dt[y, x]) - 1))
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill='white', outline=(220, 0, 0), width=2)
            d.text((x, y), str(i), fill=(220, 0, 0), font=small, anchor='mm')
    openings = []
    for si, s in enumerate(segs):
        for gi, (g0, g1) in enumerate(s.gaps):
            letter = chr(ord('A') + len(openings)) if len(openings) < 26 else 'Z' + str(len(openings) - 25)
            mid = (g0 + g1) / 2
            x, y = (mid, s.c) if s.horizontal else (s.c, mid)
            off = (0, -16) if s.horizontal else (16, 0)
            d.text((x + off[0], y + off[1]), letter, fill=(0, 90, 255), font=small, anchor='mm', stroke_width=3, stroke_fill='white')
            openings.append({'letter': letter, 'seg': si, 'gap': gi})
    img.save(out_path)
    return openings


def flatten_ai(ai):
    """One name per region: join several labels; a single label carries its dimensions."""
    for r in ai.get('rooms', []):
        ls = [l for l in r.get('labels', []) if l.get('text')]
        r['name'] = ' / '.join(l['text'].strip().capitalize() if l['text'].isupper() else l['text'].strip() for l in ls)
        r['type'] = ls[0].get('type', 'other') if ls else 'other'
        r['dimensions'] = ls[0].get('dimensions', '') if len(ls) == 1 else ''
    ai['rooms'] = [r for r in ai.get('rooms', []) if r['name']]
    return ai


def ask_gemini_rest(clean_path, marked_path, model='gemini-2.5-flash'):
    """Same request over plain HTTPS (standard library only), for the server."""
    import base64, os, urllib.request
    part = lambda p: {'inline_data': {'mime_type': 'image/png', 'data': base64.b64encode(Path(p).read_bytes()).decode()}}
    body = {'contents': [{'parts': [{'text': PROMPT}, part(clean_path), part(marked_path)]}],
            'generationConfig': {'responseMimeType': 'application/json', 'temperature': 0}}
    req = urllib.request.Request(
        f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'x-goog-api-key': os.environ['GEMINI_API_KEY']})
    with urllib.request.urlopen(req, timeout=90) as r:
        resp = json.load(r)
    text = ''.join(p.get('text', '') for c in resp.get('candidates', []) for p in c.get('content', {}).get('parts', []))
    return json.loads(text)


def ask_gemini(clean_path, marked_path):
    from google import genai
    from google.genai import types
    client = genai.Client(http_options=types.HttpOptions(api_version='v1beta'))   # JSON mode needs v1beta
    resp = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[PROMPT, Image.open(clean_path).convert('RGB'), Image.open(marked_path)],
        config=types.GenerateContentConfig(response_mime_type='application/json', temperature=0))
    return json.loads(resp.text)


def convert(path, out_json, debug_dir=None, ask=None):
    im, dark, walls, segs = geometry(path)
    lab, regs = regions(im.shape, segs)
    if len(segs) < 3 or not regs:
        raise SystemExit('No room walls found on this image; cannot set the scale.')
    debug = Path(debug_dir) if debug_dir else Path(out_json).parent
    marked = debug / 'plan-marked.png'
    clean = debug / 'plan-clean.png'
    Image.fromarray(im.astype(np.uint8)).save(clean)
    openings = annotate(im, segs, regs, marked, lab, dark)
    ai = flatten_ai((ask or ask_gemini)(clean, marked))


    # scale (px per cm) from rooms with written dimensions: compare sorted (short, long) sides
    ratios = []
    for r in ai.get('rooms', []):
        i = r.get('region', 0) - 1
        dims = [parse_ft_in(p) for p in re.split(r'\s*[x×]\s*', r.get('dimensions', '')) if p.strip()]
        if not (0 <= i < len(regs)) or len(dims) != 2 or None in dims or '/' in r.get('name', ''):
            continue
        g = regs[i]
        px = sorted([g['x1'] - g['x0'] + 1, g['y1'] - g['y0'] + 1])
        cm = sorted(dims)
        ratios += [px[0] / cm[0], px[1] / cm[1]]
    if ratios:
        scale = float(np.median(ratios))                    # px per cm
        kept = [r for r in ratios if abs(r / scale - 1) <= 0.12] or ratios
        scale = float(np.median(kept))
        spread = float(np.std(kept) / scale)
        scale_from = 'room dimensions'
    else:
        # No written sizes: assume the median doorway (a bridged gap that is not a window) is 90 cm wide.
        gaps = sorted(g1 - g0 for s in segs for g0, g1 in s.gaps if 25 < g1 - g0 < 160)
        if not gaps:
            raise SystemExit('No room dimensions or doorways found on the plan; cannot set the scale.')
        scale, spread, scale_from = float(np.median(gaps)) / 90.0, 0.0, 'door width (no dimensions on plan)'

    # Cross-check: a label whose written size does not fit its region (e.g. read into a tiny neighbour)
    # moves to the region that fits best, if that region fits well.
    def fit(r_dims, g):
        px = sorted([g['x1'] - g['x0'] + 1, g['y1'] - g['y0'] + 1]); cmv = sorted(r_dims)
        return max(abs(px[0] / scale / cmv[0] - 1), abs(px[1] / scale / cmv[1] - 1))
    by_region = {r['region']: r for r in ai.get('rooms', [])}
    for r in list(ai.get('rooms', [])):
        dims = [parse_ft_in(p) for p in re.split(r'\s*[x×]\s*', r.get('dimensions', '')) if p.strip()]
        if len(dims) != 2 or None in dims or not (1 <= r['region'] <= len(regs)):
            continue
        if fit(dims, regs[r['region'] - 1]) <= 0.3:
            continue
        best = min(range(1, len(regs) + 1), key=lambda k: fit(dims, regs[k - 1]))
        if fit(dims, regs[best - 1]) <= 0.2 and best != r['region']:
            other = by_region.get(best)
            if other is not None and other.get('dimensions'):
                continue                         # that region already has a sized label of its own
            if other is not None:
                other['region'] = r['region']; by_region[r['region']] = other
            else:
                by_region.pop(r['region'], None)
            r['region'] = best; by_region[best] = r
            r['moved'] = True

    x0 = min(min(s.a, s.c) if s.horizontal else min(s.c, s.a) for s in segs)
    xs = [v for s in segs for v in ((s.a, s.b) if s.horizontal else (s.c,))]
    ys = [v for s in segs for v in ((s.c,) if s.horizontal else (s.a, s.b))]
    ox, oy = min(xs), min(ys)
    cm = lambda v, o: round((v - o) / scale, 1)

    walls_out, doors_out, windows_out = [], [], []
    for si, s in enumerate(segs):
        wid = f'w{si + 1}'
        if s.horizontal:
            start, end = {'x': cm(s.a, ox), 'y': cm(s.c, oy)}, {'x': cm(s.b, ox), 'y': cm(s.c, oy)}
        else:
            start, end = {'x': cm(s.c, ox), 'y': cm(s.a, oy)}, {'x': cm(s.c, ox), 'y': cm(s.b, oy)}
        walls_out.append({'id': wid, 'start': start, 'end': end, 'thickness': round(min(35, max(8, s.t / scale)), 1),
                          'height': 270, 'color': '#444444'})
    kinds = {o['id']: o['kind'] for o in ai.get('openings', [])}
    drawn = classify_openings(im, segs)
    for o in openings:
        s = segs[o['seg']]; g0, g1 = s.gaps[o['gap']]
        sym = drawn[(o['seg'], o['gap'])]
        ai_kind = kinds.get(o['letter'], 'door')
        # The drawing's own symbol wins when it is clear; the AI decides the rest and window vs sliding door.
        if sym['confident'] and sym['kind'] != 'glazed':
            kind = sym['kind']
        elif sym['confident']:
            kind = 'sliding_door' if ai_kind == 'sliding_door' else 'window'
        else:
            kind = ai_kind
        o['kind'] = kind
        length = s.b - s.a
        pos = ((g0 + g1) / 2 - s.a) / length
        width = round((g1 - g0) / scale, 1)
        base = {'id': f'o{o["letter"]}', 'wallId': f'w{o["seg"] + 1}', 'position': round(pos, 4), 'width': width}
        if kind == 'window':
            windows_out.append({**base, 'height': 120, 'sillHeight': 90, 'type': 'standard'})
        elif kind == 'passage':
            doors_out.append({**base, 'height': 210, 'type': 'opening'})
        else:
            t = {'door': 'single', 'double_door': 'double', 'sliding_door': 'sliding'}.get(kind, 'single')
            # Swing from the drawn arc. Wall start = segment end 'a'. swingDirection 'left' = hinge at the
            # wall END jamb ('b'), 'right' = at the START jamb ('a'). flipSide False swings toward the
            # normal (-dy, dx): +y for a horizontal wall drawn left->right, -x for a vertical wall drawn top->bottom.
            arc = sym['arc']
            toward_plus = arc['side'] > 0                 # +y (horizontal) / +x (vertical) in image space
            flip = (not toward_plus) if s.horizontal else toward_plus
            doors_out.append({**base, 'height': 210, 'type': t, 'swingDirection': 'left' if arc['hinge'] == 'b' else 'right',
                              'flipSide': flip})

    named = {r.get('region') for r in ai.get('rooms', [])}
    for k, g in enumerate(regs, 1):
        if k not in named:
            area_m2 = g['area_px'] / scale ** 2 / 1e4
            ai.setdefault('rooms', []).append({'region': k, 'name': 'Storage' if area_m2 < 2.5 else 'Room', 'type': 'other', 'dimensions': ''})
    labels = []
    for r in ai.get('rooms', []):
        i = r.get('region', 0) - 1
        if 0 <= i < len(regs):
            g = regs[i]
            labels.append({'x': cm(g.get('nx', g['cx']), ox), 'y': cm(g.get('ny', g['cy']), oy), 'name': r.get('name', ''), 'type': r.get('type', 'other')})

    project = {
        'id': 'plan-import', 'name': Path(path).stem.replace('-', ' ').title() + ' (imported)',
        'activeFloorId': 'ground',
        'floors': [{'id': 'ground', 'name': 'Ground Floor', 'level': 0, 'elevation': 0,
                    'walls': walls_out, 'doors': doors_out, 'windows': windows_out, 'furniture': [],
                    'rooms': [], 'stairs': [], 'columns': [], 'guides': [], 'measurements': [],
                    'annotations': [], 'textAnnotations': [], 'groups': []}],
    }
    Path(out_json).write_text(json.dumps(project, indent=1))
    (debug / 'labels.json').write_text(json.dumps(labels))
    report = {'walls': len(walls_out), 'doors': len(doors_out), 'windows': len(windows_out),
              'regions': len(regs), 'px_per_cm': round(scale, 4), 'scale_spread': round(spread, 3), 'scale_from': scale_from, 'ai': ai,
              'openings_final': {o['letter']: o['kind'] for o in openings}}
    (debug / 'report.json').write_text(json.dumps(report, indent=1))
    return report


if __name__ == '__main__':
    args = sys.argv[1:]
    dbg = args[args.index('--debug') + 1] if '--debug' in args else None
    rep = convert(args[0], args[1], dbg)
    print(json.dumps({k: v for k, v in rep.items() if k != 'ai'}))
