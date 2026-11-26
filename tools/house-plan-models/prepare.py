"""Cut each showroom piece out of out/assets/<id>/original.webp (tight crop, floor shadow removed)
for AI 3D generation. Writes <id>.png and <id>.rgba to OUT. See README.md in this folder.

    .venv/bin/python tools/house-plan-models/prepare.py /tmp/hp
"""
import glob, json, os, struct, sys
import numpy as np
from PIL import Image

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for meta in sorted(glob.glob('out/assets/*/asset.json')):
    d, a = os.path.dirname(meta), json.load(open(meta))
    im = np.array(Image.open(os.path.join(d, a['looks']['original']['image'])).convert('RGBA')).astype(np.float32)
    # The shadow is a black layer with alpha <= 0.6 and the object is opaque: keep the object,
    # fade its anti-aliased rim, drop the shadow.
    obj = np.clip((im[..., 3] - 175) / (250 - 175), 0, 1)
    ys, xs = np.where(obj > 0.5)
    crop = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
    crop[..., 3] = obj[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255
    crop[..., :3] *= crop[..., 3:4] > 0
    img = Image.fromarray(crop.astype(np.uint8), 'RGBA')
    img.thumbnail((1024, 1024), Image.LANCZOS)
    img.save(f"{out}/{a['id']}.png")
    open(f"{out}/{a['id']}.rgba", 'wb').write(struct.pack('<II', *img.size) + np.array(img).tobytes())
    print(a['id'], img.size)
