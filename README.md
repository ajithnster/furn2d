# Verona showroom image pipeline

Turns product photos from `veronahomefurniture.com/collections/sofa` into the transparent,
room-positioned layers used by the virtual showroom (verona-virtual-showroom.vercel.app).

```
Shopify products.json ──► catalog.json ──► AI render: product staged in empty room
   (scrape)               (edit by hand)        │  + one recolour per fabric, from the same render
                                                ▼
                         diff vs empty-room.webp ─► feathered RGBA layer + {x,y,width,height}
                                                ▼
                         QA (room drift, size, placement) ─► auto-retry ─► products.generated.json
```

## Background-free, recolourable assets (`assets` command)

```bash
# catalogue: sofas plus chairs and coffee tables (each command adds to catalog.json)
.venv/bin/python -m pipeline scrape
.venv/bin/python -m pipeline scrape --collection chairs-1 --category chair
.venv/bin/python -m pipeline scrape --collection coffee-cocktail-tables --category coffee-table

.venv/bin/python -m pipeline assets --only raleigh-sofa,wallace-chair,dino-large-coffee-table
open out/assets/demo.html   # room composer: add pieces, move/resize/recolour, change background
```

Room styling (walls any colour, wood-tone tints, AI floor materials):

```bash
.venv/bin/python -m pipeline room            # 5 floor styles, one Gemini call each (≈ $0.20, cached)
.venv/bin/python -m pipeline room --styles herringbone-oak,walnut   # pick styles (see pipeline/room.py)
```

Bedroom and dining room (empty rooms generated in the living room's style, then masks + floors):

```bash
.venv/bin/python -m pipeline room --room bedroom --create
.venv/bin/python -m pipeline room --room dining --create
.venv/bin/python -m pipeline scrape --collection beds --category bed
.venv/bin/python -m pipeline scrape --collection dining-tables --category dining-table
.venv/bin/python -m pipeline scrape --collection dining-chairs-benches --category dining-chair
```

Room layouts and which furniture each room offers: `ROOM_SETUP` in `pipeline/demo.py`.
Note: always pass `--only` (or `--limit`) to `assets`; a bare run builds every enabled product.

Walls use `rooms/living-room/wall-mask.png`; the floor mask is detected automatically. AI floors
are pasted back through the floor mask, so walls, windows and furniture placement never shift.

Sofas and chairs get any-colour recolouring; coffee tables keep their photographed finish
(named finishes can be AI-rendered with `--fabrics`).

Per sofa, in `out/assets/<id>/` (all the same canvas size, transparent background, soft floor shadow):

| File | Use |
|---|---|
| `original.webp` | sofa as photographed; place on any background |
| `fabric-<id>.webp` | AI-rendered named fabrics (with `--fabrics`) |
| `neutral.webp` + `shading.webp` | any colour at runtime: `SofaRecolor.render(neutral, shading, "#3b5b7a")` from `pipeline/web/recolor.js` |
| `asset.json` | file list, box, floor line (86% of canvas height) |

Background removal uses BiRefNet-lite (MIT) locally: `models/birefnet_lite.onnx`
(download: `curl -L -o models/birefnet_lite.onnx https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx`).
One Gemini call per sofa for the neutral version, plus one per named fabric.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "GEMINI_API_KEY=..." > .env  # or OPENAI_API_KEY with --backend openai
```

## Run

```bash
.venv/bin/python -m pipeline scrape                        # 55 sofas -> catalog.json
.venv/bin/python -m pipeline build --only nash-sofa        # one product first
.venv/bin/python -m pipeline build --limit 5               # then a batch
.venv/bin/python -m pipeline build                         # everything enabled
.venv/bin/python -m pipeline build --no-generate           # re-cut cached renders, no API cost
```

Options for `build`: `--backend gemini|openai`, `--model <id>`, `--force` (ignore the cache),
`--retries N` (default 2).

## Edit `catalog.json` before a big run

- `enabled: false` skips a product (e.g. the comfort sleepers).
- `source_image`: the scraper prefers a 3/4-angle ("45"/"angle") shot. Change it if a
  front-on or open-sleeper photo was picked.
- `variants[].prompt`: the fabric description the model renders. Names like "Cozy Ink" mean
  little to a model, so write e.g. `"deep navy-blue soft chenille"`.
- `variants[].swatch_image`: path or URL of a real fabric swatch, for accurate colours.
- Re-running `scrape` keeps your edits (`--fresh` discards them).

## Outputs

| Path | What |
|---|---|
| `out/public/products/<id>/colour-<variant>.webp` | layer drawn over the room |
| `…-thumb.webp`, `…-swatch.webp`, `product-photo.webp` | picker thumbnail, fabric chip, catalogue photo |
| `out/products.generated.json` | product entries in the showroom's schema, including `layer` boxes |
| `out/previews/<id>--<variant>.jpg` | layer composited on the empty room. **Review these.** |
| `out/qa-report.json` | per-variant QA metrics, with failures listed |
| `cache/renders/` | raw model outputs (reused, so re-runs are free) |

Copy `out/public/products/*` into the showroom's `public/products/` and merge
`products.generated.json` into its product list.

## Tuning

- Placement: `pipeline/config.py`, sofa `Spot.hint` (prompt text) and `roi` (search box).
- Cut-out: `extract_layer()` thresholds. `object_thresh` controls what counts as the sofa,
  `shadow_thresh` how much shadow is kept, and `pad`/`feather` the soft floor margin.
- QA limits: `qa_problems()`. `outside_drift` is high when the model repainted the room.
