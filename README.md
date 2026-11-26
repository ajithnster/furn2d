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

## Remove furniture from an uploaded room (live site)

On the showroom page, **Upload your room…** then **Remove furniture** sends the photo to
`server/clear_room.py` (Gemini image edit, ≈ $0.04 per photo) and returns the empty room.
It runs on medharank.com as `showroom-api.service` (127.0.0.1:8092), proxied by nginx at
`/showroom/api/`. The key is in `/etc/showroom-api.env`; limits (`PER_IP_HOUR`, `PER_DAY`)
are in the unit file. Photos are not stored. Logs: `journalctl -u showroom-api`.

Update it: copy `server/clear_room.py` to `/opt/showroom-api/` and `systemctl restart showroom-api`.

## House plan tab (live site)

The showroom page's **House plan** tab embeds the openPlan3D floor-plan editor (separate repo,
checked out in `openPlan3D/`, not part of this one). It runs on medharank.com as
`openplan3d.service` (Node, 127.0.0.1:8093), proxied by nginx at `/showroom/plan/`; its
sharing / assistant / MCP endpoints are blocked there and analytics are off. Plans are saved
in each visitor's browser.

Update it:

```bash
cd openPlan3D && BASE_PATH=/showroom/plan BUILD_DIR=build-showroom NODE_ENV=production npm run build
# copy build-showroom/ to /opt/openplan3d/build (plus package*.json; npm ci --omit=dev if deps changed)
# then: systemctl restart openplan3d
```

## Floor plan image → editable plan (House plan tab)

**Upload floor plan** (House plan home page and welcome screen) turns a plan image into an editable
OpenPlan3D project. Walls, door/window gaps, rooms, and door types and swings come from image
processing (`tools/plan-import/plan_geometry.py`); one Gemini call reads room labels and written sizes
(scale) and tells windows from sliding doors (`plan_to_openplan.py`). Rooms are named in the browser
with the editor's own detection (`openPlan3D/src/lib/services/planImageImport.ts`).

Server: `POST /showroom/api/plan-import` in `server/clear_room.py` runs `server/plan_import_worker.py`
in `/opt/showroom-api/venv` (numpy, scipy, pillow) with `plan_import/` = `tools/plan-import/*.py`;
uploads live only in a temp dir for the request. Same per-visitor and daily limits as the other AI
features. Works best on clean plans (dark walls, light background); without printed room sizes the
scale is estimated from door widths and the project description says so.

Try locally: `.venv/bin/python tools/plan-import/plan_to_openplan.py plan.png out.json --debug DIR`
(`DIR/plan-marked.png` shows what was detected), then `name_rooms.mjs` (see its header).

## Product IDs, prices and quotes

`.venv/bin/python -m pipeline products` writes `out/showroom/products.json` from the cached store data
(cache/shopify): Shopify product id + short code (VH-xxxxxx), options, every variant's id and price,
and the House plan id. Refresh prices with `pipeline scrape --collection …` then `pipeline products`,
and copy the file to `/var/www/showroom/products.json`.

- Room composer (`pipeline/web/demo.html`) and House plan (openPlan3D `shopQuote.ts`, enabled by
  `PUBLIC_QUOTE_URL` / `PUBLIC_PRODUCTS_URL`) show code and price and have **Get a quote**.
- Quote page `/showroom/quote/` (`pipeline/web/quote.html`): pick variants, quantities, contact details;
  **Confirm** posts to `POST /showroom/api/quote`, which re-prices everything from its own products.json,
  saves `/var/lib/showroom-api/quotes/<number>.json` and returns the quote; the page downloads a PDF.
- Store list `/showroom/quotes/` (`pipeline/web/quotes-admin.html`) and `GET /showroom/api/quotes`:
  nginx basic auth, user `verona`, hash in `/etc/nginx/showroom-quotes.htpasswd`. Change the password:
  `printf 'NEW' | openssl passwd -apr1 -stdin` → `verona:<hash>` in that file, then `systemctl reload nginx`.

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
