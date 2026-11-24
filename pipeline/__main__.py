"""Verona showroom image pipeline.

  python -m pipeline scrape                      # Shopify collection -> catalog.json
  python -m pipeline build --limit 3             # render + extract + manifest
  python -m pipeline build --only nash-sofa --force
  python -m pipeline build --no-generate         # re-extract from cached renders (free)
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from . import extract, generate, scrape
from .config import CACHE, LIVING_ROOM, OUT, ROOT

CATALOG = ROOT / "catalog.json"


def save_webp(img: Image.Image, path: Path, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "WEBP", quality=quality, method=6)


def render_with_qa(make, cache_path: Path, empty: Image.Image, roi, retries: int, force: bool,
                   allow_generate: bool):
    """Return (render, layer, problems). Reuses the cached render unless forced."""
    attempts = 0
    while True:
        if cache_path.exists() and not force:
            render = Image.open(cache_path).convert("RGB")
        elif allow_generate:
            attempts += 1
            print(f"    generating ({attempts}/{retries + 1}) ...", flush=True)
            render = make()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            render.save(cache_path)
        else:
            return None, None, ["no cached render (run without --no-generate)"]
        try:
            layer = extract.extract_layer(render, empty, roi)
            problems = extract.qa_problems(layer)
        except ValueError as e:
            layer, problems = None, [str(e)]
        if not problems or not allow_generate or attempts > retries:
            return render, layer, problems
        print(f"    QA failed: {'; '.join(problems)} -> retrying")
        force = True


def build(args) -> None:
    catalog = json.loads(CATALOG.read_text())["products"]
    room = LIVING_ROOM
    empty = Image.open(room.image).convert("RGB")
    gen = generate.make_generator(args.backend, args.model) if not args.no_generate else None

    items = [p for p in catalog if p.get("enabled", True)]
    if args.only:
        wanted = set(args.only.split(","))
        items = [p for p in catalog if p["id"] in wanted]
    if args.limit:
        items = items[: args.limit]

    manifest_path = OUT / "products.generated.json"
    manifest = {p["id"]: p for p in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    report = []

    for p in items:
        pid, spot = p["id"], room.spots[p["spot"]]
        print(f"[{pid}]")
        pub = OUT / "public/products" / pid
        cache = CACHE / "renders" / pid
        photo = generate.load_image(p["source_image"])
        save_webp(photo.resize((640, round(640 * photo.height / photo.width)), Image.LANCZOS), pub / "product-photo.webp")

        base_render, base_layer, problems = render_with_qa(
            lambda: generate.render_base(gen, room, spot, photo),
            cache / "original.png", empty, spot.roi, args.retries, args.force, gen is not None)
        if problems and base_layer is None:
            print(f"  skipped: {problems}")
            report.append({"product": pid, "variant": "original", "problems": problems})
            continue

        variants_out = []
        for v in p["variants"]:
            vid = v["id"]
            if vid == "original":
                layer, vproblems = base_layer, problems
            else:
                swatch = generate.load_image(v["swatch_image"]) if (gen and v.get("swatch_image")) else None
                _, layer, vproblems = render_with_qa(
                    lambda v=v, swatch=swatch: generate.render_variant(gen, room, base_render, v["prompt"], swatch),
                    cache / f"{vid}.png", empty, spot.roi, args.retries, args.force, gen is not None)
                if layer is None:
                    print(f"  {vid}: skipped: {vproblems}")
                    report.append({"product": pid, "variant": vid, "problems": vproblems})
                    continue

            stem = f"colour-{vid}"
            save_webp(layer.rgba, pub / f"{stem}.webp")
            save_webp(extract.make_thumb(layer), pub / f"{stem}-thumb.webp")
            save_webp(extract.make_swatch(layer), pub / f"{stem}-swatch.webp")
            (OUT / "previews").mkdir(parents=True, exist_ok=True)
            extract.composite(empty, layer).save(OUT / "previews" / f"{pid}--{vid}.jpg", quality=85)
            status = "ok" if not vproblems else "needs review"
            print(f"  {vid}: layer {layer.box}  qa={layer.qa}  {status}")
            report.append({"product": pid, "variant": vid, "box": layer.box, "qa": layer.qa, "problems": vproblems})
            variants_out.append({
                "id": vid, "group": "colours", "name": v["name"],
                "image": f"/products/{pid}/{stem}.webp",
                "thumb": f"/products/{pid}/{stem}-thumb.webp",
                "swatch": f"/products/{pid}/{stem}-swatch.webp",
                "layer": layer.box,
            })

        if variants_out:
            manifest[pid] = {
                "id": pid, "name": p["name"], "brand": "Verona Home", "category": p["category"],
                "room": "living", "spot": p["spot"], "dimensions": None, "url": p["url"],
                "photo": f"/products/{pid}/product-photo.webp",
                "defaultVariant": variants_out[0]["id"],
                "groups": [{"id": "colours", "label": "Colours"}],
                "variants": variants_out,
            }

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(list(manifest.values()), indent=2))
    (OUT / "qa-report.json").write_text(json.dumps(report, indent=2))
    flagged = [r for r in report if r["problems"]]
    print(f"\nmanifest: {manifest_path}\npreviews: {OUT / 'previews'}\n"
          f"{len(report) - len(flagged)} ok, {len(flagged)} flagged (see {OUT / 'qa-report.json'})")


def build_assets(args) -> None:
    from . import assets, demo
    catalog = json.loads(CATALOG.read_text())["products"]
    gen = generate.make_generator(args.backend, args.model) if not args.no_generate else None
    items = [p for p in catalog if p.get("enabled", True)]
    if args.only:
        wanted = set(args.only.split(","))
        items = [p for p in catalog if p["id"] in wanted]
    if args.limit:
        items = items[: args.limit]

    def cached(path: Path, make):
        if path.exists() and not args.force:
            return Image.open(path).convert("RGB")
        if gen is None:
            return None
        print(f"    generating {path.stem} ...", flush=True)
        img = make()
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return img

    done = []
    for p in items:
        pid = p["id"]
        print(f"[{pid}]")
        cache = CACHE / "assets" / pid
        photo = generate.load_image(p["source_image"])
        generate.downscale(photo, 1600)
        cat = assets.CATEGORIES[p["category"]]
        neutral = cached(cache / "neutral.png", lambda: gen.edit(
            assets.NEUTRAL_PROMPT.format(thing=cat["thing"], grey=assets.NEUTRAL_GREY), [photo], photo.size)
        ) if cat["recolour"] else None
        looks = {}
        if True:  # generate fabrics only with --fabrics; always reuse ones already generated
            for v in p["variants"]:
                if v["id"] == "original":
                    continue
                swatch = generate.load_image(v["swatch_image"]) if v.get("swatch_image") else None
                note = " Image 2 is a close-up of that fabric; match its colour, weave and pattern." if swatch else ""
                path = cache / f"fabric-{v['id']}.png"
                if not (args.fabrics or path.exists()):
                    continue
                img = cached(path, lambda v=v, swatch=swatch, note=note: gen.edit(
                    assets.FABRIC_PROMPT.format(fabric=v["prompt"], swatch_note=note, thing=cat["thing"], part=cat["part"]),
                    [photo] + ([swatch] if swatch else []), photo.size))
                if img is not None:
                    looks[v["id"]] = img
        meta = assets.build_product(pid, photo, looks, neutral, OUT / "assets" / pid)
        meta["name"], meta["url"], meta["category"] = p["name"], p["url"], p["category"]
        meta["fabrics"] = {v["id"]: v["name"] for v in p["variants"]}
        (OUT / "assets" / pid / "asset.json").write_text(json.dumps(meta, indent=2))
        rec = meta.get("recolor", {})
        print(f"  looks: {list(meta['looks'])}  recolourable: {bool(rec)}"
              + (f" (upholstery {rec['upholstery_frac']:.0%} of the piece)" if rec else ""))
        done.append(meta)

    page = demo.write(OUT / "assets")
    print(f"\nassets: {OUT / 'assets'}\ndemo:   {page}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scrape", help="build catalog.json from the Shopify collection")
    s.add_argument("--collection", default="sofa")
    s.add_argument("--category", default="sofa", choices=["sofa", "chair", "coffee-table", "bed", "dining-table", "dining-chair"])
    s.add_argument("--max-variants", type=int, default=4, help="incl. the as-photographed one")
    s.add_argument("--fresh", action="store_true", help="discard hand edits in catalog.json")

    b = sub.add_parser("build", help="generate renders, extract layers, write manifest")
    b.add_argument("--backend", choices=["gemini", "openai"], default="gemini")
    b.add_argument("--model", help="override model id")
    b.add_argument("--only", help="comma-separated product ids")
    b.add_argument("--limit", type=int)
    b.add_argument("--retries", type=int, default=2)
    b.add_argument("--force", action="store_true", help="ignore cached renders")
    b.add_argument("--no-generate", action="store_true", help="only re-extract cached renders")

    a = sub.add_parser("assets", help="background-free, recolourable sofa assets + demo page")
    a.add_argument("--backend", choices=["gemini", "openai"], default="gemini")
    a.add_argument("--model", help="override model id")
    a.add_argument("--only", help="comma-separated product ids")
    a.add_argument("--limit", type=int)
    a.add_argument("--fabrics", action="store_true", help="also AI-render the named fabrics in catalog.json")
    a.add_argument("--force", action="store_true", help="ignore cached AI images")
    a.add_argument("--no-generate", action="store_true", help="no AI calls; use cache only")

    r = sub.add_parser("room", help="wall/floor styling assets for the room (+ AI floor styles)")
    r.add_argument("--backend", choices=["gemini", "openai"], default="gemini")
    r.add_argument("--model", help="override model id")
    r.add_argument("--room", default="living-room", help="living-room | bedroom | dining")
    r.add_argument("--create", action="store_true", help="generate the empty room photo first (bedroom/dining)")
    r.add_argument("--styles", default="herringbone-oak,grey-tiles,concrete,carpet,walnut")
    r.add_argument("--force", action="store_true")
    r.add_argument("--no-generate", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "scrape":
        scrape.run(args.collection, CATALOG, args.max_variants, args.category, merge=not args.fresh)
    elif args.cmd == "assets":
        build_assets(args)
    elif args.cmd == "room":
        from . import demo, room
        from .config import ROOMS
        rm = ROOMS[args.room]
        gen = None if args.no_generate else generate.make_generator(args.backend, args.model)
        if args.create:
            room.create_room(rm, gen, LIVING_ROOM.image, args.force)
        meta = room.build(rm, gen, OUT / "rooms" / rm.id, [s for s in args.styles.split(",") if s], args.force)
        print(f"floors: {[f['id'] for f in meta['floors']]}")
        print(f"demo:   {demo.write(OUT / 'assets')}")
    else:
        build(args)


if __name__ == "__main__":
    sys.exit(main())
