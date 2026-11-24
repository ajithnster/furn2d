"""Stage 1: read the Shopify collection and write an editable catalog.json."""
import json
import re
from pathlib import Path

import requests

from .config import CACHE, STORE, USER_AGENT

# Option names whose values are upholstery colours/fabrics (vs sizes, grades, wood finishes).
COLOUR_OPTIONS = ("cover", "fabric", "color", "colour")
NOT_A_COLOUR = re.compile(r"grade|\bsize\b|\d+\s*[\"″]|loveseat|sofa|chair|sleeper", re.I)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def fetch_collection(collection: str) -> list[dict]:
    products, page = [], 1
    while True:
        r = requests.get(
            f"{STORE}/collections/{collection}/products.json",
            params={"limit": 250, "page": page},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()["products"]
        if not batch:
            return products
        products += batch
        page += 1


def pick_source_image(p: dict) -> str | None:
    """Prefer a three-quarter ('45') shot: that is the angle the showroom uses."""
    srcs = [i["src"] for i in p["images"]]
    if not srcs:
        return None
    angled = [s for s in srcs if re.search(r"[-_ ]45[-_ .]|angle", s.split("/")[-1], re.I)]
    return (angled or srcs)[0]


def colour_option(p: dict) -> dict | None:
    for o in p["options"]:
        name = o["name"].lower()
        is_colour_name = any(k in name for k in COLOUR_OPTIONS)
        # Some products put fabrics under "Finish" (e.g. ryan-long-sofa: "Tasha Berry").
        looks_like_fabric = name == "finish" and all(len(v.split()) >= 2 for v in o["values"])
        if (is_colour_name or looks_like_fabric) and not any(NOT_A_COLOUR.search(v) for v in o["values"]):
            return o
    return None


# Products in a collection that are not really that category (e.g. ottomans among coffee tables).
OFF_CATEGORY = {"coffee-table": re.compile(r"ottoman|bench|end-table|stool", re.I),
                "chair": re.compile(r"ottoman|bench|stool|desk", re.I),
                "dining-chair": re.compile(r"bench|stool|table", re.I),
                "dining-table": re.compile(r"chair|bench|stool|buffet|sideboard|server|cabinet", re.I),
                "bed": re.compile(r"nightstand|night-stand|dresser|chest|mirror|bench|mattress|frame-only", re.I)}


def build_entry(p: dict, max_variants: int, category: str = "sofa") -> dict | None:
    src = pick_source_image(p)
    if not src:
        return None
    variants = [{"id": "original", "name": "As photographed", "prompt": None}]
    opt = colour_option(p)
    if opt:
        for v in opt["values"][: max_variants - 1]:
            clean = re.sub(r"\s*-\s*\d+$", "", v).strip()  # "Turbo Cream - 4" -> "Turbo Cream"
            variants.append({
                "id": slug(clean),
                "name": clean,
                # Edit freely: describe the fabric so the model can render it.
                "prompt": f"{clean} upholstery fabric (colour: {clean.split()[-1].lower()})",
                "swatch_image": None,  # optional local path or URL to a real swatch photo
            })
    return {
        "id": p["handle"],
        "name": p["title"],
        "url": f"{STORE}/products/{p['handle']}",
        "category": category,
        "spot": category,
        "source_image": src,
        "enabled": not (category in OFF_CATEGORY and OFF_CATEGORY[category].search(p["handle"])),
        "variants": variants,
    }


def run(collection: str, out_path: Path, max_variants: int, category: str = "sofa", merge: bool = True) -> None:
    """Add/refresh one collection in catalog.json; other collections' entries are kept."""
    products = fetch_collection(collection)
    (CACHE / "shopify").mkdir(parents=True, exist_ok=True)
    (CACHE / "shopify" / f"{collection}.json").write_text(json.dumps(products, indent=1))

    existing = json.loads(out_path.read_text())["products"] if out_path.exists() else []
    by_id = {e["id"]: e for e in existing}
    added = 0
    for p in products:
        if merge and p["handle"] in by_id:  # keep hand edits
            continue
        e = build_entry(p, max_variants, category)
        if e:
            by_id[p["handle"]] = e
            added += 1
    out_path.write_text(json.dumps({"products": list(by_id.values())}, indent=2))
    mine = [e for e in by_id.values() if e["category"] == category]
    print(f"{collection}: {len(products)} products, {added} new -> {len(mine)} {category} entries "
          f"({sum(e['enabled'] for e in mine)} enabled) in {out_path}")
