"""Write a self-contained demo page (images inlined, so it works from file://)."""
import base64
import json
from pathlib import Path

from .config import ROOMS

HERE = Path(__file__).parent

# Per room: name, furniture categories offered, and a starting layout. Positions are fractions
# of the stage: cx = centre x, floor = where the piece meets the floor, w = object width.
ROOM_SETUP = {
    "living-room": {"name": "Living room", "cats": ["sofa", "chair", "coffee-table"], "start": [
        {"cat": "sofa", "cx": .64, "floor": .87, "w": .50},
        {"cat": "coffee-table", "cx": .47, "floor": .985, "w": .34},
        {"cat": "chair", "cx": .19, "floor": .92, "w": .22},
        {"cat": "chair", "nth": 1, "cx": .87, "floor": .985, "w": .20, "flip": True}]},
    "bedroom": {"name": "Bedroom", "cats": ["bed", "chair"], "start": [
        {"cat": "bed", "cx": .36, "floor": .9, "w": .52},
        {"cat": "chair", "cx": .84, "floor": .93, "w": .2, "flip": True}]},
    "dining": {"name": "Dining room", "cats": ["dining-table", "dining-chair"], "start": [
        # Chair photos face the camera, so chairs go behind the table and at its ends (turned in).
        {"cat": "dining-table", "cx": .5, "floor": .9, "w": .46},
        {"cat": "dining-chair", "cx": .41, "floor": .79, "w": .115},
        {"cat": "dining-chair", "cx": .59, "floor": .79, "w": .115, "flip": True},
        {"cat": "dining-chair", "cx": .22, "floor": .87, "w": .13, "turn": 20},
        {"cat": "dining-chair", "cx": .78, "floor": .87, "w": .13, "flip": True, "turn": -20}]},
}


def _data_uri(path: Path) -> str:
    mime = {"webp": "image/webp", "png": "image/png", "jpg": "image/jpeg"}[path.suffix[1:]]
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _rooms(out_root: Path) -> list[dict]:
    rooms = []
    for rid, setup in ROOM_SETUP.items():
        room = ROOMS[rid]
        if not room.image.exists():
            continue
        d = out_root / "rooms" / rid
        entry = {"id": rid, **setup, "floors": [{"id": "original", "name": "Original", "tintable": False,
                                                  "src": _data_uri(room.image)}],
                 "wallShading": None, "floorShading": None}
        if (d / "room.json").exists():
            meta = json.loads((d / "room.json").read_text())
            entry["wallShading"] = _data_uri(d / meta["wallShading"])
            entry["floorShading"] = _data_uri(d / meta["floorShading"])
            entry["floors"] = [{"id": f["id"], "name": f["name"], "tintable": f["tintable"],
                                "src": _data_uri(d / f["image"])} for f in meta["floors"]]
        rooms.append(entry)
    return rooms


def write(assets_dir: Path) -> Path:
    shop_file = assets_dir.parent / 'showroom' / 'products.json'     # from `pipeline products`
    shop = {p['handle']: p for p in json.loads(shop_file.read_text())['products']} if shop_file.exists() else {}
    products = []
    for meta_path in sorted(assets_dir.glob("*/asset.json")):
        meta = json.loads(meta_path.read_text())
        d = meta_path.parent
        looks = [{"id": k, "name": meta.get("fabrics", {}).get(k, "As photographed" if k == "original" else k),
                  "src": _data_uri(d / v["image"])} for k, v in meta["looks"].items()]
        rec = meta.get("recolor")
        products.append({
            "id": meta["id"], "name": meta.get("name", meta["id"]), "category": meta.get("category", "sofa"),
            "looks": looks,
            "code": shop.get(meta["id"], {}).get("code"), "productId": shop.get(meta["id"], {}).get("productId"),
            "priceFrom": shop.get(meta["id"], {}).get("priceFrom"),
            "neutral": _data_uri(d / rec["base"]) if rec else None,
            "shading": _data_uri(d / rec["shading"]) if rec else None,
        })
    html = (HERE / "web/demo.html").read_text()
    html = html.replace("/*RECOLOR_JS*/", (HERE / "web/recolor.js").read_text())
    html = html.replace("\"__PRODUCTS__\"", json.dumps(products))
    html = html.replace("\"__ROOMS__\"", json.dumps(_rooms(assets_dir.parent)))
    out = assets_dir / "demo.html"
    out.write_text(html)
    return out
