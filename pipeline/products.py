"""Showroom product data (IDs, options, variants, prices) from the cached Shopify listings.

    .venv/bin/python -m pipeline products        # -> out/showroom/products.json

One file is shared by the room composer, the House plan (openPlan3D) and the quote page; the
showroom API validates every quote against its own copy. Prices are a snapshot: run
`pipeline scrape --collection ...` again to refresh cache/shopify, then this command.
"""
import json
import time
from pathlib import Path

from .config import CACHE, OUT

# House plan (openPlan3D) catalog id for each showroom product that has a 3D model.
PLAN_IDS = {
    'raleigh-sofa': 'verona_raleigh_sofa', 'nash-sofa': 'verona_nash_sofa',
    'wallace-chair': 'verona_wallace_chair', 'emma-chair': 'verona_emma_chair',
    'kava-bed-by-bdi': 'verona_kava_bed', 'carmet-tufted-platform-bed': 'verona_carmet_bed',
    'menlo-park-bed-reimagined': 'verona_menlo_park_bed',
    'amisco-asterisk-dining-table-50549-91180': 'verona_asterisk_dining_table',
    'harrison-table': 'verona_harrison_table', 'linea-dining-chair': 'verona_linea_dining_chair',
    'cove-chair-5020': 'verona_cove_dining_chair',
}
STORE = 'https://veronahomefurniture.com'


def build(out_dir: Path = OUT / 'showroom') -> Path:
    handles = {p.parent.name: json.loads(p.read_text()) for p in sorted((OUT / 'assets').glob('*/asset.json'))}
    listings = {}
    for f in sorted((CACHE / 'shopify').glob('*.json')):
        data = json.loads(f.read_text())
        for p in (data.get('products', data) if isinstance(data, dict) else data):
            if isinstance(p, dict) and p.get('handle') in handles:
                listings.setdefault(p['handle'], p)
    products = []
    for handle, asset in handles.items():
        p = listings.get(handle)
        if not p:
            continue
        options = [{'name': o['name'], 'values': o.get('values', [])} for o in p.get('options', [])
                   if not (o['name'] == 'Title' and o.get('values') == ['Default Title'])]
        variants = [{
            'id': str(v['id']),
            'title': v.get('title', ''),
            'options': [v.get(f'option{i + 1}') for i in range(len(options))],
            'price': round(float(v['price']), 2),
            'sku': v.get('sku') or None,
            'available': v.get('available', True),
        } for v in p.get('variants', [])]
        prices = [v['price'] for v in variants] or [0]
        image = next((i.get('src') for i in p.get('images', []) if i.get('src')), None)
        products.append({
            'handle': handle,                       # composer id
            'productId': str(p['id']),              # Shopify product id (unique)
            'code': f"VH-{str(p['id'])[-6:]}",      # short readable reference shown with the id
            'name': asset.get('name') or p.get('title', handle),
            'category': asset.get('category', ''),
            'vendor': p.get('vendor', ''),
            'url': f"{STORE}/products/{handle}",
            'image': image,
            'planId': PLAN_IDS.get(handle),
            'options': options,
            'variants': variants,
            'priceFrom': min(prices), 'priceTo': max(prices),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'products.json'
    path.write_text(json.dumps({'currency': 'USD', 'store': STORE,
                                'pricesAsOf': time.strftime('%Y-%m-%d', time.localtime(max(
                                    f.stat().st_mtime for f in (CACHE / 'shopify').glob('*.json')))),
                                'products': products}, indent=1))
    return path


if __name__ == '__main__':
    print(build())
