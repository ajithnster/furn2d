# House plan 3D models

AI-generated (Microsoft TRELLIS.2 on Hugging Face, via openPlan3D's own `aiFurniture.ts`)
from the showroom product cut-outs. Live: Raleigh and Nash sofas, Wallace and Emma chairs, Kava, Carmet and
Menlo Park beds, Asterisk and Harrison tables, Linea and Cove dining chairs.

```bash
.venv/bin/python tools/house-plan-models/prepare.py /tmp/hp              # tight cut-outs, no shadow
cp tools/house-plan-models/ai-gen.mjs openPlan3D/.ai-gen.mjs
(cd openPlan3D && node .ai-gen.mjs /tmp/hp raleigh-sofa nash-sofa)      # -> /tmp/hp/ai/<id>.glb
#   needs hf_token=... in openPlan3D/.env; the free GPU quota allowed ~4 pieces per day
# shrink: weld -> simplify 0.35 -> quantize (the editor's loader has no Draco/Meshopt decoder)
npx @gltf-transform/cli@4.2.1 weld in.glb a.glb
npx @gltf-transform/cli@4.2.1 simplify a.glb b.glb --ratio 0.35 --error 0.0005
npx @gltf-transform/cli@4.2.1 quantize b.glb openPlan3D/static/models/verona_<name>.glb
```

Then add the catalog entry in `openPlan3D/src/lib/utils/furnitureCatalog.ts` (width/depth/height in cm
**in the model's own proportions**: the editor stretches the model to exactly these), the id in
`furnitureModelFiles.ts` and a Portuguese name in `i18n/furnitureNames.ts`; run `npm run catalog:manifest`,
rebuild and deploy (main README, "House plan tab").

Before shrinking, check each model from above (front = +z, back/headboard at -z, width along x) and fix:
- facing: `node rotate.mjs in.glb out.glb -90` (degrees about +Y; Kava/Carmet beds -90, Asterisk table +90);
- dark, "metal" surfaces (glass tops): `node nometal.mjs in.glb out.glb`;
- glass/transparent parts can come out as invented textures: generate from the cut-out on a white
  background, and recolour the base-colour texture only (never the metallic-roughness texture).
Both scripts need `npm i @gltf-transform/core@4.2.1 @gltf-transform/extensions@4.2.1`.

Still to generate: Dino coffee table.
