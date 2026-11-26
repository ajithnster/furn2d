// node nometal.mjs in.glb out.glb — materials become non-metallic (dark "metal" from AI models).
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
const [src, dst] = process.argv.slice(2);
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const doc = await io.read(src);
for (const m of doc.getRoot().listMaterials()) { console.log(m.getName(), 'metallic was', m.getMetallicFactor()); m.setMetallicFactor(0); }
await io.write(dst, doc);
