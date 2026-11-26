// node rotate.mjs in.glb out.glb degreesAboutY — wraps the scene in a node rotated about +Y.
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
const [src, dst, deg] = process.argv.slice(2);
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const doc = await io.read(src);
const scene = doc.getRoot().getDefaultScene() ?? doc.getRoot().listScenes()[0];
const t = (Number(deg) * Math.PI) / 180;
const wrap = doc.createNode('orient').setRotation([0, Math.sin(t / 2), 0, Math.cos(t / 2)]);
for (const child of scene.listChildren()) { scene.removeChild(child); wrap.addChild(child); }
scene.addChild(wrap);
await io.write(dst, doc);
console.log('rotated', src.split('/').pop(), deg);
