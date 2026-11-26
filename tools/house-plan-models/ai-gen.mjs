import { readFileSync, writeFileSync, existsSync } from 'node:fs';
const [sp, ...pids] = process.argv.slice(2);
const token = (readFileSync('.env', 'utf8').match(/^hf_token=(.+)$/m) || [])[1]?.trim();
if (!token) throw new Error('no hf_token in .env');
const { generateFurnitureGLB } = await import('./src/lib/services/aiFurniture.ts');
for (const pid of pids) {
  if (existsSync(`${sp}/ai/${pid}.glb`)) { console.log('skip', pid); continue; }
  const t0 = Date.now(); let last = '';
  try {
    const photo = new File([readFileSync(`${sp}/${pid}.png`)], `${pid}.png`, { type: 'image/png' });
    const glb = await generateFurnitureGLB(photo, token, pid, { onProgress: p => { const s = JSON.stringify(p); if (s !== last) { last = s; console.log(pid, ((Date.now() - t0) / 1000).toFixed(0) + 's', s); } } });
    writeFileSync(`${sp}/ai/${pid}.glb`, Buffer.from(await glb.arrayBuffer()));
    console.log('DONE', pid, (glb.size / 1024).toFixed(0) + ' KB', ((Date.now() - t0) / 1000).toFixed(0) + 's');
  } catch (e) { console.log('FAIL', pid, e.message); if (/quota/i.test(e.message)) break; }
}
console.log('ALL FINISHED'); process.exit(0);
