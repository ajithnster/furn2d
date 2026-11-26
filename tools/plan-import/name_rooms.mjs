// Step 2 of plan import: name rooms with openPlan3D's own room detection, so each saved room
// record has exactly the wall-id set the editor will detect.
//   OPENPLAN3D_DIR=... node --import ./tools/plan-import/register.mjs tools/plan-import/name_rooms.mjs plan.json labels.json
import { readFileSync, writeFileSync } from 'node:fs';
const root = process.env.OPENPLAN3D_DIR;
const { detectRooms, getRoomPolygon } = await import(`${root}/src/lib/utils/roomDetection.ts`);
const [planPath, labelsPath] = process.argv.slice(2);
const plan = JSON.parse(readFileSync(planPath, 'utf8'));
const labels = JSON.parse(readFileSync(labelsPath, 'utf8'));
const floor = plan.floors[0];
const inside = (p, poly) => {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i], b = poly[j];
    if ((a.y > p.y) !== (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x) c = !c;
  }
  return c;
};
const TEXTURE = { bathroom: 'ceramic-white', kitchen: 'porcelain', laundryRoom: 'ceramic-gray', bedroom: 'carpet-beige' };
const rooms = detectRooms(floor.walls);
floor.rooms = [];
let named = 0;
rooms.forEach((room, i) => {
  const poly = getRoomPolygon(room, floor.walls);
  const label = labels.find(l => inside(l, poly));
  if (label) named++;
  const type = label?.type ?? 'other';
  floor.rooms.push({
    id: `r${i + 1}`, name: label?.name ?? `Room ${i + 1}`, walls: room.walls, area: room.area, roomType: 'indoor',
    floorTexture: TEXTURE[type] ?? (/living|kitchen/i.test(label?.name ?? '') ? 'light-oak' : 'light-oak'),
    ...(type !== 'other' ? { details: { roomType: type } } : {}),
  });
});
writeFileSync(planPath, JSON.stringify(plan, null, 1));
console.log(JSON.stringify({ detected: rooms.length, named, rooms: floor.rooms.map(r => `${r.name} ${r.area}m²`) }));
