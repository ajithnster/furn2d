// Node module hook: resolve openPlan3D's "$lib/…" alias (and extensionless .ts imports) to src/lib.
import { pathToFileURL } from 'node:url';
import { existsSync } from 'node:fs';
const root = process.env.OPENPLAN3D_DIR;
export async function resolve(spec, ctx, next) {
  if (spec.startsWith('$lib/')) spec = pathToFileURL(`${root}/src/lib/${spec.slice(5)}`).href;
  if ((spec.startsWith('file:') || spec.startsWith('.')) && !/\.[cm]?[jt]s$/.test(spec)) {
    const url = new URL(spec, ctx.parentURL);
    if (existsSync(url.pathname + '.ts')) spec = url.href + '.ts';
  }
  return next(spec, ctx);
}
