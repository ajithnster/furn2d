/* Sofa recolouring for the showroom.
 *
 *   const canvas = SofaRecolor.render(neutralImg, shadingImg, "#3b5b7a", { material: "velvet" });
 *   ctx.drawImage(canvas, x, y, w, h);
 *
 * neutralImg  = neutral.webp  (sofa with grey upholstery, legs, shadow; transparent bg)
 * shadingImg  = shading.webp  (grey = lighting, 0.5 = average; alpha = upholstery mask)
 * Pixels outside the upholstery (legs, frame, shadow) are left untouched.
 * Images must be same-origin (or CORS-enabled) so getImageData works.
 *
 * Materials (SofaRecolor.MATERIALS) change how the same lighting reads: contrast of the folds,
 * how highlights lift (sheen, tinted for velvet), glossy specular for leather, and a tiling
 * procedural surface texture (weave, loops, pile, grain). No extra assets are needed.
 */
(function (root) {
  function hexToRgb(hex) {
    const n = parseInt(hex.replace("#", ""), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map(v => v / 255);
  }
  // Fabric colours are chosen in sRGB but light behaves linearly; work in linear light.
  const toLin = c => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  const toSrgb = c => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);

  // ---------- tileable procedural textures (values roughly in [-1, 1]) ----------
  const N = 128;                                   // tile size in texture pixels
  function hash(i, j, seed) {
    let h = (i * 374761393 + j * 668265263 + seed * 982451653) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  }
  // Value noise on a cx × cy lattice that wraps at the tile edge (cx, cy divide N).
  function noise(x, y, cx, cy, seed) {
    const fx = x * cx / N, fy = y * cy / N, ix = Math.floor(fx), iy = Math.floor(fy);
    const tx = fx - ix, ty = fy - iy, sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
    const g = (i, j) => hash(((i % cx) + cx) % cx, ((j % cy) + cy) % cy, seed) * 2 - 1;
    const a = g(ix, iy) + (g(ix + 1, iy) - g(ix, iy)) * sx;
    const b = g(ix, iy + 1) + (g(ix + 1, iy + 1) - g(ix, iy + 1)) * sx;
    return a + (b - a) * sy;
  }
  const TAU = Math.PI * 2;
  const TEXTURES = {
    // Plain weave: crossing threads, alternating over/under, with slubby horizontal yarns.
    linen: (x, y) => {
      const over = ((Math.floor(x / 4) + Math.floor(y / 4)) & 1) ? Math.cos(TAU * y / 4) : Math.cos(TAU * x / 4);
      return 0.45 * over + 0.4 * noise(x, y, 4, 64, 1) + 0.15 * noise(x, y, 64, 64, 2);
    },
    // Bouclé: dense small loops of yarn (bright rims, dark centres), no large blotches.
    boucle: (x, y) => {
      const a = 1 - 2 * Math.abs(noise(x, y, 32, 32, 3)), b = 1 - 2 * Math.abs(noise(x + 2, y + 2, 32, 32, 4));
      return 0.55 * a + 0.45 * b - 0.6;
    },
    // Chenille: soft piled rows with a fuzzy, uneven surface.
    chenille: (x, y) => 0.45 * Math.cos(TAU * y * 20 / N) + 0.4 * noise(x, y, 16, 32, 5) + 0.15 * noise(x, y, 64, 64, 6),
    // Velvet: very fine pile plus large soft "crushed" patches.
    velvet: (x, y) => 0.45 * noise(x, y, 4, 4, 7) + 0.55 * noise(x, y, 64, 64, 8),
    // Leather: pebbled grain (fine creases) with faint mottling.
    leather: (x, y) => 0.8 * (0.5 - Math.abs(noise(x, y, 32, 32, 9))) * 2 + 0.2 * noise(x, y, 8, 8, 10),
  };
  const tiles = {};
  function tile(name) {
    if (!TEXTURES[name]) return null;
    if (tiles[name]) return tiles[name];
    const t = new Float32Array(N * N), fn = TEXTURES[name];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) t[y * N + x] = fn(x, y);
    return (tiles[name] = t);
  }

  // contrast: exponent on the lighting (folds deeper/flatter); sheen: how far highlights lift;
  // tint: 0 = highlights go white, 1 = highlights stay the fabric's own hue (velvet);
  // gloss: specular strength (leather); texture/amp: surface pattern and its strength;
  // flatten: 0..1, how much of the photographed fabric's own texture to remove (keeping folds).
  const MATERIALS = {
    plain:    { name: "Smooth",   flatten: 0,    contrast: 1.0, sheen: 0.55, tint: 0,    gloss: 0,    texture: null,       amp: 0 },
    linen:    { name: "Linen",    flatten: 0.8,  contrast: 1.0, sheen: 0.30, tint: 0,    gloss: 0,    texture: "linen",    amp: 0.20 },
    chenille: { name: "Chenille", flatten: 0.7,  contrast: 1.2, sheen: 0.45, tint: 0.3,  gloss: 0,    texture: "chenille", amp: 0.16 },
    boucle:   { name: "Bouclé",   flatten: 0.6,  contrast: 0.8, sheen: 0.25, tint: 0,    gloss: 0,    texture: "boucle",   amp: 0.26 },
    velvet:   { name: "Velvet",   flatten: 0.9,  contrast: 1.7, sheen: 0.90, tint: 0.65, gloss: 0,    texture: "velvet",   amp: 0.07 },
    leather:  { name: "Leather",  flatten: 1.0,  contrast: 2.0, sheen: 0.30, tint: 0.2,  gloss: 0.5,  texture: "leather",  amp: 0.035 },
  };

  function pixels(img) {
    const c = document.createElement("canvas");
    c.width = img.naturalWidth || img.width;
    c.height = img.naturalHeight || img.height;
    const x = c.getContext("2d", { willReadFrequently: true });
    x.drawImage(img, 0, 0);
    return { canvas: c, ctx: x, data: x.getImageData(0, 0, c.width, c.height) };
  }

  // Lighting in linear light, plus a mask-weighted blur of it (broad folds without the fabric
  // texture that was photographed). Blur radius scales with the image width.
  function lighting(shade, W, H) {
    const s = shade.data.data, n = W * H, L = new Float32Array(n), M = new Float32Array(n);
    for (let p = 0; p < n; p++) { const m = s[p * 4 + 3] / 255; M[p] = m; L[p] = toLin(s[p * 4] / 255) * m; }
    const r = Math.max(2, Math.round(10 * W / 1400));
    const box = (a, horiz) => {               // one separable box-blur pass (running sum)
      const o = new Float32Array(n), len = horiz ? W : H, lines = horiz ? H : W;
      for (let l = 0; l < lines; l++) {
        const at = i => horiz ? l * W + i : i * W + l;
        let sum = 0;
        for (let i = -r; i <= r; i++) sum += a[at(Math.min(len - 1, Math.max(0, i)))];
        for (let i = 0; i < len; i++) {
          o[at(i)] = sum / (2 * r + 1);
          sum += a[at(Math.min(len - 1, i + r + 1))] - a[at(Math.max(0, i - r))];
        }
      }
      return o;
    };
    const blur = a => box(box(box(box(a, true), false), true), false);
    const bl = blur(L), bm = blur(M), smooth = new Float32Array(n);
    for (let p = 0; p < n; p++) smooth[p] = bm[p] > 1e-3 ? bl[p] / bm[p] : 0;
    return smooth;
  }

  const cache = new WeakMap();
  const smooth = (a, b, v) => { const t = Math.min(1, Math.max(0, (v - a) / (b - a))); return t * t * (3 - 2 * t); };

  function render(neutralImg, shadingImg, hex, opts = {}) {
    const mat = MATERIALS[opts.material] || MATERIALS.plain;
    const sheen = opts.sheen ?? mat.sheen;     // how much highlights lift toward white
    // Cache pixel data per (base, shading) pair: one base can be shaded by several maps
    // (e.g. a room photo recoloured once for walls and once for the floor).
    let byShade = cache.get(neutralImg);
    if (!byShade) { byShade = new WeakMap(); cache.set(neutralImg, byShade); }
    let src = byShade.get(shadingImg);
    if (!src) {
      src = { base: pixels(neutralImg), shade: pixels(shadingImg) };
      byShade.set(shadingImg, src);
    }
    const { base, shade } = src;
    const W = base.canvas.width, H = base.canvas.height;
    const out = document.createElement("canvas");
    out.width = W;
    out.height = H;
    const octx = out.getContext("2d");
    const img = octx.createImageData(W, H);
    const b = base.data.data, s = shade.data.data, o = img.data;
    const col = hexToRgb(hex).map(toLin);
    // Highlight target per channel: white, or toward a brighter version of the fabric's own hue.
    const hi = col.map(c => 1 + (Math.sqrt(c) - 1) * mat.tint);
    const tex = tile(mat.texture), amp = mat.amp, flat = mat.flatten || 0;
    const soft = flat ? (src.soft ??= lighting(shade, W, H)) : null;
    const k = W / 1400;                        // texture scale: tuned for 1400px-wide assets
    const mid = toLin(0.5);
    // Per-asset lighting statistics (sampled), so contrast and gloss adapt to how strongly the
    // piece was lit: a flat-lit sofa gets the full boost, a contrasty photo gets less.
    const sample = [];
    for (let p = 0; p < W * H; p += 7) {
      if (s[p * 4 + 3] < 128) continue;
      let L = toLin(s[p * 4] / 255); if (soft) L += (soft[p] - L) * flat;
      if (L > 0) sample.push(Math.log(L / mid));
    }
    const mean = sample.reduce((a, v) => a + v, 0) / (sample.length || 1);
    const sd = Math.sqrt(sample.reduce((a, v) => a + (v - mean) ** 2, 0) / (sample.length || 1)) || 0.1;
    const REF_SD = 0.12;                       // spread of a typical, evenly lit sofa photo
    const contrast = 1 + (mat.contrast - 1) * Math.min(1, REF_SD / sd);
    sample.sort((a, b) => a - b);
    const q = t => Math.exp(contrast * (sample[Math.floor(t * (sample.length - 1))] ?? 0));
    const specLo = q(0.80), specHi = Math.max(q(0.995), specLo * 1.05);
    for (let i = 0, p = 0; i < b.length; i += 4, p++) {
      const m = s[i + 3] / 255;
      o[i + 3] = b[i + 3];
      if (m === 0) { o[i] = b[i]; o[i + 1] = b[i + 1]; o[i + 2] = b[i + 2]; continue; }
      let L = toLin(s[i] / 255);
      if (soft) L += (soft[p] - L) * flat;
      let f = L / mid;                                   // light relative to the average
      if (contrast !== 1) f = Math.pow(f, contrast);
      // Specular follows the broad lighting only, so leather shines in smooth patches, not glitter.
      const spec = mat.gloss ? mat.gloss * smooth(specLo, specHi, f) : 0;
      let t = 0;
      if (tex) {
        const x = p % W, y = (p - x) / W;
        t = tex[(Math.floor(y / k) % N) * N + (Math.floor(x / k) % N)];
        f *= 1 + amp * t;
      }
      for (let c = 0; c < 3; c++) {
        const cc = col[c];
        let lit = f <= 1 ? cc * f : cc + (hi[c] - cc) * Math.min(1, (f - 1) * sheen);
        if (spec) lit += (1 - lit) * spec;
        const v = toSrgb(Math.min(1, Math.max(0, lit))) * 255;
        o[i + c] = b[i + c] * (1 - m) + v * m;
      }
    }
    octx.putImageData(img, 0, 0);
    return out;
  }

  root.SofaRecolor = { render, MATERIALS };
})(typeof window !== "undefined" ? window : globalThis);
