/* Sofa recolouring for the showroom.
 *
 *   const canvas = SofaRecolor.render(neutralImg, shadingImg, "#3b5b7a");
 *   ctx.drawImage(canvas, x, y, w, h);
 *
 * neutralImg  = neutral.webp  (sofa with grey upholstery, legs, shadow; transparent bg)
 * shadingImg  = shading.webp  (grey = lighting, 0.5 = average; alpha = upholstery mask)
 * Pixels outside the upholstery (legs, frame, shadow) are left untouched.
 * Images must be same-origin (or CORS-enabled) so getImageData works.
 */
(function (root) {
  function hexToRgb(hex) {
    const n = parseInt(hex.replace("#", ""), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map(v => v / 255);
  }
  // Fabric colours are chosen in sRGB but light behaves linearly; work in linear light.
  const toLin = c => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  const toSrgb = c => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);

  function pixels(img) {
    const c = document.createElement("canvas");
    c.width = img.naturalWidth || img.width;
    c.height = img.naturalHeight || img.height;
    const x = c.getContext("2d", { willReadFrequently: true });
    x.drawImage(img, 0, 0);
    return { canvas: c, ctx: x, data: x.getImageData(0, 0, c.width, c.height) };
  }

  const cache = new WeakMap();

  function render(neutralImg, shadingImg, hex, opts = {}) {
    const sheen = opts.sheen ?? 0.55;     // how much highlights lift toward white
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
    const out = document.createElement("canvas");
    out.width = base.canvas.width;
    out.height = base.canvas.height;
    const octx = out.getContext("2d");
    const img = octx.createImageData(out.width, out.height);
    const b = base.data.data, s = shade.data.data, o = img.data;
    const col = hexToRgb(hex).map(toLin);
    for (let i = 0; i < b.length; i += 4) {
      const m = s[i + 3] / 255;
      o[i + 3] = b[i + 3];
      if (m === 0) { o[i] = b[i]; o[i + 1] = b[i + 1]; o[i + 2] = b[i + 2]; continue; }
      const f = toLin(s[i] / 255) / toLin(0.5);        // light relative to the average
      for (let k = 0; k < 3; k++) {
        const c = col[k];
        const lit = f <= 1 ? c * f : c + (1 - c) * Math.min(1, (f - 1) * sheen) ;
        const v = toSrgb(Math.min(1, lit)) * 255;
        o[i + k] = b[i + k] * (1 - m) + v * m;
      }
    }
    octx.putImageData(img, 0, 0);
    return out;
  }

  root.SofaRecolor = { render };
})(typeof window !== "undefined" ? window : globalThis);
