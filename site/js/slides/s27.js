// Interpolation: encode cat A and cat B, walk the straight line between their codes and decode
// every step, for ae_128 (z) and vae_128 (mu). On top, the same walk in pixel space (a crossfade):
// two ghost cats stacked, which is why we interpolate codes, not pixels. Strips of 8 frames, and a
// scrubber with a big live frame per model at the current t.

import { encode, decode, drawCat, drawTiles, inSlot, loadShowcase, getShowcaseCat, loadJSON, loadSprite } from "../models.js";
import { catPicker, slider } from "../ui.js";

const MODELS = ["ae_128", "vae_128"];
const FRAMES = 8, TILE = 64, GAP = 4;

let section, scrub;
let picked = { a: 0, b: 5 };
// The bigger AE and VAE (training/train_v2.py): every pair of the first 8 cats, precomputed.
let v2 = null; // { pairs, sprite }
const codes = {}; // model -> { a: Float32Array, b: Float32Array }
let request = 0;

export async function init(el) {
  section = el;
  await loadShowcase();
  const meta = await loadJSON("data/interp_v2.json").catch(() => null);
  if (meta) {
    v2 = { pairs: meta.pairs, sprite: await loadSprite(`data/${meta.file}`, meta) };
    picked = { a: 1, b: 7 }; // the pair where pixels, AE and VAE differ most clearly
    el.querySelector(".interp-note").hidden = false;
  }
  for (const ab of ["a", "b"]) {
    await catPicker(el.querySelector(`.ab-picker[data-ab="${ab}"]`), {
      selected: picked[ab],
      onPick: (i) => {
        picked[ab] = i;
        refresh();
      },
    });
  }
  scrub = slider(el.querySelector(".s16-slider"), {
    label: "t",
    min: 0,
    max: 1,
    step: 0.01,
    value: 0.5,
    format: (v) => v.toFixed(2),
    ticks: ["A", "B"],
    onInput: () => showFrame(),
  });
  await refresh();
}

const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

/** The pixel crossfade strip: (1 - t) * A + t * B, pixel by pixel. */
function drawPixels(a, b) {
  const canvas = section.querySelector('.interp-row[data-model="pixels"] canvas');
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let f = 0; f < FRAMES; f++) {
    const t = f / (FRAMES - 1);
    const img = new ImageData(TILE, TILE);
    for (let i = 0; i < img.data.length; i++) img.data[i] = a.data[i] * (1 - t) + b.data[i] * t;
    ctx.putImageData(img, f * (TILE + GAP), 0);
  }
}

/** Precomputed frames for the picked pair: { ae: [8 ImageData], vae: [...] }, A to B. */
function v2Frames() {
  const { a, b } = picked;
  if (a === b) return null;
  const [lo, hi] = a < b ? [a, b] : [b, a];
  const k = v2.pairs.findIndex(([x, y]) => x === lo && y === hi);
  const row = (r) => {
    const f = Array.from({ length: FRAMES }, (_, i) => v2.sprite.get((2 * k + r) * FRAMES + i));
    return a < b ? f : f.reverse();
  };
  return { ae_128: row(0), vae_128: row(1) };
}

async function refresh() {
  const mine = ++request;
  const images = [getShowcaseCat(picked.a), getShowcaseCat(picked.b)];
  drawPixels(...images);
  if (v2) {
    const frames = v2Frames();
    for (const model of MODELS) {
      const canvas = section.querySelector(`.interp-row[data-model="${model}"] canvas`);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (let f = 0; f < FRAMES; f++) ctx.putImageData(frames ? frames[model][f] : images[0], f * (TILE + GAP), 0);
    }
    return showFrame();
  }
  for (const model of MODELS) {
    const row = section.querySelector(`.interp-row[data-model="${model}"]`);
    await inSlot(row.querySelector(".strip-wrap"), async () => {
      const out = await encode(model, images);
      const z = out.z ?? out.mu; // the VAE's mean, the AE's code
      const n = z.dims[1];
      codes[model] = { a: z.data.slice(0, n), b: z.data.slice(n, 2 * n) };
      // All 8 frames in one batch.
      const batch = new Float32Array(FRAMES * n);
      for (let f = 0; f < FRAMES; f++) batch.set(lerp(codes[model].a, codes[model].b, f / (FRAMES - 1)), f * n);
      const strip = await decode(model, batch);
      if (mine !== request) return;
      drawTiles(row.querySelector("canvas"), strip, { cols: FRAMES, gap: GAP });
    });
  }
  if (mine === request) await showFrame();
}

async function showFrame() {
  const t = scrub.value;
  // The marker sits over the frame position for t along each strip.
  const width = FRAMES * (TILE + GAP) - GAP;
  const centre = (TILE / 2 + t * (FRAMES - 1) * (TILE + GAP)) / width;
  for (const m of section.querySelectorAll(".strip-marker")) m.style.left = `${centre * 100}%`;
  if (v2) {
    const frames = v2Frames();
    const i = Math.round(t * (FRAMES - 1));
    for (const model of MODELS) {
      const fig = section.querySelector(`.s16-live figure[data-model="${model}"]`);
      drawCat(fig.querySelector("canvas"), frames ? frames[model][i] : getShowcaseCat(picked.a));
    }
    return;
  }
  for (const model of MODELS) {
    const fig = section.querySelector(`.s16-live figure[data-model="${model}"]`);
    await inSlot(fig, async () => {
      // Without codes the model is missing: decoding anyway lets inSlot show the training card.
      const z = codes[model] ? lerp(codes[model].a, codes[model].b, t) : new Float32Array(128);
      drawCat(fig.querySelector("canvas"), await decode(model, z));
    });
  }
}
