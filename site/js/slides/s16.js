// Interpolation: encode cat A and cat B, walk the straight line between their codes and decode
// every step, for ae_128 (z) and vae_128 (mu). Two strips of 8 frames, and a scrubber with a big
// live frame per model at the current t.

import { encode, decode, drawCat, fromTensor, inSlot, loadShowcase, getShowcaseCat, ort } from "../models.js";
import { catPicker, slider } from "../ui.js";

const MODELS = ["ae_128", "vae_128"];
const FRAMES = 8, TILE = 64, GAP = 4;

let section, scrub;
let picked = { a: 0, b: 5 };
const codes = {}; // model -> { a: Float32Array, b: Float32Array }
let request = 0;

export async function init(el) {
  section = el;
  await loadShowcase();
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

async function refresh() {
  const mine = ++request;
  const images = [getShowcaseCat(picked.a), getShowcaseCat(picked.b)];
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
      const strip = await decode(model, new ort.Tensor("float32", batch, [FRAMES, n]));
      if (mine !== request) return;
      const ctx = row.querySelector("canvas").getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      for (let f = 0; f < FRAMES; f++) ctx.putImageData(fromTensor(strip, f), f * (TILE + GAP), 0);
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
  for (const model of MODELS) {
    const fig = section.querySelector(`.s16-live figure[data-model="${model}"]`);
    await inSlot(fig, async () => {
      // Without codes the model is missing: decoding anyway lets inSlot show the training card.
      const z = codes[model] ? lerp(codes[model].a, codes[model].b, t) : new Float32Array(128);
      drawCat(fig.querySelector("canvas"), await decode(model, z));
    });
  }
}
