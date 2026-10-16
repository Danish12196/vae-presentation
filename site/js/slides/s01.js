// Title slide: the mascot sits on a slowly drifting mosaic of held-out cats.
// Every few seconds one tile is replaced by a cat that vae_128 has just imagined.

import { loadShowcase, getShowcaseCat, hasModel, decode, fromTensor, randn } from "../models.js";

const TILE = 64;           // native pixels per tile (the CSS scales it up)
const SWAP_EVERY = 3200;   // ms between freshly generated cats
const FADE = 700;          // ms crossfade into the new cat

let section, mosaic, drift, canvas, ctx;
let cols = 0;
let rows = 0;
let canGenerate = false;
let timer = 0;
let active = false;
const tileCat = []; // which showcase cat sits in each tile

export async function init(el) {
  section = el;
  mosaic = el.querySelector(".mosaic");
  drift = el.querySelector(".mosaic-drift");
  canvas = el.querySelector(".mosaic-canvas");
  ctx = canvas.getContext("2d");
  await loadShowcase();
  layout();
  let resizeTimer = 0;
  addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(layout, 200);
  });
  canGenerate = await hasModel("vae_128"); // without it the mosaic still drifts, just never changes
}

export function enter() {
  active = true;
  mosaic.classList.add("running");
  schedule();
}

export function leave() {
  active = false;
  mosaic.classList.remove("running");
  clearTimeout(timer);
}

// The canvas holds the grid twice side by side; the CSS slides it left by exactly one
// copy and loops, so the drift never shows a seam. Changes are drawn into both copies.
function layout() {
  const tileCss = parseFloat(getComputedStyle(mosaic).getPropertyValue("--tile")) || 110;
  cols = Math.ceil(section.clientWidth / tileCss) + 1;
  rows = Math.ceil(section.clientHeight / tileCss) + 2;
  canvas.width = 2 * cols * TILE;
  canvas.height = rows * TILE;
  canvas.style.width = `${2 * cols * tileCss}px`;
  canvas.style.height = `${rows * tileCss}px`;
  mosaic.style.setProperty("--drift", `${cols * tileCss}px`);
  mosaic.style.setProperty("--drift-y", `${tileCss}px`);

  tileCat.length = 0;
  for (let i = 0; i < rows * cols; i++) {
    tileCat.push((i * 37 + Math.floor(i / cols) * 11) % 64); // scattered, no neighbours repeat
    drawTile(i, getShowcaseCat(tileCat[i]));
  }
}

function drawTile(i, image, alpha = 1) {
  const r = Math.floor(i / cols);
  const c = i % cols;
  for (const x of [c, c + cols]) {
    if (image instanceof ImageData) {
      ctx.putImageData(image, x * TILE, r * TILE); // ImageData ignores alpha: always opaque
    } else {
      ctx.globalAlpha = alpha;
      ctx.drawImage(image, x * TILE, r * TILE);
      ctx.globalAlpha = 1;
    }
  }
}

function schedule() {
  clearTimeout(timer);
  if (!active || !canGenerate) return;
  timer = setTimeout(async () => {
    if (!document.hidden) await swapOne();
    schedule();
  }, SWAP_EVERY);
}

/** Decode a random z ~ N(0, 1) with vae_128 and fade it into a random tile. */
async function swapOne() {
  const image = fromTensor(await decode("vae_128", randn(128)));
  const fresh = await createImageBitmap(image);
  const i = Math.floor(Math.random() * rows * cols);
  const old = ctx.getImageData((i % cols) * TILE, Math.floor(i / cols) * TILE, TILE, TILE);
  const start = performance.now();
  await new Promise((resolve) => {
    const step = (now) => {
      const t = Math.min(1, (now - start) / FADE);
      drawTile(i, old);
      drawTile(i, fresh, t);
      if (t < 1) requestAnimationFrame(step);
      else resolve();
    };
    requestAnimationFrame(step);
  });
  drawTile(i, image);
}
