// Variants, four boxes in a 2x2 grid.
//   Conditional VAE: text only.
//   beta-VAE: "remember this?" opens an overlay. It does not leave the slide.
//   VQ-VAE (an illustration, not a trained model): a codebook of 16 colours found with k-means on
//   one cat, and that cat rebuilt as a 16x16 mosaic where every block is one of the 16 codes.
//   Stable Diffusion (no live model): three denoising snapshots, then the image; fresh noise every
//   couple of seconds while the slide is showing.

import { drawCat, loadShowcase, getShowcaseCat, SIZE } from "../models.js";

const CODES = 16;
const MOSAIC = 16;          // 16x16 blocks of 4x4 pixels
const STEP_MS = 1800;       // fresh noise for the denoising snapshots
const NOISE = [0.85, 0.5, 0.18];

let stepCanvases = [], finalCanvas, timer = 0, sdCat;

export async function init(section) {
  const popup = document.querySelector("#remember");
  section.querySelector("[data-remember]").addEventListener("click", () => {
    popup.hidden = false;
  });
  popup.querySelector("[data-close]").addEventListener("click", () => {
    popup.hidden = true;
  });

  await loadShowcase();
  drawVQ(section, mostColourful());

  stepCanvases = [...section.querySelectorAll(".sd-step")];
  finalCanvas = section.querySelector(".sd-final");
  sdCat = getShowcaseCat(11);
  drawCat(finalCanvas, sdCat);
  drawSteps();
}

export function enter() {
  clearInterval(timer);
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  timer = setInterval(drawSteps, STEP_MS);
}

export function leave() {
  clearInterval(timer);
}

/** The showcase cat with the most colour in it, so the codebook isn't 16 greys. */
function mostColourful() {
  let best = null;
  let bestScore = -1;
  for (let i = 0; i < 64; i++) {
    const cat = getShowcaseCat(i);
    let score = 0;
    for (let p = 0; p < cat.data.length; p += 4) {
      const [r, g, b] = [cat.data[p], cat.data[p + 1], cat.data[p + 2]];
      score += Math.max(r, g, b) - Math.min(r, g, b);
    }
    if (score > bestScore) [bestScore, best] = [score, cat];
  }
  return best;
}

/** Three denoising snapshots: the image blended with less and less noise. */
function drawSteps() {
  stepCanvases.forEach((c, k) => {
    const img = new ImageData(SIZE, SIZE);
    const a = NOISE[k];
    for (let i = 0; i < img.data.length; i += 4) {
      for (let ch = 0; ch < 3; ch++) img.data[i + ch] = (1 - a) * sdCat.data[i + ch] + a * Math.random() * 255;
      img.data[i + 3] = 255;
    }
    drawCat(c, img);
  });
}

/** k-means on the cat's pixels for a 16-colour codebook, then the cat as a mosaic of those codes. */
function drawVQ(section, cat) {
  const px = [];
  for (let i = 0; i < cat.data.length; i += 4) px.push([cat.data[i], cat.data[i + 1], cat.data[i + 2]]);
  // Deterministic start: codes spread through the pixels in brightness order.
  const sorted = [...px].sort((a, b) => a[0] + a[1] + a[2] - (b[0] + b[1] + b[2]));
  let codes = Array.from({ length: CODES }, (_, k) => [...sorted[Math.floor(((k + 0.5) / CODES) * sorted.length)]]);
  const nearest = (p) => {
    let best = 0;
    let bestD = Infinity;
    codes.forEach((c, k) => {
      const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2;
      if (d < bestD) [bestD, best] = [d, k];
    });
    return best;
  };
  for (let iter = 0; iter < 12; iter++) {
    const sums = codes.map(() => [0, 0, 0, 0]);
    for (const p of px) {
      const s = sums[nearest(p)];
      s[0] += p[0];
      s[1] += p[1];
      s[2] += p[2];
      s[3]++;
    }
    codes = codes.map((c, k) => (sums[k][3] ? sums[k].slice(0, 3).map((v) => v / sums[k][3]) : c));
  }
  codes.sort((a, b) => a[0] + a[1] + a[2] - (b[0] + b[1] + b[2]));

  const book = section.querySelector(".vq-codebook");
  const bctx = book.getContext("2d");
  codes.forEach((c, k) => {
    bctx.fillStyle = `rgb(${c.map(Math.round).join(",")})`;
    bctx.fillRect(k % 4, Math.floor(k / 4), 1, 1);
  });

  const mosaic = section.querySelector(".vq-mosaic");
  mosaic.width = mosaic.height = MOSAIC;
  const mctx = mosaic.getContext("2d");
  const block = SIZE / MOSAIC;
  for (let by = 0; by < MOSAIC; by++) {
    for (let bx = 0; bx < MOSAIC; bx++) {
      const avg = [0, 0, 0];
      for (let y = 0; y < block; y++) {
        for (let x = 0; x < block; x++) {
          const i = ((by * block + y) * SIZE + bx * block + x) * 4;
          avg[0] += cat.data[i];
          avg[1] += cat.data[i + 1];
          avg[2] += cat.data[i + 2];
        }
      }
      const c = codes[nearest(avg.map((v) => v / (block * block)))];
      mctx.fillStyle = `rgb(${c.map(Math.round).join(",")})`;
      mctx.fillRect(bx, by, 1, 1);
    }
  }
}
