// Generating new cats. Left: 24 samples from vae_128 with z ~ N(0, I), the distribution its KL term
// organised the latent space around. Right: 24 samples from ae_128, either naive (z ~ N(0, I)) or
// fair (z drawn with the AE's own per-dimension code mean and std). Fair is still mostly nonsense:
// a plain AE has no reason for the space between its codes to decode to cats.
// No classifier scores: cnn_cls calls every decoder output "cat" (see SPEC), so the images speak.
// Below: a small vae_2d_b1 map where any click decodes a plausible cat.
// When data/gen_pool.json exists, the grids show samples from the bigger models in
// training/train_v2.py (precomputed: too big for the browser); re-roll draws a fresh 24 from the pool.

import { decode, drawCat, drawTiles, inSlot, loadManifest, randn, loadJSON, loadSprite } from "../models.js";
import { flipSwitch } from "../ui.js";
import { loadLatentMap, VAE_VIEW } from "../latentmap.js";

const COLS = 6, ROWS = 4, N = COLS * ROWS, LATENT = 128, GAP = 4;

let vaeCanvas, aeCanvas, vaeFig, aeFig, fair = false, aeStats = null, dice, pickCanvas, map;
let request = 0;
let pool = null; // { parts: { vae: [start, n], ae_naive, ae_fair }, sprite }

export async function init(section) {
  vaeFig = section.querySelector(".gen-vae");
  aeFig = section.querySelector(".gen-ae");
  vaeCanvas = vaeFig.querySelector("canvas");
  aeCanvas = aeFig.querySelector("canvas");
  dice = section.querySelector(".reroll");
  pickCanvas = section.querySelector(".s15-pick");

  const meta = await loadJSON("data/gen_pool.json").catch(() => null);
  if (meta) {
    pool = { parts: meta.parts, sprite: await loadSprite(`data/${meta.file}`, meta) };
    section.querySelector(".gen-note").hidden = false;
  }
  const ae = (await loadManifest()).models.ae_128;
  if (ae?.code_mean) aeStats = { mean: ae.code_mean, std: ae.code_std };

  flipSwitch(section.querySelector(".ae-mode"), {
    left: "naive N(0,1)",
    right: "fair: its own stats",
    label: "Sample the AE with its own code statistics",
    onChange: (on) => {
      fair = on;
      rollAE();
    },
  });
  dice.addEventListener("click", reroll);

  await inSlot(section.querySelector(".s15-pick-fig"), async () => {
    map = await loadLatentMap(section.querySelector(".s15-map"), "vae_2d_b1", {
      view: VAE_VIEW,
      dot: 1.6,
      label: "The VAE's latent map: click anywhere to decode a cat",
      onPick: async (z) => {
        map.setCrosshair(z);
        drawCat(pickCanvas, await decode("vae_2d_b1", z));
      },
    });
    const z = [0.4, -0.3];
    map.setCrosshair(z);
    drawCat(pickCanvas, await decode("vae_2d_b1", z));
  });
  await reroll();
}

export function leave() {
  dice?.classList.remove("rolling");
}

async function reroll() {
  // Restart the dice spin even if it is mid-turn.
  dice.classList.remove("rolling");
  void dice.offsetWidth;
  dice.classList.add("rolling");
  // One grid per frame: each batch of 24 decodes blocks the page for a moment (~30 ms, more on
  // slow phones), so yielding in between keeps the page responsive while the dice spins.
  await rollVAE();
  await new Promise((r) => requestAnimationFrame(() => setTimeout(r)));
  await rollAE();
}

/** 24 random tiles from one part of the precomputed pool. */
function drawPool(canvas, part) {
  const [start, n] = pool.parts[part];
  const picks = new Set();
  while (picks.size < N) picks.add(start + Math.floor(Math.random() * n));
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  [...picks].forEach((idx, i) => ctx.putImageData(pool.sprite.get(idx), (i % COLS) * (64 + GAP) + GAP / 2, Math.floor(i / COLS) * (64 + GAP) + GAP / 2));
}

async function rollVAE() {
  if (pool) return drawPool(vaeCanvas, "vae");
  const mine = ++request;
  await inSlot(vaeFig, async () => {
    const images = await decode("vae_128", randn(N * LATENT));
    if (mine === request) drawTiles(vaeCanvas, images, { cols: COLS, gap: GAP, inset: GAP / 2 });
  });
}

async function rollAE() {
  if (pool) return drawPool(aeCanvas, fair ? "ae_fair" : "ae_naive");
  const z = randn(N * LATENT);
  if (fair && aeStats) {
    for (let i = 0; i < z.length; i++) {
      const d = i % LATENT;
      z[i] = aeStats.mean[d] + aeStats.std[d] * z[i];
    }
  }
  await inSlot(aeFig, async () => drawTiles(aeCanvas, await decode("ae_128", z), { cols: COLS, gap: GAP, inset: GAP / 2 }));
}
