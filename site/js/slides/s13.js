// Turning the KL dial: the same 2,000 cats encoded by four 2D VAEs trained with
// beta = 0, 0.1, 1 and 4. The slider glides the dots between the four precomputed mu scatters:
// beta 0 is a plain autoencoder (spread out, gaps), beta 4 packs everything around N(0, 1).
// The rope above leans towards whichever term is winning.

import { loadJSON, renderTraining } from "../models.js";
import { slider } from "../ui.js";
import { latentMap } from "../latentmap.js";

const BETAS = [
  { beta: "0", key: "vae_2d_b0", pull: -0.85 },
  { beta: "0.1", key: "vae_2d_b0.1", pull: -0.45 },
  { beta: "1", key: "vae_2d_b1", pull: 0.05 },
  { beta: "4", key: "vae_2d_b4", pull: 0.6 },
];

let map, tug;

export async function init(section) {
  const body = section.querySelector(".s13-body");
  const data = await loadJSON("data/latent_maps.json").catch(() => null);
  const missing = BETAS.find((b) => !data?.maps?.[b.key]);
  if (missing) return renderTraining(body, missing.key);

  // One fixed view for all four, fitted to the widest (beta = 0), so the packing is visible.
  map = latentMap(section.querySelector(".s13-map"), {
    points: data.maps[BETAS[0].key],
    colours: data.colours,
    extent: 1.15,
    dot: 2.6,
    label: "Latent maps of four VAEs, from beta 0 to beta 4",
  });
  map.overlay = drawPrior;
  tug = section.querySelector(".tug");

  slider(section.querySelector(".s13-slider"), {
    label: "KL weight β",
    min: 0,
    max: BETAS.length - 1,
    step: 1,
    value: 0,
    format: (i) => `β = ${BETAS[i].beta}`,
    ticks: BETAS.map((b) => b.beta),
    onInput: (i) => {
      map.setPoints(data.maps[BETAS[i].key]);
      tug.style.setProperty("--pull", BETAS[i].pull);
    },
  });
  tug.style.setProperty("--pull", BETAS[0].pull);
}

export function leave() {
  map?.stop(); // jump to the end of any glide in progress
}

/** N(0, 1) as dashed 1 and 2 sigma rings: where the KL term pulls every cloud. */
function drawPrior(ctx, m) {
  const [ox, oy] = m.toScreen([0, 0]);
  const [r1] = m.toScreen([1, 0]);
  ctx.save();
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = "rgba(179, 136, 255, 0.85)";
  ctx.lineWidth = 2;
  for (const k of [1, 2]) {
    ctx.beginPath();
    ctx.arc(ox, oy, (r1 - ox) * k, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(179, 136, 255, 0.95)";
  ctx.font = "600 14px 'JetBrains Mono', monospace";
  ctx.fillText("N(0, 1)", ox + (r1 - ox) * 2 + 6, oy - 6);
  ctx.restore();
}
