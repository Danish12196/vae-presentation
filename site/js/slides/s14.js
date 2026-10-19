// The core idea: a VAE encodes a cat to a cloud (mu, sigma), not a point.
// The picked cat's 1 and 2 sigma ellipses glow on the vae_2d_b1 map; "sample again" draws
// z = mu + sigma * eps from that cloud, marks it on the map and decodes it. The last six samples
// stay in a row: the same cat, slightly different each time.
// The clouds are tiny next to the whole space (sigma ~0.05 on a map spanning about ±4), so after
// picking a cat the map zooms in to about 12 sigma around mu; the readout says by how much.

import { encode, decode, drawCat, hasModel, renderTraining, loadShowcase, getShowcaseCat, randn } from "../models.js";
import { catPicker } from "../ui.js";
import { loadLatentMap, VAE_VIEW } from "../latentmap.js";

const MODEL = "vae_2d_b1";
const KEEP = 6;
const ZOOM_SIGMAS = 12;

let map, inCanvas, outCanvas, row, params;
let cloud = null;     // { mu: [x, y], sigma: [sx, sy] }
let samples = [];     // recent z, newest last

export async function init(section) {
  const body = section.querySelector(".s11-body");
  if (!(await hasModel(MODEL))) return renderTraining(body, MODEL);
  inCanvas = section.querySelector(".s11-input");
  outCanvas = section.querySelector(".s11-out");
  row = section.querySelector(".s11-row");
  params = section.querySelector(".s11-params");

  map = await loadLatentMap(section.querySelector(".s11-map"), MODEL, {
    view: VAE_VIEW,
    label: "The VAE's latent map with the picked cat's cloud",
  });
  map.overlay = drawCloud;

  await loadShowcase();
  await catPicker(section.querySelector(".s11-picker"), { onPick: pickCat });
  section.querySelector("[data-sample]").addEventListener("click", sampleAgain);
  await pickCat(0);
}

async function pickCat(i) {
  const image = getShowcaseCat(i);
  drawCat(inCanvas, image);
  const { mu, logvar } = await encode(MODEL, [image]);
  cloud = {
    mu: [mu.data[0], mu.data[1]],
    sigma: [Math.exp(0.5 * logvar.data[0]), Math.exp(0.5 * logvar.data[1])],
  };
  params.textContent = `μ = (${cloud.mu.map((v) => v.toFixed(2)).join(", ")})   σ = (${cloud.sigma.map((v) => v.toFixed(2)).join(", ")})`;
  samples = [];
  row.replaceChildren();
  // Show where the cloud sits in the whole space, then glide in close enough to see it.
  map.stop();
  map.view = { ...map.home };
  map.draw();
  const half = Math.max(0.35, ZOOM_SIGMAS * Math.max(...cloud.sigma));
  setTimeout(() => map.zoomTo({ cx: cloud.mu[0], cy: cloud.mu[1], half }), 350);
  params.textContent += `   zoom ×${Math.round(map.home.half / half)}`;
  await sampleAgain();
}

export function leave() {
  map?.stop();
}

/** z = mu + sigma * eps, eps ~ N(0, 1): one more draw from the same cloud. */
async function sampleAgain() {
  if (!cloud) return;
  const eps = randn(2);
  const z = [cloud.mu[0] + cloud.sigma[0] * eps[0], cloud.mu[1] + cloud.sigma[1] * eps[1]];
  samples = [...samples, z].slice(-KEEP);
  map.draw();
  const image = await decode(MODEL, z);
  drawCat(outCanvas, image);

  const thumb = document.createElement("canvas");
  thumb.className = "cat-canvas small";
  thumb.setAttribute("role", "listitem");
  thumb.setAttribute("aria-label", "A sample");
  drawCat(thumb, image);
  row.append(thumb);
  while (row.children.length > KEEP) row.firstElementChild.remove();
}

/** Map overlay: a glowing 2-sigma and 1-sigma ellipse at mu, and the recent samples as dots. */
function drawCloud(ctx, m) {
  if (!cloud) return;
  const [cx, cy] = m.toScreen(cloud.mu);
  const [ex] = m.toScreen([cloud.mu[0] + cloud.sigma[0], cloud.mu[1]]);
  const [, ey] = m.toScreen([cloud.mu[0], cloud.mu[1] + cloud.sigma[1]]);
  const rx = Math.max(4, Math.abs(ex - cx));
  const ry = Math.max(4, Math.abs(cy - ey));
  ctx.save();
  for (const [k, alpha] of [[2, 0.22], [1, 0.4]]) {
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx * k, ry * k, 0, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(196, 98, 45, ${alpha})`;
    ctx.shadowBlur = 0;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(36, 23, 15, 0.85)";
    ctx.stroke();
  }
  ctx.shadowBlur = 0;
  samples.forEach((z, i) => {
    const [x, y] = m.toScreen(z);
    ctx.beginPath();
    ctx.arc(x, y, i === samples.length - 1 ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = i === samples.length - 1 ? "#c4622d" : "rgba(36, 23, 15, 0.55)";
    ctx.fill();
  });
  ctx.restore();
}
