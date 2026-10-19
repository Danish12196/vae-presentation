// More numbers, more room: the same 600 training cats in the latent spaces of ae_2, ae_16, ae_64
// and ae_256 (data/latent_sizes.json; codes wider than 2 are flattened to their 2 main PCA axes),
// each beside the picked cat rebuilt by that model, live. Deepia's "latent dimension" scene, with
// cats coloured by fur instead of digits coloured by class.
// Built press by press: one more model per press.

import { reconstruct, drawCat, hasModel, renderTraining, loadShowcase, getShowcaseCat, loadJSON } from "../models.js";
import { catPicker } from "../ui.js";
import { builder } from "../anim.js";

const NAMES = ["ae_2", "ae_16", "ae_64", "ae_256"];

let cols = [], data, colours, b, cat = 0;

export async function init(section) {
  const body = section.querySelector(".size-body");
  if (!(await hasModel("ae_2"))) return renderTraining(body, "ae_2");
  data = await loadJSON("data/latent_sizes.json");
  colours = (await loadJSON("data/latent_maps.json")).colours;
  const grid = section.querySelector(".size-grid");
  cols = NAMES.filter((n) => data.maps[n]).map((name, i) => {
    const k = Number(name.split("_")[1]);
    const col = document.createElement("figure");
    col.className = "size-col";
    col.dataset.step = String(i + 1);
    col.innerHTML = `
      <figcaption><strong>${k}</strong> number${k > 1 ? "s" : ""}</figcaption>
      <canvas class="size-map" role="img" aria-label="${k}-number latent space of 600 training cats"></canvas>
      <p class="size-note mono">${k === 2 ? "the real axes" : `2 of ${k} axes (PCA), ${Math.round(data.kept[name] * 100)}% of the spread`}</p>
      <canvas class="cat-canvas size-cat" width="64" height="64" aria-label="The picked cat rebuilt from ${k} numbers"></canvas>`;
    grid.append(col);
    return { name, el: col, map: col.querySelector(".size-map"), cat: col.querySelector(".size-cat") };
  });
  new ResizeObserver(() => cols.forEach(drawMap)).observe(grid);
  await loadShowcase();
  await catPicker(section.querySelector(".size-picker"), { onPick: (i) => ((cat = i), rebuild()) });
  b = builder({ last: cols.length, show, stage: grid });
  cols.forEach(drawMap);
  await rebuild();
}

export const enter = (info) => b?.enter(info);
export const step = (dir) => b?.step(dir) ?? false;

function show(n) {
  for (const c of cols) c.el.classList.toggle("shown", Number(c.el.dataset.step) <= Math.max(1, n));
}

async function rebuild() {
  const image = getShowcaseCat(cat);
  for (const c of cols) drawCat(c.cat, await reconstruct(c.name, [image]));
}

function drawMap(c) {
  const pts = data.maps[c.name];
  const size = c.map.clientWidth || 200;
  const dpr = devicePixelRatio || 1;
  c.map.width = c.map.height = Math.round(size * dpr);
  const ctx = c.map.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const q = (arr, p) => [...arr].sort((a, b) => a - b)[Math.round(p * (arr.length - 1))];
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const half = Math.max(q(xs, 0.99) - q(xs, 0.01), q(ys, 0.99) - q(ys, 0.01)) * 0.62;
  const cx = (q(xs, 0.99) + q(xs, 0.01)) / 2, cy = (q(ys, 0.99) + q(ys, 0.01)) / 2;
  ctx.clearRect(0, 0, size, size);
  pts.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(((p[0] - cx) / half / 2 + 0.5) * size, (0.5 - (p[1] - cy) / half / 2) * size, 2.4, 0, Math.PI * 2);
    ctx.fillStyle = colours[i];
    ctx.fill();
  });
}
