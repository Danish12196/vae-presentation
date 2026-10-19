// "A cat is 12,288 numbers": the 3Blue1Brown move (pixels -> a column of neurons) followed by the
// Deepia move (two bottleneck neurons -> the two axes of the latent space). Built press by press:
//   0 the cat, and the question: could you describe it with two numbers?
//   1 a pixel grid: every pixel is 3 numbers (one picked pixel shows its RGB)
//   2 pixels fly into a column: 64 x 64 x 3 = 12,288 numbers
//   3 the encoder squeezes the column to 2 neurons, with ae_2's live values
//   4 the 2 neurons become the axes; the cat lands as a dot at (z1, z2)
//   5 every training cat on the same axes: that is the latent space
// The values, the dot and the colours are real: the picked cat through ae_2.

import { encode, drawCat, hasModel, renderTraining, loadShowcase, getShowcaseCat, loadJSON, SIZE } from "../models.js";
import { catPicker } from "../ui.js";
import { tween, lerp, builder, freshToken } from "../anim.js";

const MODEL = "ae_2";
const NS = "http://www.w3.org/2000/svg";
const LAST = 5;
const CAT = { x: 40, y: 130, size: 200 };
const COL = { x: 380, top: 70, bottom: 450, n: 15 };
const CODE = { x: 700, y: [215, 285], r: 24 };
const PLOT = { x0: 840, x1: 1160, y0: 430, y1: 70 };
const FLY = 15; // pixels that visibly fly into the column

let svg, els = {}, b, cat = 0, image = null, code = [0, 0], points = [], colours = [], bounds;

const el = (name, attrs = {}, parent = svg) => {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  parent?.append(node);
  return node;
};
const text = (str, attrs, parent) => {
  const t = el("text", attrs, parent);
  t.textContent = str;
  return t;
};

export async function init(section) {
  const body = section.querySelector(".num-body");
  if (!(await hasModel(MODEL))) return renderTraining(body, MODEL);
  svg = section.querySelector("svg.num-stage");
  const maps = await loadJSON("data/latent_maps.json");
  points = maps.maps.ae_2.slice(0, 600);
  colours = maps.colours.slice(0, 600);
  bounds = viewOf(points);
  build();
  await loadShowcase();
  await catPicker(section.querySelector(".num-picker"), { onPick: (i) => pick(i) });
  b = builder({ last: LAST, show, stage: svg });
  await pick(0);
}

export const enter = (info) => b?.enter(info);
export const step = (dir) => b?.step(dir) ?? false;

function viewOf(pts) {
  const q = (arr, p) => arr[Math.round(p * (arr.length - 1))];
  const xs = pts.map((p) => p[0]).sort((a, c) => a - c);
  const ys = pts.map((p) => p[1]).sort((a, c) => a - c);
  const pad = (lo, hi) => [lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08];
  return { x: pad(q(xs, 0.005), q(xs, 0.995)), y: pad(q(ys, 0.005), q(ys, 0.995)) };
}
const toPlot = ([zx, zy]) => [
  lerp(PLOT.x0, PLOT.x1, (zx - bounds.x[0]) / (bounds.x[1] - bounds.x[0])),
  lerp(PLOT.y0, PLOT.y1, (zy - bounds.y[0]) / (bounds.y[1] - bounds.y[0])),
];

// ---------------------------------------------------------------- the static drawing

function build() {
  svg.setAttribute("viewBox", "0 0 1200 500");
  // 0: the cat (an <image> filled from a canvas) and its frame
  els.catImg = el("image", { x: CAT.x, y: CAT.y, width: CAT.size, height: CAT.size, style: "image-rendering: pixelated" });
  el("rect", { x: CAT.x, y: CAT.y, width: CAT.size, height: CAT.size, class: "num-frame" });
  els.catLabel = text("64 × 64 pixels", { x: CAT.x + CAT.size / 2, y: CAT.y + CAT.size + 34, class: "num-label", "text-anchor": "middle" });

  // 1: pixel grid and one pixel's RGB
  els.grid = el("g", { class: "num-step", "data-from": 1 });
  for (let i = 1; i < 16; i++) {
    const o = (CAT.size / 16) * i;
    el("line", { x1: CAT.x + o, y1: CAT.y, x2: CAT.x + o, y2: CAT.y + CAT.size, class: "num-grid" }, els.grid);
    el("line", { x1: CAT.x, y1: CAT.y + o, x2: CAT.x + CAT.size, y2: CAT.y + o, class: "num-grid" }, els.grid);
  }
  els.pick = el("rect", { width: CAT.size / 16, height: CAT.size / 16, class: "num-pick" }, els.grid);
  els.rgb = text("", { x: CAT.x + CAT.size / 2, y: CAT.y - 22, class: "num-mono", "text-anchor": "middle" }, els.grid);

  // 2: the column of input neurons
  els.column = el("g", { class: "num-step", "data-from": 2 });
  els.colDots = [];
  const gap = (COL.bottom - COL.top) / (COL.n - 1);
  for (let i = 0; i < COL.n; i++) {
    const y = COL.top + i * gap;
    if (i === 7) {
      text("⋮", { x: COL.x, y: y + 10, class: "num-dots", "text-anchor": "middle" }, els.column);
      continue;
    }
    els.colDots.push(el("circle", { cx: COL.x, cy: y, r: 11, class: "num-neuron" }, els.column));
  }
  el("path", { d: `M ${COL.x - 34} ${COL.top - 6} q -14 0 -14 14 V ${(COL.top + COL.bottom) / 2 - 12} l -10 12 l 10 12 V ${COL.bottom - 8} q 0 14 14 14`, class: "num-brace" }, els.column);
  text("12,288 numbers", { x: COL.x + 24, y: COL.top - 24, class: "num-big", "text-anchor": "end" }, els.column);
  els.flyers = el("g", { class: "num-flyers" });

  // 3: encoder and the two code neurons with their values
  els.encoder = el("g", { class: "num-step", "data-from": 3 });
  el("path", { d: `M 430 40 L 620 190 L 620 310 L 430 460 Z`, class: "num-encoder" }, els.encoder);
  text("encoder", { x: 525, y: 258, class: "num-label", "text-anchor": "middle" }, els.encoder);
  for (let i = 0; i < 14; i++) {
    const y = COL.top + (i < 7 ? i : i + 1) * gap;
    el("line", { x1: COL.x + 12, y1: y, x2: 430, y2: lerp(60, 440, i / 13), class: "num-wire" }, els.encoder);
  }
  els.codeGroup = el("g", { class: "num-step", "data-from": 3 });
  els.code = CODE.y.map((y, i) => {
    const g = el("g", {}, els.codeGroup);
    el("line", { x1: 620, y1: i ? 290 : 210, x2: CODE.x - CODE.r, y2: y, class: "num-wire" }, g);
    const c = el("circle", { cx: CODE.x, cy: y, r: CODE.r, class: "num-code" }, g);
    const v = text("", { x: CODE.x + 36, y: y + 8, class: "num-mono num-value" }, g);
    const name = text(`neuron ${i + 1}`, { x: CODE.x, y: i ? y + 52 : y - 36, class: "num-small", "text-anchor": "middle" }, g);
    return { g, c, v, name, home: [CODE.x, y] };
  });
  els.codeTag = text("z, the code: 2 numbers", { x: CODE.x, y: 372, class: "num-label", "text-anchor": "middle" }, els.codeGroup);

  // 4: axes, the two neurons as axis labels, the cat's dot
  els.axes = el("g", { class: "num-step", "data-from": 4 });
  el("line", { x1: PLOT.x0, y1: PLOT.y0, x2: PLOT.x1 + 10, y2: PLOT.y0, class: "num-axis", "marker-end": "url(#num-arrow)" }, els.axes);
  el("line", { x1: PLOT.x0, y1: PLOT.y0, x2: PLOT.x0, y2: PLOT.y1 - 10, class: "num-axis", "marker-end": "url(#num-arrow)" }, els.axes);
  const defs = el("defs");
  const m = el("marker", { id: "num-arrow", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
  el("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "num-arrowhead" }, m);
  text("z1 = neuron 1", { x: PLOT.x1, y: PLOT.y0 + 34, class: "num-small", "text-anchor": "end" }, els.axes);
  text("z2 = neuron 2", { x: PLOT.x0 + 12, y: PLOT.y1 - 16, class: "num-small" }, els.axes);

  // 5: every training cat
  const clip = el("clipPath", { id: "num-plot-clip" }, defs);
  el("rect", { x: PLOT.x0 + 2, y: PLOT.y1, width: PLOT.x1 - PLOT.x0, height: PLOT.y0 - PLOT.y1 - 2 }, clip);
  els.cloud = el("g", { class: "num-step num-cloud", "data-from": 5 });
  const dots = el("g", { "clip-path": "url(#num-plot-clip)" }, els.cloud);
  points.forEach((p, i) => {
    const [x, y] = toPlot(p);
    el("circle", { cx: x, cy: y, r: 3.2, fill: colours[i], class: "num-cat" }, dots);
  });
  els.cloudTag = text("600 training cats, each one a point: the latent space", { x: (PLOT.x0 + PLOT.x1) / 2, y: 490, class: "num-small", "text-anchor": "middle" }, els.cloud);
  els.dot = el("circle", { r: 9, class: "num-dot num-step", "data-from": 4 });
}

// ---------------------------------------------------------------- one cat

async function pick(i) {
  cat = i;
  image = getShowcaseCat(i);
  const c = document.createElement("canvas");
  drawCat(c, image);
  els.catImg.setAttribute("href", c.toDataURL());
  // the pixel the grid points at: near the left eye of an aligned cat face
  const px = 20, py = 26, k = (py * SIZE + px) * 4;
  const cell = CAT.size / SIZE;
  els.pick.setAttribute("x", CAT.x + px * cell - CAT.size / 32 + cell / 2);
  els.pick.setAttribute("y", CAT.y + py * cell - CAT.size / 32 + cell / 2);
  els.rgb.textContent = `one pixel = (${image.data[k]}, ${image.data[k + 1]}, ${image.data[k + 2]})`;
  // column colours: real pixels sampled down the image
  els.colDots.forEach((d, j) => {
    const q = (Math.floor(((j + 0.5) / els.colDots.length) * SIZE) * SIZE + 32) * 4;
    d.setAttribute("fill", `rgb(${image.data[q]}, ${image.data[q + 1]}, ${image.data[q + 2]})`);
  });
  const { z } = await encode(MODEL, [image]);
  code = [z.data[0], z.data[1]];
  els.code.forEach((n, j) => (n.v.textContent = code[j].toFixed(2)));
  b.redraw();
}

// ---------------------------------------------------------------- the build

function show(n, { animate }) {
  const alive = freshToken();
  for (const g of svg.querySelectorAll(".num-step")) g.classList.toggle("on", n >= Number(g.dataset.from));
  const [dx, dy] = toPlot(code);
  els.dot.setAttribute("cx", dx);
  els.dot.setAttribute("cy", dy);
  // step 4 moves the two neuron values next to their axes; earlier steps keep them in the column
  const target = n >= 4;
  const place = (t) => {
    els.code.forEach((nn, j) => {
      const to = j === 0 ? [dx, PLOT.y0 + 2] : [PLOT.x0 - 2, dy];
      const [x, y] = [lerp(nn.home[0], to[0], t), lerp(nn.home[1], to[1], t)];
      nn.c.setAttribute("cx", x);
      nn.c.setAttribute("cy", y);
      nn.c.setAttribute("r", lerp(CODE.r, 12, t));
      nn.v.setAttribute("x", x + (j === 0 ? -22 : -60));
      nn.v.setAttribute("y", y + (j === 0 ? 40 : 8));
      nn.name.setAttribute("opacity", 1 - t);
      nn.g.querySelector("line").setAttribute("opacity", 1 - t);
    });
    els.codeTag.setAttribute("opacity", 1 - t);
  };
  if (animate && n === 4) tween(900, (t) => alive() && place(t));
  else place(target ? 1 : 0);
  if (animate && n === 2) flyPixels(alive);
  else els.flyers.replaceChildren();
}

/** A handful of real pixels leave the image and land on column neurons. */
function flyPixels(alive) {
  els.flyers.replaceChildren();
  const cell = CAT.size / SIZE;
  const gap = (COL.bottom - COL.top) / (COL.n - 1);
  const flights = Array.from({ length: FLY }, (_, j) => {
    const px = 8 + ((j * 23) % 48), py = 6 + ((j * 37) % 52);
    const k = (py * SIZE + px) * 4;
    const r = el("rect", { width: 10, height: 10, fill: `rgb(${image.data[k]}, ${image.data[k + 1]}, ${image.data[k + 2]})`, class: "num-flyer" }, els.flyers);
    const slot = j < 7 ? j : j + 1;
    return { r, from: [CAT.x + px * cell, CAT.y + py * cell], to: [COL.x - 5, COL.top + Math.min(slot, COL.n - 1) * gap - 5], delay: j * 0.035 };
  });
  tween(1100, (t) => {
    if (!alive()) return;
    for (const f of flights) {
      const u = Math.max(0, Math.min(1, (t - f.delay) / (1 - FLY * 0.035)));
      f.r.setAttribute("x", lerp(f.from[0], f.to[0], u));
      f.r.setAttribute("y", lerp(f.from[1], f.to[1], u) - Math.sin(u * Math.PI) * 60);
      f.r.setAttribute("opacity", u < 1 ? 1 : 0);
    }
  }, (t) => t);
}
