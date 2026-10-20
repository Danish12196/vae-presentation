// Backpropagation, briefly (3Blue1Brown style): a tiny two-weight network with real numbers.
//   x → (× w₁) → h → (× w₂) → ŷ → L = (ŷ − y)²
// Presses: 0 the network · 1 forward values · 2 the loss · 3 the blame flowing back · 4 one step downhill.
// "take a step" keeps doing gradient descent; the landscape on the right shows the ball rolling into the valley.

import { builder } from "../anim.js";

const NS = "http://www.w3.org/2000/svg";
const LAST = 4;
const X = 1.5, Y = 2, ETA = 0.1;
const W0 = { w1: 0.8, w2: 0.5 };
const RANGE = 2; // the landscape shows w₁, w₂ in [0, 2]

let svg, section, build, pathSvg, taken = 0;
const nodes = {};

const fmt = (v) => (Math.abs(v) < 0.005 ? "0.00" : v.toFixed(2)).replace("-", "−");

function forward({ w1, w2 }) {
  const h = X * w1;
  const yhat = h * w2;
  const L = (yhat - Y) ** 2;
  const dy = 2 * (yhat - Y); // ∂L/∂ŷ
  return { w1, w2, h, yhat, L, dy, dw2: dy * h, dh: dy * w2, dw1: dy * w2 * X };
}

/** Weights after k gradient-descent steps from the start. */
function weightsAfter(k) {
  let w = { ...W0 };
  const trail = [{ ...w }];
  for (let i = 0; i < k; i++) {
    const f = forward(w);
    w = { w1: w.w1 - ETA * f.dw1, w2: w.w2 - ETA * f.dw2 };
    trail.push({ ...w });
  }
  return { w, trail };
}

function el(name, attrs = {}, parent = svg) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent.appendChild(e);
  return e;
}
function text(x, y, cls, parent = svg, anchor = "middle") {
  return el("text", { x, y, class: cls, "text-anchor": anchor }, parent);
}

export function init(el_) {
  section = el_;
  svg = section.querySelector(".bp2-graph");
  pathSvg = section.querySelector(".bp2-path");
  drawGraph();
  drawLandscape(section.querySelector(".bp2-heat"));

  section.querySelector("[data-bp-step]").addEventListener("click", () => {
    while (build.shown < LAST) build.step(1);
    taken = Math.max(1, taken) + 1;
    update(LAST);
  });
  section.querySelector("[data-bp-reset]").addEventListener("click", () => {
    taken = 0;
    update(build.shown);
  });

  build = builder({ last: LAST, show: (n) => update(n), stage: section.querySelector(".bp2-left") });
  update(LAST);
}

function drawGraph() {
  const yF = 112; // the forward row
  const box = (cx, label, key) => {
    const g = el("g", { class: `bp2-node bp2-${key}` });
    el("rect", { x: cx - 52, y: yF - 40, width: 104, height: 80 }, g);
    text(cx, yF - 10, "bp2-name", g).textContent = label;
    nodes[key] = text(cx, yF + 24, "bp2-val", g);
    return g;
  };
  const op = (cx, label, key) => {
    const g = el("g", { class: `bp2-op bp2-${key}` });
    el("circle", { cx, cy: yF, r: 38 }, g);
    text(cx, yF + 9, "bp2-name", g).textContent = label;
    nodes[key] = text(cx, yF - 52, "bp2-weight", g);
    nodes[`g${key}`] = text(cx, yF + 172, "bp2-grad", g);
    return g;
  };
  const arrow = (x1, x2, y, cls) => {
    el("line", { x1, y1: y, x2, y2: y, class: cls, "marker-end": `url(#${cls}-head)` });
  };
  const defs = el("defs");
  for (const cls of ["bp2-fwd", "bp2-back"]) {
    const m = el("marker", { id: `${cls}-head`, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M 0 0 L 10 5 L 0 10 z", class: `${cls}-tip` }, m);
  }

  box(60, "x", "x");
  op(215, "× w₁", "w1");
  box(370, "h", "h");
  op(525, "× w₂", "w2");
  box(680, "ŷ", "yhat");
  const lossBox = box(830, "loss", "L");
  lossBox.classList.add("bp2-lossbox");
  nodes.target = text(830, yF - 52, "bp2-target");
  nodes.target.textContent = `target y = ${Y}`;

  for (const [a, b] of [[112, 177], [253, 318], [422, 487], [563, 628], [732, 778]]) arrow(a, b, yF, "bp2-fwd");
  // The blame, flowing right to left under the forward row: one arrow per box, each value under its box.
  const yB = 206;
  const back = el("g", { class: "bp2-backrow" });
  for (const [a, b] of [[810, 700], [660, 545], [505, 390], [350, 235]]) {
    el("line", { x1: a, y1: yB, x2: b, y2: yB, class: "bp2-back", "marker-end": "url(#bp2-back-head)" }, back);
  }
  text(755, yB - 14, "bp2-why", back).textContent = "2(ŷ − y)";
  text(447, yB - 14, "bp2-why", back).textContent = "× w₂";
  nodes.dy = text(680, yB + 44, "bp2-blame", back);
  nodes.dh = text(370, yB + 44, "bp2-blame", back);
  nodes.gw1.classList.add("bp2-back-only");
  nodes.gw2.classList.add("bp2-back-only");
}

function update(n) {
  section.dataset.step = n;
  const { w, trail } = weightsAfter(n >= LAST ? Math.max(1, taken) : 0);
  // Before the step, show the starting weights; at the step, show where they moved.
  const f = forward(n >= LAST ? trail[trail.length - 2] : w);
  const now = forward(w);
  nodes.x.textContent = fmt(X);
  nodes.w1.textContent = `w₁ = ${fmt(n >= LAST ? now.w1 : f.w1)}`;
  nodes.w2.textContent = `w₂ = ${fmt(n >= LAST ? now.w2 : f.w2)}`;
  const shownF = n >= LAST ? now : f;
  nodes.h.textContent = n >= 1 ? fmt(shownF.h) : "?";
  nodes.yhat.textContent = n >= 1 ? fmt(shownF.yhat) : "?";
  nodes.L.textContent = n >= 2 ? fmt(shownF.L) : "?";
  nodes.dy.textContent = `∂L/∂ŷ = ${fmt(f.dy)}`;
  nodes.dh.textContent = `∂L/∂h = ${fmt(f.dh)}`;
  nodes.gw1.textContent = `∂L/∂w₁ = ${fmt(f.dw1)}  (× x)`;
  nodes.gw2.textContent = `∂L/∂w₂ = ${fmt(f.dw2)}  (× h)`;
  drawPath(n >= LAST ? trail : trail.slice(0, 1));
}

const toPx = (w) => ({ x: (w.w1 / RANGE) * 100, y: 100 - (w.w2 / RANGE) * 100 });

function drawPath(trail) {
  pathSvg.replaceChildren();
  const pts = trail.map(toPx);
  const line = document.createElementNS(NS, "polyline");
  line.setAttribute("points", pts.map((p) => `${p.x},${p.y}`).join(" "));
  line.setAttribute("class", "bp2-trail");
  pathSvg.appendChild(line);
  pts.forEach((p, i) => {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", p.x);
    c.setAttribute("cy", p.y);
    c.setAttribute("r", i === pts.length - 1 ? 4.2 : 1.6);
    c.setAttribute("class", i === pts.length - 1 ? "bp2-ball" : "bp2-dot");
    pathSvg.appendChild(c);
  });
}

/** The loss over (w₁, w₂): dark lilac is low, paper is high. The valley is the curve w₁·w₂ = y/x. */
function drawLandscape(canvas) {
  const ctx = canvas.getContext("2d");
  const n = canvas.width;
  const img = ctx.createImageData(n, n);
  const low = [0x6b, 0x30, 0x50], high = [0xff, 0xfa, 0xf3];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const w1 = ((i + 0.5) / n) * RANGE, w2 = (1 - (j + 0.5) / n) * RANGE;
      const L = (X * w1 * w2 - Y) ** 2;
      const t = Math.min(1, Math.sqrt(L) / 2.5);
      const band = Math.abs(((Math.sqrt(L) * 4) % 1) - 0.5) < 0.05 ? 0.9 : 1; // faint contour lines
      const k = (j * n + i) * 4;
      for (let c = 0; c < 3; c++) img.data[k + c] = (low[c] + (high[c] - low[c]) * t) * band;
      img.data[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

export const enter = (info) => build.enter(info);
export const step = (dir) => build.step(dir);
