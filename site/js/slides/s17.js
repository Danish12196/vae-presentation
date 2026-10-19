// Probability in one minute (Deepia's VAE video, 3:25-6:00): a joint p(x, z) as a heatmap, its two
// marginals on the edges, and conditionals as normalised slices. Drag the crosshair:
//   the horizontal line at z0 gives p(x | z0)  (what a decoder draws for a given code)
//   the vertical line at x0 gives p(z | x0)    (the posterior: which codes explain this image)
// The joint is a toy mixture of three tilted Gaussians; x stands in for a whole image.
// Built press by press: joint, marginals, decoder slice, posterior slice and why it's hard.

import { builder } from "../anim.js";
import { renderMath } from "../math.js";

const NS = "http://www.w3.org/2000/svg";
const N = 28;                               // grid cells per side
const J = { x: 170, y: 150, size: 400 };    // the joint's square
const BLOBS = [                             // mean x, mean z, sx, sz, correlation, weight
  [0.28, 0.32, 0.11, 0.09, 0.5, 1.0],
  [0.62, 0.62, 0.12, 0.1, -0.4, 0.8],
  [0.72, 0.25, 0.07, 0.08, 0.2, 0.5],
];
const LAST = 4;

let svg, cells = [], joint = [], px = [], pz = [], cross = { i: 9, j: 9 }, parts = {}, b, section;

const el = (tag, attrs = {}, parent = svg) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  parent.append(n);
  return n;
};

export function init(sec) {
  section = sec;
  svg = sec.querySelector("svg.prob-stage");
  svg.setAttribute("viewBox", "0 0 640 600");
  computeJoint();
  draw();
  renderMath(sec);
  b = builder({ last: LAST, show });
  const move = (e) => {
    const r = svg.getBoundingClientRect();
    const s = 640 / r.width;
    const gx = ((e.clientX - r.left) * s - J.x) / J.size;
    const gy = ((e.clientY - r.top) * s - J.y) / J.size;
    if (gx < 0 || gx > 1 || gy < 0 || gy > 1) return;
    cross = { i: Math.min(N - 1, Math.floor(gx * N)), j: Math.min(N - 1, Math.floor((1 - gy) * N)) };
    update();
  };
  parts.hit.addEventListener("pointermove", (e) => e.buttons && move(e));
  parts.hit.addEventListener("pointerdown", (e) => {
    parts.hit.setPointerCapture(e.pointerId);
    if (b.shown < 3) b.step(1);
    move(e);
  });
  update();
}

export const enter = (info) => b?.enter(info);
export const step = (dir) => b?.step(dir) ?? false;

function computeJoint() {
  joint = Array.from({ length: N }, () => new Array(N).fill(0)); // joint[i = x][j = z]
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const x = (i + 0.5) / N, z = (j + 0.5) / N;
      for (const [mx, mz, sx, sz, r, w] of BLOBS) {
        const dx = (x - mx) / sx, dz = (z - mz) / sz;
        joint[i][j] += w * Math.exp(-(dx * dx - 2 * r * dx * dz + dz * dz) / (2 * (1 - r * r)));
      }
    }
  }
  const total = joint.flat().reduce((a, v) => a + v, 0);
  for (const row of joint) row.forEach((v, j) => (row[j] = v / total));
  px = joint.map((row) => row.reduce((a, v) => a + v, 0));
  pz = Array.from({ length: N }, (_, j) => joint.reduce((a, row) => a + row[j], 0));
}

function draw() {
  const c = J.size / N;
  const max = Math.max(...joint.flat());
  parts.joint = el("g", { class: "prob-step", "data-from": 0 });
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const t = Math.sqrt(joint[i][j] / max);
      el("rect", { x: J.x + i * c, y: J.y + (N - 1 - j) * c, width: c + 0.5, height: c + 0.5, class: "prob-cell", style: `--t: ${t.toFixed(3)}` }, parts.joint);
    }
  }
  el("rect", { x: J.x, y: J.y, width: J.size, height: J.size, class: "prob-frame" }, parts.joint);
  const t1 = el("text", { x: J.x + J.size / 2, y: J.y + J.size + 34, class: "prob-axis", "text-anchor": "middle" }, parts.joint);
  t1.textContent = "x  (an image, squashed to one number)";
  const t2 = el("text", { x: J.x - 24, y: J.y + J.size / 2, class: "prob-axis", "text-anchor": "middle", transform: `rotate(-90 ${J.x - 24} ${J.y + J.size / 2})` }, parts.joint);
  t2.textContent = "z  (a code)";
  const t3 = el("text", { x: J.x + J.size - 10, y: J.y + 28, class: "prob-name", "text-anchor": "end" }, parts.joint);
  t3.textContent = "p(x, z)";

  // marginals: p(x) above, p(z) to the left
  parts.marg = el("g", { class: "prob-step", "data-from": 1 });
  const mx = Math.max(...px), mz = Math.max(...pz);
  px.forEach((v, i) => el("rect", { x: J.x + i * c + 1, y: J.y - 8 - (v / mx) * 110, width: c - 2, height: (v / mx) * 110, class: "prob-px" }, parts.marg));
  pz.forEach((v, j) => el("rect", { x: J.x - 50 - (v / mz) * 100, y: J.y + (N - 1 - j) * c + 1, width: (v / mz) * 100, height: c - 2, class: "prob-pz" }, parts.marg));
  const a = el("text", { x: J.x + J.size + 14, y: J.y - 60, class: "prob-name px" }, parts.marg);
  a.textContent = "p(x)";
  const bz = el("text", { x: J.x - 60, y: J.y + J.size + 34, class: "prob-name pz", "text-anchor": "end" }, parts.marg);
  bz.textContent = "p(z)";

  // slices
  parts.hline = el("g", { class: "prob-step", "data-from": 2 });
  parts.hrow = el("rect", { x: J.x, width: J.size, height: c, class: "prob-row" }, parts.hline);
  parts.hbars = el("g", {}, parts.hline);
  parts.vline = el("g", { class: "prob-step", "data-from": 3 });
  parts.vcol = el("rect", { y: J.y, width: c, height: J.size, class: "prob-col" }, parts.vline);
  parts.vbars = el("g", {}, parts.vline);
  parts.hit = el("rect", { x: J.x, y: J.y, width: J.size, height: J.size, class: "prob-hit interactive" });
}

function update() {
  const c = J.size / N;
  const { i, j } = cross;
  const row = joint.map((col) => col[j]);           // p(x, z0) over x
  const col = joint[i];                             // p(x0, z) over z
  const cond = (arr) => {
    const s = arr.reduce((a, v) => a + v, 0) || 1;
    return arr.map((v) => v / s);
  };
  const pxz = cond(row), pzx = cond(col);
  const m1 = Math.max(...pxz), m2 = Math.max(...pzx);
  const y0 = J.y + (N - 1 - j) * c;
  parts.hrow.setAttribute("y", y0);
  parts.hbars.replaceChildren();
  pxz.forEach((v, k) => el("rect", { x: J.x + k * c + 1, y: y0 - (v / m1) * 90, width: c - 2, height: (v / m1) * 90, class: "prob-cond-x" }, parts.hbars));
  const x0 = J.x + i * c;
  parts.vcol.setAttribute("x", x0);
  parts.vbars.replaceChildren();
  pzx.forEach((v, k) => el("rect", { x: x0 + c, y: J.y + (N - 1 - k) * c + 1, width: (v / m2) * 90, height: c - 2, class: "prob-cond-z" }, parts.vbars));
}

function show(n) {
  for (const g of svg.querySelectorAll(".prob-step")) g.classList.toggle("on", n >= Number(g.dataset.from));
  section.querySelectorAll(".prob-notes [data-step]").forEach((p) => p.classList.toggle("on", n >= Number(p.dataset.step)));
}
