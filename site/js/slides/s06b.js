// "Every face is a point" (inspired by Deepia's VAE video): describe a face with two numbers,
// put everyone on a chart, walk along one axis, then take the axis names away.
// The faces come from a hand-written SVG "decoder" (js/faces.js), an illustration, not a model.
// Presses: 0 one face and its two numbers · 1 the crowd · 2 walk one axis · 3 the axes lose their names.
// Drag (or tap) anywhere on the chart to redraw the big face from that point.

import { builder, tween, reducedMotion } from "../anim.js";
import { faceSVG, seeded } from "../faces.js";

const NS = "http://www.w3.org/2000/svg";
const LAST = 3;
const W = 600, PAD = 56, SPAN = W - PAD * 2;
const CROWD = 26;
const WALK = 6;
const LOOK = 2; // the colours of "you", the big face

let chart, bigFace, readS, readH, walkEl, walkLine, you, labels, build, note;
let point = { s: 0.8, h: 0.3 };

const px = (s) => PAD + s * SPAN;
const py = (h) => W - PAD - h * SPAN;

function el(name, attrs = {}, parent) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent?.appendChild(e);
  return e;
}

export function init(section) {
  chart = section.querySelector(".faces-chart");
  bigFace = section.querySelector(".faces-big");
  readS = section.querySelector("[data-read=s]");
  readH = section.querySelector("[data-read=h]");
  walkEl = section.querySelector(".faces-walk");
  note = section.querySelector(".faces-note");

  chart.setAttribute("viewBox", `0 0 ${W} ${W}`);
  // Axes.
  const axes = el("g", { class: "faces-axes" }, chart);
  el("line", { x1: PAD, y1: W - PAD, x2: W - PAD + 14, y2: W - PAD }, axes);
  el("line", { x1: PAD, y1: W - PAD, x2: PAD, y2: PAD - 14 }, axes);
  el("path", { d: `M ${W - PAD + 22} ${W - PAD} l -14 -7 v 14 z`, class: "faces-arrow" }, axes);
  el("path", { d: `M ${PAD} ${PAD - 22} l -7 14 h 14 z`, class: "faces-arrow" }, axes);
  labels = {
    x: el("text", { x: W - PAD, y: W - PAD + 38, "text-anchor": "end", class: "faces-axis-label" }, axes),
    y: el("text", { x: PAD - 18, y: PAD + 4, "text-anchor": "end", class: "faces-axis-label", transform: `rotate(-90 ${PAD - 18} ${PAD + 4})` }, axes),
  };
  setAxisNames(false);

  // The crowd: everyone in the room, as dots with faces.
  const rand = seeded(7);
  const crowd = el("g", { class: "faces-crowd" }, chart);
  for (let i = 0; i < CROWD; i++) {
    const s = 0.06 + rand() * 0.88;
    const h = 0.06 + rand() * 0.88;
    const g = el("g", { transform: `translate(${px(s) - 26} ${py(h) - 26}) scale(0.52)` }, crowd);
    g.innerHTML = faceSVG(s, h, i, { detail: false });
  }
  walkLine = el("line", { class: "faces-walkline" }, chart);
  you = el("g", { class: "faces-you" }, chart);
  el("circle", { r: 13 }, you);
  el("text", { y: -20, "text-anchor": "middle" }, you).textContent = "you";

  // Drag on the chart to move "you".
  const move = (e) => {
    const r = chart.getBoundingClientRect();
    const s = ((e.clientX - r.left) / r.width * W - PAD) / SPAN;
    const h = (W - PAD - (e.clientY - r.top) / r.height * W) / SPAN;
    point = { s: Math.min(1, Math.max(0, s)), h: Math.min(1, Math.max(0, h)) };
    draw();
  };
  chart.addEventListener("pointerdown", (e) => {
    chart.setPointerCapture(e.pointerId);
    move(e);
  });
  chart.addEventListener("pointermove", (e) => {
    if (e.buttons & 1) move(e);
  });

  for (let i = 0; i < WALK; i++) walkEl.insertAdjacentHTML("beforeend", `<svg viewBox="0 0 100 100" aria-hidden="true"></svg>`);

  build = builder({ last: LAST, show, stage: section.querySelector(".faces-body") });
  draw();
  show(LAST, { animate: false });
}

function setAxisNames(hidden) {
  labels.x.textContent = hidden ? "z₁ = ?" : "smile →";
  labels.y.textContent = hidden ? "z₂ = ?" : "hair length →";
}

function draw() {
  bigFace.innerHTML = faceSVG(point.s, point.h, LOOK);
  readS.textContent = point.s.toFixed(2);
  readH.textContent = point.h.toFixed(2);
  you.setAttribute("transform", `translate(${px(point.s)} ${py(point.h)})`);
  walkLine.setAttribute("x1", px(0));
  walkLine.setAttribute("x2", px(1));
  walkLine.setAttribute("y1", py(point.h));
  walkLine.setAttribute("y2", py(point.h));
  [...walkEl.children].forEach((svg, i) => (svg.innerHTML = faceSVG(i / (WALK - 1), point.h, LOOK)));
}

function show(n, { animate }) {
  const root = chart.closest(".slide");
  root.dataset.step = n;
  setAxisNames(n >= 3);
  // At the walk step, slide "you" along the smile axis once, so the strip reads as a walk.
  if (n === 2 && animate && !reducedMotion()) {
    const h = point.h;
    tween(1400, (t) => {
      point = { s: t, h };
      draw();
    });
  }
}

export const enter = (info) => build.enter(info);
export const step = (dir) => build.step(dir);
