// Two worlds (Deepia's p(x) <-> p(z) picture, made live with vae_2d_b1):
//   left, p(x): the data, a lumpy shape filled with real training-cat photos
//   right, p(z) = N(0, I): the latent world, a round glow we choose ourselves
//   the decoder p(x | z) maps a code to a cat: tap anywhere in p(z) and that exact z is decoded
//   the encoder q(z | x) ≈ p(z | x) maps a cat to a small cloud: pick a cat and its cloud appears
// Built press by press: the two worlds, the decoder arrow (and a sample), the encoder arrow.

import { encode, decode, drawCat, hasModel, renderTraining, loadShowcase, getShowcaseCat, loadJSON, loadSprite, randn } from "../models.js";
import { catPicker } from "../ui.js";
import { tween, lerp, builder } from "../anim.js";
import { renderMath } from "../math.js";

const MODEL = "vae_2d_b1";
const NS = "http://www.w3.org/2000/svg";
const Z = { cx: 900, cy: 255, r: 185, span: 3 };   // p(z): radius = 3 standard deviations
const X = { cx: 290, cy: 255 };
const SLOTS = [[205, 190], [330, 300], [250, 330], [345, 175]]; // where new cats land in p(x)
const LAST = 2;

let svg, stage, b, slot = 0, cloud = null, cats = [];

const el = (tag, attrs = {}, parent = svg) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  parent.append(n);
  return n;
};

export async function init(section) {
  const body = section.querySelector(".worlds-body");
  if (!(await hasModel(MODEL))) return renderTraining(body, MODEL);
  stage = section.querySelector(".worlds-stage");
  svg = section.querySelector("svg.worlds-svg");
  svg.setAttribute("viewBox", "0 0 1200 510");
  await build();
  renderMath(section);
  await loadShowcase();
  await catPicker(section.querySelector(".worlds-picker"), { onPick: (i) => encodeCat(i), selected: -1 });
  b = builder({ last: LAST, show });
  section.querySelector("[data-sample]").addEventListener("click", () => sampleAt(randn(2)));
}

export const enter = (info) => b?.enter(info);
export const step = (dir) => b?.step(dir) ?? false;

async function build() {
  const defs = el("defs");
  // p(x): a mosaic of real training-cat photos, clipped to a lumpy blob
  const maps = await loadJSON("data/latent_maps.json");
  const thumbs = await loadSprite(`data/${maps.thumbs.file}`, maps.thumbs);
  const mosaic = document.createElement("canvas");
  mosaic.width = 16 * 32;
  mosaic.height = 14 * 32;
  const mctx = mosaic.getContext("2d");
  for (let i = 0; i < 16 * 14; i++) mctx.putImageData(thumbs.get(i * 7), (i % 16) * 32, Math.floor(i / 16) * 32);
  const blob = "M 160 120 C 210 40 300 120 330 80 C 380 20 470 90 430 160 C 400 210 480 240 460 300 C 440 380 360 330 330 400 C 300 460 210 440 220 380 C 230 330 120 360 110 290 C 100 230 170 220 150 180 C 140 150 140 140 160 120 Z";
  const clip = el("clipPath", { id: "worlds-blob" }, defs);
  el("path", { d: blob }, clip);
  el("image", { href: mosaic.toDataURL(), x: 80, y: 20, width: 16 * 32, height: 14 * 32, "clip-path": "url(#worlds-blob)", style: "image-rendering: pixelated", opacity: 0.9 });
  el("path", { d: blob, class: "worlds-blob" });

  // p(z): a radial glow, with faint axes
  const g = el("radialGradient", { id: "worlds-glow" }, defs);
  el("stop", { offset: "0%", "stop-color": "var(--sky)", "stop-opacity": 0.85 }, g);
  el("stop", { offset: "55%", "stop-color": "var(--sky)", "stop-opacity": 0.35 }, g);
  el("stop", { offset: "100%", "stop-color": "var(--sky)", "stop-opacity": 0 }, g);
  el("circle", { cx: Z.cx, cy: Z.cy, r: Z.r, fill: "url(#worlds-glow)" });
  el("line", { x1: Z.cx - Z.r, y1: Z.cy, x2: Z.cx + Z.r, y2: Z.cy, class: "worlds-axis" });
  el("line", { x1: Z.cx, y1: Z.cy - Z.r, x2: Z.cx, y2: Z.cy + Z.r, class: "worlds-axis" });
  const hit = el("circle", { cx: Z.cx, cy: Z.cy, r: Z.r, class: "worlds-hit interactive" });
  hit.addEventListener("pointerdown", (e) => {
    const r = svg.getBoundingClientRect();
    const s = 1200 / r.width;
    const x = (e.clientX - r.left) * s, y = (e.clientY - r.top) * s;
    sampleAt([((x - Z.cx) / Z.r) * Z.span, (-(y - Z.cy) / Z.r) * Z.span]);
  });
  svg.cloudLayer = el("g");
  svg.dotLayer = el("g");

  // arrows (drawn as steps)
  const m = el("marker", { id: "worlds-head", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
  el("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "worlds-head" }, m);
  svg.dec = el("path", { d: "M 700 330 C 640 420 540 420 480 340", class: "worlds-arrow dec worlds-step", "data-from": 1, "marker-end": "url(#worlds-head)" });
  svg.enc = el("path", { d: "M 480 160 C 540 70 640 70 700 160", class: "worlds-arrow enc worlds-step", "data-from": 2, "marker-end": "url(#worlds-head)" });
}

function show(n) {
  for (const s of svg.querySelectorAll(".worlds-step")) s.classList.toggle("on", n >= Number(s.dataset.from));
  stage.querySelectorAll("[data-from]").forEach((s) => s.classList.toggle("on", n >= Number(s.dataset.from)));
  if (n >= 1 && cats.length === 0) sampleAt(randn(2));
}

const toSvg = ([zx, zy]) => [Z.cx + (zx / Z.span) * Z.r, Z.cy - (zy / Z.span) * Z.r];

/** Decode one code: a dot at z in p(z), travelling along the decoder arrow to a new cat in p(x). */
async function sampleAt(z) {
  if (b.shown < 1) b.step(1);
  const [x, y] = toSvg(z);
  const dot = el("circle", { cx: x, cy: y, r: 9, class: "worlds-dot" }, svg.dotLayer);
  const image = await decode(MODEL, [z[0], z[1]]);
  const path = svg.dec;
  const len = path.getTotalLength();
  await tween(700, (t) => {
    const p = t < 0.15 ? { x: lerp(x, path.getPointAtLength(0).x, t / 0.15), y: lerp(y, path.getPointAtLength(0).y, t / 0.15) } : path.getPointAtLength(((t - 0.15) / 0.85) * len);
    dot.setAttribute("cx", p.x);
    dot.setAttribute("cy", p.y);
  });
  dot.remove();
  const [sx, sy] = SLOTS[slot++ % SLOTS.length];
  const fig = document.createElement("canvas");
  fig.className = "cat-canvas worlds-cat";
  fig.style.left = `${(sx / 1200) * 100}%`;
  fig.style.top = `${(sy / 510) * 100}%`;
  fig.setAttribute("aria-label", `A cat decoded from z = (${z[0].toFixed(2)}, ${z[1].toFixed(2)})`);
  drawCat(fig, image);
  stage.append(fig);
  cats.push(fig);
  if (cats.length > SLOTS.length) cats.shift().remove();
  // leave a small mark where this code was
  el("circle", { cx: x, cy: y, r: 6, class: "worlds-mark" }, svg.dotLayer);
}

/** Encode a showcase cat: its cloud q(z | x) appears in p(z) (sigma drawn to scale, at least 5px). */
async function encodeCat(i) {
  if (b.shown < 2) b.step(1), b.step(1);
  const image = getShowcaseCat(i);
  const { mu, logvar } = await encode(MODEL, [image]);
  const m = [mu.data[0], mu.data[1]];
  const s = [Math.exp(0.5 * logvar.data[0]), Math.exp(0.5 * logvar.data[1])];
  const [x, y] = toSvg(m);
  svg.cloudLayer.replaceChildren();
  const path = svg.enc;
  const len = path.getTotalLength();
  const dot = el("circle", { r: 9, class: "worlds-dot enc" }, svg.cloudLayer);
  await tween(650, (t) => {
    const p = t < 0.85 ? path.getPointAtLength((t / 0.85) * len) : { x: lerp(path.getPointAtLength(len).x, x, (t - 0.85) / 0.15), y: lerp(path.getPointAtLength(len).y, y, (t - 0.85) / 0.15) };
    dot.setAttribute("cx", p.x);
    dot.setAttribute("cy", p.y);
  });
  dot.remove();
  const rx = Math.max(5, (s[0] / Z.span) * Z.r), ry = Math.max(5, (s[1] / Z.span) * Z.r);
  el("ellipse", { cx: x, cy: y, rx: rx * 2, ry: ry * 2, class: "worlds-cloud" }, svg.cloudLayer);
  el("circle", { cx: x, cy: y, r: 4, class: "worlds-mu" }, svg.cloudLayer);
}
