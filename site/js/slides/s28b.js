// GAN as an alternative? A forger (generator) and a detective (discriminator), drawn as a diagram,
// beside nine big samples from Gary, the real GAN trained on our cats (the same precomputed pool #s30
// uses, data/compare_pool.json). "new samples" draws nine more and swaps the fake/real pair in the
// diagram. Without the pool: the still-training card, and the diagram keeps empty frames.

import { loadJSON, loadSprite, loadShowcase, getShowcaseCat, renderTraining } from "../models.js";

const GAP = 4;
const NS = "http://www.w3.org/2000/svg";

let svg, grid, sprite, part, showcaseCount = 0;

export async function init(section) {
  svg = section.querySelector("svg.gan-diagram");
  drawDiagram();

  const meta = await loadJSON("data/compare_pool.json").catch(() => null);
  if (!meta?.parts?.gan) return renderTraining(section.querySelector(".gan-right"), "gan");
  sprite = await loadSprite(`data/${meta.file}`, meta);
  part = meta.parts.gan; // [start, count]
  showcaseCount = (await loadShowcase()).count;
  grid = section.querySelector(".gan-grid");
  section.querySelector("[data-gan]").addEventListener("click", draw);
  draw();
}

function el(name, attrs, text) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text) node.textContent = text;
  return node;
}

/** A square frame with the deck's hard offset shadow. */
function frame(x, y, w, h, cls) {
  const g = el("g", { class: `gan-box ${cls}` });
  g.append(el("rect", { class: "gan-shadow", x: x + 5, y: y + 5, width: w, height: h }));
  g.append(el("rect", { class: "gan-face", x, y, width: w, height: h }));
  return g;
}

function arrow(d, cls = "") {
  return el("path", { class: `gan-arrow ${cls}`, d, "marker-end": `url(#gan-tip${cls ? "-" + cls : ""})` });
}

function drawDiagram() {
  svg.replaceChildren();
  const defs = el("defs", {});
  for (const [id, cls] of [["gan-tip", "gan-tip"], ["gan-tip-back", "gan-tip back"]]) {
    const m = el("marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
    m.append(el("path", { class: cls, d: "M0 0 L10 5 L0 10 Z" }));
    defs.append(m);
  }
  svg.append(defs);

  // Top row: noise -> generator -> a fake cat.
  svg.append(frame(10, 60, 100, 60, "noise"), el("text", { class: "gan-label", x: 60, y: 96 }, "noise"));
  svg.append(arrow("M112 90 H146"));
  svg.append(frame(150, 50, 140, 80, "gen"), el("text", { class: "gan-name", x: 220, y: 96 }, "generator"));
  svg.append(arrow("M292 90 H326"));
  svg.append(frame(330, 50, 80, 80, "img"), el("image", { class: "gan-fake", x: 330, y: 50, width: 80, height: 80, preserveAspectRatio: "none" }));
  svg.append(el("text", { class: "gan-label small", x: 370, y: 150 }, "fake"));

  // Bottom: a real cat.
  svg.append(frame(330, 196, 80, 80, "img"), el("image", { class: "gan-real", x: 330, y: 196, width: 80, height: 80, preserveAspectRatio: "none" }));
  svg.append(el("text", { class: "gan-label small", x: 370, y: 296 }, "real cat"));

  // Both go to the discriminator, which answers real or fake.
  svg.append(arrow("M412 92 L466 150"), arrow("M412 236 L466 190"));
  svg.append(frame(470, 130, 160, 80, "disc"), el("text", { class: "gan-name", x: 550, y: 176 }, "discriminator"));
  svg.append(arrow("M550 216 V246"));
  svg.append(el("text", { class: "gan-verdict", x: 550, y: 272 }, "real or fake?"));

  // The feedback loop: the detective's verdict trains the forger.
  svg.append(arrow("M550 128 V20 H220 V46", "back"));
  svg.append(el("text", { class: "gan-back-label", x: 385, y: 14 }, "fool me better"));
}

function dataURL(image) {
  const c = document.createElement("canvas");
  c.width = image.width;
  c.height = image.height;
  c.getContext("2d").putImageData(image, 0, 0);
  return c.toDataURL();
}

function draw() {
  const [start, n] = part;
  const picks = new Set();
  while (picks.size < 9) picks.add(start + Math.floor(Math.random() * n));
  const ctx = grid.getContext("2d");
  ctx.clearRect(0, 0, grid.width, grid.height);
  [...picks].forEach((idx, i) => ctx.putImageData(sprite.get(idx), GAP + (i % 3) * (64 + GAP), GAP + Math.floor(i / 3) * (64 + GAP)));
  // The diagram's pair: one of these fakes, and a real cat.
  svg.querySelector(".gan-fake").setAttribute("href", dataURL(sprite.get([...picks][0])));
  svg.querySelector(".gan-real").setAttribute("href", dataURL(getShowcaseCat(Math.floor(Math.random() * showcaseCount))));
}
