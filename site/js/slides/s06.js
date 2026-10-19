// The bottleneck: squeeze the cat through k numbers and rebuild it (ae_2, ae_16, ae_64, ae_256, live).
// The last stop removes the bottleneck: a perfect copy, and Photocopier Cat is back.

import { drawCat, reconstruct, inSlot, loadShowcase, getShowcaseCat } from "../models.js";
import { catPicker, slider } from "../ui.js";

const INPUT = 12288;
const STOPS = [
  { k: 2, open: 0.06 },
  { k: 16, open: 0.16 },
  { k: 64, open: 0.3 },
  { k: 256, open: 0.5 },
  { k: null, open: 1 }, // no bottleneck
];
const fmt = (n) => n.toLocaleString("en-US");

let inCanvas, outCanvas, outSlot, diagram, kEl, ratioEl, copier, picker, stop;
let cat = 0;
let request = 0;

export async function init(section) {
  [inCanvas, outCanvas] = section.querySelectorAll(".bn-img canvas");
  outSlot = section.querySelector(".bn-out");
  diagram = section.querySelector(".bn-diagram");
  kEl = section.querySelector(".bn-k");
  ratioEl = section.querySelector(".bn-ratio");
  copier = section.querySelector(".bn-copier");

  await loadShowcase();
  stop = slider(section.querySelector(".bn-slider"), {
    label: "bottleneck size",
    min: 0,
    max: STOPS.length - 1,
    step: 1,
    value: 0,
    format: (i) => (STOPS[i].k ? `${STOPS[i].k} numbers` : "no bottleneck"),
    ticks: ["2", "16", "64", "256", "none"],
    onInput: update,
  });
  picker = await catPicker(section.querySelector(".bn-picker"), {
    onPick: (i) => {
      cat = i;
      update();
    },
  });
  await update();
}

async function update() {
  const { k, open } = STOPS[stop.value];
  const image = getShowcaseCat(cat);
  const mine = ++request; // the slider can move faster than the model: only the latest result is drawn

  drawCat(inCanvas, image);
  diagram.style.setProperty("--open", open);
  kEl.textContent = fmt(k ?? INPUT);
  ratioEl.textContent = k ? `${fmt(INPUT / k)}× smaller` : "no compression: 1×";
  copier.hidden = k !== null;

  if (k === null) {
    // Nothing to squeeze through: the network can just copy.
    await inSlot(outSlot, async () => drawCat(outCanvas, image));
    return;
  }
  await inSlot(outSlot, async () => {
    const out = await reconstruct(`ae_${k}`, [image]);
    if (mine === request) drawCat(outCanvas, out);
  });
}
