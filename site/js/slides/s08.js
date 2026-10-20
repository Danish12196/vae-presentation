// The bottleneck: squeeze the cat through k numbers and rebuild it (ae_2, ae_16, ae_64, ae_256, live).
// The last stop removes the bottleneck (overcomplete): a perfect copy and a loss of 0, Photocopier Cat.
// Loss is the summed squared error over all 12,288 numbers, the same measure as the reconstruction-loss slide.

import { drawCat, reconstruct, inSlot, loadShowcase, getShowcaseCat, toTensor } from "../models.js";
import { catPicker, slider, readout } from "../ui.js";

const INPUT = 12288;
const STOPS = [
  { k: 2, open: 0.06 },
  { k: 16, open: 0.16 },
  { k: 64, open: 0.3 },
  { k: 256, open: 0.5 },
  { k: null, open: 1 }, // no bottleneck
];
const fmt = (n) => n.toLocaleString("en-US");

let inCanvas, outCanvas, outSlot, diagram, kEl, ratioEl, copier, stamps, picker, stop, loss, kindEl, illusEl;
let cat = 0;
let request = 0;

export async function init(section) {
  [inCanvas, outCanvas] = section.querySelectorAll(".bn-img canvas");
  outSlot = section.querySelector(".bn-out");
  diagram = section.querySelector(".bn-diagram");
  kEl = section.querySelector(".bn-k");
  ratioEl = section.querySelector(".bn-ratio");
  copier = section.querySelector(".bn-copier");
  stamps = section.querySelector(".bn-stamps");
  kindEl = section.querySelector(".bn-kind-tag");
  illusEl = section.querySelector(".bn-illus");
  const lossEl = section.querySelector(".bn-loss-v");
  loss = readout(lossEl.parentElement, { format: (v) => (v === 0 ? "0.000" : v.toFixed(1)) });
  lossEl.replaceWith(loss.el);

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

/** Smooth hourglass: full height at the ends, a neck whose opening follows the slider. */
function drawGlass(diagram, open) {
  const mid = 110;
  const neck = 12 + open * 86;
  const top = mid - neck;
  const bot = mid + neck;
  diagram.querySelector(".bn-enc-shape").setAttribute(
    "d",
    `M 14 14 C 78 14 100 ${top} 128 ${top} L 128 ${bot} C 100 ${bot} 78 206 14 206 Z`,
  );
  diagram.querySelector(".bn-dec-shape").setAttribute(
    "d",
    `M 286 14 C 222 14 200 ${top} 172 ${top} L 172 ${bot} C 200 ${bot} 222 206 286 206 Z`,
  );
}

async function update() {
  const { k, open } = STOPS[stop.value];
  const image = getShowcaseCat(cat);
  const mine = ++request; // the slider can move faster than the model: only the latest result is drawn

  drawCat(inCanvas, image);
  drawGlass(diagram, open);
  kEl.textContent = fmt(k ?? INPUT);
  ratioEl.textContent = k ? `${fmt(INPUT / k)}× smaller` : "no compression: 1×";
  copier.hidden = stamps.hidden = k !== null;
  kindEl.textContent = k ? "undercomplete" : "overcomplete";
  kindEl.classList.toggle("is-over", !k);
  illusEl.hidden = k !== null;

  if (k === null) {
    // Nothing to squeeze through: the network can just copy. A perfect score that means nothing.
    await inSlot(outSlot, async () => drawCat(outCanvas, image));
    loss.set(0);
    return;
  }
  await inSlot(outSlot, async () => {
    const out = await reconstruct(`ae_${k}`, [image]);
    if (mine !== request) return;
    drawCat(outCanvas, out);
    const x = toTensor([image]).data;
    let sq = 0;
    for (let i = 0; i < x.length; i++) sq += (out.data[i] - x[i]) ** 2;
    loss.set(sq);
  });
}
