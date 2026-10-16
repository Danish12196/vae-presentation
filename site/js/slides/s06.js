// Reconstruction loss: input, ae_64's output and the squared error of every pixel.
// MSE and BCE computed live. The only "label" is the input itself.

import { drawCat, reconstruct, inSlot, loadShowcase, getShowcaseCat, toTensor } from "../models.js";
import { catPicker, tabs, toggle, readout } from "../ui.js";
import { drawErrorHeatmap, heatLegend } from "../viz.js";

const FORMULAS = {
  mse: "MSE = <sup>1</sup>&frasl;<sub>n</sub> Σ<sub>i</sub> (x̂<sub>i</sub> − x<sub>i</sub>)²",
  bce: "BCE = −<sup>1</sup>&frasl;<sub>n</sub> Σ<sub>i</sub> [ x<sub>i</sub> log x̂<sub>i</sub> + (1 − x<sub>i</sub>) log(1 − x̂<sub>i</sub>) ]",
};
const EPS = 1e-6;

let canvases, outSlot, formula, counter, readouts, lossTabs, showFormula;
let cat = 0;
let request = 0;

export async function init(section) {
  canvases = section.querySelectorAll(".loss-row canvas");
  outSlot = section.querySelector(".loss-out");
  formula = section.querySelector(".formula");
  counter = section.querySelector(".labels-count");
  heatLegend(section.querySelector(".legend"));

  const box = section.querySelector(".loss-readouts");
  readouts = {
    mse: readout(box, { label: "MSE (mean)", format: (v) => v.toFixed(4) }),
    sum: readout(box, { label: "MSE (summed)", format: (v) => v.toFixed(1) }),
    bce: readout(box, { label: "BCE (mean)", format: (v) => v.toFixed(4) }),
  };
  readouts.mse.el.dataset.loss = readouts.sum.el.dataset.loss = "mse";
  readouts.bce.el.dataset.loss = "bce";

  lossTabs = tabs(section.querySelector(".loss-tabs"), {
    tabs: [
      { id: "mse", label: "MSE" },
      { id: "bce", label: "BCE" },
    ],
    onChange: showSelected,
  });
  showFormula = toggle(section.querySelector(".formula-toggle"), { label: "show formula", onChange: showSelected });

  await loadShowcase();
  await catPicker(section.querySelector(".s06-picker"), {
    onPick: (i) => {
      cat = i;
      update();
    },
  });
  showSelected();
  await update();
}

function showSelected() {
  const loss = lossTabs.value;
  for (const r of Object.values(readouts)) r.el.classList.toggle("dim", r.el.dataset.loss !== loss);
  formula.innerHTML = FORMULAS[loss];
  formula.hidden = !showFormula.checked;
}

async function update() {
  const image = getShowcaseCat(cat);
  const mine = ++request;
  const [input, output, heat] = canvases;
  drawCat(input, image);

  // Same counter, every time: nobody had to label this cat.
  counter.classList.remove("pulse");
  void counter.offsetWidth;
  counter.classList.add("pulse");

  await inSlot(outSlot, async () => {
    const out = await reconstruct("ae_64", [image]);
    if (mine !== request) return;
    drawCat(output, out);
    drawErrorHeatmap(heat, image, out);

    const x = toTensor(image).data;
    const y = out.data;
    let sq = 0;
    let bce = 0;
    for (let i = 0; i < x.length; i++) {
      const p = Math.min(1 - EPS, Math.max(EPS, y[i]));
      sq += (y[i] - x[i]) ** 2;
      bce -= x[i] * Math.log(p) + (1 - x[i]) * Math.log(1 - p);
    }
    readouts.mse.set(sq / x.length);
    readouts.sum.set(sq);
    readouts.bce.set(bce / x.length);
  });
}
