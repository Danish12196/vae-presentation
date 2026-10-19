// PCA vs autoencoder at the same k: the same four cats, rebuilt from k numbers by PCA
// (precomputed, site/data/pca_k*.png) and by ae_k (live). Lower error is highlighted.
// Four cats can go either way, so the side panel also shows the average over all 64 showcase cats.

import { drawCat, reconstruct, squaredError, inSlot, loadShowcase, getShowcaseCat, loadPCA } from "../models.js";
import { tabs } from "../ui.js";

const CATS = [0, 3, 9, 13];
const KS = [2, 16, 64, 256];

let rows = [];
let request = 0;
let avgEl;

export async function init(section) {
  await loadShowcase();
  avgEl = section.querySelector(".pca-avg");
  const grid = section.querySelector(".pca-grid");
  // Rows are display: contents, so the grid lays out the cells while screen readers get a table.
  grid.innerHTML = `<div class="pca-row" role="row">
    <span role="columnheader" class="pca-head">original</span>
    <span role="columnheader" class="pca-head">PCA</span>
    <span role="columnheader" class="pca-head">autoencoder</span></div>`;
  rows = CATS.map((i) => {
    const cell = (cls) => {
      const fig = document.createElement("figure");
      fig.className = `pca-cell ${cls}`;
      fig.setAttribute("role", "cell");
      fig.innerHTML = `<canvas class="cat-canvas small" width="64" height="64"></canvas><figcaption class="mono"></figcaption>`;
      return fig;
    };
    const original = cell("orig");
    const pca = cell("pca");
    const ae = cell("ae");
    drawCat(original.querySelector("canvas"), getShowcaseCat(i));
    original.querySelector("figcaption").textContent = " "; // keeps the rows aligned
    const row = document.createElement("div");
    row.className = "pca-row";
    row.setAttribute("role", "row");
    row.append(original, pca, ae);
    grid.append(row);
    return { i, pca, ae };
  });

  tabs(section.querySelector(".k-tabs"), {
    tabs: KS.map((k) => ({ id: String(k), label: `k = ${k}` })),
    selected: "16",
    onChange: (k) => update(Number(k)),
  });
  await update(16);
}

async function update(k) {
  const mine = ++request;
  let pca = null;
  try {
    pca = await loadPCA(k);
  } catch {
    /* PCA data not exported: leave those cells empty */
  }
  for (const row of rows) {
    const image = getShowcaseCat(row.i);
    const pcaErr = pca ? pca.errors[row.i] : NaN;
    if (pca && mine === request) {
      drawCat(row.pca.querySelector("canvas"), pca.sprite.get(row.i));
      row.pca.querySelector("figcaption").textContent = pcaErr.toFixed(0);
    }
    const aeErr = await inSlot(row.ae, async () => {
      const out = await reconstruct(`ae_${k}`, [image]);
      if (mine === request) drawCat(row.ae.querySelector("canvas"), out);
      return squaredError(image, out);
    });
    if (mine !== request) return;
    row.ae.querySelector("figcaption").textContent = aeErr == null ? "" : aeErr.toFixed(0);
    row.pca.classList.toggle("better", aeErr != null && pcaErr < aeErr);
    row.ae.classList.toggle("better", aeErr != null && aeErr < pcaErr);
  }
  await showAverage(k, pca, mine);
}

async function showAverage(k, pca, mine) {
  if (!pca) return;
  const cats = Array.from({ length: 64 }, (_, i) => getShowcaseCat(i));
  const pcaMean = pca.errors.reduce((a, b) => a + b, 0) / pca.errors.length;
  const aeMean = await inSlot(avgEl, async () => {
    const out = await reconstruct(`ae_${k}`, cats);
    return cats.reduce((sum, cat, i) => sum + squaredError(cat, out, i), 0) / cats.length;
  });
  if (mine !== request || aeMean == null) return;
  avgEl.textContent = `average of 64 cats: PCA ${Math.round(pcaMean)} · AE ${Math.round(aeMean)}`;
}
