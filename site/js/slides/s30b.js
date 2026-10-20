// Measure it, then use it well: PSNR of each autoencoder on the 64 held-out showcase cats,
// computed live in the browser (so it stays right when the models are retrained),
// next to where autoencoders are used and what to watch out for.

import { loadShowcase, getShowcaseCat, reconstruct, squaredError } from "../models.js";

const MODELS = [
  { name: "ae_2", label: "AE · 2 numbers" },
  { name: "ae_16", label: "AE · 16" },
  { name: "ae_64", label: "AE · 64" },
  { name: "ae_256", label: "AE · 256" },
  { name: "vae_128", label: "VAE · 128" },
];
const PIXELS = 64 * 64 * 3;
const MAX_DB = 30;

let bars, measured = false;

export function init(section) {
  bars = section.querySelector(".eval-bars");
  bars.innerHTML = MODELS.map(
    (m) => `<div class="eval-row" data-model="${m.name}"><span class="eval-name">${m.label}</span><span class="eval-track"><span class="eval-fill"></span></span><span class="eval-db mono">…</span></div>`,
  ).join("");
}

export async function enter() {
  if (measured) return;
  measured = true;
  const { count } = await loadShowcase();
  const cats = Array.from({ length: count ?? 64 }, (_, i) => getShowcaseCat(i));
  for (const m of MODELS) {
    const row = bars.querySelector(`[data-model="${m.name}"]`);
    try {
      const recon = await reconstruct(m.name, cats);
      let psnr = 0;
      cats.forEach((cat, i) => {
        const mse = squaredError(cat, recon, i) / PIXELS;
        psnr += 10 * Math.log10(1 / Math.max(mse, 1e-10));
      });
      psnr /= cats.length;
      row.querySelector(".eval-db").textContent = `${psnr.toFixed(1)} dB`;
      row.querySelector(".eval-fill").style.width = `${Math.min(100, (psnr / MAX_DB) * 100)}%`;
    } catch (err) {
      console.warn("s30b:", m.name, err);
      row.querySelector(".eval-db").textContent = "still training";
    }
  }
}
