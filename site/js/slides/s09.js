// Other kinds of autoencoders, one tab each, all on the same picked cat.
//   Sparse:        the 1024-neuron code as a 32x32 grid (precomputed: the sparse models are too big
//                  to ship), with the L1 penalty on (sparse_l1) or off (sparse_none).
//   Denoising:     noisy cat -> dae_128 (trained on noisy inputs) vs ae_128 (never saw noise), live.
//   Contractive:   jiggle the input a little, ~4 times a second, and measure how far the code moves
//                  for ae_64 and cae_64, each divided by that model's code spread so the bars compare fairly.
//   Convolutional: the architecture we have used all along.

import {
  drawCat, encode, decode, reconstruct, inSlot, renderTraining, loadShowcase, getShowcaseCat,
  loadManifest, loadJSON, loadSprite, toTensor, randn, ort, SIZE,
} from "../models.js";
import { catPicker, tabs, toggle, slider, readout } from "../ui.js";
import { netDiagram } from "../netdiagram.js";

const PIXELS = 3 * SIZE * SIZE;
const JIGGLE_EVERY = 250; // ms
const BAR_MAX = 0.5;      // code movement / code spread that fills a bar

/** base + scale * noise, each value clamped into [0, 1], as one image tensor. */
function noisyImage(base, scale, noise) {
  const noisy = new Float32Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) noisy[i] = Math.min(1, Math.max(0, base[i] + scale * noise[i]));
  return new ort.Tensor("float32", noisy, [1, 3, SIZE, SIZE]);
}

let section, cat = 0, tab = "sparse", active = false;
let manifest;

export async function init(el) {
  section = el;
  manifest = (await loadManifest()).models;
  await loadShowcase();

  tabs(el.querySelector(".s09-tabs"), {
    tabs: [
      { id: "sparse", label: "Sparse" },
      { id: "denoise", label: "Denoising" },
      { id: "contract", label: "Contractive" },
      { id: "conv", label: "Convolutional" },
    ],
    onChange: (id) => {
      tab = id;
      for (const p of el.querySelectorAll(".s09-panel")) p.hidden = p.dataset.tab !== id;
      refresh();
    },
  });
  await catPicker(el.querySelector(".s09-picker"), {
    onPick: (i) => {
      cat = i;
      refresh();
    },
  });

  await Promise.all([sparse.init(), denoise.init(), contract.init(), conv.init()]);
  refresh();
}

export function enter() {
  active = true;
  refresh();
}

export function leave() {
  active = false;
  contract.stop();
}

function refresh() {
  contract.stop();
  ({ sparse, denoise, contract, conv })[tab].show();
}

// ================================================================ sparse

const sparse = {
  data: null,
  penalty: null,
  scale: {},

  async init() {
    const panel = section.querySelector('[data-tab="sparse"]');
    try {
      this.data = await loadJSON("data/sparse_codes.json");
      this.recon = {};
      for (const name of ["sparse_l1", "sparse_none"]) {
        this.recon[name] = await loadSprite(`data/${this.data.recon[name]}`);
        // Brightness scale: the 99th percentile of this model's positive activations.
        const values = this.data[name].codes.flat().filter((v) => v > 0).sort((a, b) => a - b);
        this.scale[name] = values[Math.floor(values.length * 0.99)] || 1;
      }
    } catch {
      renderTraining(panel.querySelector(".s09-row"), "sparse_l1");
      return;
    }
    this.penalty = toggle(panel.querySelector(".sp-toggle"), { label: "L1 penalty", checked: true, onChange: () => this.show() });
    const box = panel.querySelector(".sp-readouts");
    this.onReadout = readout(box, { label: "neurons on (this cat)", format: (v) => `${v.toFixed(1)}%` });
    const avg = document.createElement("p");
    avg.className = "sp-avg";
    const pct = (name) => `${((manifest[name]?.active_fraction ?? 0) * 100).toFixed(1)}%`;
    avg.textContent = `average over the data: ${pct("sparse_l1")} on with penalty, ${pct("sparse_none")} without`;
    box.append(avg);
    this.canvas = panel.querySelector(".sp-grid");
    this.panel = panel;
  },

  show() {
    if (!this.data) return;
    const name = this.penalty.checked ? "sparse_l1" : "sparse_none";
    const codes = this.data[name].codes[cat];
    drawGrid(this.canvas, codes, this.scale[name]);
    this.onReadout.set(this.data[name].active[cat] * 100);
    drawCat(this.panel.querySelector(".sp-input"), getShowcaseCat(cat));
    drawCat(this.panel.querySelector(".sp-on"), this.recon.sparse_l1.get(cat));
    drawCat(this.panel.querySelector(".sp-off"), this.recon.sparse_none.get(cat));
    this.panel.querySelector(".sp-on").closest("figure").classList.toggle("current", name === "sparse_l1");
    this.panel.querySelector(".sp-off").closest("figure").classList.toggle("current", name === "sparse_none");
  },
};

/** 1024 activations as a 32x32 grid of cells: dark = off, bright mint = strongly on. */
function drawGrid(canvas, codes, scale) {
  const ctx = canvas.getContext("2d");
  const cell = canvas.width / 32;
  ctx.fillStyle = "#0e0c20";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  codes.forEach((v, i) => {
    const x = (i % 32) * cell;
    const y = Math.floor(i / 32) * cell;
    if (v > 0) {
      const t = Math.min(1, v / scale);
      ctx.fillStyle = `rgba(6, 214, 160, ${0.25 + 0.75 * t})`; // mint
    } else {
      ctx.fillStyle = "#231d47";
    }
    ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
  });
}

// ================================================================ denoising

const denoise = {
  noise: null,
  noiseCat: -1,
  request: 0,

  async init() {
    const panel = section.querySelector('[data-tab="denoise"]');
    this.panel = panel;
    this.sigma = slider(panel.querySelector(".dn-slider"), {
      label: "noise σ",
      min: 0,
      max: 0.5,
      step: 0.01,
      value: 0.3, // from about 0.25 up, the denoising AE clearly wins
      format: (v) => v.toFixed(2),
      onInput: () => this.show(),
    });
  },

  async show() {
    const mine = ++this.request;
    // One fixed noise pattern per cat, so the slider only changes its strength.
    if (this.noiseCat !== cat) {
      this.noise = randn(PIXELS);
      this.noiseCat = cat;
    }
    const x = toTensor(getShowcaseCat(cat)).data;
    const input = noisyImage(x, this.sigma.value, this.noise);
    drawCat(this.panel.querySelector(".dn-noisy"), input);
    for (const [cls, model] of [[".dn-dae", "dae_128"], [".dn-ae", "ae_128"]]) {
      const fig = this.panel.querySelector(cls);
      await inSlot(fig, async () => {
        const out = await reconstruct(model, input);
        if (mine === this.request) drawCat(fig.querySelector("canvas"), out);
      });
    }
  },
};

// ================================================================ contractive

const contract = {
  timer: 0,
  gen: 0, // only the newest update loop may schedule the next jiggle
  base: {},
  baseCat: -1,

  async init() {
    const panel = section.querySelector('[data-tab="contract"]');
    this.panel = panel;
    this.level = slider(panel.querySelector(".ct-slider"), {
      label: "jiggle",
      min: 0,
      max: 0.1,
      step: 0.005,
      value: 0.05,
      format: (v) => (v === 0 ? "still" : v.toFixed(3)),
      onInput: () => this.show(),
    });
    const bars = panel.querySelector(".ct-bars");
    bars.innerHTML = `<p class="ct-title">how far z moves</p>`;
    this.rows = {};
    for (const [model, label] of [["ae_64", "plain AE"], ["cae_64", "contractive AE"]]) {
      const row = document.createElement("div");
      row.className = "ct-row";
      row.innerHTML = `<span class="ct-label">${label}</span><span class="ct-track"><span class="ct-fill"></span></span><span class="ct-value mono">–</span>`;
      bars.append(row);
      this.rows[model] = row;
    }
  },

  /** Average size of one code dimension's natural variation, so moves are comparable across models. */
  spread(model) {
    const std = manifest[model]?.code_std;
    return std ? std.reduce((a, b) => a + b, 0) / std.length : 1;
  },

  stop() {
    clearTimeout(this.timer);
    this.timer = 0;
    this.gen++;
  },

  async show() {
    this.stop();
    const gen = ++this.gen;
    const image = getShowcaseCat(cat);
    const x = toTensor(image);
    if (this.baseCat !== cat) {
      this.base = {};
      this.baseCat = cat;
    }
    const level = this.level.value;
    const jiggled = noisyImage(x.data, level, randn(PIXELS));
    drawCat(this.panel.querySelector(".ct-input"), jiggled);

    for (const [model, cls] of [["ae_64", ".ct-ae"], ["cae_64", ".ct-cae"]]) {
      const fig = this.panel.querySelector(cls);
      const moved = await inSlot(fig, async () => {
        this.base[model] ??= (await encode(model, x)).z;
        const z = (await encode(model, jiggled)).z;
        drawCat(fig.querySelector("canvas"), await decode(model, z));
        let d = 0;
        for (let i = 0; i < z.data.length; i++) d += (z.data[i] - this.base[model].data[i]) ** 2;
        return Math.sqrt(d) / this.spread(model);
      });
      const row = this.rows[model];
      row.querySelector(".ct-fill").style.width = moved == null ? "0%" : `${Math.min(100, (moved / BAR_MAX) * 100)}%`;
      row.querySelector(".ct-value").textContent = moved == null ? "–" : moved.toFixed(2);
    }
    // Keep jiggling while the slide is showing this tab and the slider is above zero.
    if (gen === this.gen && active && tab === "contract" && level > 0) {
      this.timer = setTimeout(() => this.show(), JIGGLE_EVERY);
    }
  },
};

// ================================================================ convolutional

const conv = {
  async init() {
    this.net = netDiagram(section.querySelector(".cv-stage"), { ending: "decoder", note: false });
    this.net.showAll();
  },

  async show() {
    const image = getShowcaseCat(cat);
    drawCat(this.net.input, image);
    await inSlot(this.net.output.parentElement, async () => drawCat(this.net.output, await reconstruct("dae_128", [image])));
  },
};
