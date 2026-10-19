// VAE vs GAN vs diffusion.
//   The comparison table: every cell starts hidden and appears when clicked (a row header shows its row).
//   The trilemma: pick a model and its two corners of quality / variety / speed light up.
//   Meet Gary, a mode-collapsed GAN (an illustration: his cats are drawn by our VAE from one fixed
//   code, with tiny jitter), next to nine genuinely different vae_128 samples.

import { decode, drawTiles, inSlot, randn, loadJSON, loadSprite } from "../models.js";
import { tabs } from "../ui.js";

const ROWS = [
  ["Image quality", ["Blurry", false], ["Sharp", true], ["Sharp", true]],
  ["Variety of outputs", ["High", true], ["Can suffer mode collapse", false], ["High", true]],
  ["Generation speed", ["Fast, one pass", true], ["Fast, one pass", true], ["Slow, many steps", false]],
  ["Training", ["Stable", true], ["Unstable", false], ["Stable", true]],
  ["Encoder for real images", ["Yes", true], ["No", false], ["No (noising instead)", false]],
];
const CORNERS = { quality: [160, 46], variety: [36, 230], speed: [284, 230] };
const GETS = { vae: ["variety", "speed"], gan: ["quality", "speed"], diffusion: ["quality", "variety"] };
const WHO = { quality: "GAN · diffusion", variety: "VAE · diffusion", speed: "VAE · GAN" };
const LATENT = 128, GAP = 4;

let tri, grids;

export async function init(section) {
  buildTable(section.querySelector(".cmp-table tbody"));

  tri = section.querySelector("svg.tri");
  buildTriangle();
  tabs(section.querySelector(".tri-models"), {
    tabs: [
      { id: "vae", label: "VAE" },
      { id: "gan", label: "GAN" },
      { id: "diffusion", label: "Diffusion" },
    ],
    onChange: highlight,
  });
  highlight("vae");

  // Real samples from a VAE, a GAN and a diffusion model, all trained on the same cats
  // (training/train_v2.py, precomputed in data/compare_pool.json). Without them: the Gary illustration.
  const meta = await loadJSON("data/compare_pool.json").catch(() => null);
  if (meta) return realSamples(section, meta);
  grids = [...section.querySelectorAll(".gary-grid")];
  section.querySelector("[data-gary]").addEventListener("click", generate);
  await generate();
}

function buildTable(tbody) {
  for (const [label, ...cells] of ROWS) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    const rowButton = document.createElement("button");
    rowButton.type = "button";
    rowButton.className = "cmp-row";
    rowButton.textContent = label;
    th.append(rowButton);
    tr.append(th);
    const buttons = cells.map(([text, good]) => {
      const td = document.createElement("td");
      const b = document.createElement("button");
      b.type = "button";
      b.className = `cmp-cell ${good ? "good" : "bad"}`;
      b.setAttribute("aria-label", `${label}: reveal`);
      b.innerHTML = `<span class="cmp-q" aria-hidden="true">?</span><span class="cmp-a">${text}</span>`;
      b.addEventListener("click", () => reveal(b));
      td.append(b);
      tr.append(td);
      return b;
    });
    rowButton.addEventListener("click", () => buttons.forEach(reveal));
    tbody.append(tr);
  }
}

function reveal(b) {
  b.classList.add("revealed");
  b.removeAttribute("aria-label");
}

function buildTriangle() {
  const [q, v, s] = [CORNERS.quality, CORNERS.variety, CORNERS.speed];
  const edge = (a, b, k) => `<line class="tri-edge" data-edge="${k}" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" />`;
  const corner = (name, [x, y], label, nameDy, whoDy) =>
    `<g class="tri-corner" data-corner="${name}"><circle cx="${x}" cy="${y}" r="8" /><text class="tri-name" x="${x}" y="${y + nameDy}" text-anchor="middle">${label}</text><text class="tri-who" x="${x}" y="${y + whoDy}" text-anchor="middle">${WHO[name]}</text></g>`;
  tri.setAttribute("viewBox", "-80 -46 480 358");
  tri.innerHTML = `
    <polygon class="tri-fill" points="${q} ${v} ${s}" />
    ${edge(q, v, "quality-variety")}${edge(v, s, "speed-variety")}${edge(q, s, "quality-speed")}
    ${corner("quality", q, "Quality", -52, -18)}${corner("variety", v, "Variety", 36, 66)}${corner("speed", s, "Speed", 36, 66)}`;
}

function highlight(model) {
  const on = GETS[model];
  for (const c of tri.querySelectorAll(".tri-corner")) c.classList.toggle("on", on.includes(c.dataset.corner));
  const key = [...on].sort().join("-");
  for (const e of tri.querySelectorAll(".tri-edge")) e.classList.toggle("on", e.dataset.edge === key);
}

/** Gary's nine cats (one code, tiny jitter) and nine independent VAE samples. */
async function generate() {
  const gary = randn(LATENT);
  const garyBatch = new Float32Array(9 * LATENT);
  const jitter = randn(9 * LATENT);
  for (let i = 0; i < garyBatch.length; i++) garyBatch[i] = gary[i % LATENT] + 0.04 * jitter[i];
  const batches = [garyBatch, randn(9 * LATENT)];
  for (let g = 0; g < 2; g++) {
    await inSlot(grids[g].closest("figure"), async () => {
      const images = await decode("vae_128", batches[g]);
      drawTiles(grids[g], images, { cols: 3, gap: GAP, inset: GAP / 2 });
    });
  }
}

/** The comparison with real models: nine samples each from the VAE, the GAN (Gary) and diffusion. */
async function realSamples(section, meta) {
  const sprite = await loadSprite(`data/${meta.file}`, meta);
  const labels = { vae: "VAE", gan: "GAN", diffusion: "Diffusion" };
  const notes = { vae: "one pass, blurry", gan: "one pass, sharp", diffusion: "100 steps, sharp and varied" };
  const row = section.querySelector(".gary");
  row.classList.add("trio");
  row.innerHTML = Object.keys(meta.parts).map((k) => `
    <figure class="trio-fig ${k}">
      <figcaption><strong>${labels[k]}</strong> <span>${notes[k]}</span></figcaption>
      <canvas class="trio-grid" width="${3 * 64 + 4 * GAP}" height="${3 * 64 + 4 * GAP}" data-part="${k}" aria-label="Nine ${labels[k]} samples"></canvas>
    </figure>`).join("") + `<button type="button" class="btn primary" data-trio>new samples</button>`;
  section.querySelector(".gary-caption").textContent =
    "All three were trained on the same 29,331 cats on one GPU. Same data, three ways to generate. The GAN is Gary, now a real one.";
  const draw = () => {
    for (const c of row.querySelectorAll(".trio-grid")) {
      const [start, n] = meta.parts[c.dataset.part];
      const picks = new Set();
      while (picks.size < 9) picks.add(start + Math.floor(Math.random() * n));
      const ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
      [...picks].forEach((idx, i) => ctx.putImageData(sprite.get(idx), GAP + (i % 3) * (64 + GAP), GAP + Math.floor(i / 3) * (64 + GAP)));
    }
  };
  row.querySelector("[data-trio]").addEventListener("click", draw);
  draw();
}
