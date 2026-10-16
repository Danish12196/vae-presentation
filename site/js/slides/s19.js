// VAE vs GAN vs diffusion.
//   The comparison table: every cell starts hidden and appears when clicked (a row header shows its row).
//   The trilemma: pick a model and its two corners of quality / variety / speed light up.
//   Meet Gary, a mode-collapsed GAN (an illustration: his cats are drawn by our VAE from one fixed
//   code, with tiny jitter), next to nine genuinely different vae_128 samples.

import { decode, fromTensor, inSlot, randn, ort } from "../models.js";
import { tabs } from "../ui.js";

const ROWS = [
  ["Image quality", ["Blurry", false], ["Sharp", true], ["Sharp", true]],
  ["Variety of outputs", ["High", true], ["Can suffer mode collapse", false], ["High", true]],
  ["Generation speed", ["Fast, one pass", true], ["Fast, one pass", true], ["Slow, many steps", false]],
  ["Training", ["Stable", true], ["Unstable", false], ["Stable", true]],
  ["Encoder for real images", ["Yes", true], ["No", false], ["No (noising instead)", false]],
];
const CORNERS = { quality: [160, 34], variety: [36, 248], speed: [284, 248] };
const GETS = { vae: ["variety", "speed"], gan: ["quality", "speed"], diffusion: ["quality", "variety"] };
const LATENT = 128, TILE = 64, GAP = 4;

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
  const corner = (name, [x, y], label, dy) =>
    `<g class="tri-corner" data-corner="${name}"><circle cx="${x}" cy="${y}" r="16" /><text x="${x}" y="${y + dy}" text-anchor="middle">${label}</text></g>`;
  tri.innerHTML = `
    <polygon class="tri-fill" points="${q} ${v} ${s}" />
    ${edge(q, v, "quality-variety")}${edge(v, s, "speed-variety")}${edge(q, s, "quality-speed")}
    ${corner("quality", q, "Quality", -26)}${corner("variety", v, "Variety", 38)}${corner("speed", s, "Speed", 38)}`;
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
      const images = await decode("vae_128", new ort.Tensor("float32", batches[g], [9, LATENT]));
      const ctx = grids[g].getContext("2d");
      for (let i = 0; i < 9; i++) ctx.putImageData(fromTensor(images, i), (i % 3) * (TILE + GAP) + GAP / 2, Math.floor(i / 3) * (TILE + GAP) + GAP / 2);
    });
  }
}
