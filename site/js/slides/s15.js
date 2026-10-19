// Holes in the latent space. The ae_2 codes of 2,000 training cats, coloured by fur.
// Click anywhere to decode that point with ae_2. The verdict is about the map, not the picture:
// is this z somewhere the decoder has seen training cats, or out in a gap? (We tried judging the
// pictures with cnn_cls, but it calls every decoder output "cat", even flat green noise.)
// "Random point" samples the view uniformly and keeps score: most of the space is gaps.

import { decode, drawCat, hasModel, renderTraining, loadJSON, loadSprite, loadManifest } from "../models.js";
import { latentMap } from "../latentmap.js";
import { mascot } from "../mascot.js";
import { decodeGrid } from "../decodegrid.js";

let map, out, zLabel, chip, dist, tally, tip, tipCanvas, thumbs;
let seenRadius = 0; // a z this close to a training cat counts as "seen territory"
let spread = 1;     // typical size of one code dimension, for readable distances
const score = { seen: 0, gaps: 0 };
let request = 0;

export async function init(section) {
  const body = section.querySelector(".s10-body");
  let data;
  try {
    data = await loadJSON("data/latent_maps.json");
  } catch {
    data = null;
  }
  if (!data?.maps?.ae_2 || !(await hasModel("ae_2"))) {
    renderTraining(body, "ae_2");
    return;
  }

  out = section.querySelector(".s10-out canvas");
  zLabel = section.querySelector(".s10-z");
  chip = section.querySelector(".verdict-chip");
  dist = section.querySelector(".s10-dist");
  tally = section.querySelector(".s10-tally");
  tip = section.querySelector(".map-tip");
  tipCanvas = tip.querySelector("canvas");

  const points = data.maps.ae_2;
  const std = (await loadManifest()).models.ae_2?.code_std ?? [1, 1];
  spread = (std[0] + std[1]) / 2;
  seenRadius = typicalSpacing(points);

  map = latentMap(section.querySelector(".s10-map"), {
    points,
    colours: data.colours,
    onPick: (z) => pick(z),
    onHover: showTip,
  });
  thumbs = await loadSprite(`data/${data.thumbs.file}`, data.thumbs);

  // Old-map labels in the biggest empty regions.
  for (const z of map.emptySpots(3)) {
    const label = document.createElement("div");
    label.className = "monster-label";
    label.innerHTML = `${mascot({ expression: "monster" })}<span>here be monsters</span>`;
    label.setAttribute("aria-hidden", "true");
    map.addLabel(z, label);
  }

  section.querySelector("[data-random]").addEventListener("click", () => pick(map.randomPoint(), { counted: true }));
  gridButton(section, decodeGrid(map, () => "ae_2", { n: 12 }));
  // Start in the middle of the crowd, so the output square shows a cat instead of an empty frame.
  const median = (k) => points.map((p) => p[k]).sort((x, y) => x - y)[points.length >> 1];
  pick([median(0), median(1)]);
}

/** "What lives here?": the button covers the map with a grid of decodes, and back. */
function gridButton(section, grid) {
  const btn = section.querySelector("[data-grid]");
  btn.addEventListener("click", async () => {
    const on = !grid.on;
    btn.setAttribute("aria-pressed", String(on));
    btn.textContent = on ? "back to the dots" : "what lives here?";
    await grid.show(on);
  });
}

export function leave() {
  if (tip) tip.hidden = true;
}

/**
 * How close training cats usually are to their nearest neighbour: the 95th percentile of those
 * distances. A clicked z within this radius of some training cat is in "seen territory".
 */
function typicalSpacing(points) {
  const nearest = points.map((p, i) => {
    let best = Infinity;
    for (let j = 0; j < points.length; j++) {
      if (j === i) continue;
      const d = (p[0] - points[j][0]) ** 2 + (p[1] - points[j][1]) ** 2;
      if (d < best) best = d;
    }
    return Math.sqrt(best);
  });
  nearest.sort((a, b) => a - b);
  return nearest[Math.floor(nearest.length * 0.95)];
}

function nearestTraining(z) {
  let best = Infinity;
  for (const p of map.points) best = Math.min(best, (p[0] - z[0]) ** 2 + (p[1] - z[1]) ** 2);
  return Math.sqrt(best);
}

async function pick(z, { counted = false } = {}) {
  const mine = ++request;
  map.setCrosshair(z);
  zLabel.textContent = `z = (${z[0].toFixed(1)}, ${z[1].toFixed(1)})`;
  const d = nearestTraining(z);
  const seen = d <= seenRadius;
  chip.textContent = seen ? "seen territory" : "here be monsters";
  chip.dataset.level = seen ? "ok" : "bad";
  dist.textContent = `nearest training cat: ${(d / spread).toFixed(2)}σ away`;
  if (counted) {
    score[seen ? "seen" : "gaps"]++;
    tally.textContent = `in the data: ${score.seen} / in the gaps: ${score.gaps}`;
  }
  const image = await decode("ae_2", z);
  if (mine === request) drawCat(out, image);
}

/** On hover, the nearest training cat's thumbnail (only when the pointer is close to a dot). */
function showTip(nearest, at) {
  if (!nearest || nearest.distance > 28 || !thumbs) {
    tip.hidden = true;
    return;
  }
  tipCanvas.getContext("2d").putImageData(thumbs.get(nearest.index), 0, 0);
  const [x, y] = map.toScreen(map.points[nearest.index]);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
  tip.hidden = false;
}
