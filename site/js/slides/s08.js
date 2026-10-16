// What autoencoders are good for: three small cards and a live anomaly detector.
// ae_64 was trained on normal cats only. Draw on a cat (or pick a "not a cat") and watch
// its reconstruction error land on the histogram of 512 held-out normal cats.
//   above the 99th percentile        -> ANOMALY (can't rebuild it)
//   between the 95th and 99th        -> hmm, suspicious
//   below the 1st percentile         -> ANOMALY (suspiciously easy: flat or blurry images)
// The upside-down and colour-inverted cats pass as normal: the honest blind spot.

import { drawCat, reconstruct, squaredError, inSlot, loadShowcase, getShowcaseCat, loadNotCats, loadJSON } from "../models.js";
import { imagePicker } from "../ui.js";
import { drawErrorHeatmap, histogram } from "../viz.js";

const COLOURS = [
  { name: "coral", value: "#ff6b6b" },
  { name: "mint", value: "#06d6a0" },
  { name: "ink", value: "#14112b" },
];
const BRUSH = 5;        // in the image's own 64x64 pixels: thick
const THROTTLE = 120;   // ms between live reconstructions while drawing
const HIST_MAX = 360;

let canvas, ctx, reconCanvas, heatCanvas, reconSlot, chip, reason, hist;
let stats = null;
let base = null;        // the picked image, before any drawing
let source = null;      // { label, notCat }
let colour = COLOURS[0].value;
let undoStack = [];
let drawing = false;
let last = null;
let timer = 0;
let request = 0;

export async function init(section) {
  canvas = section.querySelector("canvas.draw");
  ctx = canvas.getContext("2d", { willReadFrequently: true });
  reconSlot = section.querySelector(".recon-col figure");
  [reconCanvas, heatCanvas] = section.querySelectorAll(".recon-col canvas");
  chip = section.querySelector(".verdict-chip");
  reason = section.querySelector(".verdict-reason");

  await loadShowcase();
  drawDenoiseCard(section);
  buildTools(section.querySelector(".draw-tools"));

  try {
    stats = await loadJSON("data/anomaly.json");
    hist = histogram(section.querySelector(".anomaly-hist"), stats.errors, {
      max: HIST_MAX,
      lines: [
        { at: stats.p1, label: "1%" },
        { at: stats.p95, label: "95%" },
        { at: stats.p99, label: "99%" },
      ],
    });
  } catch {
    /* no statistics exported: the verdict can't be judged */
  }

  // Six typical held-out cats (error inside the normal range; a few real cats are false alarms,
  // which would muddle the demo), then the six "not a cat" images.
  // anomaly.json lists the 512 held-out cats in order; the showcase is the first 64 of them.
  const typical = Array.from({ length: 64 }, (_, i) => i)
    .filter((i) => !stats || (stats.errors[i] > stats.p1 && stats.errors[i] < stats.p95))
    .slice(0, 6);
  const items = typical.map((i) => ({ image: getShowcaseCat(i), label: `Cat ${i + 1}` }));
  try {
    const notCats = await loadNotCats();
    notCats.labels.forEach((label, i) => items.push({ image: notCats.sprite.get(i), label, notCat: true }));
  } catch {
    /* cats only */
  }
  imagePicker(section.querySelector(".anomaly-picker"), {
    items,
    columns: items.length,
    onPick: (_, item) => load(item),
  });

  canvas.addEventListener("pointerdown", startStroke);
  canvas.addEventListener("pointermove", moveStroke);
  canvas.addEventListener("pointerup", endStroke);
  canvas.addEventListener("pointercancel", endStroke);

  load(items[0]);
}

export function leave() {
  clearTimeout(timer);
  drawing = false;
}

// ---------------------------------------------------------------- picking and drawing

function load(item) {
  base = item.image;
  source = item;
  undoStack = [];
  drawCat(canvas, base);
  check();
}

function point(e) {
  const r = canvas.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * 64, ((e.clientY - r.top) / r.height) * 64];
}

function startStroke(e) {
  try {
    canvas.setPointerCapture(e.pointerId); // keep the stroke even if the finger leaves the canvas
  } catch {
    /* capture is a nicety */
  }
  undoStack.push(ctx.getImageData(0, 0, 64, 64));
  if (undoStack.length > 30) undoStack.shift();
  drawing = true;
  last = point(e);
  stroke(last, last);
}

function moveStroke(e) {
  if (!drawing) return;
  const p = point(e);
  stroke(last, p);
  last = p;
}

function endStroke() {
  if (!drawing) return;
  drawing = false;
  clearTimeout(timer);
  timer = 0;
  check();
}

function stroke(a, b) {
  ctx.strokeStyle = colour;
  ctx.lineWidth = BRUSH;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(...a);
  ctx.lineTo(...b);
  ctx.stroke();
  // Live while drawing, but at most every THROTTLE ms.
  if (!timer) {
    timer = setTimeout(() => {
      timer = 0;
      check();
    }, THROTTLE);
  }
}

function buildTools(el) {
  const swatches = COLOURS.map(({ name, value }, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "swatch";
    b.style.setProperty("--swatch", value);
    b.setAttribute("aria-label", `${name} brush`);
    b.setAttribute("aria-pressed", String(i === 0));
    b.addEventListener("click", () => {
      colour = value;
      swatches.forEach((s) => s.setAttribute("aria-pressed", String(s === b)));
    });
    return b;
  });
  const undo = button("undo", () => {
    const prev = undoStack.pop();
    if (prev) {
      ctx.putImageData(prev, 0, 0);
      check();
    }
  });
  const clear = button("clear", () => {
    undoStack.push(ctx.getImageData(0, 0, 64, 64));
    drawCat(canvas, base);
    check();
  });
  el.append(...swatches, undo, clear);
}

function button(label, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "btn tool";
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

// ---------------------------------------------------------------- judging

async function check() {
  const mine = ++request;
  const image = ctx.getImageData(0, 0, 64, 64);
  const error = await inSlot(reconSlot, async () => {
    const out = await reconstruct("ae_64", [image]);
    if (mine !== request) return null;
    drawCat(reconCanvas, out);
    drawErrorHeatmap(heatCanvas, image, out);
    return squaredError(image, out);
  });
  if (mine !== request) return;
  showVerdict(error);
}

function showVerdict(error) {
  if (error == null || !stats) {
    chip.textContent = "";
    reason.textContent = "";
    return;
  }
  hist?.mark(error);
  let verdict;
  if (error > stats.p99) verdict = ["ANOMALY", "can't rebuild it", "bad"];
  else if (error > stats.p95) verdict = ["hmm, suspicious", "worse than 95% of cats", "warn"];
  else if (error < stats.p1) verdict = ["ANOMALY", "suspiciously easy", "bad"];
  else if (source?.notCat) verdict = ["normal cat", "blind spot: not a cat, but easy to rebuild", "ok"];
  else verdict = ["normal cat", "", "ok"];
  const [text, why, level] = verdict;
  chip.textContent = text;
  chip.dataset.level = level;
  reason.textContent = why ? `${why} · error ${Math.round(error)}` : `error ${Math.round(error)}`;
}

// ---------------------------------------------------------------- the small cards

function drawDenoiseCard(section) {
  const [noisy, clean] = section.querySelectorAll(".use-denoise canvas");
  const cat = getShowcaseCat(7);
  drawCat(clean, cat);
  const n = new ImageData(new Uint8ClampedArray(cat.data), 64, 64);
  for (let i = 0; i < n.data.length; i += 4) {
    const k = (Math.random() - 0.5) * 140;
    n.data[i] += k;
    n.data[i + 1] += k;
    n.data[i + 2] += k;
  }
  drawCat(noisy, n);
}
