// Small visualisations shared by the demos: an error heatmap with its colour legend,
// and a histogram with a marker and threshold lines.

import { toTensor, SIZE } from "./models.js";

const PIXELS = SIZE * SIZE;

// Navy -> lilac -> coral -> sunflower -> white: dark means "rebuilt well", bright means "missed".
const STOPS = [
  [0, [20, 17, 43]],
  [0.25, [125, 92, 207]],
  [0.5, [255, 107, 107]],
  [0.78, [255, 209, 102]],
  [1, [255, 255, 255]],
];

function heatColour(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i];
    const [t0, c0] = STOPS[i - 1];
    if (t <= t1) {
      const k = (t - t0) / (t1 - t0);
      return c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
    }
  }
  return STOPS.at(-1)[1];
}

/** CSS gradient matching the heatmap colours, for legends. */
const HEAT_GRADIENT = `linear-gradient(90deg, ${STOPS.map(([t, c]) => `rgb(${c}) ${t * 100}%`).join(", ")})`;

/**
 * Paint the squared error per pixel (averaged over R, G, B) between an image and its
 * reconstruction. Errors at or above `vmax` are white.
 */
export function drawErrorHeatmap(canvas, image, recon, { vmax = 0.05, index = 0 } = {}) {
  canvas.width = canvas.height = SIZE;
  const x = toTensor(image).data;
  const y = recon.data ?? recon;
  const base = index * 3 * PIXELS;
  const out = new ImageData(SIZE, SIZE);
  for (let i = 0; i < PIXELS; i++) {
    let e = 0;
    for (let c = 0; c < 3; c++) e += (x[c * PIXELS + i] - y[base + c * PIXELS + i]) ** 2;
    const [r, g, b] = heatColour(e / 3 / vmax);
    out.data.set([r, g, b, 255], i * 4);
  }
  canvas.getContext("2d").putImageData(out, 0, 0);
}

/** A small colour bar with end labels. */
export function heatLegend(el, { low = "0", high = "0.05+" } = {}) {
  el.innerHTML = `<div class="heat-legend"><span class="mono">${low}</span><span class="heat-bar" style="background:${HEAT_GRADIENT}"></span><span class="mono">${high}</span></div>`;
}

/**
 * Histogram of `values` as SVG, with labelled threshold lines and a movable marker.
 *   const h = histogram(el, values, { max, bins, lines: [{ at, label }] })
 *   h.mark(value)   moves the marker (values beyond max sit at the edge with an arrow)
 */
export function histogram(el, values, { max = Math.max(...values), bins = 32, lines = [], width = 600, height = 180 } = {}) {
  const pad = { l: 8, r: 8, t: 28, b: 26 };
  const w = width - pad.l - pad.r;
  const h = height - pad.t - pad.b;
  const counts = new Array(bins).fill(0);
  for (const v of values) counts[Math.min(bins - 1, Math.floor((v / max) * bins))]++;
  const top = Math.max(...counts);
  const x = (v) => pad.l + (Math.min(v, max) / max) * w;

  const bars = counts
    .map((c, i) => {
      const bh = (c / top) * h;
      return `<rect class="hist-bar" x="${pad.l + (i * w) / bins + 1}" y="${pad.t + h - bh}" width="${w / bins - 2}" height="${bh}" rx="2" />`;
    })
    .join("");
  const lineEls = lines
    .map(
      ({ at, label }) =>
        `<line class="hist-line" x1="${x(at)}" x2="${x(at)}" y1="${pad.t - 6}" y2="${pad.t + h}" />` +
        `<text class="hist-line-label" x="${x(at)}" y="${pad.t - 10}" text-anchor="middle">${label}</text>`,
    )
    .join("");
  el.innerHTML = `
    <svg class="histogram" viewBox="0 0 ${width} ${height}" role="img" aria-label="Histogram of reconstruction errors of normal cats">
      ${bars}
      ${lineEls}
      <line class="hist-axis" x1="${pad.l}" x2="${pad.l + w}" y1="${pad.t + h}" y2="${pad.t + h}" />
      <text class="hist-tick" x="${pad.l}" y="${height - 6}">0</text>
      <text class="hist-tick" x="${pad.l + w}" y="${height - 6}" text-anchor="end">${max}</text>
      <g class="hist-marker" style="opacity:0">
        <line x1="0" x2="0" y1="${pad.t - 4}" y2="${pad.t + h + 4}" />
        <circle cx="0" cy="${pad.t + h}" r="7" />
        <text class="hist-marker-label" x="0" y="${pad.t + h - 10}" text-anchor="middle"></text>
      </g>
    </svg>`;
  const marker = el.querySelector(".hist-marker");
  const label = el.querySelector(".hist-marker-label");
  return {
    mark(value) {
      marker.style.opacity = 1;
      marker.style.transform = `translateX(${x(value)}px)`;
      label.textContent = value > max ? `${Math.round(value)} →` : "";
      label.setAttribute("text-anchor", value > max ? "end" : "middle");
    },
  };
}
