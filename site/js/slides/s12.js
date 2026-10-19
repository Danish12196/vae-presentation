// Loss = reconstruction + KL. Hover or tap a term for a one-line meaning.
// Below: drag a Gaussian against N(0, 1) (its peak moves mu, its side handles change sigma)
// and watch KL( N(mu, sigma^2) || N(0, 1) ) = 0.5 * (sigma^2 + mu^2 - 1 - ln sigma^2).

import { readout } from "../ui.js";

const EXPLAIN = {
  recon: "reconstruction: how far the rebuilt cat x̂ is from the input x",
  kl: "KL: how far the cloud is from N(0, 1). Zero only when μ = 0 and σ = 1",
};
const HINT = "hover or tap a term";

// Plot geometry (SVG units): x from -4 to 4, density from 0 to 1.
const W = 800, H = 300, X0 = 40, X1 = 760, Y0 = 262, Y1 = 22;
const MU = [-3, 3], SIGMA = [0.4, 2.5];
const sx = (x) => X0 + ((x + 4) / 8) * (X1 - X0);
const ux = (px) => ((px - X0) / (X1 - X0)) * 8 - 4;
const sy = (p) => Y0 - p * (Y0 - Y1);
const pdf = (x, mu, s) => Math.exp(-(((x - mu) / s) ** 2) / 2) / (s * Math.sqrt(2 * Math.PI));
const clamp = (v, [a, b]) => Math.min(b, Math.max(a, v));

const kl = (mu, sigma) => 0.5 * (sigma ** 2 + mu ** 2 - 1 - Math.log(sigma ** 2));

let svg, qPath, qFill, peak, sides, klOut, paramsEl;
let mu = 1.4;
let sigma = 0.6;
let snapFrame = 0;

export function init(section) {
  // The equation's terms.
  const explain = section.querySelector(".eq-explain");
  const terms = [...section.querySelectorAll(".eq-term")];
  const show = (term) => {
    terms.forEach((t) => t.classList.toggle("active", t === term));
    explain.textContent = term ? EXPLAIN[term.dataset.term] : HINT;
    explain.dataset.term = term?.dataset.term ?? "";
  };
  for (const t of terms) {
    t.addEventListener("pointerenter", () => show(t));
    t.addEventListener("focus", () => show(t));
    t.addEventListener("click", () => show(t));
  }
  section.querySelector(".loss-eq").addEventListener("pointerleave", () => show(null));

  // The plot.
  svg = section.querySelector("svg.gauss");
  svg.innerHTML = `
    <line class="g-axis" x1="${X0}" x2="${X1}" y1="${Y0}" y2="${Y0}" />
    ${[-3, -2, -1, 0, 1, 2, 3].map((t) => `<text class="g-tick" x="${sx(t)}" y="${Y0 + 26}" text-anchor="middle">${t}</text>`).join("")}
    <path class="g-prior-fill" d="${area(0, 1)}" />
    <path class="g-prior" d="${curve(0, 1)}" />
    <text class="g-label prior" x="${sx(0)}" y="${sy(pdf(0, 0, 1)) - 10}" text-anchor="middle">N(0, 1)</text>
    <path class="g-q-fill" />
    <path class="g-q" />
    <circle class="g-handle side" r="13" tabindex="0" role="slider" aria-label="spread σ" data-side="-1" />
    <circle class="g-handle side" r="13" tabindex="0" role="slider" aria-label="spread σ" data-side="1" />
    <circle class="g-handle peak" r="15" tabindex="0" role="slider" aria-label="mean μ" />`;
  qPath = svg.querySelector(".g-q");
  qFill = svg.querySelector(".g-q-fill");
  peak = svg.querySelector(".peak");
  sides = [...svg.querySelectorAll(".side")];
  draggable(peak, (x) => (mu = clamp(x, MU)));
  for (const s of sides) draggable(s, (x) => (sigma = clamp(Math.abs(x - mu), SIGMA)));
  keyboard(peak, (d) => (mu = clamp(mu + d * 0.1, MU)));
  for (const s of sides) keyboard(s, (d) => (sigma = clamp(sigma + d * Number(s.dataset.side) * 0.05, SIGMA)));

  klOut = readout(section.querySelector(".kl-readout"), { label: "KL", value: kl(mu, sigma), format: (v) => v.toFixed(3), duration: 150 });
  paramsEl = section.querySelector(".gauss-params");
  section.querySelector("[data-snap]").addEventListener("click", snap);
  update();
}

export function leave() {
  cancelAnimationFrame(snapFrame);
}

function curve(m, s) {
  const pts = [];
  for (let i = 0; i <= 160; i++) {
    const x = -4 + (i / 160) * 8;
    pts.push(`${sx(x).toFixed(1)},${sy(Math.min(1.05, pdf(x, m, s))).toFixed(1)}`);
  }
  return `M${pts.join(" L")}`;
}
function area(m, s) {
  return `${curve(m, s)} L${X1},${Y0} L${X0},${Y0} Z`;
}

function update() {
  qPath.setAttribute("d", curve(mu, sigma));
  qFill.setAttribute("d", area(mu, sigma));
  const top = Math.min(1.05, pdf(mu, mu, sigma));
  peak.setAttribute("cx", sx(mu));
  peak.setAttribute("cy", sy(top));
  peak.setAttribute("aria-valuenow", mu.toFixed(2));
  sides.forEach((s) => {
    const x = mu + Number(s.dataset.side) * sigma;
    s.setAttribute("cx", sx(x));
    s.setAttribute("cy", sy(Math.min(1.05, pdf(x, mu, sigma))));
    s.setAttribute("aria-valuenow", sigma.toFixed(2));
  });
  klOut.set(kl(mu, sigma));
  paramsEl.textContent = `μ = ${mu.toFixed(2)}   σ = ${sigma.toFixed(2)}`;
}

/** Drag a handle horizontally; onX gets the pointer's position in data units. */
function draggable(el, onX) {
  el.addEventListener("pointerdown", (e) => {
    cancelAnimationFrame(snapFrame);
    el.setPointerCapture(e.pointerId);
    const move = (ev) => {
      const r = svg.getBoundingClientRect();
      onX(ux(((ev.clientX - r.left) / r.width) * W));
      update();
    };
    const up = () => {
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    e.preventDefault();
  });
}

/** Arrow keys nudge a focused handle (main.js leaves the arrows to role="slider" elements). */
function keyboard(el, onStep) {
  el.addEventListener("keydown", (e) => {
    const d = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
    if (!d) return;
    onStep(d);
    update();
    e.preventDefault();
  });
}

/** Animate back to mu = 0, sigma = 1, where KL is exactly zero. */
function snap() {
  cancelAnimationFrame(snapFrame);
  const [m0, s0] = [mu, sigma];
  const start = performance.now();
  const ms = matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 600;
  const step = (now) => {
    const t = ms ? Math.min(1, (now - start) / ms) : 1;
    const k = 1 - (1 - t) ** 3;
    mu = m0 + (0 - m0) * k;
    sigma = s0 + (1 - s0) * k;
    update();
    if (t < 1) snapFrame = requestAnimationFrame(step);
  };
  snapFrame = requestAnimationFrame(step);
}
