// The reparameterization trick, as a computation graph with the backward pass animated.
//   Naive sampling: z is drawn by a dice roll. Gradient particles flow back from the loss and
//   stop at the dice: you can't differentiate a dice roll.
//   Reparameterization: eps ~ N(0, 1) comes in from the side and z = mu + sigma * eps, so the
//   particles flow on through mu and sigma into the encoder.
// The graph runs left to right on wide screens and top to bottom on phones.
// Below: mu and sigma sliders plus "resample eps" drive a live vae_2d_b1 decode and a mini map.

import { decode, drawCat, hasModel, renderTraining, randn } from "../models.js";
import { flipSwitch, slider } from "../ui.js";
import { loadLatentMap, VAE_VIEW } from "../latentmap.js";

const MODEL = "vae_2d_b1";
const SPAWN_MS = 260;
const PX_SPEED = 0.38; // graph units per ms, the same on every route

// Node centres in SVG units for each layout, and each node's half width and height (for arrows).
const LAYOUTS = {
  wide: {
    view: [1200, 290],
    vertical: false,
    at: { x: [70, 120], enc: [225, 120], mu: [375, 72], sigma: [375, 168], sampler: [530, 120], eps: [530, 262], z: [680, 120], dec: [820, 120], xhat: [965, 120], loss: [1110, 120] },
  },
  tall: {
    view: [420, 900],
    vertical: true,
    at: { x: [140, 40], enc: [140, 160], mu: [80, 290], sigma: [200, 290], sampler: [140, 410], eps: [320, 410], z: [140, 530], dec: [140, 650], xhat: [140, 770], loss: [140, 860] },
  },
};
const HALF = { x: [38, 28], enc: [62, 50], mu: [28, 25], sigma: [28, 25], sampler: [60, 28], z: [28, 25], dec: [62, 50], xhat: [30, 28], loss: [40, 28] };
const narrow = matchMedia("(max-width: 760px)");

let svg, layout, particlesLayer, mode = "naive";
let frame = 0, lastSpawn = 0, lastTime = 0, running = false;
let particles = [];
let mu = [0, 0], sigma = 0.6, eps = [0.8, -0.5];
let outCanvas, zLabel, map, request = 0;

export async function init(section) {
  svg = section.querySelector("svg.rep-graph");
  narrow.addEventListener("change", () => setMode(mode));
  flipSwitch(section.querySelector(".s14-switch"), {
    left: "Naive sampling",
    right: "Reparameterization",
    label: "Use the reparameterization trick",
    onChange: (on) => setMode(on ? "reparam" : "naive"),
  });
  setMode("naive");

  // The live demo below the graph.
  const demo = section.querySelector(".s14-demo");
  if (!(await hasModel(MODEL))) return renderTraining(demo, MODEL);
  outCanvas = section.querySelector(".s14-out");
  zLabel = section.querySelector(".s14-z");
  const sliders = section.querySelector(".s14-sliders");
  const resample = sliders.querySelector("[data-eps]");
  const make = (label, min, max, value, set) =>
    slider(sliders, { label, min, max, step: 0.01, value, format: (v) => v.toFixed(2), onInput: (v) => (set(v), update()) });
  make("μ1", -3, 3, mu[0], (v) => (mu[0] = v));
  make("μ2", -3, 3, mu[1], (v) => (mu[1] = v));
  make("σ", 0.05, 1.5, sigma, (v) => (sigma = v));
  sliders.append(resample); // keep the button under the sliders
  resample.addEventListener("click", () => {
    eps = Array.from(randn(2));
    update();
  });
  map = await loadLatentMap(section.querySelector(".s14-map"), MODEL, {
    view: VAE_VIEW,
    dot: 1.8,
    label: "Mini map of the VAE's latent space with mu, sigma and z",
  });
  map.overlay = drawZ;
  update();
}

export function enter() {
  running = true;
  lastTime = performance.now();
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(tick);
}

export function leave() {
  running = false;
  cancelAnimationFrame(frame);
}

// ---------------------------------------------------------------- the graph

function half(name) {
  if (name === "sampler" && mode === "naive") return [38, 38]; // the dice is square
  return HALF[name];
}

/** Where the line from a node's centre towards (tx, ty) leaves the node's box, plus a small gap. */
function edgePoint(name, [tx, ty], pad = 6) {
  const [cx, cy] = layout.at[name];
  const [hx, hy] = half(name);
  const [dx, dy] = [tx - cx, ty - cy];
  const t = Math.min(dx ? (hx + pad) / Math.abs(dx) : Infinity, dy ? (hy + pad) / Math.abs(dy) : Infinity);
  return [cx + dx * t, cy + dy * t];
}

function arrow(from, to, cls = "") {
  const [x1, y1] = edgePoint(from, layout.at[to]);
  const [x2, y2] = edgePoint(to, layout.at[from]);
  return `<line class="g-edge ${cls}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#rep-arrow)" />`;
}

function box(name, label, cls = "") {
  const [cx, cy] = layout.at[name];
  const [hx, hy] = HALF[name];
  return `<g class="g-node ${cls}"><rect x="${cx - hx}" y="${cy - hy}" width="${2 * hx}" height="${2 * hy}" rx="12" /><text x="${cx}" y="${cy + 7}" text-anchor="middle">${label}</text></g>`;
}

/** The encoder narrows along the flow, the decoder widens. */
function trapezoid(name, label, narrowing) {
  const [cx, cy] = layout.at[name];
  const [a, b] = narrowing ? [50, 26] : [26, 50];
  const pts = layout.vertical
    ? [[cx - a, cy - 62], [cx + a, cy - 62], [cx + b, cy + 62], [cx - b, cy + 62]]
    : [[cx - 62, cy - a], [cx + 62, cy - b], [cx + 62, cy + b], [cx - 62, cy + a]];
  return `<g class="g-node net"><polygon points="${pts.map((p) => p.join(",")).join(" ")}" /><text x="${cx}" y="${cy + 7}" text-anchor="middle">${label}</text></g>`;
}

function drawGraph() {
  layout = narrow.matches ? LAYOUTS.tall : LAYOUTS.wide;
  const [w, h] = layout.view;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const v = layout.vertical;
  const [sx, sy] = layout.at.sampler;
  const [ex, ey] = layout.at.eps;
  // Where the gradient gets stuck: the side of the dice that faces the loss.
  const stuck = v ? [sx, sy + 46] : [sx + 46, sy];
  const blockLabel = v
    ? `<text class="block-label" x="${sx + 42}" y="${sy + 70}"><tspan x="${sx + 42}">can't differentiate</tspan><tspan x="${sx + 42}" dy="26">a dice roll</tspan></text>`
    : `<text class="block-label" x="${sx + 20}" y="${sy + 92}" text-anchor="middle">can't differentiate a dice roll</text>`;
  const epsEdge = v ? [ex - 72, ey, sx + 66, sy] : [ex, ey - 10, sx, sy + 34];
  svg.innerHTML = `
    <defs><marker id="rep-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" /></marker></defs>
    ${arrow("x", "enc")}${arrow("enc", "mu")}${arrow("enc", "sigma")}${arrow("mu", "sampler")}${arrow("sigma", "sampler")}
    ${arrow("sampler", "z")}${arrow("z", "dec")}${arrow("dec", "xhat")}${arrow("xhat", "loss")}
    ${box("x", "x")}
    ${trapezoid("enc", "Encoder", true)}
    ${box("mu", "μ", "param")}${box("sigma", "σ", "param")}
    <g class="sampler naive-only">
      <rect class="dice-body" x="${sx - 38}" y="${sy - 38}" width="76" height="76" rx="14" />
      ${[[-18, -18], [18, -18], [0, 0], [-18, 18], [18, 18]].map(([dx, dy]) => `<circle class="dice-pip" cx="${sx + dx}" cy="${sy + dy}" r="6" />`).join("")}
    </g>
    <g class="reparam-only">
      ${box("sampler", "μ + σ·ε", "formula")}
      <line class="g-edge eps-edge" x1="${epsEdge[0]}" y1="${epsEdge[1]}" x2="${epsEdge[2]}" y2="${epsEdge[3]}" marker-end="url(#rep-arrow)" />
      <text class="g-eps" x="${ex}" y="${v ? ey + 6 : ey + 16}" text-anchor="middle">ε ~ N(0, 1)</text>
    </g>
    ${box("z", "z", "param")}
    ${trapezoid("dec", "Decoder", false)}
    ${box("xhat", "x̂")}${box("loss", "Loss", "loss")}
    <g class="particles"></g>
    <g class="blocked naive-only">
      <path class="block-x" d="M${stuck[0] - 18},${stuck[1] - 18} l36,36 M${stuck[0] + 18},${stuck[1] - 18} l-36,36" />
      ${blockLabel}
    </g>`;
  particlesLayer = svg.querySelector(".particles");
  const at = layout.at;
  // Backward routes, from the loss towards the encoder.
  layout.routes = {
    naive: [[at.loss, at.xhat, at.dec, at.z, stuck]],
    reparam: [
      [at.loss, at.xhat, at.dec, at.z, at.sampler, at.mu, at.enc],
      [at.loss, at.xhat, at.dec, at.z, at.sampler, at.sigma, at.enc],
    ],
  };
}

function routeLength(route) {
  let total = 0;
  for (let i = 1; i < route.length; i++) total += Math.hypot(route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1]);
  return total;
}

function pointOn(route, t) {
  let d = t * routeLength(route);
  for (let i = 1; i < route.length; i++) {
    const len = Math.hypot(route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1]);
    if (d <= len) {
      const k = len ? d / len : 0;
      return [route[i - 1][0] + (route[i][0] - route[i - 1][0]) * k, route[i - 1][1] + (route[i][1] - route[i - 1][1]) * k];
    }
    d -= len;
  }
  return route.at(-1);
}

function setMode(m) {
  mode = m;
  svg.dataset.mode = m;
  drawGraph(); // arrows into the sampler clip to the dice or to the formula box
  particles = [];
}

function tick(now) {
  if (!running) return;
  const dt = Math.min(64, now - lastTime);
  lastTime = now;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduced && now - lastSpawn > SPAWN_MS) {
    lastSpawn = now;
    const routes = layout.routes[mode];
    const route = routes[Math.floor(Math.random() * routes.length)];
    const el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    el.setAttribute("r", "6");
    el.setAttribute("class", "grad-particle");
    particlesLayer.append(el);
    particles.push({ el, t: 0, route, length: routeLength(route) });
  }
  for (const p of particles) {
    p.t += (dt * PX_SPEED) / p.length;
    const [x, y] = pointOn(p.route, Math.min(1, p.t));
    p.el.setAttribute("cx", x);
    p.el.setAttribute("cy", y);
    if (p.t >= 1) p.el.classList.add(mode === "naive" ? "stuck" : "arrived");
  }
  // Drop finished particles once they have had 300 ms to fade.
  particles = particles.filter((p) => {
    if ((p.t - 1) * (p.length / PX_SPEED) < 300) return true;
    p.el.remove();
    return false;
  });
  frame = requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- the live demo

async function update() {
  const mine = ++request;
  const z = [mu[0] + sigma * eps[0], mu[1] + sigma * eps[1]];
  zLabel.textContent = `z = μ + σ·ε = (${z[0].toFixed(2)}, ${z[1].toFixed(2)})`;
  map.draw();
  const image = await decode(MODEL, z);
  if (mine === request) drawCat(outCanvas, image);
}

function drawZ(ctx, m) {
  const [mx, my] = m.toScreen(mu);
  const [rx] = m.toScreen([mu[0] + sigma, mu[1]]);
  const z = [mu[0] + sigma * eps[0], mu[1] + sigma * eps[1]];
  const [zx, zy] = m.toScreen(z);
  ctx.save();
  ctx.fillStyle = "rgba(179, 136, 255, 0.25)";
  ctx.strokeStyle = "#b388ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(mx, my, Math.max(3, rx - mx), 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(mx - 6, my);
  ctx.lineTo(mx + 6, my);
  ctx.moveTo(mx, my - 6);
  ctx.lineTo(mx, my + 6);
  ctx.stroke();
  ctx.fillStyle = "#ffd166";
  ctx.beginPath();
  ctx.arc(zx, zy, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}
