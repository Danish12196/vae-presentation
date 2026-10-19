// "Watch it learn": a replayed training run (training/snapshots.py, data/training.json).
// Loss curves, an epoch slider with play, four cats as the model rebuilt them at that epoch,
// and the latent map at that epoch (600 training cats, coloured by fur). Shared by s11 (ae_2)
// and s22 (vae_2d_b1). Arriving forward with the keys replays from epoch 0.

import { loadJSON, loadSprite } from "./models.js";
import { slider } from "./ui.js";
import { reducedMotion } from "./anim.js";

const NS = "http://www.w3.org/2000/svg";
const PLAY_MS = 6000;

export async function learnView(root, name, { kl = false } = {}) {
  const data = await loadJSON("data/training.json");
  const run = data.models?.[name];
  if (!run) throw new Error(`no training replay for ${name}`);
  const maps = await loadJSON("data/latent_maps.json");
  const colours = maps.colours;
  const sprite = await loadSprite(`data/${data.recons.file}`, data.recons);
  const rowOf = Object.keys(data.models).indexOf(name); // which block of rows in the sprite
  const firstRow = 1 + rowOf * (run.epochs + 1);
  const cols = data.recons.cols;

  // ---------------------------------------------------------------- layout
  root.innerHTML = `
    <div class="learn-curves"><svg class="learn-svg" viewBox="0 0 520 300" role="img" aria-label="Training loss per epoch"></svg></div>
    <div class="learn-middle">
      <div class="learn-cats">
        <p class="learn-row-label">original</p><div class="learn-row learn-orig"></div>
        <p class="learn-row-label">rebuilt at this epoch</p><div class="learn-row learn-now"></div>
      </div>
      <div class="learn-controls">
        <button type="button" class="btn primary learn-play">▶ replay training</button>
        <div class="learn-slider"></div>
      </div>
    </div>
    <div class="learn-map"><canvas class="learn-canvas" role="img" aria-label="The latent map at this epoch"></canvas><p class="learn-map-label mono"></p></div>`;
  const svg = root.querySelector(".learn-svg");
  const orig = root.querySelector(".learn-orig");
  const now = root.querySelector(".learn-now");
  const canvas = root.querySelector(".learn-canvas");
  const mapLabel = root.querySelector(".learn-map-label");

  const tile = () => {
    const c = document.createElement("canvas");
    c.width = c.height = data.recons.tile;
    c.className = "cat-canvas";
    return c;
  };
  const origTiles = Array.from({ length: cols }, () => orig.appendChild(tile()));
  const nowTiles = Array.from({ length: cols }, () => now.appendChild(tile()));
  origTiles.forEach((c, i) => c.getContext("2d").putImageData(sprite.get(i), 0, 0));

  // ---------------------------------------------------------------- loss curves
  const W = 520, H = 300, L = 70, R = 20, T = 20, B = 50;
  const x = (e) => L + ((e - 1) / (run.epochs - 1)) * (W - L - R);
  const el = (tag, attrs, parent = svg) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    parent.append(n);
    return n;
  };
  const series = [{ key: "recon", label: "reconstruction", cls: "recon", values: run.recon }];
  if (kl && run.kl) series.push({ key: "kl", label: "KL", cls: "kl", values: run.kl });
  // Each curve gets its own log scale: the VAE's KL starts in the thousands during warm-up.
  for (const s of series) {
    const logs = s.values.map((v) => Math.log10(Math.max(v, 1e-3)));
    const [lo, hi] = [Math.min(...logs), Math.max(...logs)];
    s.y = (v) => T + (1 - (Math.log10(Math.max(v, 1e-3)) - lo) / (hi - lo || 1)) * (H - T - B);
  }
  el("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, class: "learn-axis" });
  el("line", { x1: L, y1: T, x2: L, y2: H - B, class: "learn-axis" });
  const label = (str, attrs) => (el("text", attrs).textContent = str);
  label("epoch", { x: W - R, y: H - 12, class: "learn-tick", "text-anchor": "end" });
  label("1", { x: x(1), y: H - B + 22, class: "learn-tick", "text-anchor": "middle" });
  label(String(run.epochs), { x: x(run.epochs), y: H - B + 22, class: "learn-tick", "text-anchor": "middle" });
  label("loss (log scale)", { x: L - 24, y: (T + H - B) / 2, class: "learn-tick", "text-anchor": "middle", transform: `rotate(-90 ${L - 24} ${(T + H - B) / 2})` });
  for (const [i, s] of series.entries()) {
    el("path", { d: s.values.map((v, k) => `${k ? "L" : "M"} ${x(k + 1)} ${s.y(v)}`).join(" "), class: `learn-line ${s.cls}` });
    s.dot = el("circle", { r: 7, class: `learn-dot ${s.cls}` });
    s.read = el("text", { x: W - R, y: T + 18 + i * 26, class: `learn-read ${s.cls}`, "text-anchor": "end" });
  }
  const cursor = el("line", { y1: T, y2: H - B, class: "learn-cursor" });

  // ---------------------------------------------------------------- the map at one epoch
  const final = run.codes[run.epochs];
  const q = (arr, p) => [...arr].sort((a, b) => a - b)[Math.round(p * (arr.length - 1))];
  const xs = final.map((p) => p[0]), ys = final.map((p) => p[1]);
  const half = Math.max(q(xs, 0.99) - q(xs, 0.01), q(ys, 0.99) - q(ys, 0.01)) * 0.68;
  const view = { cx: (q(xs, 0.99) + q(xs, 0.01)) / 2, cy: (q(ys, 0.99) + q(ys, 0.01)) / 2, half };
  if (name.startsWith("vae")) Object.assign(view, { cx: 0, cy: 0, half: 4.5 }); // the Part 5 map's view

  function drawMap(epoch) {
    const size = canvas.clientWidth || 300;
    const dpr = devicePixelRatio || 1;
    if (canvas.width !== Math.round(size * dpr)) {
      canvas.width = canvas.height = Math.round(size * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    const toPx = ([zx, zy]) => [((zx - view.cx) / view.half / 2 + 0.5) * size, (0.5 - (zy - view.cy) / view.half / 2) * size];
    ctx.strokeStyle = "rgba(36,23,15,0.35)";
    ctx.lineWidth = 1;
    const [ox, oy] = toPx([0, 0]);
    ctx.beginPath();
    ctx.moveTo(0, oy); ctx.lineTo(size, oy); ctx.moveTo(ox, 0); ctx.lineTo(ox, size);
    ctx.stroke();
    run.codes[epoch].forEach((p, i) => {
      const [px, py] = toPx(p);
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fillStyle = colours[i];
      ctx.fill();
      ctx.strokeStyle = "rgba(36,23,15,0.4)";
      ctx.lineWidth = 0.6;
      ctx.stroke();
    });
    mapLabel.textContent = epoch === 0 ? "before training" : `after epoch ${epoch}`;
  }

  // ---------------------------------------------------------------- one epoch
  let epoch = run.epochs;
  const sliderEl = root.querySelector(".learn-slider");
  slider(sliderEl, {
    label: "epoch",
    min: 0,
    max: run.epochs,
    step: 1,
    value: run.epochs,
    format: (v) => (v === 0 ? "before training" : `${v} of ${run.epochs}`),
    onInput: (v) => {
      stop();
      show(v);
    },
  });
  const input = sliderEl.querySelector("input");

  function show(e) {
    epoch = e;
    input.value = e;
    input.dispatchEvent(new Event("input-silent"));
    sliderEl.querySelector(".slider-value").textContent = e === 0 ? "before training" : `${e} of ${run.epochs}`;
    input.style.setProperty("--fill", `${(e / run.epochs) * 100}%`);
    nowTiles.forEach((c, i) => c.getContext("2d").putImageData(sprite.get((firstRow + e) * cols + i), 0, 0));
    const shown = Math.max(1, e);
    cursor.setAttribute("x1", x(shown));
    cursor.setAttribute("x2", x(shown));
    cursor.style.opacity = e === 0 ? 0 : 1;
    for (const s of series) {
      const v = s.values[shown - 1];
      s.dot.setAttribute("cx", x(shown));
      s.dot.setAttribute("cy", s.y(v));
      s.dot.style.opacity = e === 0 ? 0 : 1;
      s.read.textContent = e === 0 ? "" : `${s.label} ${v >= 100 ? Math.round(v) : v.toFixed(1)}`;
    }
    drawMap(e);
  }

  // ---------------------------------------------------------------- play
  let raf = 0;
  function stop() {
    cancelAnimationFrame(raf);
    raf = 0;
  }
  function play() {
    stop();
    if (reducedMotion()) return show(run.epochs);
    const start = performance.now();
    const frame = (t) => {
      const e = Math.min(run.epochs, Math.floor(((t - start) / PLAY_MS) * (run.epochs + 1)));
      if (e !== epoch) show(e);
      if (e < run.epochs) raf = requestAnimationFrame(frame);
      else raf = 0;
    };
    show(0);
    raf = requestAnimationFrame(frame);
  }
  root.querySelector(".learn-play").addEventListener("click", play);
  new ResizeObserver(() => drawMap(epoch)).observe(canvas);
  show(run.epochs);

  return {
    enter({ direction = 1, byKey = false } = {}) {
      if (byKey && direction === 1) play();
      else show(run.epochs);
    },
    leave: stop,
  };
}
