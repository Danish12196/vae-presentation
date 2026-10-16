// A 2D latent space drawn as a scatter of training cats, shared by Part 4 (ae_2) and Part 5 (VAEs).
//
//   const map = latentMap(container, { points, colours, onPick, onHover })
//   map.toData(px, py) / map.toScreen([z1, z2])   convert between canvas pixels and latent space
//   map.setCrosshair([z1, z2]) / map.setCrosshair(null)
//   map.setPoints(newPoints, { animate: true })     e.g. sliding between the beta maps
//   map.overlay = (ctx, map) => { ... }             extra drawing on top (clouds, samples)
//   map.emptySpots(k)                               k far-apart points in regions with no data
//   map.addLabel(z, element)                        a DOM label pinned to a latent point
//   map.randomPoint()                               uniform random point in the view
//
// The view is square with the same scale on both axes, centred on the data and
// `extent` times as wide as it (measured between the 1st and 99th percentiles, so a few
// outliers don't shrink everything else), so the empty regions around the data are visible.

const DOT = 3.2;          // dot radius in CSS pixels
const MORPH_MS = 900;

function percentile(sorted, q) {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))];
}

export function latentMap(container, { points, colours, extent = 2, view = null, label = "Latent space map: click a point to decode it", onPick = () => {}, onHover = () => {} }) {
  const wrap = document.createElement("div");
  wrap.className = "latent-map interactive";
  const canvas = document.createElement("canvas");
  canvas.className = "latent-canvas";
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", label);
  const labels = document.createElement("div");
  labels.className = "latent-labels";
  wrap.append(canvas, labels);
  container.append(wrap);
  const ctx = canvas.getContext("2d");

  const api = { canvas, wrap, overlay: null, points, view: null };
  let crosshair = null;
  let morph = null; // { from, to, start }
  let frame = 0;

  // ---------------------------------------------------------------- view and coordinates

  function fitView(pts) {
    const xs = pts.map((p) => p[0]).sort((a, b) => a - b);
    const ys = pts.map((p) => p[1]).sort((a, b) => a - b);
    const [x0, x1] = [percentile(xs, 0.01), percentile(xs, 0.99)];
    const [y0, y1] = [percentile(ys, 0.01), percentile(ys, 0.99)];
    const half = (Math.max(x1 - x0, y1 - y0) / 2) * extent;
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, half };
  }
  api.view = view ?? fitView(points);

  const size = () => canvas.clientWidth;
  api.toScreen = ([x, y]) => {
    const { cx, cy, half } = api.view;
    const s = size();
    return [((x - cx) / half + 1) * (s / 2), (1 - (y - cy) / half) * (s / 2)];
  };
  api.toData = (px, py) => {
    const { cx, cy, half } = api.view;
    const s = size();
    return [cx + (px / (s / 2) - 1) * half, cy - (py / (s / 2) - 1) * half];
  };
  api.randomPoint = () => {
    const { cx, cy, half } = api.view;
    return [cx + (Math.random() * 2 - 1) * half, cy + (Math.random() * 2 - 1) * half];
  };

  // ---------------------------------------------------------------- drawing

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const s = size();
    if (!s) return;
    canvas.width = canvas.height = Math.round(s * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    placeLabels();
    draw();
  }

  function currentPoints(now = performance.now()) {
    if (!morph) return api.points;
    const t = Math.min(1, (now - morph.start) / MORPH_MS);
    const k = t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2; // ease in-out
    if (t >= 1) {
      morph = null;
      return api.points;
    }
    return morph.from.map((p, i) => [p[0] + (morph.to[i][0] - p[0]) * k, p[1] + (morph.to[i][1] - p[1]) * k]);
  }

  function drawAxes() {
    const s = size();
    const [ox, oy] = api.toScreen([0, 0]);
    ctx.strokeStyle = "rgba(201, 195, 236, 0.22)";
    ctx.lineWidth = 1;
    // A faint grid every quarter of the view, then the two axes through zero.
    ctx.setLineDash([3, 6]);
    for (let i = 1; i < 4; i++) {
      const p = (i * s) / 4;
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, s);
      ctx.moveTo(0, p);
      ctx.lineTo(s, p);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(201, 195, 236, 0.5)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, oy);
    ctx.lineTo(s, oy);
    ctx.moveTo(ox, 0);
    ctx.lineTo(ox, s);
    ctx.stroke();
    ctx.fillStyle = "rgba(247, 245, 255, 0.85)";
    ctx.font = "600 16px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText("z1", s - 8, Math.min(s - 8, Math.max(18, oy - 8)));
    ctx.textAlign = "left";
    ctx.fillText("z2", Math.min(s - 30, Math.max(8, ox + 8)), 20);
  }

  function draw() {
    const s = size();
    ctx.clearRect(0, 0, s, s);
    drawAxes();
    const pts = currentPoints();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.28)"; // a light rim so dark cats stay visible
    pts.forEach((p, i) => {
      const [x, y] = api.toScreen(p);
      if (x < -DOT || y < -DOT || x > s + DOT || y > s + DOT) return;
      ctx.beginPath();
      ctx.arc(x, y, DOT, 0, Math.PI * 2);
      ctx.fillStyle = colours[i];
      ctx.fill();
      ctx.stroke();
    });
    api.overlay?.(ctx, api);
    if (crosshair) drawCrosshair(crosshair);
    if (morph) frame = requestAnimationFrame(draw);
  }

  function drawCrosshair(z) {
    const [x, y] = api.toScreen(z);
    ctx.strokeStyle = "#ffd166";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 11, 0, Math.PI * 2);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      ctx.moveTo(x + dx * 6, y + dy * 6);
      ctx.lineTo(x + dx * 18, y + dy * 18);
    }
    ctx.stroke();
  }

  api.draw = draw;
  api.setCrosshair = (z) => {
    crosshair = z;
    draw();
  };
  api.setPoints = (next, { animate = true } = {}) => {
    cancelAnimationFrame(frame);
    if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
      morph = { from: currentPoints(), to: next, start: performance.now() };
    }
    api.points = next;
    draw();
  };
  api.stop = () => {
    cancelAnimationFrame(frame);
    morph = null;
    draw();
  };

  // ---------------------------------------------------------------- empty regions and labels

  /** k points in empty regions: cells of a coarse grid with no data in them or next to them. */
  api.emptySpots = (k = 3, cells = 10) => {
    const counts = Array.from({ length: cells }, () => new Array(cells).fill(0));
    const cellOf = ([x, y]) => {
      const { cx, cy, half } = api.view;
      return [Math.floor(((x - cx) / half + 1) * (cells / 2)), Math.floor(((cy - y) / half + 1) * (cells / 2))];
    };
    for (const p of api.points) {
      const [c, r] = cellOf(p);
      if (c >= 0 && r >= 0 && c < cells && r < cells) counts[r][c]++;
    }
    const empty = [];
    for (let r = 1; r < cells - 1; r++) {
      for (let c = 1; c < cells - 1; c++) {
        let around = 0;
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) around += counts[r + dr][c + dc];
        if (around === 0) empty.push([c, r]);
      }
    }
    // Greedy: keep picking the empty cell farthest from the ones already chosen.
    const chosen = [];
    const centre = cells / 2 - 0.5;
    while (chosen.length < k && empty.length) {
      let best = 0;
      let bestScore = -1;
      empty.forEach(([c, r], i) => {
        const d = chosen.length
          ? Math.min(...chosen.map(([cc, rr]) => Math.hypot(c - cc, r - rr)))
          : -Math.hypot(c - centre, r - centre); // first one: as close to the middle as possible
        if (d > bestScore) [bestScore, best] = [d, i];
      });
      chosen.push(empty.splice(best, 1)[0]);
    }
    const { cx, cy, half } = api.view;
    return chosen.map(([c, r]) => [cx + (((c + 0.5) / cells) * 2 - 1) * half, cy - (((r + 0.5) / cells) * 2 - 1) * half]);
  };

  const pinned = []; // [{ z, el }]
  api.addLabel = (z, el) => {
    el.classList.add("latent-label");
    labels.append(el);
    pinned.push({ z, el });
    placeLabels();
    return el;
  };
  function placeLabels() {
    const s = size();
    for (const { z, el } of pinned) {
      const [x, y] = api.toScreen(z);
      el.style.left = `${(x / s) * 100}%`;
      el.style.top = `${(y / s) * 100}%`;
    }
  }

  // ---------------------------------------------------------------- pointer

  /** Index of the training point nearest to a canvas position, and its distance in pixels. */
  api.nearest = (px, py) => {
    let best = -1;
    let bestD = Infinity;
    api.points.forEach((p, i) => {
      const [x, y] = api.toScreen(p);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < bestD) [bestD, best] = [d, i];
    });
    return { index: best, distance: Math.sqrt(bestD) };
  };

  const local = (e) => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  canvas.addEventListener("pointerdown", (e) => {
    const [px, py] = local(e);
    onPick(api.toData(px, py), e);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") return; // no hover on touch screens
    const [px, py] = local(e);
    onHover(api.nearest(px, py), [px, py], e);
  });
  canvas.addEventListener("pointerleave", () => onHover(null));

  new ResizeObserver(resize).observe(canvas);
  return api;
}
