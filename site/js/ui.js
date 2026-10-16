// Small widgets the demos share. Each takes a container element, builds its DOM,
// and returns a little object to read or update it.

import { loadShowcase, getShowcaseCat, drawCat } from "./models.js";

const h = (tag, attrs = {}, ...children) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  el.append(...children);
  return el;
};

/** A grid of images to choose from: items = [{ image: ImageData, label }]. onPick(index, item). */
export function imagePicker(el, { items, selected = 0, columns = items.length, onPick = () => {} } = {}) {
  const grid = h("div", { class: "cat-picker image-picker", role: "group", "aria-label": "Pick an image" });
  grid.style.setProperty("--picker-cols", columns);
  const buttons = items.map((item, i) => {
    const canvas = h("canvas", { width: 64, height: 64 });
    drawCat(canvas, item.image);
    const button = h("button", { type: "button", class: "cat-pick", "aria-label": item.label ?? `Image ${i + 1}`, "aria-pressed": "false" }, canvas);
    button.addEventListener("click", () => pick(i));
    return button;
  });
  grid.append(...buttons);
  el.append(grid);

  let current = selected;
  function pick(i, notify = true) {
    current = i;
    buttons.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
    if (notify) onPick(i, items[i]);
  }
  pick(selected, false);
  return {
    el: grid,
    get value() {
      return current;
    },
    set: pick,
  };
}

/** A strip of showcase cats to choose from. onPick(index) runs on every choice. */
export async function catPicker(el, { count = 8, start = 0, selected = 0, onPick = () => {} } = {}) {
  await loadShowcase();
  const items = Array.from({ length: count }, (_, i) => ({ image: getShowcaseCat(start + i), label: `Cat ${start + i + 1}` }));
  const picker = imagePicker(el, { items, selected: selected - start, onPick: (i) => onPick(start + i) });
  return {
    el: picker.el,
    get value() {
      return start + picker.value;
    },
    set: (i, notify) => picker.set(i - start, notify),
  };
}

/**
 * A switch between two named states, e.g. "Discriminative" / "Generative".
 * value false = left, true = right. Clicking either name also flips it.
 */
export function flipSwitch(el, { left, right, value = false, label = `${left} or ${right}`, onChange = () => {} } = {}) {
  const input = h("input", { type: "checkbox", role: "switch", "aria-label": label });
  const leftEl = h("button", { type: "button", class: "flip-side", tabindex: "-1" }, left);
  const rightEl = h("button", { type: "button", class: "flip-side", tabindex: "-1" }, right);
  const wrap = h("div", { class: "flip" }, leftEl, h("label", { class: "toggle" }, input, h("span", { class: "toggle-track" })), rightEl);
  el.append(wrap);

  const set = (v, notify = true) => {
    input.checked = v;
    leftEl.classList.toggle("on", !v);
    rightEl.classList.toggle("on", v);
    if (notify) onChange(v);
  };
  input.addEventListener("change", () => set(input.checked));
  leftEl.addEventListener("click", () => input.checked && set(false));
  rightEl.addEventListener("click", () => !input.checked && set(true));
  set(value, false);
  return {
    el: wrap,
    get value() {
      return input.checked;
    },
    set,
  };
}

/**
 * Draw `image` appearing out of random noise over `ms` milliseconds.
 * Resolves when done; pass an AbortSignal to stop early (it then jumps to the final image).
 */
export function revealFromNoise(canvas, image, { ms = 1200, signal } = {}) {
  drawCat(canvas, image);
  const target = canvas.getContext("2d").getImageData(0, 0, 64, 64);
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return Promise.resolve();
  const noise = new Uint8ClampedArray(target.data.length).map(() => Math.random() * 255);
  const frame = new ImageData(64, 64);
  const ctx = canvas.getContext("2d");
  return new Promise((resolve) => {
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / ms);
      const k = t * t * (3 - 2 * t); // smoothstep
      if (signal?.aborted || t >= 1) {
        ctx.putImageData(target, 0, 0);
        return resolve();
      }
      for (let i = 0; i < frame.data.length; i += 4) {
        frame.data[i] = noise[i] + (target.data[i] - noise[i]) * k;
        frame.data[i + 1] = noise[i + 1] + (target.data[i + 1] - noise[i + 1]) * k;
        frame.data[i + 2] = noise[i + 2] + (target.data[i + 2] - noise[i + 2]) * k;
        frame.data[i + 3] = 255;
      }
      ctx.putImageData(frame, 0, 0);
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

/** A big slider with a label and a live value. format(v) turns the value into text. */
export function slider(el, { label, min = 0, max = 1, step = 0.01, value = min, format = String, ticks, onInput = () => {} } = {}) {
  const input = h("input", { type: "range", min, max, step, value, "aria-label": label });
  const valueEl = h("span", { class: "slider-value" });
  const wrap = h("div", { class: "slider" }, h("div", { class: "slider-top" }, h("span", {}, label), valueEl), input);
  if (ticks) wrap.append(h("div", { class: "slider-ticks" }, ...ticks.map((t) => h("span", {}, t))));
  el.append(wrap);

  const update = () => {
    const v = Number(input.value);
    valueEl.textContent = format(v);
    input.style.setProperty("--fill", `${((v - min) / (max - min)) * 100}%`);
    return v;
  };
  input.addEventListener("input", () => onInput(update()));
  update();
  return {
    el: wrap,
    input,
    get value() {
      return Number(input.value);
    },
    set value(v) {
      input.value = v;
      update();
    },
  };
}

/** An on/off switch. onChange(checked) runs when it flips. */
export function toggle(el, { label, checked = false, onChange = () => {} } = {}) {
  const input = h("input", { type: "checkbox", role: "switch" });
  input.checked = checked;
  input.addEventListener("change", () => onChange(input.checked));
  const wrap = h("label", { class: "toggle" }, input, h("span", { class: "toggle-track" }), h("span", {}, label));
  el.append(wrap);
  return {
    el: wrap,
    get checked() {
      return input.checked;
    },
    set checked(v) {
      input.checked = v;
    },
  };
}

/** A row of tabs: tabs = [{ id, label }]. Arrow keys move between them when one has focus. */
export function tabs(el, { tabs: items, selected = items[0].id, onChange = () => {} } = {}) {
  const bar = h("div", { class: "tabs", role: "tablist" });
  const buttons = items.map(({ id, label }) =>
    h("button", { type: "button", class: "tab", role: "tab", "data-id": id, onclick: () => select(id) }, label),
  );
  bar.append(...buttons);
  bar.addEventListener("keydown", (e) => {
    const i = buttons.indexOf(document.activeElement);
    if (i < 0 || !["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const next = buttons[(i + (e.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length];
    next.focus();
    select(next.dataset.id);
    e.preventDefault();
  });
  el.append(bar);

  let current;
  function select(id, notify = true) {
    current = id;
    for (const b of buttons) {
      const on = b.dataset.id === id;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    if (notify) onChange(id);
  }
  select(selected, false);
  return {
    el: bar,
    get value() {
      return current;
    },
    select,
  };
}

/** Horizontal probability bars, one per class. update({ cat: 0.9, dog: 0.07, ... }). */
export function probBars(el, classes, { colours = {} } = {}) {
  const grid = h("div", { class: "prob-bars" });
  const rows = classes.map((c) => {
    const label = h("span", { class: "prob-label" }, c);
    const fill = h("div", { class: "prob-fill" });
    const value = h("span", { class: "prob-value" }, "–");
    if (colours[c]) [label, fill, value].forEach((n) => n.style.setProperty("--bar-colour", colours[c]));
    grid.append(label, h("div", { class: "prob-track" }, fill), value);
    return { c, label, fill, value };
  });
  el.append(grid);
  return {
    el: grid,
    update(probs) {
      const best = Math.max(...classes.map((c) => probs[c] ?? 0));
      for (const r of rows) {
        const p = probs[r.c] ?? 0;
        r.fill.style.width = `${p * 100}%`;
        r.value.textContent = `${Math.round(p * 100)}%`;
        r.label.classList.toggle("top", p === best);
        r.value.classList.toggle("top", p === best);
      }
    },
  };
}

/** A number that glides to each new value. format(v) turns it into text. */
export function readout(el, { label = "", value = 0, format = (v) => v.toFixed(2), duration = 300 } = {}) {
  const valueEl = h("span", { class: "readout-value" }, format(value));
  const wrap = h("div", { class: "readout" });
  if (label) wrap.append(h("span", { class: "readout-label" }, label));
  wrap.append(valueEl);
  el.append(wrap);

  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  let shown = value;
  let frame = 0;
  return {
    el: wrap,
    set(target) {
      cancelAnimationFrame(frame);
      if (reduced.matches || duration === 0 || !Number.isFinite(shown)) {
        shown = target;
        valueEl.textContent = format(target);
        return;
      }
      const from = shown;
      const start = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - start) / duration);
        shown = from + (target - from) * (1 - (1 - t) ** 3); // ease out
        valueEl.textContent = format(shown);
        if (t < 1) frame = requestAnimationFrame(step);
      };
      frame = requestAnimationFrame(step);
    },
  };
}
