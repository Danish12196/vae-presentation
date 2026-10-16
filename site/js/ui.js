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

/** A strip of showcase cats to choose from. onPick(index) runs on every choice. */
export async function catPicker(el, { count = 8, start = 0, selected = 0, onPick = () => {} } = {}) {
  await loadShowcase();
  const strip = h("div", { class: "cat-picker", role: "group", "aria-label": "Pick a cat" });
  const buttons = [];
  for (let i = start; i < start + count; i++) {
    const canvas = h("canvas", { width: 64, height: 64 });
    drawCat(canvas, getShowcaseCat(i));
    const button = h("button", { type: "button", class: "cat-pick", "aria-label": `Cat ${i + 1}`, "aria-pressed": "false" }, canvas);
    button.addEventListener("click", () => pick(i));
    buttons.push(button);
    strip.append(button);
  }
  el.append(strip);

  let current = selected;
  function pick(i, notify = true) {
    current = i;
    buttons.forEach((b, j) => b.setAttribute("aria-pressed", String(j + start === i)));
    if (notify) onPick(i);
  }
  pick(selected, false);
  return {
    el: strip,
    get value() {
      return current;
    },
    set: pick,
  };
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
