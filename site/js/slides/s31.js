// Wrap-up: the story of the talk, six rows revealed one at a time.
// When the presenter arrives with the keys or the buttons going forward, the rows start hidden
// and each next press reveals one (main.js asks step() first). Arriving any other way
// (scrolling, swiping, or coming back) shows everything, so no one gets stuck.
// A click or tap anywhere on the slide also reveals the next row.

const LAST = 6;

let section, items = [], dots = [], shown = LAST;

export function init(el) {
  section = el;
  items = [...el.querySelectorAll("[data-step]")];
  const dotBox = el.querySelector(".wrap-dots");
  dots = Array.from({ length: LAST }, () => dotBox.appendChild(document.createElement("span")));
  el.querySelector(".s20-body").addEventListener("click", (e) => {
    if (e.target.closest("a, button")) return;
    if (shown < LAST) show(shown + 1);
  });
  show(LAST);
}

export function enter({ direction = 1, byKey = false } = {}) {
  show(byKey && direction === 1 ? 0 : LAST);
}

/** Next/previous inside the slide: reveal or hide one row; at either end, let the deck move. */
export function step(dir) {
  if (dir > 0 && shown < LAST) return show(shown + 1), true;
  if (dir < 0 && shown > 0 && shown < LAST) return show(shown - 1), true;
  return false;
}

function show(n) {
  shown = n;
  for (const el of items) el.classList.toggle("shown", Number(el.dataset.step) <= n);
  dots.forEach((d, i) => d.classList.toggle("on", i < n));
  section.classList.toggle("revealing", n < LAST);
}
