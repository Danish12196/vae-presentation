// The presentation shell: navigation, top bar, overlays, loading, and lazy slide modules.
//
// Each slide can have a module js/slides/sNN.js exporting:
//   init(sectionEl)   once, when the slide first comes near the screen
//   enter(info)       every time it becomes the current slide (optional); info is
//                     { direction: 1 | -1, byKey } where byKey means the presenter used
//                     the keys or the prev/next buttons (rather than scrolling or swiping)
//   leave()           every time it stops being current, e.g. to pause animations (optional)
//   step(dir)         optional: return true to use a next/previous press inside the slide
//                     (e.g. reveal its next row) instead of moving to another slide

import { mountAll } from "./mascot.js";
import { runLoader } from "./loader.js";
import { startOffline, showOfflineStatus } from "./offline.js";

const slides = [...document.querySelectorAll("#deck > .slide")];
const progressFill = document.querySelector(".progress-fill");
const partEl = document.querySelector(".topbar-part");
const countEl = document.querySelector(".slide-count");
const navButtons = document.querySelector(".nav-buttons");
const help = document.getElementById("help");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

let current = -1;     // the slide on screen
let target = 0;       // where key presses are heading (runs ahead of `current` during a scroll)
let autoScrolling = false;
let autoScrollTimer = 0;
let keyTarget = -1;    // the slide a key press or nav button is heading for (-1: none)

// ---------------------------------------------------------------- moving between slides

function goTo(index, { instant = false } = {}) {
  target = Math.max(0, Math.min(slides.length - 1, index));
  autoScrolling = true;
  clearTimeout(autoScrollTimer);
  autoScrollTimer = setTimeout(() => (autoScrolling = false), 1000); // fallback for browsers without scrollend
  slides[target].scrollIntoView({ behavior: instant || reducedMotion.matches ? "instant" : "smooth", block: "start" });
}
/** One step forward or back: the current slide may use it first (see step(dir) above). */
function step(dir) {
  if (target === current && ready.get(slides[current].id)?.step?.(dir)) return;
  goTo(target + dir);
  keyTarget = target;
}
const next = () => step(1);
const prev = () => step(-1);

for (const type of ["wheel", "touchstart"]) addEventListener(type, () => (keyTarget = -1), { passive: true });

addEventListener("scrollend", () => {
  autoScrolling = false;
  if (current >= 0) target = current;
});

// The current slide is whichever one crosses the middle of the screen.
const currentObserver = new IntersectionObserver(
  (entries) => {
    for (const e of entries) if (e.isIntersecting) setCurrent(slides.indexOf(e.target));
  },
  { rootMargin: "-50% 0px -50% 0px" },
);
slides.forEach((s) => currentObserver.observe(s));

function setCurrent(index) {
  if (index === current) return;
  const previous = current;
  current = index;
  if (!autoScrolling) target = index;

  const slide = slides[index];
  document.documentElement.dataset.accent = slide.dataset.accent;
  progressFill.style.width = `${((index + 1) / slides.length) * 100}%`;
  partEl.textContent = slide.querySelector(".part-label")?.textContent ?? "";
  countEl.textContent = `${String(index + 1).padStart(2, "0")} / ${slides.length}`;
  navButtons.querySelector('[data-nav="prev"]').disabled = index === 0;
  navButtons.querySelector('[data-nav="next"]').disabled = index === slides.length - 1;
  history.replaceState(null, "", `#${slide.id}`);

  const info = { direction: previous < 0 || index > previous ? 1 : -1, byKey: index === keyTarget };
  if (index === keyTarget) keyTarget = -1;
  if (previous >= 0) callHook(slides[previous], "leave");
  callHook(slide, "enter", info);
}

function indexFromHash() {
  const i = slides.findIndex((s) => `#${s.id}` === location.hash);
  return i < 0 ? null : i;
}
addEventListener("hashchange", () => {
  const i = indexFromHash();
  if (i !== null && i !== current) goTo(i);
});

// ---------------------------------------------------------------- keyboard

const TYPING = 'input:not([type="range"]):not([type="checkbox"]):not([type="radio"]), textarea, select, [contenteditable]';
const ACTIVATES_ON_SPACE = 'button, a[href], summary, input[type="checkbox"], input[type="radio"], [role="tab"]';
const USES_ARROWS = 'input[type="range"], [role="tab"], [role="slider"]';

addEventListener("keydown", (e) => {
  if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target instanceof Element ? e.target : document.body;
  if (e.key === "Escape") return closeOverlays();
  if (t.closest(TYPING)) return;

  const handled = (() => {
    switch (e.key) {
      case "PageDown": // clickers send PageUp/PageDown: always navigate
        next();
        return true;
      case "PageUp":
        prev();
        return true;
      case " ":
        if (t.closest(ACTIVATES_ON_SPACE)) return false;
        e.shiftKey ? prev() : next();
        return true;
      case "ArrowDown":
      case "ArrowRight":
        if (t.closest(USES_ARROWS)) return false;
        next();
        return true;
      case "ArrowUp":
      case "ArrowLeft":
        if (t.closest(USES_ARROWS)) return false;
        prev();
        return true;
      case "Home":
      case "End":
        if (t.closest(USES_ARROWS)) return false;
        goTo(e.key === "Home" ? 0 : slides.length - 1);
        return true;
      case "f":
      case "F":
        toggleFullscreen();
        return true;
      case "?":
        help.hidden = !help.hidden;
        return true;
    }
    return false;
  })();
  if (handled) e.preventDefault();
});

// After a slider or button is used with the mouse or a finger, hand focus back to the page,
// so the arrow keys and space move slides again instead of nudging the slider.
document.addEventListener(
  "pointerup",
  () =>
    setTimeout(() => {
      const a = document.activeElement;
      if (a && a !== document.body && a.matches('input[type="range"], button, [role="tab"], [role="slider"], input[type="checkbox"], select')) a.blur();
    }),
  true,
);

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen?.();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}

function closeOverlays() {
  help.hidden = true;
  const remember = document.querySelector("#remember");
  if (remember) remember.hidden = true;
}
help.addEventListener("click", (e) => {
  if (e.target === help) closeOverlays();
});
document.querySelector("#remember")?.addEventListener("click", (e) => {
  if (e.target.id === "remember") closeOverlays();
});

// ---------------------------------------------------------------- touch: prev/next buttons
// Swiping between slides is the browser's own scroll snapping. Demos mark draggable areas
// with class="interactive" (touch-action: none), so a swipe that starts there stays in the demo.

navButtons.addEventListener("click", (e) => {
  const dir = e.target.closest("[data-nav]")?.dataset.nav;
  if (dir === "next") next();
  if (dir === "prev") prev();
});

let idleTimer = 0;
function wake() {
  navButtons.classList.remove("idle");
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => navButtons.classList.add("idle"), 2500);
}
// Only a mouse or a finger wakes them. Keys and clickers move without them, so they don't sit on
// top of a slide's bottom caption for the first seconds after every press while presenting.
for (const type of ["pointermove", "pointerdown", "touchstart", "wheel"]) {
  addEventListener(type, wake, { passive: true });
}
wake();

// ---------------------------------------------------------------- slide modules

const modules = new Map(); // slide id -> Promise<module | null>
const ready = new Map();   // slide id -> module, once its init has finished

function loadModule(slide) {
  if (!modules.has(slide.id)) {
    const promise = import(`./slides/${slide.id}.js`)
      .then(async (mod) => {
        await mod.init?.(slide);
        ready.set(slide.id, mod);
        return mod;
      })
      .catch((err) => {
        // No module yet for this slide is fine; a broken module is worth a warning.
        if (!/dynamically imported module|Importing a module script failed/i.test(err.message)) {
          console.warn(`slide ${slide.id}:`, err);
        }
        return null;
      });
    modules.set(slide.id, promise);
  }
  return modules.get(slide.id);
}

async function callHook(slide, hook, arg) {
  if (!modulesStarted) return;
  const mod = await loadModule(slide);
  // Only enter if it is still the current slide by the time the module is ready.
  if (hook === "enter" && slides[current] !== slide) return;
  mod?.[hook]?.(arg);
}

let modulesStarted = false;
function startModules() {
  modulesStarted = true;
  // Load a slide's module when it comes within one screen of the viewport.
  const nearObserver = new IntersectionObserver(
    (entries) => {
      for (const e of entries) if (e.isIntersecting) loadModule(e.target);
    },
    { rootMargin: "100% 0px" },
  );
  slides.forEach((s) => nearObserver.observe(s));
  if (current >= 0) callHook(slides[current], "enter", { direction: 1, byKey: false });
}

// ---------------------------------------------------------------- start

mountAll();
const start = indexFromHash();
if (start !== null) goTo(start, { instant: true });

(async () => {
  await startOffline();      // the service worker must be in control before the big downloads
  await runLoader();
  startModules();
  showOfflineStatus();
})();
