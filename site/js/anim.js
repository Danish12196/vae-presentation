// Small animation helpers for the step-by-step "explainer" slides (the 3Blue1Brown-style builds).
//
//   await tween(500, (t) => ...)       t eases from 0 to 1; instant under prefers-reduced-motion
//   const b = builder({ last, show })  press-by-press build-up with the deck's usual rules:
//     arriving with the keys going forward starts at 0 and every next press shows one more step;
//     arriving any other way (scroll, swipe, coming back) shows the finished picture.
//     show(n, { animate }) draws step n; export { enter, step } from the slide via b.enter / b.step.

export const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
export const easeInOut = (t) => (t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2);
export const lerp = (a, b, t) => a + (b - a) * t;

let generation = 0;

/** Run fn(t) for t from 0 to 1 over ms. Resolves when done (or at once with reduced motion). */
export function tween(ms, fn, ease = easeInOut) {
  if (reducedMotion() || ms <= 0) {
    fn(1);
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const start = performance.now();
    const frame = (now) => {
      const t = Math.min(1, (now - start) / ms);
      fn(ease(t));
      if (t < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });
}

/** A token that goes stale when a newer animation starts, so an old one can stop drawing. */
export function freshToken() {
  const mine = ++generation;
  return () => mine === generation;
}

/**
 * Step-by-step builds. `show(n, { animate })` must draw the state after n steps (0..last).
 * `stage` (optional): clicking it shows the next step too.
 */
export function builder({ last, show, stage = null }) {
  let shown = last;
  const go = (n, animate) => {
    shown = n;
    show(n, { animate });
  };
  stage?.addEventListener("click", (e) => {
    if (e.target.closest("a, button, input, .interactive")) return;
    if (shown < last) go(shown + 1, true);
  });
  return {
    get shown() {
      return shown;
    },
    redraw: () => show(shown, { animate: false }),
    enter({ direction = 1, byKey = false } = {}) {
      go(byKey && direction === 1 ? 0 : last, false);
    },
    step(dir) {
      if (dir > 0 && shown < last) return go(shown + 1, true), true;
      if (dir < 0 && shown > 0 && shown < last) return go(shown - 1, false), true;
      return false;
    },
  };
}
