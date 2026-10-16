// The image the audience picked last (on slide 2). Slides 3 and 4 run their
// diagrams on the same image, so a dog picked on slide 2 stays a dog.

import { loadShowcase, getShowcaseCat } from "./models.js";

let current = null; // { image: ImageData, label: "cat" | "dog" | "wild" }
const listeners = new Set();

export async function getCurrent() {
  if (!current) {
    await loadShowcase();
    current = { image: getShowcaseCat(0), label: "cat" };
  }
  return current;
}

export function setCurrent(next) {
  current = next;
  for (const fn of listeners) fn(current);
}

/** Call fn(current) whenever the picked image changes. Returns an unsubscribe function. */
export function onCurrentChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
