// Questions: the thinking mascot, credits, and new vae_128 cats gently floating up behind it all.
// Each cat floats with a CSS animation; when one leaves the top it comes back as a fresh cat.
// Without vae_128 the background just stays empty (it's decoration).

import { decode, fromTensor, drawCat, hasModel, randn, ort } from "../models.js";

const COUNT = 10;
const LATENT = 128;

let layer, cats = [];

export async function init(section) {
  layer = section.querySelector(".float-cats");
  if (!(await hasModel("vae_128"))) return;
  const images = await decode("vae_128", new ort.Tensor("float32", randn(COUNT * LATENT), [COUNT, LATENT]));
  for (let i = 0; i < COUNT; i++) {
    const c = document.createElement("canvas");
    c.className = "float-cat";
    drawCat(c, fromTensor(images, i));
    place(c, true);
    c.addEventListener("animationiteration", () => refresh(c));
    layer.append(c);
    cats.push(c);
  }
}

export function enter() {
  layer?.classList.add("running");
}

export function leave() {
  layer?.classList.remove("running");
}

/** Random column, size and speed; `first` spreads the starting heights so they don't rise in a line. */
function place(c, first = false) {
  const duration = 16 + Math.random() * 12;
  c.style.left = `${3 + Math.random() * 90}%`;
  c.style.setProperty("--size", `${56 + Math.random() * 56}px`);
  c.style.setProperty("--drift", `${(Math.random() - 0.5) * 80}px`);
  c.style.animationDuration = `${duration}s`;
  if (first) c.style.animationDelay = `${-Math.random() * duration}s`;
}

async function refresh(c) {
  drawCat(c, await decode("vae_128", randn(LATENT)));
  c.style.left = `${3 + Math.random() * 90}%`;
}
