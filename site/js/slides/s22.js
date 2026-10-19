// Watch a VAE learn: a replay of vae_2d_b1's training (js/learnview.js), with the KL curve.
// KL warm-up lets the cats spread out first; then KL pulls everything into a round N(0, 1) blob.

import { renderTraining } from "../models.js";
import { learnView } from "../learnview.js";

let view;

export async function init(section) {
  const host = section.querySelector(".learn-body");
  try {
    view = await learnView(host, "vae_2d_b1", { kl: true });
  } catch {
    renderTraining(host, "vae_2d_b1");
  }
}

export const enter = (info) => view?.enter(info);
export const leave = () => view?.leave();
