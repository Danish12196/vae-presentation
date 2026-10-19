// Watch it learn: a replay of ae_2's training (js/learnview.js). The loss falls, the four cats go
// from mush to blurry cats, and the 2-D latent map spreads out and organises itself.

import { renderTraining } from "../models.js";
import { learnView } from "../learnview.js";

let view;

export async function init(section) {
  const host = section.querySelector(".learn-body");
  try {
    view = await learnView(host, "ae_2");
  } catch {
    renderTraining(host, "ae_2");
  }
}

export const enter = (info) => view?.enter(info);
export const leave = () => view?.leave();
