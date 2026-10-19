// Discriminative vs generative.
// Discriminative: picked image -> CNN -> p(label | image), live with cnn_cls.
// Generative: the arrows flip: "cat" -> ? -> an image, sampled from vae_128 and resolving from noise.
// "Ask the classifier to draw a cat" plays the precomputed gradient-ascent frames:
// the classifier grows ever more confident in a psychedelic mess.

import { loadShowcase, getShowcaseCat, loadAnimals, loadDream, classify, decode, drawCat, randn, ModelMissing, renderTraining, hasModel } from "../models.js";
import { imagePicker, flipSwitch, probBars, revealFromNoise } from "../ui.js";
import { setCurrent } from "../state.js";

const DREAM_FRAME_MS = 320;

let flow, imageCanvas, imageSlot, bars, barsEl, picker, flip, dreamButton, sticker;
let items = [];
let generative = false;
let busy = null; // AbortController of the running animation

export async function init(section) {
  flow = section.querySelector(".flow");
  imageSlot = section.querySelector(".flow-image");
  imageCanvas = imageSlot.querySelector("canvas");
  barsEl = section.querySelector(".flow-bars");
  dreamButton = section.querySelector("[data-dream]");
  sticker = section.querySelector(".dream-sticker");

  // Six held-out cats, then three held-out dogs and three wild animals.
  await loadShowcase();
  items = Array.from({ length: 6 }, (_, i) => ({ image: getShowcaseCat(i), label: "cat" }));
  try {
    const animals = await loadAnimals();
    for (const label of ["dog", "wild"]) {
      animals.labels
        .map((l, i) => [l, i])
        .filter(([l]) => l === label)
        .slice(0, 3)
        .forEach(([, i]) => items.push({ image: animals.sprite.get(i), label }));
    }
  } catch {
    /* no animals exported: cats only */
  }
  picker = imagePicker(section.querySelector(".s02-picker"), {
    items,
    columns: 4,
    onPick: (i) => {
      setCurrent(items[i]);
      if (generative) flip.set(false); // picking an image means "classify this"
      else showDiscriminative();
    },
  });

  if (await hasModel("cnn_cls")) {
    bars = probBars(barsEl, ["cat", "dog", "wild"]);
  } else {
    renderTraining(barsEl, "cnn_cls", { compact: true });
  }

  flip = flipSwitch(section.querySelector(".s02-switch"), {
    left: "Discriminative",
    right: "Generative",
    label: "Generative mode",
    onChange: (on) => (on ? showGenerative() : showDiscriminative()),
  });

  dreamButton.addEventListener("click", playDream);
  await showDiscriminative();
}

export function leave() {
  busy?.abort();
}

function startTask() {
  busy?.abort();
  busy = new AbortController();
  sticker.hidden = true;
  return busy.signal;
}

function clearMissing() {
  imageSlot.querySelector(".training-card")?.remove();
  imageCanvas.hidden = false;
}

async function showDiscriminative() {
  startTask();
  generative = false;
  flow.classList.remove("generative");
  clearMissing();
  const item = items[picker.value];
  drawCat(imageCanvas, item.image);
  await updateBars(item.image);
}

async function updateBars(image) {
  if (!bars) return;
  const [result] = await classify([image]);
  bars.update(result.probs);
}

async function showGenerative() {
  const signal = startTask();
  generative = true;
  flow.classList.add("generative");
  clearMissing();
  try {
    const cat = await decode("vae_128", randn(128));
    if (!signal.aborted) await revealFromNoise(imageCanvas, cat, { ms: 1400, signal });
  } catch (err) {
    if (!(err instanceof ModelMissing)) throw err;
    imageCanvas.hidden = true;
    const holder = document.createElement("div");
    renderTraining(holder, "vae_128", { compact: true });
    imageSlot.append(holder.firstElementChild);
  }
}

/** Play the classifier's gradient-ascent "drawing" of a cat, re-classifying every frame live. */
async function playDream() {
  if (generative) flip.set(false, false);
  generative = false;
  flow.classList.remove("generative");
  clearMissing();
  const signal = startTask();
  let dream;
  try {
    dream = await loadDream();
  } catch {
    dreamButton.disabled = true;
    return;
  }
  const cat = dream.classes.indexOf("cat");
  dreamButton.disabled = true;
  for (let f = 0; f < dream.cols; f++) {
    if (signal.aborted) break;
    const frame = dream.frame(cat, f);
    drawCat(imageCanvas, frame);
    await updateBars(frame);
    await new Promise((r) => setTimeout(r, DREAM_FRAME_MS));
  }
  dreamButton.disabled = false;
  if (!signal.aborted) sticker.hidden = false;
}
