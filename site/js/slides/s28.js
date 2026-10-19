// Why VAE cats are blurry.
// Left: a cat and its vae_128 reconstruction, with a loupe that magnifies the same spot on both.
// Right: the averaging problem. If n different sharp cats are all plausible answers, the guess
// with the lowest MSE is their pixel average, and the average of sharp cats is a blurry cat.

import { drawCat, reconstruct, inSlot, loadShowcase, getShowcaseCat, SIZE, loadJSON, loadSprite } from "../models.js";
import { catPicker, slider } from "../ui.js";

const LOUPE_PX = 12; // the loupe shows a 12x12 pixel patch of the 64x64 image

let orig, recon, hosts, avgCanvas, avgLabel, strip, count, stops = [1], bigSprite = null;
let cat = 0;

export async function init(section) {
  orig = section.querySelector(".s17-orig");
  recon = section.querySelector(".s17-recon");
  hosts = [...section.querySelectorAll(".loupe-host")];
  avgCanvas = section.querySelector(".s17-avg");
  avgLabel = section.querySelector(".s17-avg-label");
  strip = section.querySelector(".s17-strip");

  await loadShowcase();
  await catPicker(section.querySelector(".s17-picker"), {
    onPick: (i) => {
      cat = i;
      showPair();
      showAverage();
    },
  });
  // Up to 64 the average is computed live from the showcase cats; beyond that it was
  // precomputed from the training cats (data/averages.json), all the way to every one of them.
  const big = await loadJSON("data/averages.json").catch(() => null);
  bigSprite = big ? await loadSprite(`data/${big.file}`, big) : null;
  stops = [1, 2, 4, 8, 16, 32, 64, ...(big?.counts ?? [])];
  count = slider(section.querySelector(".s17-slider"), {
    label: "possible answers to hedge between",
    min: 0,
    max: stops.length - 1,
    step: 1,
    value: 0,
    format: (i) => `${stops[i].toLocaleString("en")} cat${stops[i] > 1 ? "s" : ""}`,
    onInput: showAverage,
  });

  // The loupe: hover with a mouse, or press and drag with a finger.
  for (const c of [orig, recon]) {
    c.addEventListener("pointermove", (e) => (e.pointerType !== "touch" || e.buttons) && loupe(e, c));
    c.addEventListener("pointerdown", (e) => loupe(e, c));
    c.addEventListener("pointerleave", hideLoupe);
    c.addEventListener("pointerup", (e) => e.pointerType === "touch" && hideLoupe());
  }

  await showPair();
  showAverage();
}

export function leave() {
  hideLoupe();
}

async function showPair() {
  const image = getShowcaseCat(cat);
  drawCat(orig, image);
  await inSlot(recon.closest("figure"), async () => drawCat(recon, await reconstruct("vae_128", [image])));
}

/** Magnify the same spot on both images. */
function loupe(e, from) {
  const r = from.getBoundingClientRect();
  const u = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  const v = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
  const sx = Math.min(SIZE - LOUPE_PX, Math.max(0, u * SIZE - LOUPE_PX / 2));
  const sy = Math.min(SIZE - LOUPE_PX, Math.max(0, v * SIZE - LOUPE_PX / 2));
  hosts.forEach((host, i) => {
    const source = i === 0 ? orig : recon;
    const lens = host.querySelector(".loupe");
    const ctx = lens.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, lens.width, lens.height);
    ctx.drawImage(source, sx, sy, LOUPE_PX, LOUPE_PX, 0, 0, lens.width, lens.height);
    lens.style.left = `${u * 100}%`;
    lens.style.top = `${v * 100}%`;
    lens.classList.add("on");
  });
}

function hideLoupe() {
  for (const l of document.querySelectorAll("#s22 .loupe")) l.classList.remove("on");
}

/** Pixel average of n cats, starting with the picked one (precomputed past 64). */
function showAverage() {
  const n = stops[count.value];
  const cats = Array.from({ length: Math.min(n, 64) }, (_, k) => getShowcaseCat((cat + k * 7) % 64));
  if (n > 64) {
    drawCat(avgCanvas, bigSprite.get(stops.indexOf(n) - 7));
  } else {
    const avg = new ImageData(SIZE, SIZE);
    for (let i = 0; i < avg.data.length; i++) {
      let sum = 0;
      for (const c of cats) sum += c.data[i];
      avg.data[i] = sum / n;
    }
    drawCat(avgCanvas, avg);
  }
  avgLabel.textContent = n === 1 ? "one sharp cat" : `average of ${n.toLocaleString("en")}${n > 64 ? " training cats" : ""}`;
  // the strip shows at most 8 of the cats being averaged, then how many more
  const more = document.createElement("span");
  more.className = "s17-more mono";
  more.textContent = n > 8 ? `+ ${(n - 8).toLocaleString("en")} more` : "";
  strip.replaceChildren(
    more,
    ...cats.slice(0, 8).map((c) => {
      const t = document.createElement("canvas");
      t.className = "cat-canvas small";
      drawCat(t, c);
      return t;
    }),
  );
}
