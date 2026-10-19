// The identity trap. With a bottleneck the decoder has to rebuild the cat from a squeezed
// summary (ae_64, live). Make every layer as wide as the input and the network can just copy:
// perfect loss, nothing learned. The "on" state is an illustration, not a trained model.

import { encode, decode, drawCat, squaredError, withModel, SIZE } from "../models.js";
import { toggle, readout } from "../ui.js";
import { mountMascot } from "../mascot.js";
import { netDiagram } from "../netdiagram.js";
import { getCurrent, onCurrentChange } from "../state.js";

let net, mascotEl, stamp, loss, features, newCats;
let wide = false;
let bottleneckLoss = NaN;

export async function init(section) {
  net = netDiagram(section.querySelector(".net-stage"), { ending: "decoder", note: false });
  net.showAll();
  mascotEl = section.querySelector(".s04-mascot [data-mascot]");
  stamp = section.querySelector(".stamp");

  const readouts = section.querySelector(".s04-readouts");
  loss = readout(readouts, { label: "loss", value: 0, format: (v) => (Number.isFinite(v) ? v.toFixed(3) : "–"), duration: 700 });
  features = textReadout(readouts, "features learned");
  newCats = textReadout(readouts, "can make new cats");

  toggle(section.querySelector(".s04-toggle"), {
    label: "make every layer as wide as the input",
    onChange: (on) => {
      wide = on;
      update();
    },
  });

  onCurrentChange(update);
  await update();
}

export function enter() {
  update();
}

function textReadout(parent, label) {
  const wrap = document.createElement("div");
  wrap.className = "readout";
  wrap.innerHTML = `<span class="readout-label">${label}</span><span class="readout-value readout-text"></span>`;
  parent.append(wrap);
  const value = wrap.lastElementChild;
  return {
    set(text, bad) {
      if (value.textContent === text) return;
      value.classList.add("swap");
      setTimeout(() => {
        value.textContent = text;
        value.classList.toggle("bad", bad);
        value.classList.remove("swap");
      }, 150);
    },
  };
}

async function update() {
  const { image } = await getCurrent();
  drawCat(net.input, image);
  net.setWide(wide);
  mountMascot(mascotEl, { expression: wide ? "photocopier" : "happy" });
  stamp.hidden = !wide;

  if (wide) {
    drawCat(net.output, image); // the pipe copies the input exactly
    loss.set(0);
    features.set("none", true);
    newCats.set("no", true);
    return;
  }
  features.set("edges, fur, faces", false);
  newCats.set("not yet", false);
  await withModel(net.output.parentElement, async () => {
    const { z } = await encode("ae_64", [image]);
    const out = await decode("ae_64", z);
    drawCat(net.output, out);
    // Mean squared error per pixel value between the input and the reconstruction.
    bottleneckLoss = squaredError(image, out) / (3 * SIZE * SIZE);
  }, { compact: true });
  loss.set(bottleneckLoss);
}
