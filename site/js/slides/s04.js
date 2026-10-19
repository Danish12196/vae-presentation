// Encoder and decoder: build the network one layer at a time with its real shapes,
// then swap what comes after the encoder: a classifier head (live cnn_cls) or a decoder
// (transposed convolutions back up to an image, here ae_256's reconstruction).

import { classify, encode, decode, drawCat, withModel, hasModel } from "../models.js";
import { flipSwitch, probBars } from "../ui.js";
import { netDiagram } from "../netdiagram.js";
import { getCurrent, onCurrentChange } from "../state.js";

const HINT = "hover or tap a layer";

let net, info, ending = "classifier";

export async function init(section) {
  info = section.querySelector(".net-info");
  info.textContent = HINT;
  net = netDiagram(section.querySelector(".net-stage"), {
    ending,
    built: 1,
    onInfo: (text) => (info.textContent = text),
  });

  flipSwitch(section.querySelector(".net-switch"), {
    left: "classifier head",
    right: "decoder",
    label: "Decoder ending",
    onChange: (decoder) => {
      ending = decoder ? "decoder" : "classifier";
      net.setEnding(ending);
      net.play();
      fill();
    },
  });

  onCurrentChange(fill);
  await fill();
  net.play();
}

export function enter() {
  fill();
}

export function leave() {
  net?.stop();
}

/** Draw the current image into the diagram, and run the ending that is showing. */
async function fill() {
  const { image } = await getCurrent();
  drawCat(net.input, image);
  if (ending === "classifier" && net.labels) {
    const labels = net.labels;
    if (!labels.firstChild) {
      if (await hasModel("cnn_cls")) labels.bars = probBars(labels, ["cat", "dog", "wild"]);
    }
    await withModel(labels, async () => {
      const [result] = await classify([image]);
      labels.bars?.update(result.probs);
    }, { compact: true });
  }
  if (ending === "decoder" && net.output) {
    const out = net.output;
    await withModel(out.parentElement, async () => {
      const { z } = await encode("ae_256", [image]);
      drawCat(out, await decode("ae_256", z));
    }, { compact: true });
  }
}
