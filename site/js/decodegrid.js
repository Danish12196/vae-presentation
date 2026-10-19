// "What lives here?": cover a latent map with an n x n grid of decoded images, one per cell, so
// the whole space is visible at once (the classic picture from the VAE paper, and Deepia's
// decoded axes). One batched decode per grid; drawn on a canvas laid over the map.
//
//   const grid = decodeGrid(map, () => "ae_2", { n: 12 });
//   await grid.show(true) / grid.show(false); grid.refresh() after the model or view changes

import { decode, fromTensor, SIZE } from "./models.js";

export function decodeGrid(map, modelName, { n = 12 } = {}) {
  const layer = document.createElement("canvas");
  layer.className = "decode-grid";
  layer.setAttribute("aria-hidden", "true");
  layer.hidden = true;
  map.wrap.append(layer);
  let on = false, request = 0;

  async function render() {
    const mine = ++request;
    const { cx, cy, half } = map.view;
    const codes = [];
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        // cell centres, top row = highest z2
        codes.push([cx + ((c + 0.5) / n * 2 - 1) * half, cy - ((r + 0.5) / n * 2 - 1) * half]);
      }
    }
    const images = await decode(modelName(), codes);
    if (mine !== request || !on) return;
    const size = map.canvas.clientWidth;
    const dpr = devicePixelRatio || 1;
    layer.width = layer.height = Math.round(size * dpr);
    const ctx = layer.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    const tile = document.createElement("canvas");
    tile.width = tile.height = SIZE;
    const tctx = tile.getContext("2d");
    const cell = (size * dpr) / n;
    for (let i = 0; i < n * n; i++) {
      tctx.putImageData(fromTensor(images, i), 0, 0);
      ctx.drawImage(tile, (i % n) * cell, Math.floor(i / n) * cell, cell + 0.5, cell + 0.5);
    }
  }

  return {
    get on() {
      return on;
    },
    async show(value) {
      on = value;
      layer.hidden = !value;
      if (value) await render();
    },
    refresh: () => (on ? render() : Promise.resolve()),
  };
}
