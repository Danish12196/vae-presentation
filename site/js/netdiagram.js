// The network diagram shared by slides 3 and 4: the real shapes of our conv trunk,
// followed by either a classifier head or a decoder.
//
//   const net = netDiagram(stageEl, { ending: "decoder", onInfo: (text) => ..., note: true })
//                         (without onInfo the layers are not hoverable)
//   net.next()            show one more layer (returns false when everything is shown)
//   net.showAll()
//   net.setEnding("classifier" | "decoder")
//   net.setWide(true)     every layer as wide as the input (slide 4's identity trap)
//   net.input / net.output (canvases), net.labels (element for the class bars)

const ENCODER = [
  { name: "input image", c: 3, s: 64, image: true },
  { name: "conv 1", c: 32, s: 32 },
  { name: "conv 2", c: 64, s: 16 },
  { name: "conv 3", c: 128, s: 8 },
  { name: "conv 4", c: 256, s: 4 },
];
const DECODER = [
  { name: "transposed conv 1", c: 128, s: 8, decoder: true },
  { name: "transposed conv 2", c: 64, s: 16, decoder: true },
  { name: "transposed conv 3", c: 32, s: 32, decoder: true },
  { name: "output image", c: 3, s: 64, image: true, decoder: true },
];
const CLASSIFIER = [
  { name: "flatten", n: 4096, vector: true },
  { name: "label head", n: 3, labels: true },
];

const INPUT_NUMBERS = 3 * 64 * 64;
const fmt = (n) => n.toLocaleString("en-US");

// Pixel sizes before the responsive --u scale: height follows the spatial size,
// thickness follows the number of channels (square roots keep big layers on screen).
const heightFor = (s) => 34 * (s / 4) ** 0.62;
const depthFor = (c) => 4 * Math.sqrt(c);
const IMAGE = heightFor(64);

function numbers(layer) {
  return layer.n ?? layer.c * layer.s * layer.s;
}
function shapeText(layer) {
  if (layer.labels) return `${layer.n} labels`;
  return layer.n ? fmt(layer.n) : `${layer.c}×${layer.s}×${layer.s}`;
}
function infoText(layer, wide) {
  if (wide && !layer.image && !layer.n) return `${layer.name} · as wide as the input · ${fmt(INPUT_NUMBERS)} numbers`;
  const shape = layer.n ? `${fmt(layer.n)} numbers` : `${layer.c} × ${layer.s} × ${layer.s} = ${fmt(numbers(layer))} numbers`;
  return `${layer.name} · ${shape}`;
}

export function netDiagram(stage, { ending = "decoder", onInfo = null, built = Infinity, note = true } = {}) {
  const net = document.createElement("div");
  net.className = "net";
  const enc = document.createElement("div");
  enc.className = "net-group net-enc";
  const end = document.createElement("div");
  end.className = "net-group net-end";
  net.append(enc, end);
  stage.append(net);

  const api = { input: null, output: null, labels: null };
  let columns = []; // [{ el, layer, block }]
  let shown = built;
  let wide = false;
  let endingName = ending;

  function column(layer) {
    const el = document.createElement("div");
    el.className = "net-col";
    let block;
    if (layer.image) {
      block = document.createElement("canvas");
      block.width = block.height = 64;
      block.className = "cat-canvas layer";
    } else if (layer.labels) {
      block = document.createElement("div");
      block.className = "layer labels";
    } else {
      block = document.createElement("div");
      block.className = `layer block${layer.vector ? " vector" : ""}${layer.decoder ? " dec" : ""}`;
    }
    const label = document.createElement("span");
    label.className = "layer-shape mono";
    el.append(block, label);
    if (onInfo) {
      // Hover, focus or tap a layer to read its shape.
      el.tabIndex = 0;
      const show = () => {
        for (const c of columns) c.el.classList.toggle("active", c.el === el);
        onInfo(infoText(layer, wide));
      };
      el.addEventListener("pointerenter", show);
      el.addEventListener("focus", show);
      el.addEventListener("click", show); // touch
    }
    return { el, layer, block, label };
  }

  function size(c) {
    const { layer, block, label } = c;
    let h, w;
    if (layer.image || layer.labels) [h, w] = [IMAGE, IMAGE];
    else if (layer.vector) [h, w] = [IMAGE, 12];
    else if (wide) [h, w] = [IMAGE, 76]; // wide enough that the blocks touch: one straight pipe
    else [h, w] = [heightFor(layer.s), depthFor(layer.c)];
    if (layer.labels) w = IMAGE * 1.15;
    block.style.setProperty("--h", `${h}px`);
    block.style.setProperty("--w", `${w}px`);
    label.textContent = wide && !layer.image && !layer.n ? `${fmt(INPUT_NUMBERS)} numbers each` : shapeText(layer);
  }

  function build() {
    enc.replaceChildren();
    end.replaceChildren();
    columns = [...ENCODER.map(column)];
    enc.append(...columns.map((c) => c.el));
    const tail = (endingName === "decoder" ? DECODER : CLASSIFIER).map(column);
    if (endingName === "decoder" && note) {
      const noteEl = document.createElement("p");
      noteEl.className = "net-note";
      noteEl.textContent = "transposed convolution = convolution in reverse: grows the image";
      end.append(noteEl);
    }
    end.append(...tail.map((c) => c.el));
    columns.push(...tail);
    net.dataset.ending = endingName;
    for (const c of columns) size(c);
    api.input = columns[0].block;
    api.output = endingName === "decoder" ? tail.at(-1).block : null;
    api.labels = endingName === "classifier" ? tail.at(-1).block : null;
    update();
  }

  function update() {
    columns.forEach((c, i) => c.el.classList.toggle("unbuilt", i >= shown));
    end.classList.toggle("started", shown > ENCODER.length);
  }

  api.next = () => {
    if (shown >= columns.length) return false;
    shown++;
    update();
    const c = columns[shown - 1];
    for (const o of columns) o.el.classList.toggle("active", o === c);
    onInfo?.(infoText(c.layer, wide));
    return shown < columns.length;
  };
  api.reset = () => {
    shown = 1;
    update();
    onInfo?.(infoText(columns[0].layer, wide));
  };
  api.showAll = () => {
    shown = Infinity;
    update();
  };
  api.complete = () => shown >= columns.length;
  api.setEnding = (name) => {
    endingName = name;
    build();
    return api;
  };
  api.setWide = (on) => {
    wide = on;
    net.classList.toggle("wide", on);
    for (const c of columns) size(c);
  };
  build();
  return api;
}
