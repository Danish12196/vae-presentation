// Everything the demos need to talk to the models and the precomputed data.
//
// Images travel as float32 tensors [B, 3, 64, 64] with values in [0, 1],
// exactly what the PyTorch models were trained on.
//
//   await encode("ae_64", imgs)      -> { z }            (VAEs: { mu, logvar })
//   await decode("ae_64", z)         -> image tensor [B, 3, 64, 64]
//   await classify(imgs)             -> [{ probs: {cat, dog, wild}, top }]
//   drawCat(canvas, image)           -> paints a 64x64 image
//   drawTiles(canvas, batch, opts)   -> paints a grid of those images
//
// Sessions are created on first use and cached. If a model hasn't been exported,
// the helpers throw ModelMissing and the demo can call renderTraining().

import * as ort from "../vendor/ort-1.30.0/ort.wasm.bundle.min.js";
import { mascot } from "./mascot.js";

export const SIZE = 64;
const PIXELS = SIZE * SIZE;
const ROOT = new URL("../", import.meta.url); // the site root, wherever it is hosted
const url = (path) => new URL(path, ROOT).href;

// WASM only: our models are tiny (64x64 inputs, 1-3M weights), so each call takes a few
// milliseconds on the CPU, and the WebGPU runtime would not fit the 25 MB per-file limit.
// Threads need a cross-origin isolated page (SharedArrayBuffer); otherwise run single-threaded.
// Only the .wasm path: the glue code is already inside the bundle.
ort.env.wasm.wasmPaths = { wasm: url("vendor/ort-1.30.0/ort-wasm-simd-threaded.wasm") };
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
const SESSION_OPTIONS = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };

export class ModelMissing extends Error {
  constructor(name) {
    super(`model "${name}" has no browser files (not trained or not exported yet)`);
    this.name = "ModelMissing";
    this.model = name;
  }
}

// ---------------------------------------------------------------- manifest and files

const FILES = "data/files.json";
const MANIFEST = "models/manifest.json";

// Models and data files have a content hash in their names (showcase.3fa9c1.png) so the host can
// cache them forever. Code asks for the logical name (data/showcase.png); files.json maps it.
let fileMap = null;
async function resolve(path) {
  if (!/^(data|models)\//.test(path) || path === FILES || path === MANIFEST) return path;
  fileMap ??= loadJSON(FILES).then(({ files }) => new Map(files.map((f) => [f.name, f.path])));
  return (await fileMap).get(path) ?? path;
}

const jsonCache = new Map();
export function loadJSON(path) {
  if (!jsonCache.has(path)) {
    jsonCache.set(
      path,
      resolve(path)
        .then((real) => fetch(url(real)))
        .then((r) => {
          if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
          return r.json();
        }),
    );
  }
  return jsonCache.get(path);
}

export const loadManifest = () => loadJSON(MANIFEST);

/** The manifest entry for a model that can run in the browser; throws ModelMissing otherwise. */
export async function getModel(name) {
  const entry = (await loadManifest()).models?.[name];
  if (!entry || !entry.files || Object.keys(entry.files).length === 0) throw new ModelMissing(name);
  return entry;
}

export async function hasModel(name) {
  try {
    await getModel(name);
    return true;
  } catch {
    return false;
  }
}

// The loading screen hands over bytes it already downloaded, so nothing is fetched twice.
const preloaded = new Map();
export function provideBytes(path, buffer) {
  preloaded.set(path, buffer);
}
export function setWasmBinary(buffer) {
  ort.env.wasm.wasmBinary = buffer;
}

// ---------------------------------------------------------------- sessions

const sessions = new Map(); // "name/role" -> Promise<{ session, queue }>

async function getSession(name, role) {
  const key = `${name}/${role}`;
  if (!sessions.has(key)) {
    const promise = (async () => {
      const entry = await getModel(name);
      const file = entry.files[role]?.file;
      if (!file) throw new ModelMissing(`${name} (${role})`);
      const path = `models/${file}`;
      let bytes = preloaded.get(path);
      preloaded.delete(path);
      if (!bytes) bytes = await (await fetch(url(path))).arrayBuffer();
      const session = await ort.InferenceSession.create(new Uint8Array(bytes), SESSION_OPTIONS);
      return { session, queue: Promise.resolve() };
    })();
    promise.catch(() => sessions.delete(key)); // let a later call retry
    sessions.set(key, promise);
  }
  return sessions.get(key);
}

/** Run one session. Calls on the same session are queued, because a session runs one call at a time. */
async function run(name, role, feeds) {
  const s = await getSession(name, role);
  const result = s.queue.then(() => s.session.run(feeds));
  s.queue = result.catch(() => {});
  return result;
}

// ---------------------------------------------------------------- the model helpers

export async function encode(name, imgs) {
  return run(name, "encoder", { image: asImageTensor(imgs) }); // { z } or { mu, logvar }
}

export async function decode(name, z) {
  const entry = await getModel(name);
  const out = await run(name, "decoder", { z: asCodeTensor(z, entry.latent) });
  return out.image;
}

/** Encode then decode with an autoencoder: the reconstruction as an image tensor [B, 3, 64, 64]. */
export async function reconstruct(name, imgs) {
  const codes = await encode(name, imgs);
  return decode(name, codes.z ?? codes.mu); // VAEs: decode the mean
}

/** Summed squared error between an image and a reconstruction (same measure as the training loss). */
export function squaredError(image, recon, index = 0) {
  const x = toTensor(image).data;
  const y = recon instanceof ort.Tensor ? recon.data : recon;
  const base = index * x.length;
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += (x[i] - y[base + i]) ** 2;
  return sum;
}

export async function classify(imgs) {
  const entry = await getModel("cnn_cls");
  const { logits } = await run("cnn_cls", "model", { image: asImageTensor(imgs) });
  const [batch, k] = logits.dims;
  const results = [];
  for (let b = 0; b < batch; b++) {
    const row = softmax(logits.data.subarray(b * k, (b + 1) * k));
    const probs = Object.fromEntries(entry.classes.map((c, i) => [c, row[i]]));
    const top = entry.classes[row.indexOf(Math.max(...row))];
    results.push({ probs, top });
  }
  return results;
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = Array.from(values, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Create every session and run it once on a dummy input, so no demo stalls the first time. */
export async function warmUp(onProgress = () => {}) {
  const { models } = await loadManifest();
  const jobs = [];
  for (const entry of Object.values(models)) {
    for (const role of Object.keys(entry.files ?? {})) jobs.push([entry, role]);
  }
  let done = 0;
  for (const [entry, role] of jobs) {
    const feeds =
      role === "decoder"
        ? { z: new ort.Tensor("float32", new Float32Array(entry.latent), [1, entry.latent]) }
        : { image: new ort.Tensor("float32", new Float32Array(3 * PIXELS).fill(0.5), [1, 3, SIZE, SIZE]) };
    await run(entry.name, role, feeds);
    onProgress(++done, jobs.length);
  }
}

// ---------------------------------------------------------------- tensors and canvases

/** ImageData (or an array of them) -> float32 tensor [B, 3, 64, 64] in [0, 1]. */
export function toTensor(imageDatas) {
  const list = Array.isArray(imageDatas) ? imageDatas : [imageDatas];
  const data = new Float32Array(list.length * 3 * PIXELS);
  list.forEach((img, b) => {
    const px = img.data;
    const base = b * 3 * PIXELS;
    for (let i = 0; i < PIXELS; i++) {
      data[base + i] = px[i * 4] / 255;
      data[base + PIXELS + i] = px[i * 4 + 1] / 255;
      data[base + 2 * PIXELS + i] = px[i * 4 + 2] / 255;
    }
  });
  return new ort.Tensor("float32", data, [list.length, 3, SIZE, SIZE]);
}

/** One image of a [B, 3, 64, 64] tensor (or a flat CHW Float32Array) -> ImageData. */
export function fromTensor(image, index = 0) {
  const data = image instanceof ort.Tensor ? image.data : image;
  const base = index * 3 * PIXELS;
  const out = new ImageData(SIZE, SIZE);
  for (let i = 0; i < PIXELS; i++) {
    out.data[i * 4] = clamp255(data[base + i]);
    out.data[i * 4 + 1] = clamp255(data[base + PIXELS + i]);
    out.data[i * 4 + 2] = clamp255(data[base + 2 * PIXELS + i]);
    out.data[i * 4 + 3] = 255;
  }
  return out;
}

const clamp255 = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));

/** Paint a cat (ImageData, tensor or CHW floats) onto a canvas at its native 64x64. CSS scales it up. */
export function drawCat(canvas, image, index = 0) {
  if (canvas.width !== SIZE || canvas.height !== SIZE) {
    canvas.width = SIZE;
    canvas.height = SIZE;
  }
  const img = image instanceof ImageData ? image : fromTensor(image, index);
  canvas.getContext("2d").putImageData(img, 0, 0);
}

/**
 * Paint a batch of images (an [N, 3, 64, 64] tensor) as a grid of native-size tiles.
 * `gap` is the space between tiles; `inset` shifts the grid so tiles sit in the middle of their cells.
 */
export function drawTiles(canvas, images, { cols, gap = 0, inset = 0, tile = SIZE } = {}) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < images.dims[0]; i++) {
    ctx.putImageData(fromTensor(images, i), (i % cols) * (tile + gap) + inset, Math.floor(i / cols) * (tile + gap) + inset);
  }
}

function asImageTensor(imgs) {
  if (imgs instanceof ort.Tensor) return imgs;
  if (imgs instanceof Float32Array) return new ort.Tensor("float32", imgs, [imgs.length / (3 * PIXELS), 3, SIZE, SIZE]);
  return toTensor(imgs);
}

function asCodeTensor(z, latent) {
  if (z instanceof ort.Tensor) return z;
  if (Array.isArray(z) && Array.isArray(z[0])) z = z.flat(); // [[...], [...]] -> one batch
  const data = z instanceof Float32Array ? z : Float32Array.from(z);
  return new ort.Tensor("float32", data, [data.length / latent, latent]);
}

// ---------------------------------------------------------------- sprites

const imageCache = new Map();

/** A sprite sheet of 64px (or 32px) tiles, read back as ImageData per tile. */
export function loadSprite(path, { tile = SIZE, cols = 8, count } = {}) {
  if (!imageCache.has(path)) {
    imageCache.set(
      path,
      (async () => {
        // fetch + createImageBitmap rather than <img>.decode(), which can wait for a rendered
        // frame and so stalls while the page isn't being painted.
        const res = await fetch(url(await resolve(path)));
        if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
        const img = await createImageBitmap(await res.blob());
        const sheet = document.createElement("canvas");
        sheet.width = img.width;
        sheet.height = img.height;
        const ctx = sheet.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        const total = count ?? cols * Math.floor(img.height / tile);
        return {
          tile,
          count: total,
          image: img,
          get(i) {
            const r = Math.floor(i / cols);
            const c = i % cols;
            return ctx.getImageData(c * tile, r * tile, tile, tile);
          },
        };
      })(),
    );
  }
  return imageCache.get(path);
}

let showcase = null;

/** The 64 held-out cats every demo starts from. */
export async function loadShowcase() {
  if (!showcase) {
    const meta = await loadJSON("data/showcase.json");
    showcase = { ...meta, sprite: await loadSprite("data/showcase.png", meta) };
  }
  return showcase;
}

/** ImageData of showcase cat i (call loadShowcase() first). */
export function getShowcaseCat(i) {
  if (!showcase) throw new Error("call loadShowcase() first");
  return showcase.sprite.get(i);
}

/** 8 held-out dogs and 8 wild animals for the classifier demo. */
export async function loadAnimals() {
  const meta = await loadJSON("data/animals.json");
  return { ...meta, sprite: await loadSprite("data/animals.png", meta) };
}

/** PCA reconstructions of the showcase cats with k components, plus their squared errors. */
export async function loadPCA(k) {
  const meta = await loadJSON("data/pca.json");
  const sprite = await loadSprite(`data/pca_k${k}.png`, { count: 64 });
  return { k, sprite, errors: meta.squared_error[k] };
}

/** The six "not a cat" images, with ae_64's errors and verdicts. */
export async function loadNotCats() {
  const meta = await loadJSON("data/not_cats.json");
  return { ...meta, sprite: await loadSprite("data/not_cats.png", meta) };
}

/** The classifier "drawing" each class: frame(classIndex, frameIndex), last frame = final image. */
export async function loadDream() {
  const meta = await loadJSON("data/cls_dream.json");
  const sprite = await loadSprite("data/cls_dream.png", meta);
  return { ...meta, sprite, frame: (c, f) => sprite.get(c * meta.cols + f) };
}

/** n samples from N(0, 1) (Box-Muller), e.g. a random latent code for a VAE. */
export function randn(n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * Math.log(1 - Math.random()));
    const a = 2 * Math.PI * Math.random();
    out[i] = r * Math.cos(a);
    if (i + 1 < n) out[i + 1] = r * Math.sin(a);
  }
  return out;
}

// ---------------------------------------------------------------- missing models

/**
 * The friendly card a demo shows when its model hasn't been trained or exported yet.
 * compact: a small version that fits inside a diagram or a single panel slot.
 */
export function renderTraining(el, modelName, { compact = false } = {}) {
  el.innerHTML = `
    <div class="card training-card${compact ? " compact" : ""}">
      ${mascot({ expression: "sleepy" })}
      <div>
        <h3>This cat is still training</h3>
        ${compact ? `<p><code>${modelName}</code></p>` : `<p>The <code>${modelName}</code> model isn't exported yet. Check back after its next nap.</p>`}
      </div>
    </div>`;
}

/**
 * Run fn() for one slot of a demo (a figure holding a canvas). If a model is missing, a compact
 * training card covers the slot; the next successful run removes it again. Returns fn's result or null.
 */
export async function inSlot(slot, fn) {
  try {
    const result = await fn();
    slot.querySelector(":scope > .missing")?.remove();
    slot.classList.remove("is-missing");
    return result;
  } catch (err) {
    if (!(err instanceof ModelMissing)) throw err;
    let holder = slot.querySelector(":scope > .missing");
    if (!holder) {
      holder = document.createElement("div");
      holder.className = "missing";
      slot.append(holder);
    }
    renderTraining(holder, err.model, { compact: true });
    slot.classList.add("is-missing");
    return null;
  }
}

/** Run fn(); if a model is missing, show the training card in el instead. Returns fn's result or null. */
export async function withModel(el, fn, options) {
  try {
    return await fn();
  } catch (err) {
    if (!(err instanceof ModelMissing)) throw err;
    renderTraining(el, err.model, options);
    return null;
  }
}
