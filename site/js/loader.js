// First-visit loading screen: download every model and data file with a progress bar,
// then create every ONNX session and run it once, so no demo stalls the first time.
// On later visits the service worker serves everything from the cache in a second or two.

import { loadJSON, provideBytes, setWasmBinary, warmUp } from "./models.js";

const WASM = "vendor/ort-1.30.0/ort-wasm-simd-threaded.wasm";
const PARALLEL = 4;
const MB = 1e6;

export async function runLoader(root = document.getElementById("loader")) {
  const fill = root.querySelector(".loader-fill");
  const text = root.querySelector(".loader-text");
  const errorEl = root.querySelector(".loader-error");
  const setBar = (fraction) => (fill.style.width = `${Math.min(100, fraction * 100)}%`);

  try {
    // What to download: the models and data (files.json, from export.py), plus the runtime.
    const [{ files }, precache] = await Promise.all([loadJSON("data/files.json"), loadJSON("precache.json").catch(() => null)]);
    const wasmBytes = precache?.files.find((f) => f.path === WASM)?.bytes ?? 14.3 * MB;
    const items = [...files, { path: WASM, bytes: wasmBytes }];
    const total = items.reduce((sum, f) => sum + f.bytes, 0);

    let loaded = 0;
    const show = () => {
      setBar((0.85 * loaded) / total); // the last 15% is the warm-up
      text.textContent = `downloading cats: ${Math.round(loaded / MB)} / ${Math.round(total / MB)} MB`;
    };
    show();

    await parallel(items, PARALLEL, async (item) => {
      const buffer = await download(item.path, (n) => {
        loaded += n;
        show();
      });
      if (item.path === WASM) setWasmBinary(buffer);
      else if (item.path.endsWith(".onnx")) provideBytes(item.path, buffer);
      // data files: reading them once is enough, they are now in the cache.
    });

    await warmUp((done, count) => {
      setBar(0.85 + (0.15 * done) / count);
      text.textContent = `waking up the models: ${done} / ${count}`;
    });
    text.textContent = "ready!";
    setBar(1);
    finish(root);
  } catch (err) {
    console.error(err);
    errorEl.hidden = false;
    errorEl.textContent = "Some cats couldn't be fetched (are you offline?). The demos that need them will say so.";
    const button = document.createElement("button");
    button.className = "btn primary";
    button.textContent = "Continue anyway";
    button.addEventListener("click", () => finish(root));
    errorEl.after(button);
    await new Promise((resolve) => button.addEventListener("click", resolve, { once: true }));
  }
}

function finish(root) {
  root.classList.add("done");
  setTimeout(() => (root.hidden = true), 450);
}

/** Fetch a file, reporting progress in bytes as they arrive. */
async function download(path, onBytes) {
  const res = await fetch(new URL(`../${path}`, import.meta.url));
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const reader = res.body.getReader();
  const chunks = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    size += value.length;
    onBytes(value.length);
  }
  const out = new Uint8Array(size);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out.buffer;
}

/** Run fn over items with at most `limit` running at once. */
async function parallel(items, limit, fn) {
  const queue = [...items];
  const workers = Array.from({ length: limit }, async () => {
    while (queue.length) await fn(queue.shift());
  });
  await Promise.all(workers);
}
