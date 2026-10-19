// Service worker: keeps the whole presentation working offline (classroom Wi-Fi can fail).
//
// - install: precache the site shell (HTML, CSS, JS, fonts) listed in precache.json
// - the big files (models, data, the ONNX runtime) are cached as the loading screen downloads
//   them through this worker; a "status" message tops up anything still missing
// - every request is served cache-first
// - a new VERSION means a new cache; old caches are deleted on activation
//
// VERSION is rewritten by tools/stamp_site.py whenever any site file changes (models and data
// have a content hash in their names, so a hash of data/files.json covers them).

const VERSION = "678f50a7b0b4";
const CACHE = `cats-${VERSION}`;
const BIG = /^(models|data)\/|\.wasm$/;

const scoped = (path) => new URL(path, self.registration.scope).href;
// The page is cached as "./": hosts with clean URLs redirect /index.html to /, and a
// redirected response can't be used to answer a navigation.
const key = (path) => scoped(path === "index.html" ? "./" : path);

async function precacheList() {
  const cached = await caches.match(scoped("precache.json"));
  const res = cached ?? (await fetch(scoped("precache.json"), { cache: "no-store" }));
  return (await res.json()).files.map((f) => f.path);
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const res = await fetch(scoped("precache.json"), { cache: "no-store" });
      const { files } = await res.clone().json();
      const cache = await caches.open(CACHE);
      await cache.put(scoped("precache.json"), res);
      const shell = files.map((f) => f.path).filter((p) => !BIG.test(p));
      await Promise.all(
        shell.map(async (p) => {
          const res = await fetch(new Request(key(p), { cache: "no-cache" }));
          if (!res.ok) throw new Error(`${p}: HTTP ${res.status}`);
          await cache.put(key(p), res);
        }),
      );
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      for (const key of await caches.keys()) {
        if (key.startsWith("cats-") && key !== CACHE) await caches.delete(key);
      }
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin || !req.url.startsWith(self.registration.scope)) return;
  event.respondWith(cacheFirst(req, url));
});

async function cacheFirst(req, url) {
  const cache = await caches.open(CACHE);
  // Every page navigation is the one-page presentation; query strings don't matter.
  const k = req.mode === "navigate" ? key("index.html") : url.origin + url.pathname;
  const hit = await cache.match(k);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok && res.type === "basic" && !res.redirected) cache.put(k, res.clone());
  return res;
}

// The page asks "is everything cached?" after loading; fetch whatever is missing, then answer.
self.addEventListener("message", (event) => {
  if (event.data?.type !== "status") return;
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      const missing = [];
      for (const path of await precacheList()) {
        if (await cache.match(key(path))) continue;
        try {
          const res = await fetch(key(path));
          if (res.ok) await cache.put(key(path), res);
          else missing.push(path);
        } catch {
          missing.push(path);
        }
      }
      event.source?.postMessage({ type: "status", complete: missing.length === 0, missing, version: VERSION });
    })(),
  );
});
