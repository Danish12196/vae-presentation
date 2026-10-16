// Service worker: keeps the whole presentation working offline (classroom Wi-Fi can fail).
//
// - install: precache the site shell (HTML, CSS, JS, fonts) listed in precache.json
// - the big files (models, data, the ONNX runtime) are cached as the loading screen downloads
//   them through this worker; a "status" message tops up anything still missing
// - every request is served cache-first
// - a new VERSION means a new cache; old caches are deleted on activation
//
// VERSION is rewritten by tools/stamp_site.py whenever any site file changes.

const VERSION = "63c003451905";
const CACHE = `cats-${VERSION}`;
const BIG = /^(models|data)\/|\.wasm$/;

const scoped = (path) => new URL(path, self.registration.scope).href;

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
      await cache.addAll(shell.map((p) => new Request(scoped(p), { cache: "no-cache" })));
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
  const key = req.mode === "navigate" ? scoped("index.html") : url.origin + url.pathname;
  const hit = await cache.match(key);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok && res.type === "basic") cache.put(key, res.clone());
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
        if (await cache.match(scoped(path))) continue;
        try {
          const res = await fetch(scoped(path));
          if (res.ok) await cache.put(scoped(path), res);
          else missing.push(path);
        } catch {
          missing.push(path);
        }
      }
      event.source?.postMessage({ type: "status", complete: missing.length === 0, missing, version: VERSION });
    })(),
  );
});
