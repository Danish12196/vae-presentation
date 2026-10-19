// Service worker: keeps the whole presentation working offline (classroom Wi-Fi can fail).
//
// - install: precache the site shell (HTML, CSS, JS, fonts) listed in precache.json
// - the big files (models, data, the ONNX runtime) are cached as the loading screen downloads
//   them through this worker; a "status" message tops up anything still missing
// - every request is served cache-first
// - a navigation to / or /index.html (any query string) is the presentation, cached as "./"
// - any other navigation that isn't a real file is 404.html with status 404, online and offline
// - a new VERSION means a new cache; old caches are deleted on activation
//
// VERSION is rewritten by tools/stamp_site.py whenever any site file changes (models and data
// have a content hash in their names, so a hash of data/files.json covers them).

const VERSION = "22727d0679d3";
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

// The presentation lives at the scope root and at index.html. Slide numbers are hashes
// (#s05), so a path like /s05 is a missing address, not a slide.
function isAppNavigation(url) {
  const scopePath = new URL(self.registration.scope).pathname;
  const withSlash = scopePath.endsWith("/") ? scopePath : `${scopePath}/`;
  const noSlash = withSlash.replace(/\/$/, "") || "/";
  return url.pathname === withSlash || url.pathname === noSlash || url.pathname === `${withSlash}index.html`;
}

// A cached 404.html was stored as HTTP 200. Rebuild it so the address still reports 404,
// and never store that copy under the missing URL.
async function notFoundResponse() {
  const cached = await (await caches.open(CACHE)).match(key("404.html"));
  if (!cached) {
    return new Response("Not found", {
      status: 404,
      statusText: "Not Found",
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }
  const headers = new Headers(cached.headers);
  headers.delete("Content-Length");
  return new Response(await cached.blob(), { status: 404, statusText: "Not Found", headers });
}

async function cacheFirst(req, url) {
  const cache = await caches.open(CACHE);
  const navigate = req.mode === "navigate";
  const app = navigate && isAppNavigation(url);
  const k = app ? key("index.html") : url.origin + url.pathname;
  const hit = await cache.match(k);
  if (hit) return hit;
  try {
    const res = await fetch(req);
    if (res.ok && res.type === "basic" && !res.redirected) cache.put(k, res.clone());
    if (navigate && !app && !res.ok) return notFoundResponse();
    return res;
  } catch {
    if (navigate && !app) return notFoundResponse();
    return Response.error();
  }
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
