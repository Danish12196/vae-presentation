// Registers the service worker (sw.js) and shows "available offline" once every file is cached.
//
// On localhost the service worker is off by default, so edits show up on reload without
// a stale cache. Turn it on with ?sw=on (remembered) and off again with ?sw=off.

const LOCAL = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);

function wanted() {
  if (!("serviceWorker" in navigator)) return false;
  if (!LOCAL) return true;
  try {
    const param = new URLSearchParams(location.search).get("sw");
    if (param) localStorage.setItem("sw", param);
    return localStorage.getItem("sw") === "on";
  } catch {
    return false;
  }
}

/**
 * Start the service worker. Resolves once it controls the page (so the loader's downloads
 * go through it and get cached), or straight away if there is none.
 */
export async function startOffline() {
  if (!wanted()) {
    if ("serviceWorker" in navigator) {
      for (const reg of await navigator.serviceWorker.getRegistrations()) await reg.unregister();
    }
    return false;
  }
  try {
    await navigator.serviceWorker.register(new URL("../sw.js", import.meta.url));
  } catch (err) {
    console.warn("service worker not available:", err);
    return false;
  }
  if (!navigator.serviceWorker.controller) {
    // First visit: wait for the new worker to take over (it claims the page on activation).
    await Promise.race([
      new Promise((resolve) => navigator.serviceWorker.addEventListener("controllerchange", resolve, { once: true })),
      new Promise((resolve) => setTimeout(resolve, 8000)),
    ]);
  }
  return true;
}

/** Ask the worker to check (and top up) its cache, and show the tick once everything is there. */
export async function showOfflineStatus(tick = document.querySelector(".offline-tick")) {
  if (!("serviceWorker" in navigator) || !navigator.serviceWorker.controller) return;
  navigator.serviceWorker.addEventListener("message", (e) => {
    if (e.data?.type === "status") tick.classList.toggle("on", e.data.complete);
  });
  const reg = await navigator.serviceWorker.ready;
  reg.active?.postMessage({ type: "status" });
}
