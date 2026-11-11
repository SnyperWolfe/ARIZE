// ARIZE phone app: makes it installable, and shows a friendly page when ARIZE isn't
// running (for example after Android closed Termux). The page itself is never cached:
// it carries a secret that changes every time ARIZE starts.
const CACHE = "arize-v1";
const OFFLINE = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARIZE is asleep</title><style>
body{margin:0;min-height:100vh;display:flex;flex-direction:column;align-items:center;
justify-content:center;gap:16px;padding:24px;box-sizing:border-box;background:#04070c;
color:#d8f3ff;font-family:system-ui,sans-serif;text-align:center}
img{width:120px;height:120px;opacity:.7}h1{margin:0;font-size:24px;letter-spacing:.06em}
p{margin:0;color:#8fb0c2;line-height:1.5;max-width:300px}
button{margin-top:8px;height:52px;padding:0 28px;border-radius:26px;border:0;
background:#46dcff;color:#03222b;font:600 16px system-ui,sans-serif}
</style></head><body><img src="/icon-192.png" alt=""><h1>ARIZE is asleep</h1>
<p>Tap the <strong>ARIZE</strong> widget on your home screen to wake it up, then come back.</p>
<button onclick="location.reload()">Try again</button></body></html>`;

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(["/icon-192.png", "/icon-512.png"])));
  self.skipWaiting();
});
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.pathname.startsWith("/api/")) return;
  if (e.request.mode === "navigate") {
    e.respondWith(fetch(e.request).catch(() =>
      new Response(OFFLINE, { headers: { "Content-Type": "text/html; charset=utf-8" } })));
    return;
  }
  e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request)));
});
