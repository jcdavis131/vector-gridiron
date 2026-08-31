/* gridiron PWA v67 human-v6 — CORE20 offline13k */
const CACHE='vector-gridiron-v67-human-v6-real646';
const CORE=['/','/index.html','/manifest.json','/offline.html','/assets/human-v6/tokens.css','/assets/human-v6/base.css','/assets/human-v6/navigation.css','/assets/human-v6/individual.css','/assets/human-v6/peers.css','/assets/human-v6/map.css','/assets/human-v6/evidence.css','/assets/human-v6/states.css','/assets/human-v6/motion.css','/assets/human-v6/human-v6.js','/assets/data/gridiron.json','/assets/icon-192.png','/assets/icon-512.png'];
self.addEventListener('install',e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(CORE)).then(()=>self.skipWaiting()))});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()))});
self.addEventListener('fetch',e=>{
  const url=new URL(e.request.url);
  if(url.pathname.endsWith('.json')){e.respondWith(fetch(e.request).then(r=>{const cc=r.clone();caches.open(CACHE).then(c=>c.put(e.request,cc));return r;}).catch(()=>caches.match(e.request)));return;}
  e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request).catch(()=>caches.match('/offline.html'))));
});
