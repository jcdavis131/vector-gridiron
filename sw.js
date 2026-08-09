/* vector-gridiron PWA v66 — PWA shell-only, CORE immutable stale-while-revalidate, large JSON_ONNX deny-cached
   - CORE only shell (~19 files), no large JSON/models
   - network-first for js/css/img assets with 1MB cache cap
   - JSON is deliberately never SW-cached (network only, browser HTTP cache still applies)
     => offline mode is shell-only; data pages need a connection
   - stale-while-revalidate for immutable CORE
   - Matches hoops v66 parity
*/

const CACHE_NAME = 'vector-gridiron-v66-hoops-parity';

const CORE = [
  '/',
  '/play',
  '/model',
  '/players',
  '/methods',
  '/trends',
  '/dashboard',
  '/manifest.json',
  '/offline.html',
  '/assets/shell.css',
  '/assets/responsive.css',
  '/assets/final-qa.css',
  '/assets/gridiron.css',
  '/assets/unified.css',
  '/assets/motion.css',
  '/assets/player-profile-v28.css',
  '/assets/trading-card.css',
  '/assets/site-nav.js',
  '/assets/error-boundary.js',
  '/assets/keyboard-a11y.js',
  '/assets/pwa-install.js',
  '/assets/og-embed.png',
  '/assets/og-1200x630.png',
  '/assets/icon-192.png',
  '/assets/icon-512.png'
];

const DENY_CACHE = [
  '/assets/vectors.json',
  '/assets/mtnn.onnx',
  '/assets/mtnn.onnx.data',
  '/assets/mtnn_heads.f32',
  '/assets/mtnn_embeddings.f32',
  '/assets/vectors_full.json',
  '/assets/nextgame.json',
  '/assets/projections.json'
];

const FULL_MTNN = [
  '/assets/mtnn_embeddings.f32',
  '/assets/mtnn_heads.f32',
  '/assets/mtnn_arch.json',
  '/assets/mtnn_meta.json',
  '/assets/mtnn_map.json',
  '/assets/mtnn-full.js',
  '/assets/mtnn-worker.js',
  '/assets/mtnn-onnx.js',
  '/assets/vectors_lite.json',
  '/assets/archetype_lite.json',
  '/assets/vectors.json',
  '/assets/skills.json',
  '/assets/archetype_assignments.json',
  '/assets/playoffs.json',
  '/assets/pedigree.json'
];

function isDenied(p) { return DENY_CACHE.some(x=>p.includes(x)); }
function isImmutable(url){ return CORE.includes(url.pathname); }
function isAsset(url){
  const p=url.pathname;
  if(!p.startsWith('/assets/')) return false;
  return (p.endsWith('.js')||p.endsWith('.css')||p.endsWith('.png')||p.endsWith('.svg')||p.endsWith('.webp'));
}

self.addEventListener('install',e=>{
  self.skipWaiting();
  e.waitUntil((async()=>{
    const cache=await caches.open(CACHE_NAME);
    const results=await Promise.allSettled(CORE.map(u=>cache.add(new Request(u,{cache:'reload'}))));
    const failed=results.filter(r=>r.status==='rejected');
    if(failed.length) console.warn('[sw v66 gridiron] CORE precache partial failures',failed.length);
  })());
});

self.addEventListener('activate',e=>{
  e.waitUntil((async()=>{
    if('navigationPreload' in self.registration){ try{await self.registration.navigationPreload.enable()}catch{}}
    const keys=await caches.keys();
    await Promise.all(keys.filter(k=>k!==CACHE_NAME).map(k=>caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch',e=>{
  const req=e.request; if(req.method!=='GET') return;
  const url=new URL(req.url);
  if(url.origin!==location.origin) return;

  if(isDenied(url.pathname)){
    e.respondWith(fetch(req).catch(()=>new Response('',{status:504,statusText:'Denied asset offline'})));
    return;
  }

  const isNavigate = req.mode==='navigate' || (req.headers.get('accept')||'').includes('text/html');
  if(isNavigate){
    e.respondWith((async()=>{
      try{
        const preload=await e.preloadResponse;
        if(preload){ const c=await caches.open(CACHE_NAME); c.put(req,preload.clone()).catch(()=>{}); return preload; }
        const net=await fetch(req);
        if(net&&net.ok){ const c=await caches.open(CACHE_NAME); c.put(req,net.clone()).catch(()=>{}); }
        return net;
      }catch{
        const cached=await caches.match(req);
        if(cached) return cached;
        const off=await caches.match('/offline.html');
        if(off) return off;
        return caches.match('/')||new Response('Offline',{status:503});
      }
    })());
    return;
  }

  if(isImmutable(url)){
    e.respondWith((async()=>{
      const cache=await caches.open(CACHE_NAME);
      const cached=await cache.match(req);
      const fp=fetch(req).then(r=>{ if(r&&r.ok) cache.put(req,r.clone()).catch(()=>{}); return r; }).catch(()=>null);
      if(cached){ e.waitUntil(fp); return cached; }
      const net=await fp; return net||cached||Response.error();
    })());
    return;
  }

  if(isAsset(url)){
    e.respondWith((async()=>{
      const cache=await caches.open(CACHE_NAME);
      try{
        const net=await fetch(req);
        if(net&&net.ok){
          const size=parseInt(net.headers.get('content-length')||'0',10);
          if(size<1_000_000) cache.put(req,net.clone()).catch(()=>{});
        }
        return net;
      }catch{
        const cached=await cache.match(req);
        if(cached) return cached;
        return new Response('',{status:504,statusText:'Asset offline'});
      }
    })());
    return;
  }

  e.respondWith((async()=>{
    const cached=await caches.match(req);
    if(cached) return cached;
    try{return await fetch(req);}catch{return new Response('',{status:504,statusText:'Offline'});}
  })());
});

self.addEventListener('message',e=>{ if(e.data&&e.data.type==='SKIP_WAITING') self.skipWaiting(); });
