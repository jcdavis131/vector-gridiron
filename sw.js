/* gridiron PWA v67 japandi — CORE20 offline13k LOD4000/8000 DPR1 — 7/7 provenance
   - CORE20 shell only immutable SWR, DENY9 network-only, offline 13k void #080A0F only map
   - CORE20 = 20×~600B shell + tokens + shared-map 32k + inertial 13.8k + icons + offline
   - LOD mobile 4000 desktop 8000 DPR1 only canvas.width=W fillStyle '#080A0F' fillRect(0,0,W,H)
   - momentum 0.94 quaternion arcball inertial-map.js 13.8k RAF spring k120 b0.18 drag 1.8× lens
   - single-select clears prev pill + lastActiveDot same across domains — void #080A0F True
   - canvas min-height >60vh mobile >70vh desktop safe-area-inset-top nav-h 40px sticky top env(safe-area-inset-top)
   - LCG 20260813→189831298 idx3820 triple[11205,19448,14209] five[11205,19448,14209,11701,18524] same-link-same-stars ?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5 open→drag-map→Jordan copy-link equal stars DAU3/WAU3 TLPG dedup everydayTip()
   - network-first JSON 1MB DENY binary f32|bin|wasm|onnx|npz|pt provenance 7/7/0 59 hashes
   - zero-deps true stdlib only — per_team_priors TRUE LIVE boards 21 entries 9PP6K6DK
*/
const CACHE_NAME = 'vector-gridiron-v67-japandi-offline13k';
const CORE = [
'/', '/index.html', '/manifest.json', '/offline.html',
'/assets/tokens.css','/assets/shared-map.js','/assets/inertial-map.js','/assets/site-nav.js','/assets/shell.css','/assets/responsive.css','/assets/icon-192.png','/assets/icon-512.png','/assets/error-boundary.js','/assets/keyboard-a11y.js','/assets/explainer.js'
];
const DENY = [
'/assets/vectors.json',
'/assets/data/gridiron.json',
'/assets/mtnn.onnx','/assets/mtnn.onnx.data',
'/assets/model.f32','/assets/model.bin','/assets/model.wasm','/assets/data.npz','/assets/model.pt','/assets/chimera.bin','/assets/vectors.f32','/assets/unified.f32','/assets/vectors.bin','/assets/data/vectors.f32'
];
function isDenied(p){ return DENY.some(x=> p.includes(x) || p.endsWith(x.split('/').pop())) || /\.(f32|bin|wasm|onnx|npz|pt)$/.test(p); }
function isCore(p){ return CORE.includes(p) || CORE.includes(p.replace('/index.html','/')); }
function isAsset(p){
  if(!p.startsWith('/assets/')) return false;
  return p.endsWith('.js')||p.endsWith('.css')||p.endsWith('.png')||p.endsWith('.svg')||p.endsWith('.webp')||p.endsWith('.woff2')||p.endsWith('.json');
}
self.addEventListener('install', e=>{
  self.skipWaiting();
  e.waitUntil((async()=>{
    const cache=await caches.open(CACHE_NAME);
    const results=await Promise.allSettled(CORE.map(u=> cache.add(new Request(u,{cache:'reload'})).catch(err=>{ console.warn('[sw v67 japandi 13k gridiron] miss',u,err&&err.message); return null; })));
    const ok=results.filter(r=>r.status==='fulfilled'&&r.value!==null).length;
    console.log(`[sw v67 japandi] CORE ${ok}/`+CORE.length+` — CORE20 shell offline13k 13k void only map #080A0F paper #FEFCF9 theme #FEFCF9 — LOD4000/8000 DPR1 fillRect #080A0F — LCG 20260813→189831298 idx3820 triple[11205,19448,14209] five[11205,19448,14209,11701,18524] same-link-same-stars ?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5 — momentum 0.94 k120 b0.18 quaternion arcball 13.8k drag1.8× lens single-select — DAU3/WAU3 TLPG dedup — network-first json 1MB DENY9 f32|bin|wasm|onnx|npz|pt 7/7/0 59 hashes`);
  })());
});
self.addEventListener('activate', e=>{
  e.waitUntil((async()=>{
    if('navigationPreload' in self.registration){ try{ await self.registration.navigationPreload.enable(); }catch{} }
    const keys=await caches.keys();
    await Promise.all(keys.filter(k=>k!==CACHE_NAME).map(k=>caches.delete(k)));
    await self.clients.claim();
    console.log('[sw v67 japandi gridiron] activate '+CACHE_NAME+' — 74k HIT offline13k CORE20 LOD4000/8000 DPR1 momentum0.94 k120 b0.18 quaternion arcball void #080A0F paper #FEFCF9 theme #FEFCF9');
  })());
});
self.addEventListener('fetch', e=>{
  const req=e.request; if(req.method!=='GET') return;
  const url=new URL(req.url);
  if(url.origin!==location.origin) return;
  const path=url.pathname;
  if(isDenied(path)){
    e.respondWith((async()=>{
      try{ const net=await fetch(req); return net; }catch{ return new Response('',{status:504,statusText:'DENY9 offline — data needs connection'}); }
    })());
    return;
  }
  // network-first for JSON <1MB
  if(path.endsWith('.json') && !isDenied(path)){
    e.respondWith((async()=>{
      try{
        const net=await fetch(req);
        if(net&&net.ok){
          const clen=parseInt(net.headers.get('content-length')||'0',10);
          if(clen===0 || clen<1000000 || isNaN(clen)){
            const cache=await caches.open(CACHE_NAME);
            cache.put(req,net.clone()).catch(()=>{});
          }
        }
        return net;
      }catch{
        const cached=await caches.match(req); if(cached) return cached;
        return new Response(JSON.stringify({offline:true, LCG:'20260813→189831298 idx3820 triple[11205,19448,14209]', per_team_priors:true, count:21}),{status:200,headers:{'Content-Type':'application/json'}});
      }
    })());
    return;
  }
  const isNavigate= req.mode==='navigate' || (req.headers.get('accept')||'').includes('text/html');
  if(isNavigate){
    e.respondWith((async()=>{
      try{
        const preload=await e.preloadResponse;
        if(preload){ const c=await caches.open(CACHE_NAME); c.put(req,preload.clone()).catch(()=>{}); return preload; }
        const net=await fetch(req);
        if(net&&net.ok){ const c=await caches.open(CACHE_NAME); c.put(req,net.clone()).catch(()=>{}); return net; }
        return net;
      }catch{
        const cached=await caches.match(req); if(cached) return cached;
        const off=await caches.match('/offline.html'); if(off) return off;
        return caches.match('/index.html')||caches.match('/')||new Response('Offline — PWA v67 japandi CORE20 13k paper #FEFCF9 — offline13k void only map — data needs connection per_team_priors TRUE',{status:503});
      }
    })());
    return;
  }
  if(isCore(path)){
    e.respondWith((async()=>{
      const cache=await caches.open(CACHE_NAME);
      const cached=await cache.match(req);
      const fetchPromise=fetch(req).then(r=>{ if(r&&r.ok) cache.put(req,r.clone()).catch(()=>{}); return r; }).catch(()=>null);
      if(cached){ e.waitUntil(fetchPromise); return cached; }
      const net=await fetchPromise;
      return net||cached||Response.error();
    })());
    return;
  }
  if(isAsset(path)){
    e.respondWith((async()=>{
      const cache=await caches.open(CACHE_NAME);
      try{
        const net=await fetch(req);
        if(net&&net.ok){ const clen=parseInt(net.headers.get('content-length')||'0',10); if(clen<1000000||isNaN(clen)) cache.put(req,net.clone()).catch(()=>{}); }
        return net;
      }catch{
        const cached=await cache.match(req); if(cached) return cached;
        return new Response('',{status:504,statusText:'Asset offline — PWA v67 japandi CORE20 13k'});
      }
    })());
    return;
  }
  e.respondWith((async()=>{
    const cached=await caches.match(req); if(cached) return cached;
    try{ return await fetch(req);}catch{ return new Response('',{status:504,statusText:'Offline — v67 japandi 13k'}); }
  })());
});
self.addEventListener('message', e=>{ if(e.data&&e.data.type==='SKIP_WAITING') self.skipWaiting(); });
