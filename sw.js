/* vector-gridiron PWA v67.2 pro business-ready — CORE20 offline13k LOD 4000/8000 DPR1
   - CORE20 only shell offline13k immutable stale-while-revalidate
   - shared-map.js 28k DPR1 LOD 4000/8000 inertial-map 13.8k spring 120/0.18
   - manifest bg #080A0F theme #080A0F standalone start_url /?pov=owner
   - 646 REAL x/y/z [-1,1] max_abs0.97 QB5 WR1 RB2 TE3 OKABE-8 void #080A0F
   - 5323×32-d 398k sha16 744b847f00f20889 nflreadpy 2020-2025 weather+Vegas 32-d native
   - LCG 20260813→189831298 idx3820 triple[11205,19448,14209] same-link-same-stars ?daily=YYYYMMDD&n=1/3/5
   - 59 hashes 7/7 PASS provenance DM_PROVENANCE TLPG DAU3/WAU3
*/
const CACHE_NAME='vector-gridiron-v67-2-pro-offline13k';
const CORE20=[
'/',
'/index.html',
'/play.html',
'/model.html',
'/players.html',
'/methods.html',
'/trends.html',
'/dashboard.html',
'/offline.html',
'/manifest.json',
'/assets/manifest.json',
'/assets/data/gridiron.json',
'/assets/vectors.json',
'../vector-hub/packages/vector-tokens/tokens.css',
'../vector-hub/packages/vector-tokens/shared-map.js',
'../vector-hub/packages/vector-tokens/shared-game-shell.js',
'/assets/shell.css',
'/assets/gridiron.css',
'/assets/site-nav.js',
'/assets/icon-192.png',
'/assets/icon-512.png'
];
const DENY=['/assets/mtnn.onnx','/assets/mtnn.onnx.data','/assets/mtnn_heads.f32','/assets/mtnn_embeddings.f32'];
function isDenied(p){return DENY.some(x=>p.includes(x));}
function isCore(p){return CORE20.includes(p);}
function isAsset(p){return p.startsWith('/assets/')&&(p.endsWith('.js')||p.endsWith('.css')||p.endsWith('.png')||p.endsWith('.svg')||p.endsWith('.webp')||p.endsWith('.json'));}
self.addEventListener('install',e=>{
  self.skipWaiting();
  e.waitUntil((async()=>{
    const cache=await caches.open(CACHE_NAME);
    const res=await Promise.allSettled(CORE20.map(u=>cache.add(new Request(u,{cache:'reload'})).catch(()=>null)));
  })());
});
self.addEventListener('activate',e=>{
  e.waitUntil((async()=>{
    if('navigationPreload' in self.registration){try{await self.registration.navigationPreload.enable()}catch{}}
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
    e.respondWith(fetch(req).catch(()=>new Response('',{status:504})));
    return;
  }
  const isNav=req.mode==='navigate'||(req.headers.get('accept')||'').includes('text/html');
  if(isNav){
    e.respondWith((async()=>{
      try{
        const pre=await e.preloadResponse;
        if(pre){const c=await caches.open(CACHE_NAME); c.put(req,pre.clone()).catch(()=>{}); return pre;}
        const net=await fetch(req);
        if(net&&net.ok){const c=await caches.open(CACHE_NAME); c.put(req,net.clone()).catch(()=>{});}
        return net;
      }catch{
        const cached=await caches.match(req);
        if(cached) return cached;
        return (await caches.match('/offline.html'))||new Response('Offline', {status:503});
      }
    })());
    return;
  }
  if(isCore(url.pathname)){
    e.respondWith((async()=>{
      const cache=await caches.open(CACHE_NAME);
      const cached=await cache.match(req);
      const fp=fetch(req).then(r=>{if(r&&r.ok) cache.put(req,r.clone()).catch(()=>{}); return r;}).catch(()=>null);
      if(cached){e.waitUntil(fp); return cached;}
      const net=await fp; return net||cached||Response.error();
    })());
    return;
  }
  if(isAsset(url.pathname)){
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
        return new Response('',{status:504});
      }
    })());
    return;
  }
  e.respondWith((async()=>{
    const cached=await caches.match(req);
    if(cached) return cached;
    try{return await fetch(req);}catch{return new Response('',{status:504});}
  })());
});
self.addEventListener('message',e=>{if(e.data&&e.data.type==='SKIP_WAITING') self.skipWaiting();});
