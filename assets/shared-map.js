/* shared-map.js — LOD 8000/4000, quaternion arcball, momentum 0.94, spring 120/0.18, lens 1.8x, DPR1 no devicePixelRatio, single-select clear prev, void #080A0F */
export async function mountSharedMap(canvas, opts={}){
  const DPR=1;
  const LOD_HIGH=8000, LOD_LOW=4000;
  const w=canvas.width=Math.floor(canvas.clientWidth||900)*DPR;
  const h=canvas.height=Math.floor(canvas.clientHeight||520)*DPR;
  canvas.style.width=(w/DPR)+'px'; canvas.style.height=(h/DPR)+'px';
  const ctx=canvas.getContext('2d',{alpha:false});
  let points=[]; let sel=null; let rot={x:-0.18,y:0.62,z:0}; let vel={x:0,y:0};
  let momentum=0.94, springK=120, damp=0.18, lens=1.8;
  let lod=LOD_HIGH; let paused=false;
  function quatFromEuler(rx,ry){ const cx=Math.cos(rx/2),sx=Math.sin(rx/2),cy=Math.cos(ry/2),sy=Math.sin(ry/2); return {w:cx*cy,x:sx*cy,y:cx*sy,z:-sx*sy}; }
  function applyQuat(p,q){ const {w,x,y,z}=q; const vx=p.x,vy=p.y,vz=p.z; const ix=w*vx + y*vz - z*vy, iy=w*vy + z*vx - x*vz, iz=w*vz + x*vy - y*vx; return {x:ix*w + -x*0 + iy*-z - iz*-y, y:iy*w + -y*0 + iz*-x - ix*-z, z:iz*w + -z*0 + ix*-y - iy*-x}; }
  function project(p){ const q=quatFromEuler(rot.x,rot.y); const r=applyQuat(p,q); const d=2.8/(2.8 - r.z*0.9); return {sx:(r.x*d*lens*0.42+0.5)*w, sy:(-r.y*d*lens*0.42+0.5)*h, depth:r.z, d}; }
  function draw(){ ctx.fillStyle='#080A0F'; ctx.fillRect(0,0,w,h); ctx.strokeStyle='rgba(30,42,68,.22)'; ctx.lineWidth=1; const N=Math.min(lod, points.length); const sorted=points.slice(0,N).map(p=>{const pr=project(p); return {...p,...pr}}).sort((a,b)=>a.depth-b.depth); for(const pt of sorted){ const isSel=sel&&pt.id===sel.id; const c=pt.c||'#56B4E9'; const s=isSel?6:(pt.z*0.5+2.2)*(pt.d||1); ctx.beginPath(); ctx.fillStyle=c; ctx.globalAlpha=isSel?1:0.86; ctx.arc(pt.sx,pt.sy,Math.max(1.2,Math.min(7,s)),0,Math.PI*2); ctx.fill(); if(isSel){ ctx.strokeStyle='#FEFCF9'; ctx.lineWidth=1.8; ctx.stroke(); } } ctx.globalAlpha=1; }
  function tick(){ if(!paused){ rot.y+=vel.y*0.016; rot.x+=vel.x*0.016; vel.x*=momentum; vel.y*=momentum; rot.x+=(-0.18-rot.x)*0.015*springK*0.01; vel.x*=(1-damp*0.02);} draw(); requestAnimationFrame(tick); }
  let dragging=false,last={x:0,y:0}; canvas.addEventListener('pointerdown',e=>{dragging=true;canvas.setPointerCapture(e.pointerId);canvas.classList.add('grabbing');last={x:e.clientX,y:e.clientY};paused=false;});
  canvas.addEventListener('pointermove',e=>{ if(!dragging) return; const dx=e.clientX-last.x, dy=e.clientY-last.y; vel.y+=dx*0.008; vel.x+=dy*0.006; rot.y+=dx*0.006; rot.x+=dy*0.004; rot.x=Math.max(-1.2,Math.min(1.2,rot.x)); last={x:e.clientX,y:e.clientY};});
  canvas.addEventListener('pointerup',()=>{dragging=false;canvas.classList.remove('grabbing');});
  canvas.addEventListener('click',e=>{const rect=canvas.getBoundingClientRect(); const mx=(e.clientX-rect.left)*DPR,my=(e.clientY-rect.top)*DPR; let best=null,bd=18; for(const pt of points.slice(0,lod)){ const pr=project(pt); const dist=Math.hypot(pr.sx-mx,pr.sy-my); if(dist<bd){bd=dist;best=pt;}} if(best){ sel=best; canvas.dispatchEvent(new CustomEvent('point-select',{detail:best,bubbles:true})); } else { sel=null; canvas.dispatchEvent(new CustomEvent('point-deselect',{bubbles:true})); }});
  canvas.addEventListener('wheel',e=>{e.preventDefault(); lens=Math.max(0.9,Math.min(2.6,lens+Math.sign(e.deltaY)*-0.06));},{passive:false});
  tick(); return {setPoints(arr){ points=arr.map(p=>({x:(p.x??Math.random()*2-1),y:(p.y??Math.random()*2-1),z:(p.z??Math.random()*2-1),c:p.c||p.color||'#56B4E9',id:p.id||p.ticker||p.pid||Math.random(),...p})); draw();}, setTarget(id){ sel=points.find(p=>p.id===id||p.ticker===id)||null; }, clearSel(){ sel=null; }, resize(){ const nw=Math.floor(canvas.clientWidth)*DPR, nh=Math.floor(canvas.clientHeight)*DPR; canvas.width=nw; canvas.height=nh; draw(); }, setLOD(v){ lod=v===8000?LOD_HIGH:LOD_LOW; }, pause(){paused=true}, resume(){paused=false}, getSelected(){return sel}};
}
export function pickOKABE(pos){ const map={PG:'#E69F00',SG:'#56B4E9',SF:'#009E73',PF:'#F0E442',C:'#0072B2',QB:'#E69F00',WR:'#56B4E9',RB:'#009E73',TE:'#0072B2',DEF:'#0072B2',FWD:'#E69F00',MID:'#56B4E9',GK:'#FFFEF7'}; return map[pos]||'#56B4E9'; }
