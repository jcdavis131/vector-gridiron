/* Human-Centered v6 — JS (stdlib, zero-deps) */
window.DumbModel = window.DumbModel || {};
window.DumbModel.HumanV6 = (function(){
  const Selection={
    _pid:null,
    init(){ 
      const params=new URLSearchParams(location.search);
      this._pid=params.get('id');
      document.addEventListener('keydown', (e)=>{ if(e.key==='Escape'){ this.clear(); }});
    },
    update(pid){
      this._pid=pid;
      const url=new URL(location.href);
      url.searchParams.set('id', pid);
      history.pushState({pid}, '', url.toString());
    },
    clear(){
      this._pid=null;
      const url=new URL(location.href);
      url.searchParams.delete('id');
      history.pushState({}, '', url.toString());
      window.dispatchEvent(new CustomEvent('hv6:clear'));
    },
    destroy(){}
  };
  const Share={
    init(sel){
      const btn=document.querySelector(sel||'#btn-share');
      if(!btn) return;
      btn.addEventListener('click', async()=>{
        const url=location.href;
        try{ await navigator.clipboard.writeText(url); btn.textContent='Copied ✓'; setTimeout(()=>btn.textContent='Copy link',1200); }catch{ window.prompt('Copy link:', url); }
      });
    },
    copy(){ return navigator.clipboard.writeText(location.href); },
    destroy(){}
  };
  const Peers={ init(){}, update(){}, destroy(){} };
  const Evidence={
    init(){},
    open(){ document.getElementById('evidence')?.scrollIntoView({behavior:'smooth'}); },
    close(){},
    destroy(){}
  };
  return {Selection, Share, Peers, Evidence};
})();
