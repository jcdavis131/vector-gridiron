export async function GET(request){
  const feats=["rushing","usage","form","redzone","snaps","age","weather","vegas","rest","def_vs_pos"];
  const vals=feats.map((_,i)=>0.2+i*0.05);
  const coeffs=[0.28,0.21,0.28,0.16,0.09,0.01,0.02,0.06,0.01,0.04];
  const pred=vals.reduce((s,v,i)=>s+v*coeffs[i]*10,0)+3.5;
  const shap={}; feats.forEach((n,i)=>shap[n]=coeffs[i]*vals[i]*10);
  const lime={}; feats.forEach((n,i)=>lime[n]=coeffs[i]*9);
  return new Response(JSON.stringify({pred,shap,lime,narrative:{generic:`Gridiron projects ${pred.toFixed(2)} pts because form ${shap["form"].toFixed(2)} + usage ${shap["usage"].toFixed(2)}.`}, domain:"gridiron", lcg:"20260813→189831298"}),{headers:{'content-type':'application/json'}}));
}
