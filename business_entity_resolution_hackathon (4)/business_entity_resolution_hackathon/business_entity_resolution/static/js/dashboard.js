
function toast(msg,error=false){
 const t=document.getElementById('toast'); if(!t)return;
 t.textContent=msg; t.className=error?'show error':'show';
 setTimeout(()=>t.className='',3000);
}
async function post(url){
 try{
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  const d=await r.json();
  if(!d.ok){toast(d.error||'Operation failed',true);return d}
  toast(d.message||'Completed successfully');
  setTimeout(()=>location.reload(),600);
  return d;
 }catch(e){toast(e.message,true)}
}
function table(cols,rows){
 let h='<div class="tablewrap"><table><thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>';
 h+=rows.map(r=>'<tr>'+cols.map(c=>`<td>${String(r[c]??'').replaceAll('<','&lt;')}</td>`).join('')+'</tr>').join('');
 return h+'</tbody></table></div>';
}
