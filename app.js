// MedSort frontend. Classification, counts and records all come from the Flask API (SQLite). No filename matching.
const API = window.API_BASE || "";
const BINS = [
 {id:'y',k:'yellow',n:'Yellow',ic:'🧻',d:'Soiled dressings, gauze, anatomical waste',x:'Items: soiled gauze, used bandages, cotton, human tissue. Goes for incineration or deep burial.'},
 {id:'r',k:'red',n:'Red',ic:'🧪',d:'Contaminated recyclables: tubing, catheters, syringes without needles',x:'Items: IV tubing, catheters, gloves, plastic syringes without needles. Autoclave, then recycle.'},
 {id:'wh',k:'white',n:'White translucent',ic:'🪡',d:'Needles, scalpels, metal sharps',x:'Items: needles, scalpel blades, metal sharps. Puncture-proof container; autoclave or shred.'},
 {id:'b',k:'blue',n:'Blue',ic:'🏺',d:'Glass vials, ampoules, metallic implants',x:'Items: glass vials, ampoules, broken glass, metallic implants. Disinfect, then recycle.'}];
const DEMO = [{n:'Soiled gauze',ic:'🧻',b:0},{n:'IV tubing',ic:'🧵',b:1},{n:'Hypodermic needle',ic:'💉',b:2},{n:'Glass vial',ic:'🧪',b:3}];
const $ = id => document.getElementById(id);
let file=null, subId=null, demo=null, running=false, timer=null, runId=0, modelOk=false;
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now())+Math.random().toString(36).slice(2,12));
const say = (t,k='') => { const m=$('msg'); m.textContent=t; m.className='msg '+k };
const stage = n => { for(let i=0;i<5;i++){ const e=$('st'+i); e.classList.toggle('on',i===n); e.classList.toggle('done',i<n) } };
const pct = v => Math.round(v*100)+'%';
async function api(path, opt){
  let r; try{ r=await fetch(API+path,opt) }catch(_){ throw new Error('Cannot reach the backend. Start it with python app.py.') }
  let j=null; try{ j=await r.json() }catch(_){}
  if(!r.ok||!j||j.ok===false) throw new Error((j&&j.error&&j.error.message)||('Server error '+r.status));
  return j;
}

// ---------- navigation ----------
const burger=$('burger'), menu=$('menu');
burger.onclick=()=>{const o=menu.classList.toggle('open'); burger.setAttribute('aria-expanded',o)};
menu.querySelectorAll('a').forEach(a=>a.onclick=()=>{menu.classList.remove('open');burger.setAttribute('aria-expanded',false)});
const links=[...menu.querySelectorAll('a')];
if('IntersectionObserver' in window){
  const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)links.forEach(a=>a.classList.toggle('act',a.getAttribute('href')==='#'+e.target.id))}),{rootMargin:'-40% 0px -55% 0px'});
  document.querySelectorAll('main section[id]').forEach(s=>io.observe(s));
}

// ---------- bins ----------
$('bins').innerHTML=BINS.map((b,i)=>`<button class="bin" id="bin${i}" aria-label="${b.n} bin details"><div class="bk ${b.id}">${b.ic}</div><b>${b.n}</b><div class="cnt" id="c${i}">0</div><small>${b.d}</small></button>`).join('');
BINS.forEach((b,i)=>$('bin'+i).onclick=()=>{
  BINS.forEach((_,j)=>$('bin'+j).classList.toggle('pick',j===i));
  const h=document.createElement('h3'), p=document.createElement('p'); h.textContent=b.ic+' '+b.n+' bin'; p.textContent=b.x; $('bininfo').replaceChildren(h,p);
});
$('mcat').insertAdjacentHTML('beforeend',BINS.map(b=>`<option value="${b.k}">${b.n}</option>`).join(''));
$('fcat').insertAdjacentHTML('beforeend',BINS.map(b=>`<option value="${b.k}">${b.n}</option>`).join(''));
$('samples').innerHTML=DEMO.map((t,i)=>`<button class="s" data-i="${i}"><span>${t.ic}</span>${t.n}</button>`).join('');
$('bins').insertAdjacentHTML('afterend','<div id="item"></div>');
document.querySelectorAll('.s').forEach(el=>el.onclick=()=>{
  document.querySelectorAll('.s').forEach(x=>x.classList.remove('sel')); el.classList.add('sel');
  demo=DEMO[el.dataset.i]; file=null; $('up').value=''; $('prev').style.display='none';
  say('Animation demo: '+demo.n+'. This is not AI and nothing is recorded. Press Classify & sort to watch it.','warn');
});

// ---------- upload ----------
$('up').onchange=e=>pickFile(e.target.files[0]);
$('capin').onchange=e=>pickFile(e.target.files[0]);
$('capbtn').onclick=()=>$('capin').click();
function pickFile(f){
  file=null; demo=null; document.querySelectorAll('.s').forEach(x=>x.classList.remove('sel'));
  if(!f){$('prev').style.display='none';return}
  if(!/^image\/(jpeg|png|webp|bmp)$/.test(f.type)){say('Unsupported file type. Use JPEG, PNG, WEBP or BMP.','err');$('prev').style.display='none';return}
  if(f.size>8*1024*1024){say('Image is larger than 8 MB.','err');return}
  file=f; subId=uid(); $('prev').src=URL.createObjectURL(f); $('prev').style.display='block'; say('Image ready. Press Classify & sort.');
}

// ---------- camera + live auto-detection ----------
let stream=null, autoOn=false, autoTimer=null, streak=0, lastK=null, armed=true, busy=false;
const info=t=>{$('autostat').textContent=t};
function autoStop(){ autoOn=false; clearInterval(autoTimer); streak=0; lastK=null; armed=true; $('auto').textContent='Auto-detect (live)'; info('') }
const camOff=()=>{ autoStop(); if(stream){stream.getTracks().forEach(t=>t.stop());stream=null} $('vid').style.display='none'; $('snap').style.display='none'; $('camon').textContent='Use camera' };
async function camOn(){
  if(stream) return true;
  if(!window.isSecureContext||!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){
    say('The browser blocks the camera on this address. Open the site as http://127.0.0.1:5000 (not file:// and not a network IP), or use "Take photo (phone camera)".','err'); return false }
  try{
    try{ stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}}) }
    catch(e){ if(e&&(e.name==='OverconstrainedError'||e.name==='NotFoundError')) stream=await navigator.mediaDevices.getUserMedia({video:true}); else throw e }
    $('vid').srcObject=stream; await $('vid').play();
    $('vid').style.display='block'; $('snap').style.display='inline-block'; $('camon').textContent='Stop camera'; return true
  }catch(e){ stream=null; const n=e&&e.name;
    say(n==='NotAllowedError'||n==='SecurityError' ? 'Camera permission is blocked. Click the camera/lock icon in the address bar, set Camera to Allow, then reload. On Windows also check Settings > Privacy & security > Camera. Or use "Take photo (phone camera)".'
      : n==='NotFoundError' ? 'No camera was found on this device. Use "Take photo (phone camera)" or upload an image.'
      : n==='NotReadableError' ? 'The camera is in use by another app (Zoom, Teams or another tab). Close it and try again.'
      : 'Could not start the camera ('+(n||'unknown error')+'). Use "Take photo (phone camera)" or upload an image.','err');
    return false }
}
$('camon').onclick=async()=>{ if(stream){camOff();return} if(await camOn()) say('Show the waste item to the camera, then press Capture photo.') };
const frameBlob=(max)=>{ const v=$('vid'); if(!v.videoWidth) return Promise.resolve(null);
  const c=document.createElement('canvas'); c.width=Math.min(v.videoWidth,max); c.height=Math.round(c.width*v.videoHeight/v.videoWidth); c.getContext('2d').drawImage(v,0,0,c.width,c.height);
  return new Promise(r=>c.toBlob(r,'image/jpeg',.9)) };
$('snap').onclick=async()=>{
  const b=await frameBlob(1280); if(!b){say('Camera is not ready yet.','err');return}
  file=new File([b],'camera.jpg',{type:'image/jpeg'}); demo=null; subId=uid(); $('prev').src=URL.createObjectURL(b); $('prev').style.display='block'; camOff(); say('Photo captured. Press Classify & sort.');
};
async function autoTick(){
  if(!autoOn||busy||running) return; busy=true;
  try{
    const b=await frameBlob(640); if(!b) return;
    const fd=new FormData(); fd.append('image',new File([b],'live.jpg',{type:'image/jpeg'}));
    const j=await api('/api/detect',{method:'POST',body:fd}); if(!autoOn) return;
    if(j.confident){
      if(!armed){ info('Item still in view. Remove it to detect the next one.'); return }
      if(j.category===lastK) streak++; else { lastK=j.category; streak=1 }
      info(`Seeing ${j.category} (${pct(j.confidence)}) — confirming ${streak}/3`);
      if(streak>=3){                                      // same class on 3 frames in a row -> record + sort once
        armed=false; streak=0; file=new File([b],'live.jpg',{type:'image/jpeg'}); subId=uid(); demo=null;
        $('prev').src=URL.createObjectURL(b); $('prev').style.display='block'; await start();
      }
    }else{ streak=0; lastK=null; armed=true; info('Waiting for a waste item…') }
  }catch(e){ info(e.message) } finally{ busy=false }
}
$('auto').onclick=async()=>{
  if(autoOn){ autoStop(); return }
  if(!modelOk){ say('Auto-detect needs an AI engine (trained model or zero-shot CLIP). Currently manual mode.','err'); return }
  if(!(await camOn())) return;
  autoOn=true; $('auto').textContent='Stop auto-detect'; say('Auto-detect is on. Hold a waste item in front of the camera; it is sorted once when the same class is seen 3 times in a row.','warn'); info('Waiting for a waste item…');
  autoTimer=setInterval(autoTick,1500);
};

// ---------- sorting ----------
const sleep=ms=>new Promise(r=>timer=setTimeout(r,ms));
async function animate(ic,b,alive){
  $('bins').classList.add('run'); $('bin'+b).classList.add('on'); $('sel').textContent=BINS[b].n; $('status').textContent='Sorting…';
  const it=$('item'), from=$('up').getBoundingClientRect(), d=$('bin'+b).querySelector('.bk').getBoundingClientRect();
  const sx=from.left+from.width/2, sy=Math.max(70,from.top);
  it.textContent=ic; it.style.cssText=`display:block;opacity:1;transition:none;left:${sx}px;top:${sy}px;transform:none`; it.offsetWidth;
  it.style.transition='transform 1.4s ease-in-out,opacity .4s'; it.style.transform=`translate(${d.left+d.width/2-sx-18}px,${d.top-sy+20}px)`;
  await sleep(1500); if(alive()) it.style.opacity=0;
}
function clearRun(){ $('bins').classList.remove('run'); BINS.forEach((_,i)=>$('bin'+i).classList.remove('on')) }
async function start(){
  if(running)return;
  if(!file&&!demo){say('Choose an image first.','err');return}
  running=true; $('start').disabled=true; const id=++runId, alive=()=>running&&id===runId; setGate(true); clearRun();
  try{
    if(demo){ $('status').textContent='Demo animation'; await sleep(500); if(!alive())return; await animate(demo.ic,demo.b,alive); say('Demo finished (not AI, nothing recorded).','warn'); return }
    if(!modelOk&&!$('mcat').value){ say('No AI engine is loaded. Choose the bin in the dropdown (manual mode), or enable AI with setup_ai.bat (see the yellow box above).','err'); $('status').textContent='Choose a bin'; return }
    stage(0); await sleep(400); if(!alive())return; stage(1); $('status').textContent=modelOk?'Classifying…':'Manual selection…';
    const fd=new FormData(); fd.append('image',file); fd.append('submission_id',subId); if(!modelOk) fd.append('manual_category',$('mcat').value);
    const j=await api('/api/classify',{method:'POST',body:fd}); if(!alive())return; stage(2);
    $('last').textContent=file.name; $('conf').textContent=j.source==='manual'?`${j.category} (manual selection, not AI)`:`${j.category} (${pct(j.confidence)}) · threshold ${pct(j.threshold)}${j.demo?' · DEMO (not AI)':j.source==='zeroshot'?' · zero-shot AI (unvalidated)':''}`;
    if(j.duplicate){ say('This image was already submitted and counted once. No new item added.','warn'); $('status').textContent='Duplicate'; return }
    if(j.status==='manual_review'){ say(j.message,'warn'); $('status').textContent='Manual Verification Required'; $('sel').textContent='–'; return }
    $('status').textContent='Classified'; await sleep(600); if(!alive())return; stage(3); await sleep(500); if(!alive())return;
    stage(4); await animate('🗑️',j.class_id,alive); if(!alive())return;
    say(j.message,'ok'); $('status').textContent='Virtually Sorted';
  }catch(e){ say(e.message,'err'); $('status').textContent='Error' }
  finally{ if(id===runId){ setTimeout(()=>stage(-1),2500); running=false; $('start').disabled=false; setGate(false); await refresh() } }
}
function stop(){ running=false; runId++; clearTimeout(timer); $('item').style.display='none'; $('start').disabled=false; clearRun(); $('status').textContent='Stopped'; setGate(false) }
function clearView(){
  stop(); camOff(); file=null; demo=null; $('up').value=''; $('prev').style.display='none'; document.querySelectorAll('.s').forEach(x=>x.classList.remove('sel'));
  ['last','sel','conf'].forEach(x=>$(x).textContent='–'); $('status').textContent='Idle'; say('Choose an image, then press Classify & sort.'); tU=tG=0; draw();
}
$('start').onclick=start; $('stop').onclick=stop; $('reset').onclick=clearView;

// ---------- monitoring + records (all from API) ----------
function td(tr,t){const c=document.createElement('td'); c.textContent=t; tr.appendChild(c)}
async function loadBins(){
  const j=await api('/api/bins'); $('mon').replaceChildren();
  j.bins.forEach(b=>{
    $('c'+b.class_id).textContent=b.total_sorted;
    const m=BINS[b.class_id], card=document.createElement('div'); card.className='card';
    const cls=b.status==='Full'?'full':b.status==='Almost Full'?'almost':'';
    card.innerHTML=`<h3><span class="sw ${m.id}"></span>${m.n}</h3><div>Sorted: <b></b></div><div class="prog" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${b.fullness_percent}"><i class="${cls}" style="width:${b.fullness_percent}%"></i></div><div class="note"></div><div><b class="stt"></b></div><button class="o">Mark as emptied</button>`;
    card.querySelector('b').textContent=b.total_sorted;
    card.querySelector('.note').textContent=`${b.fullness_percent}% full (simulated) · ${b.items_since_emptied}/${b.capacity}`;
    card.querySelector('.stt').textContent=b.status;
    card.querySelector('button').onclick=async()=>{
      if(!confirm('Record that the '+m.n+' bin was emptied? History is kept.'))return;
      try{const r=await api('/api/bins/'+b.bin+'/empty',{method:'POST'}); say(r.message,'ok'); await refresh()}catch(e){say(e.message,'err')}
    };
    $('mon').appendChild(card);
  });
}
async function loadRecords(){
  const q=new URLSearchParams(); if($('fcat').value)q.set('category',$('fcat').value); if($('fd1').value)q.set('date_from',$('fd1').value); if($('fd2').value)q.set('date_to',$('fd2').value);
  const [h,s]=await Promise.all([api('/api/history?'+q),api('/api/summary')]); const body=$('rec'); body.replaceChildren();
  $('tot').textContent=s.total_sorted; $('sumline').textContent=`${s.total_sorted} virtually sorted · ${s.manual_review} need manual verification · ${s.emptying_events} emptying events recorded`;
  if(!h.records.length){body.innerHTML='<tr><td colspan="6" class="note">No records match.</td></tr>';return}
  h.records.forEach(r=>{const tr=document.createElement('tr'); td(tr,new Date(r.created_at).toLocaleString()); td(tr,r.category); td(tr,r.bin?BINS[r.class_id].n:'–'); td(tr,r.source==='manual'?'Manual':pct(r.confidence)); td(tr,r.status); td(tr,r.collection_status+(r.bin?' · Disposal Not Confirmed':'')); body.appendChild(tr)});
}
async function refresh(){ try{ await Promise.all([loadBins(),loadRecords()]) }catch(e){ $('rec').innerHTML='<tr><td colspan="6" class="note">Backend unavailable: '+e.message.replace(/</g,'&lt;')+'</td></tr>'; $('mon').textContent=e.message } }
const fq=()=>{const q=new URLSearchParams(); if($('fcat').value)q.set('category',$('fcat').value); if($('fd1').value)q.set('date_from',$('fd1').value); if($('fd2').value)q.set('date_to',$('fd2').value); return q};
$('fdl').onclick=async()=>{
  try{
    let r; try{ r=await fetch(API+'/api/export?'+fq()) }catch(_){ throw new Error('Cannot reach the backend. Start it with python app.py.') }
    if(!r.ok) throw new Error('Export failed (server error '+r.status+').');
    const b=await r.blob(), a=document.createElement('a'); a.href=URL.createObjectURL(b); a.download='medsort_records.csv'; document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(a.href),1500);
  }catch(e){ say(e.message,'err'); alert(e.message) }
};
$('fgo').onclick=()=>loadRecords().catch(e=>say(e.message,'err'));
(async()=>{
  try{
    const h=await api('/api/health'); modelOk=h.model_loaded||h.demo_mode;
    const same=JSON.stringify(h.classes)===JSON.stringify(BINS.map(b=>b.k)); if(!same){modelOk=false; say('Class order mismatch between frontend and backend.','err')}
    $('manualbox').style.display=modelOk?'none':'block'; $('aihelp').style.display=modelOk?'none':'block'; $('aireason').textContent=h.engine_error?'Reason: '+h.engine_error:''; $('modelstat').textContent=h.demo_mode?'DEMO MODE: colour heuristic, NOT AI and not accurate. Records are kept in a separate demo database.':modelOk?h.engine==='zeroshot'?`Zero-shot AI (CLIP) active: not trained on medical waste, accuracy unvalidated · threshold ${pct(h.confidence_threshold)}`:`Trained model loaded · confidence threshold ${pct(h.confidence_threshold)}`:'No trained model: manual-selection mode (you choose the bin; recorded as manual, not AI).';
  }catch(e){ $('modelstat').textContent='Backend not reachable: start python app.py'; }
  refresh();
})();

// ---------- clock gating simulation (browser simulation, not measured power) ----------
let en=false,tU=0,tG=0; const cv=$('wave'),cx=cv.getContext('2d');
function setGate(v){en=v;$('cgs').textContent=v?'Clock enabled (active)':'Gated (idle)'}
setInterval(()=>{tU++;if(en)tG++;draw()},120);
function draw(){
  const css=getComputedStyle(document.documentElement),col=css.getPropertyValue('--teal').trim()||'#0f9d9a',ink=css.getPropertyValue('--ink').trim()||'#14213d';
  cx.clearRect(0,0,600,190); cx.lineWidth=2;
  const N=24,w=600/N,hist=draw.h=(draw.h||[]); hist.push(en); if(hist.length>N/2)hist.shift();
  [['clk',0],['en',1],['gclk',2]].forEach(([n,r])=>{
    const y=30+r*58; cx.strokeStyle=r==2?col:'#8aa'; cx.fillStyle=ink; cx.font='12px sans-serif'; cx.fillText(n,4,y-14); cx.beginPath();
    let x=40; hist.forEach(e=>{const lo=y+18,hi=y-8;
      if(r==1){cx.lineTo(x,e?hi:lo);cx.lineTo(x+w*2,e?hi:lo)}
      else{cx.lineTo(x,lo);cx.lineTo(x,r==2&&!e?lo:hi);cx.lineTo(x+w,r==2&&!e?lo:hi);cx.lineTo(x+w,lo);cx.lineTo(x+w*2,lo)}
      x+=w*2}); cx.stroke();
  });
  $('sw').innerHTML=`<div>Without gating: ${tU} toggles</div><div class="bar" style="width:100%;background:#0b2447">${tU}</div><div style="margin-top:8px">With gating: ${tG} toggles</div><div class="bar" style="width:${Math.max(4,tU?tG/tU*100:0)}%;background:#0f9d9a">${tG}</div><div class="note">Toggles avoided: ${tU?Math.round((1-tG/tU)*100):0}% (simulated activity, not measured power)</div>`;
}
draw();
