/* cleaner.js — erase the original fighters and text from a real poster and rebuild the background.
   Runs entirely in the browser: LaMa inpainting (ONNX, via onnxruntime-web), fighter detection (@imgly/background-removal),
   optional text detection (tesseract.js). Hands the cleaned poster back to the composer in index.html. */
(function(){
'use strict';
const ORT_VER='1.29.0';
const MODEL_URL='https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx';
const N=512;                       // LaMa's fixed input size
let ortLoading=null, session=null, sessionEP=null;

/* ---------------- model loading ---------------- */
function loadScript(src){ return new Promise((res,rej)=>{ const s=document.createElement('script'); s.src=src; s.onload=res; s.onerror=()=>rej(new Error('Failed to load '+src)); document.head.appendChild(s); }); }
async function getModel(prog){
  const cache=await (window.caches?caches.open('kotr-models'):Promise.reject()).catch(()=>null);
  const hit=cache&&await cache.match(MODEL_URL).catch(()=>null);
  if(hit){ prog('AI model loaded from cache'); return await hit.arrayBuffer(); }
  const r=await fetch(MODEL_URL); if(!r.ok) throw new Error('Model download failed ('+r.status+')');
  const total=+r.headers.get('content-length')||93e6; const rd=r.body.getReader(); const chunks=[]; let got=0;
  while(true){ const {done,value}=await rd.read(); if(done) break; chunks.push(value); got+=value.length; prog(`Downloading AI model (one time) … ${Math.round(got/1e6)} / ${Math.round(total/1e6)} MB`, got/total); }
  const buf=new Uint8Array(got); let o=0; for(const c of chunks){ buf.set(c,o); o+=c.length; }
  if(cache) cache.put(MODEL_URL,new Response(buf.slice().buffer)).catch(()=>{});
  return buf.buffer;
}
async function getSession(prog){
  if(session) return session;
  const wantGPU=!!navigator.gpu;
  if(!ortLoading) ortLoading=loadScript(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/${wantGPU?'ort.webgpu.min.js':'ort.min.js'}`);
  await ortLoading;
  ort.env.wasm.wasmPaths=`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
  ort.env.wasm.numThreads=self.crossOriginIsolated?Math.min(4,navigator.hardwareConcurrency||1):1;
  const buf=await getModel(prog);
  prog('Starting AI model…');
  for(const ep of (wantGPU?['webgpu','wasm']:['wasm'])){
    try{ session=await ort.InferenceSession.create(buf,{executionProviders:[ep],graphOptimizationLevel:'all'}); sessionEP=ep; break; }
    catch(e){ console.warn('execution provider failed',ep,e); }
  }
  if(!session) throw new Error('Could not start the AI model in this browser');
  return session;
}

/* ---------------- canvas helpers ---------------- */
function mk(w,h){ const c=document.createElement('canvas'); c.width=Math.max(1,Math.round(w)); c.height=Math.max(1,Math.round(h)); return c; }
function dilate(mask,px){ // grow white areas by ~px
  const c=mk(mask.width,mask.height), x=c.getContext('2d'); x.fillStyle='#000'; x.fillRect(0,0,c.width,c.height);
  x.filter=`blur(${px}px)`; x.drawImage(mask,0,0); x.filter='none';
  const d=x.getImageData(0,0,c.width,c.height); for(let i=0;i<d.data.length;i+=4){ const v=d.data[i]>18?255:0; d.data[i]=d.data[i+1]=d.data[i+2]=v; d.data[i+3]=255; }
  x.putImageData(d,0,0); return c;
}
// connected components of the mask (on a downscaled copy) → bboxes in full-res px
function components(mask){
  const s=Math.min(1,160/Math.max(mask.width,mask.height)); const w=Math.max(1,Math.round(mask.width*s)), h=Math.max(1,Math.round(mask.height*s));
  const c=mk(w,h), x=c.getContext('2d'); x.drawImage(mask,0,0,w,h); const d=x.getImageData(0,0,w,h).data;
  const on=new Uint8Array(w*h); for(let i=0;i<w*h;i++) on[i]=d[i*4]>127?1:0;
  const seen=new Uint8Array(w*h); const out=[]; const stack=[];
  for(let i=0;i<w*h;i++){ if(!on[i]||seen[i]) continue; let x0=w,y0=h,x1=0,y1=0,n=0; stack.push(i); seen[i]=1;
    while(stack.length){ const p=stack.pop(); const px=p%w, py=(p-px)/w; n++; if(px<x0)x0=px; if(px>x1)x1=px; if(py<y0)y0=py; if(py>y1)y1=py;
      const nb=[p-1,p+1,p-w,p+w]; for(const q of nb){ if(q<0||q>=w*h) continue; if((q===p-1&&px===0)||(q===p+1&&px===w-1)) continue; if(on[q]&&!seen[q]){ seen[q]=1; stack.push(q); } } }
    if(n<3) continue;
    const pad=0.04*Math.max(mask.width,mask.height);
    out.push({ x:Math.max(0,x0/s-pad), y:Math.max(0,y0/s-pad), w:Math.min(mask.width,(x1+1)/s+pad)-Math.max(0,x0/s-pad), h:Math.min(mask.height,(y1+1)/s+pad)-Math.max(0,y0/s-pad) });
  }
  return out;
}
function maskOnly(mask,comps){ const c=mk(mask.width,mask.height), x=c.getContext('2d'); x.fillStyle='#000'; x.fillRect(0,0,c.width,c.height); comps.forEach(r=>x.drawImage(mask,r.x,r.y,r.w,r.h,r.x,r.y,r.w,r.h)); return c; }

/* ---------------- inpainting ---------------- */
// Inpaint one rectangle of `work` (full-res canvas) using `mask` (full-res, white = erase). Paints the result back, feathered, only inside the mask.
async function inpaintRect(sess,work,mask,r){
  const c=mk(N,N), x=c.getContext('2d'); x.drawImage(work,r.x,r.y,r.w,r.h,0,0,N,N);
  const m=mk(N,N), mx=m.getContext('2d'); mx.fillStyle='#000'; mx.fillRect(0,0,N,N); mx.drawImage(mask,r.x,r.y,r.w,r.h,0,0,N,N);
  const id=x.getImageData(0,0,N,N).data, md=mx.getImageData(0,0,N,N).data;
  const im=new Float32Array(3*N*N), mkd=new Float32Array(N*N); let any=false;
  for(let i=0;i<N*N;i++){ im[i]=id[i*4]/255; im[N*N+i]=id[i*4+1]/255; im[2*N*N+i]=id[i*4+2]/255; mkd[i]=md[i*4]>127?1:0; if(mkd[i]) any=true; }
  if(!any) return;
  const out=await sess.run({image:new ort.Tensor('float32',im,[1,3,N,N]),mask:new ort.Tensor('float32',mkd,[1,1,N,N])});
  const o=out[sess.outputNames[0]].data; const scaleOut=(o[0]<=1.5&&o[N*N]<=1.5&&o[2*N*N]<=1.5)?255:1;
  const od=x.createImageData(N,N); for(let i=0;i<N*N;i++){ od.data[i*4]=o[i]*scaleOut; od.data[i*4+1]=o[N*N+i]*scaleOut; od.data[i*4+2]=o[2*N*N+i]*scaleOut; od.data[i*4+3]=255; }
  x.putImageData(od,0,0);
  // feathered composite back
  const res=mk(r.w,r.h), rx=res.getContext('2d'); rx.imageSmoothingQuality='high'; rx.drawImage(c,0,0,r.w,r.h);
  const fe=mk(r.w,r.h), fx=fe.getContext('2d'); fx.fillStyle='#000'; fx.fillRect(0,0,fe.width,fe.height); fx.filter='blur(2px)'; fx.drawImage(mask,r.x,r.y,r.w,r.h,0,0,r.w,r.h); fx.filter='none';
  const fd=fx.getImageData(0,0,fe.width,fe.height).data, rd=rx.getImageData(0,0,res.width,res.height);
  for(let i=0;i<fd.length;i+=4) rd.data[i+3]=fd[i];
  rx.putImageData(rd,0,0); work.getContext('2d').drawImage(res,r.x,r.y);
}
async function cleanPoster(baseImg,mask,prog){
  const sess=await getSession(prog);
  const W=baseImg.width,H=baseImg.height; const work=mk(W,H); work.getContext('2d').drawImage(baseImg,0,0);
  const comps=components(mask); if(!comps.length) throw new Error('Nothing is marked for removal');
  prog(`Rebuilding background (${sessionEP==='webgpu'?'GPU':'CPU'}) …`);
  await inpaintRect(sess,work,mask,{x:0,y:0,w:W,h:H});                       // global pass
  const small=comps.filter(c=>c.w*c.h<=0.12*W*H);                             // detail passes at higher effective resolution
  for(let i=0;i<small.length;i++){
    const c=small[i]; prog(`Refining detail ${i+1}/${small.length} …`);
    let side=Math.max(N,Math.max(c.w,c.h)*1.8), w=Math.min(W,side), h=Math.min(H,side);
    let x=Math.max(0,Math.min(W-w,c.x+c.w/2-w/2)), y=Math.max(0,Math.min(H-h,c.y+c.h/2-h/2));
    await inpaintRect(sess,work,maskOnly(mask,[c]),{x:Math.round(x),y:Math.round(y),w:Math.round(w),h:Math.round(h)});
  }
  return work;
}

/* ---------------- detectors ---------------- */
async function detectPeople(src,W,H,prog){
  prog('Loading fighter detector…');
  const mod=await import('https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.7.0/+esm');
  const blob=await mod.removeBackground(src,{progress:(k,c,t)=>prog(`Detecting fighters… ${t?Math.round(c/t*100):0}%`)});
  const url=URL.createObjectURL(blob); const im=new Image(); im.src=url; await im.decode(); URL.revokeObjectURL(url);
  const c=mk(W,H), x=c.getContext('2d'); x.drawImage(im,0,0,W,H); const d=x.getImageData(0,0,W,H);
  for(let i=0;i<d.data.length;i+=4){ const v=d.data[i+3]>50?255:0; d.data[i]=d.data[i+1]=d.data[i+2]=v; d.data[i+3]=255; }
  x.putImageData(d,0,0); return dilate(c,Math.max(4,Math.round(Math.max(W,H)*0.006)));
}
async function detectText(img,prog){
  prog('Loading text detector…');
  if(!window.Tesseract) await loadScript('https://cdn.jsdelivr.net/npm/tesseract.js@7.0.0/dist/tesseract.min.js');
  const worker=await Tesseract.createWorker('eng',1,{ logger:m=>{ if(m.status) prog(`Detecting text… ${m.status} ${Math.round((m.progress||0)*100)}%`); } });
  const s=Math.min(1,1600/Math.max(img.width,img.height)); const c=mk(img.width*s,img.height*s); c.getContext('2d').drawImage(img,0,0,c.width,c.height);
  let words=[];
  try{
    const {data}=await worker.recognize(c,{},{blocks:true,text:false});
    (data.blocks||[]).forEach(b=>(b.paragraphs||[]).forEach(p=>(p.lines||[]).forEach(l=>(l.words||[]).forEach(w=>words.push(w)))));
    if(!words.length&&data.words) words=data.words;
  } finally { await worker.terminate(); }
  const minH=c.height*0.012;
  return words.filter(w=>(w.confidence||0)>35&&(w.bbox.y1-w.bbox.y0)>minH&&/[A-Za-z0-9]{2,}/.test(w.text||'')).map(w=>{ const h=(w.bbox.y1-w.bbox.y0)/s, pad=h*0.35; return {x:w.bbox.x0/s-pad,y:w.bbox.y0/s-pad,w:(w.bbox.x1-w.bbox.x0)/s+2*pad,h:h+2*pad}; });
}

/* ---------------- editor UI ---------------- */
const UI={};
let cur=null, baseImg=null, base=null, mask=null, result=null, tool='brush', size=40, undo=[], drag=null, busy=false;
function q(id){ return document.getElementById(id); }
function status(t,frac){ UI.status.textContent=t||''; UI.bar.style.width=(frac!=null?Math.round(frac*100):(t?100:0))+'%'; UI.prog.style.opacity=frac!=null?1:0; }
function pushUndo(){ const c=mk(mask.width,mask.height); c.getContext('2d').drawImage(mask,0,0); undo.push(c); if(undo.length>15) undo.shift(); UI.undo.disabled=false; }
function draw(){
  const cc=UI.canvas, x=cc.getContext('2d'); x.clearRect(0,0,cc.width,cc.height);
  x.drawImage(result||base,0,0);
  if(!result){ const t=mk(mask.width,mask.height), tx=t.getContext('2d'); tx.drawImage(mask,0,0); tx.globalCompositeOperation='multiply'; tx.fillStyle='#ff2a2a'; tx.fillRect(0,0,t.width,t.height); tx.globalCompositeOperation='destination-in'; tx.drawImage(mask,0,0);
    x.globalAlpha=0.55; x.drawImage(t,0,0); x.globalAlpha=1; }
  if(drag&&tool==='rect'){ x.strokeStyle='#ffd166'; x.lineWidth=Math.max(2,cc.width/400); x.setLineDash([12,8]); x.strokeRect(drag.x0,drag.y0,drag.x1-drag.x0,drag.y1-drag.y0); x.setLineDash([]); }
  q('cl_keep').hidden=!result; q('cl_back').hidden=!result; q('cl_refine').hidden=!result; q('cl_run').hidden=!!result; UI.tools.style.visibility=result?'hidden':'visible';
}
function pos(e){ const r=UI.canvas.getBoundingClientRect(); return {x:(e.clientX-r.left)/r.width*UI.canvas.width, y:(e.clientY-r.top)/r.height*UI.canvas.height}; }
function paint(p,last){ const x=mask.getContext('2d'); x.globalCompositeOperation=tool==='erase'?'destination-out':'source-over'; x.strokeStyle=x.fillStyle='#fff'; x.lineWidth=size*(mask.width/1000); x.lineCap='round'; x.lineJoin='round';
  x.beginPath(); x.moveTo(last?last.x:p.x,last?last.y:p.y); x.lineTo(p.x,p.y); x.stroke(); x.globalCompositeOperation='source-over'; }
function onDown(e){ if(busy||result) return; e.preventDefault(); UI.canvas.setPointerCapture(e.pointerId); const p=pos(e); pushUndo();
  if(tool==='rect'){ drag={x0:p.x,y0:p.y,x1:p.x,y1:p.y}; } else { drag={last:p}; paint(p,null); } draw(); }
function onMove(e){ if(!drag) return; const p=pos(e); if(tool==='rect'){ drag.x1=p.x; drag.y1=p.y; } else { paint(p,drag.last); drag.last=p; } draw(); }
function onUp(e){ if(!drag) return; if(tool==='rect'){ const x=mask.getContext('2d'); x.globalCompositeOperation='source-over'; x.fillStyle='#fff'; x.fillRect(Math.min(drag.x0,drag.x1),Math.min(drag.y0,drag.y1),Math.abs(drag.x1-drag.x0),Math.abs(drag.y1-drag.y0)); } drag=null; draw(); }
function setTool(t){ tool=t; ['brush','rect','erase'].forEach(k=>q('cl_'+k).classList.toggle('on',k===t)); }
function setBusy(b){ busy=b; UI.root.classList.toggle('busy',b); }
async function guarded(fn){ if(busy) return; setBusy(true); try{ await fn(); }catch(e){ console.error(e); status('Failed: '+e.message); } finally{ setBusy(false); } }

async function open(){
  cur=typeof currentPoster==='function'?currentPoster():null; if(!cur){ setStatus('Pick a poster first'); return; }
  const src=cur.clean||cur.src; baseImg=await loadImg(src); if(!baseImg) return;
  base=mk(baseImg.width,baseImg.height); base.getContext('2d').drawImage(baseImg,0,0);
  mask=mk(baseImg.width,baseImg.height); mask.getContext('2d').fillStyle='#000'; mask.getContext('2d').fillRect(0,0,mask.width,mask.height);
  if(cur.maskData&&!cur.clean){ const mi=await loadImg(cur.maskData); if(mi) mask.getContext('2d').drawImage(mi,0,0,mask.width,mask.height); }
  result=null; undo=[]; UI.undo.disabled=true; UI.canvas.width=base.width; UI.canvas.height=base.height;
  UI.root.hidden=false; q('cl_title').textContent=cur.name+(cur.clean?'  ·  already cleaned, you can refine it further':''); status(cur.clean?'':'Mark what to erase: Auto-detect, brush or rectangle. Then press Erase & rebuild.'); setTool('brush'); draw();
}
function close(){ UI.root.hidden=true; result=null; }
async function runClean(){ await guarded(async()=>{ const out=await cleanPoster(base,mask,status); result=out; status('Done. Keep it, or go back and adjust the mask.'); draw(); }); }
async function keep(){ if(!result) return; await guarded(async()=>{
  status('Saving…'); cur.clean=result.toDataURL('image/jpeg',0.92); cur.maskData=null; cur.thumb=shrinkCanvas(result,240);
  if(typeof dbPut==='function') await dbPut(cur);
  if(typeof S!=='undefined'&&S.pfx){ S.pfx.coverTop=0; S.pfx.coverBottom=0; S.pfx.darken=Math.min(S.pfx.darken,0.12); S.pfx.blur=0; }
  if(typeof RANK!=='undefined') RANK=null;
  if(typeof rebuildUI==='function') rebuildUI(); else if(typeof queue==='function') queue();
  close(); setStatus('Cleaned poster saved — now place the KOTR fighters'); }); }
function shrinkCanvas(c,max){ const r=Math.min(1,max/Math.max(c.width,c.height)); const t=mk(c.width*r,c.height*r); t.getContext('2d').drawImage(c,0,0,t.width,t.height); return t.toDataURL('image/jpeg',0.85); }
async function resetOriginal(){ if(!cur||!cur.clean) return; if(!confirm('Discard the cleaned version and go back to the original poster?')) return; cur.clean=null; cur.maskData=null; const im=await loadImg(cur.src); cur.thumb=im?shrinkCanvas((()=>{const c=mk(im.width,im.height);c.getContext('2d').drawImage(im,0,0);return c;})(),240):cur.thumb; if(typeof dbPut==='function') await dbPut(cur); close(); if(typeof rebuildUI==='function') rebuildUI(); }

function init(){
  UI.root=q('cleaner'); UI.canvas=q('cl_canvas'); UI.status=q('cl_status'); UI.prog=q('cl_prog'); UI.bar=UI.prog.firstElementChild; UI.undo=q('cl_undo'); UI.tools=q('cl_tools');
  UI.canvas.addEventListener('pointerdown',onDown); UI.canvas.addEventListener('pointermove',onMove); UI.canvas.addEventListener('pointerup',onUp); UI.canvas.addEventListener('pointercancel',onUp);
  q('cl_brush').onclick=()=>setTool('brush'); q('cl_rect').onclick=()=>setTool('rect'); q('cl_erase').onclick=()=>setTool('erase');
  q('cl_size').oninput=e=>size=+e.target.value;
  q('cl_undo').onclick=()=>{ const m=undo.pop(); if(m){ mask=m; UI.undo.disabled=!undo.length; draw(); } };
  q('cl_clear').onclick=()=>{ pushUndo(); const x=mask.getContext('2d'); x.globalCompositeOperation='source-over'; x.fillStyle='#000'; x.fillRect(0,0,mask.width,mask.height); draw(); };
  q('cl_people').onclick=()=>guarded(async()=>{ pushUndo(); const m=await detectPeople(cur.clean||cur.src,mask.width,mask.height,status); mask.getContext('2d').drawImage(m,0,0); status('Fighters marked. Add anything missing with the brush, then Erase & rebuild.'); draw(); });
  q('cl_text').onclick=()=>guarded(async()=>{ pushUndo(); const boxes=await detectText(base,status); const x=mask.getContext('2d'); x.fillStyle='#fff'; boxes.forEach(b=>x.fillRect(b.x,b.y,b.w,b.h)); status(boxes.length?`${boxes.length} text block(s) marked. Stylised fonts are often missed — use the rectangle tool for those.`:'No text found — use the rectangle tool to mark the text areas.'); draw(); });
  q('cl_run').onclick=runClean; q('cl_keep').onclick=keep;
  q('cl_back').onclick=()=>{ result=null; status('Adjust the mask and run again.'); draw(); };
  q('cl_refine').onclick=()=>{ if(!result) return; base=result; result=null; pushUndo(); const x=mask.getContext('2d'); x.fillStyle='#000'; x.fillRect(0,0,mask.width,mask.height); status('Mark any leftovers and run again.'); draw(); };
  q('cl_cancel').onclick=close; q('cl_reset').onclick=resetOriginal;
  window.addEventListener('keydown',e=>{ if(UI.root.hidden) return; if(e.key==='Escape') close(); if((e.ctrlKey||e.metaKey)&&e.key==='z'){ q('cl_undo').click(); } });
}
if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init); else init();
window.openCleaner=open;
})();
