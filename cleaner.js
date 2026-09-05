/* cleaner.js — erase the original fighters, text and logos from a real poster and rebuild the background.
   Runs entirely in the browser:
     - LaMa inpainting (ONNX via onnxruntime-web) rebuilds the background; a second tiled pass adds detail to big areas
     - @imgly/background-removal finds the fighters
     - PP-OCRv4 text detector (ONNX, 4.7 MB) finds text and text logos in any font or orientation
     - colour wand, brush, rectangle and edge-zone tools for anything the detectors miss
   Hands the cleaned poster back to the composer in index.html. */
(function(){
'use strict';
const ORT_VER='1.29.0';
const LAMA_URL='https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx';
const DET_URL='https://huggingface.co/deepghs/paddleocr/resolve/main/det/ch_PP-OCRv4_det/model.onnx';
const N=512;                       // LaMa's fixed input size
let ortLoading=null, lama=null, det=null, sessionEP=null;

/* ---------------- model loading ---------------- */
function loadScript(src){ return new Promise((res,rej)=>{ const s=document.createElement('script'); s.src=src; s.onload=res; s.onerror=()=>rej(new Error('Failed to load '+src)); document.head.appendChild(s); }); }
async function fetchModel(url,label,prog){
  const cache=await (window.caches?caches.open('kotr-models'):Promise.reject()).catch(()=>null);
  const hit=cache&&await cache.match(url).catch(()=>null);
  if(hit){ prog(label+' loaded from cache'); return await hit.arrayBuffer(); }
  const r=await fetch(url); if(!r.ok) throw new Error(label+' download failed ('+r.status+')');
  const total=+r.headers.get('content-length')||+r.headers.get('x-linked-size')||0; const rd=r.body.getReader(); const chunks=[]; let got=0;
  while(true){ const {done,value}=await rd.read(); if(done) break; chunks.push(value); got+=value.length; prog(`Downloading ${label} (one time) … ${Math.round(got/1e6)}${total?' / '+Math.round(total/1e6):''} MB`, total?got/total:null); }
  const buf=new Uint8Array(got); let o=0; for(const c of chunks){ buf.set(c,o); o+=c.length; }
  if(cache) cache.put(url,new Response(buf.slice().buffer)).catch(()=>{});
  return buf.buffer;
}
async function ensureOrt(){
  const wantGPU=!!navigator.gpu;
  if(!ortLoading) ortLoading=loadScript(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/${wantGPU?'ort.webgpu.min.js':'ort.min.js'}`);
  await ortLoading;
  ort.env.wasm.wasmPaths=`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
  ort.env.wasm.numThreads=self.crossOriginIsolated?Math.min(4,navigator.hardwareConcurrency||1):1;
  ort.env.logLevel='error';
  return wantGPU;
}
async function getLama(prog){
  if(lama) return lama;
  const wantGPU=await ensureOrt(); const buf=await fetchModel(LAMA_URL,'AI eraser model',prog); prog('Starting AI eraser…');
  for(const ep of (wantGPU?['webgpu','wasm']:['wasm'])){ try{ lama=await ort.InferenceSession.create(buf,{executionProviders:[ep],graphOptimizationLevel:'all'}); sessionEP=ep; break; }catch(e){ console.warn('execution provider failed',ep,e); } }
  if(!lama) throw new Error('Could not start the AI eraser in this browser');
  return lama;
}
async function getDet(prog){
  if(det) return det;
  await ensureOrt(); const buf=await fetchModel(DET_URL,'text detector',prog);
  det=await ort.InferenceSession.create(buf,{executionProviders:['wasm']}); return det;
}

/* ---------------- canvas helpers ---------------- */
function mk(w,h){ const c=document.createElement('canvas'); c.width=Math.max(1,Math.round(w)); c.height=Math.max(1,Math.round(h)); return c; }
function blank(w,h){ const c=mk(w,h); const x=c.getContext('2d'); x.fillStyle='#000'; x.fillRect(0,0,c.width,c.height); return c; }
function dilate(mask,px){ // grow white areas by ~px
  const c=blank(mask.width,mask.height), x=c.getContext('2d');
  x.filter=`blur(${px}px)`; x.drawImage(mask,0,0); x.filter='none';
  const d=x.getImageData(0,0,c.width,c.height); for(let i=0;i<d.data.length;i+=4){ const v=d.data[i]>14?255:0; d.data[i]=d.data[i+1]=d.data[i+2]=v; d.data[i+3]=255; }
  x.putImageData(d,0,0); return c;
}
function union(dst,src){ const x=dst.getContext('2d'); x.globalCompositeOperation='lighten'; x.drawImage(src,0,0,dst.width,dst.height); x.globalCompositeOperation='source-over'; }
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
    const X0=Math.max(0,x0/s-pad), Y0=Math.max(0,y0/s-pad);
    out.push({ x:X0, y:Y0, w:Math.min(mask.width,(x1+1)/s+pad)-X0, h:Math.min(mask.height,(y1+1)/s+pad)-Y0 });
  }
  return out;
}
function maskOnly(mask,comps){ const c=blank(mask.width,mask.height), x=c.getContext('2d'); comps.forEach(r=>x.drawImage(mask,r.x,r.y,r.w,r.h,r.x,r.y,r.w,r.h)); return c; }

/* ---------------- inpainting ---------------- */
// Inpaint one rectangle of `work` (full-res canvas) using `mask` (full-res, white = erase). Paints the result back only inside the mask,
// feathered at the mask edge and (when `edge` > 0) faded at the window borders so overlapping windows blend.
async function inpaintRect(sess,work,mask,r,edge=0){
  const c=mk(N,N), x=c.getContext('2d'); x.drawImage(work,r.x,r.y,r.w,r.h,0,0,N,N);
  const m=blank(N,N), mx=m.getContext('2d'); mx.drawImage(mask,r.x,r.y,r.w,r.h,0,0,N,N);
  const id=x.getImageData(0,0,N,N).data, md=mx.getImageData(0,0,N,N).data;
  const im=new Float32Array(3*N*N), mkd=new Float32Array(N*N); let any=0;
  for(let i=0;i<N*N;i++){ im[i]=id[i*4]/255; im[N*N+i]=id[i*4+1]/255; im[2*N*N+i]=id[i*4+2]/255; mkd[i]=md[i*4]>127?1:0; any+=mkd[i]; }
  if(any<4) return false;
  const out=await sess.run({image:new ort.Tensor('float32',im,[1,3,N,N]),mask:new ort.Tensor('float32',mkd,[1,1,N,N])});
  const o=out[sess.outputNames[0]].data; const k=(o[0]<=1.5&&o[N*N]<=1.5&&o[2*N*N]<=1.5)?255:1;
  const od=x.createImageData(N,N); for(let i=0;i<N*N;i++){ od.data[i*4]=o[i]*k; od.data[i*4+1]=o[N*N+i]*k; od.data[i*4+2]=o[2*N*N+i]*k; od.data[i*4+3]=255; }
  x.putImageData(od,0,0);
  const res=mk(r.w,r.h), rx=res.getContext('2d'); rx.imageSmoothingQuality='high'; rx.drawImage(c,0,0,r.w,r.h);
  const fe=blank(r.w,r.h), fx=fe.getContext('2d'); fx.filter='blur(2px)'; fx.drawImage(mask,r.x,r.y,r.w,r.h,0,0,r.w,r.h); fx.filter='none';
  const fd=fx.getImageData(0,0,fe.width,fe.height).data, rd=rx.getImageData(0,0,res.width,res.height);
  const W=work.width,H=work.height, rw=res.width, rh=res.height;
  for(let yy=0;yy<rh;yy++){
    let ry=1; if(edge){ if(r.y>0) ry=Math.min(ry,yy/edge); if(r.y+r.h<H) ry=Math.min(ry,(rh-1-yy)/edge); }
    for(let xx=0;xx<rw;xx++){ let a=fd[(yy*rw+xx)*4]; if(edge){ let rxp=1; if(r.x>0) rxp=Math.min(rxp,xx/edge); if(r.x+r.w<W) rxp=Math.min(rxp,(rw-1-xx)/edge); a*=Math.max(0,Math.min(1,Math.min(rxp,ry))); } rd.data[(yy*rw+xx)*4+3]=a; }
  }
  rx.putImageData(rd,0,0); work.getContext('2d').drawImage(res,r.x,r.y); return true;
}
async function cleanPoster(baseImg,mask,hq,prog){
  const sess=await getLama(prog);
  const W=baseImg.width,H=baseImg.height; const work=mk(W,H); work.getContext('2d').drawImage(baseImg,0,0);
  const comps=components(mask); if(!comps.length) throw new Error('Nothing is marked for removal');
  const dev=sessionEP==='webgpu'?'GPU':'CPU';
  prog(`Rebuilding background · pass 1 (${dev}) …`);
  await inpaintRect(sess,work,mask,{x:0,y:0,w:W,h:H});                               // 1. global pass: coarse fill with full context
  const big=comps.filter(c=>c.w*c.h>0.12*W*H), small=comps.filter(c=>c.w*c.h<=0.12*W*H);
  if(hq&&big.length){                                                                // 2. tiled detail pass over big areas
    const wins=[]; let S=Math.max(N,Math.min(W,H)*0.55);
    for(;;){ wins.length=0; for(const c of big){ const step=S*0.62; for(let y=c.y;;y+=step){ const yy=Math.max(0,Math.min(H-S,y)); for(let x=c.x;;x+=step){ const xx=Math.max(0,Math.min(W-S,x)); if(!wins.some(w=>Math.abs(w.x-xx)<4&&Math.abs(w.y-yy)<4)) wins.push({x:xx,y:yy,w:Math.min(S,W),h:Math.min(S,H)}); if(x+S>=c.x+c.w||xx===W-S) break; } if(y+S>=c.y+c.h||yy===H-S) break; } }
      if(wins.length<=6||S>=Math.min(W,H)) break; S=Math.min(Math.min(W,H),S*1.25); }
    const bm=maskOnly(mask,big);
    for(let i=0;i<wins.length;i++){ prog(`Adding detail · ${i+1}/${wins.length} …`); const w=wins[i]; await inpaintRect(sess,work,bm,{x:Math.round(w.x),y:Math.round(w.y),w:Math.round(w.w),h:Math.round(w.h)},Math.round(S*0.12)); }
  }
  for(let i=0;i<small.length;i++){                                                   // 3. small areas at higher effective resolution
    const c=small[i]; prog(`Refining detail ${i+1}/${small.length} …`);
    const side=Math.max(N,Math.max(c.w,c.h)*1.8), w=Math.min(W,side), h=Math.min(H,side);
    const x=Math.max(0,Math.min(W-w,c.x+c.w/2-w/2)), y=Math.max(0,Math.min(H-h,c.y+c.h/2-h/2));
    await inpaintRect(sess,work,maskOnly(mask,[c]),{x:Math.round(x),y:Math.round(y),w:Math.round(w),h:Math.round(h)});
  }
  return work;
}

/* ---------------- detectors ---------------- */
async function detectPeople(srcCanvas,W,H,prog){
  prog('Loading fighter detector…');
  const mod=await import('https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.7.0/+esm');
  const input=await new Promise(r=>srcCanvas.toBlob(r,'image/png'));
  const blob=await mod.removeBackground(input,{progress:(k,c,t)=>prog(`Detecting fighters… ${t?Math.round(c/t*100):0}%`)});
  const url=URL.createObjectURL(blob); const im=new Image(); im.src=url; await im.decode(); URL.revokeObjectURL(url);
  const c=mk(W,H), x=c.getContext('2d'); x.drawImage(im,0,0,W,H); const d=x.getImageData(0,0,W,H);
  for(let i=0;i<d.data.length;i+=4){ const v=d.data[i+3]>50?255:0; d.data[i]=d.data[i+1]=d.data[i+2]=v; d.data[i+3]=255; }
  x.putImageData(d,0,0); return dilate(c,Math.max(4,Math.round(Math.max(W,H)*0.006)));
}
// text / text-logo mask: PP-OCRv4 DB detector at two scales (big display type needs the small scale), probability map → dilated mask
async function detectText(img,prog){
  const sess=await getDet(prog); const W=img.width,H=img.height; const out=blank(W,H);
  const mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225];
  for(const lim of [960,480]){
    prog(`Detecting text… (${lim===960?'small':'large'} type)`);
    const r=Math.min(1,lim/Math.max(W,H)); const w=Math.max(32,Math.round(W*r/32)*32), h=Math.max(32,Math.round(H*r/32)*32);
    const c=mk(w,h), x=c.getContext('2d'); x.drawImage(img,0,0,w,h); const d=x.getImageData(0,0,w,h).data; const im=new Float32Array(3*w*h);
    for(let i=0;i<w*h;i++){ im[i]=(d[i*4]/255-mean[0])/std[0]; im[w*h+i]=(d[i*4+1]/255-mean[1])/std[1]; im[2*w*h+i]=(d[i*4+2]/255-mean[2])/std[2]; }
    const res=await sess.run({[sess.inputNames[0]]:new ort.Tensor('float32',im,[1,3,h,w])}); const o=res[sess.outputNames[0]]; const oh=o.dims[2],ow=o.dims[3];
    const m=mk(ow,oh), mx=m.getContext('2d'), md=mx.createImageData(ow,oh);
    for(let i=0;i<ow*oh;i++){ const p=o.data[i]>0.3?255:0; md.data[i*4]=md.data[i*4+1]=md.data[i*4+2]=p; md.data[i*4+3]=255; }
    mx.putImageData(md,0,0);
    const full=blank(W,H); full.getContext('2d').drawImage(m,0,0,W,H);           // DB predicts a shrunk core of each word: grow it back
    union(out,dilate(full,Math.max(6,Math.round(Math.max(W,H)/ (lim===960?90:45)))));
  }
  return out;
}
// colour wand: every pixel on the poster close to the clicked colour
function wandMask(base,px,py,tol){
  const W=base.width,H=base.height; const d=base.getContext('2d').getImageData(0,0,W,H).data; const i0=(py*W+px)*4; const r0=d[i0],g0=d[i0+1],b0=d[i0+2];
  const c=blank(W,H), x=c.getContext('2d'), md=x.getImageData(0,0,W,H); const t2=tol*tol;
  for(let i=0;i<W*H;i++){ const dr=d[i*4]-r0,dg=d[i*4+1]-g0,db=d[i*4+2]-b0; if(dr*dr+dg*dg+db*db<=t2){ md.data[i*4]=md.data[i*4+1]=md.data[i*4+2]=255; } }
  x.putImageData(md,0,0); return dilate(c,Math.max(3,Math.round(Math.max(W,H)*0.003)));
}

/* ---------------- editor UI ---------------- */
const UI={};
let cur=null, base=null, mask=null, result=null, tool='brush', size=40, undo=[], drag=null, busy=false;
function q(id){ return document.getElementById(id); }
function status(t,frac){ UI.status.textContent=t||''; UI.bar.style.width=(frac!=null?Math.round(frac*100):(t?100:0))+'%'; UI.prog.style.opacity=frac!=null?1:0; }
function pushUndo(){ const c=mk(mask.width,mask.height); c.getContext('2d').drawImage(mask,0,0); undo.push(c); if(undo.length>15) undo.shift(); UI.undo.disabled=false; }
function draw(){
  const cc=UI.canvas, x=cc.getContext('2d'); x.clearRect(0,0,cc.width,cc.height);
  x.drawImage(result||base,0,0);
  if(!result){ // red tint only where the mask is white
    const t=mk(mask.width,mask.height), tx=t.getContext('2d'); tx.drawImage(mask,0,0); const td=tx.getImageData(0,0,t.width,t.height);
    for(let i=0;i<td.data.length;i+=4){ const a=td.data[i]>127?150:0; td.data[i]=255; td.data[i+1]=42; td.data[i+2]=42; td.data[i+3]=a; }
    tx.putImageData(td,0,0); x.drawImage(t,0,0); }
  if(drag&&tool==='rect'){ x.strokeStyle='#ffd166'; x.lineWidth=Math.max(2,cc.width/400); x.setLineDash([12,8]); x.strokeRect(drag.x0,drag.y0,drag.x1-drag.x0,drag.y1-drag.y0); x.setLineDash([]); }
  q('cl_keep').hidden=!result; q('cl_back').hidden=!result; q('cl_refine').hidden=!result; q('cl_run').hidden=!!result; UI.tools.style.visibility=result?'hidden':'visible';
}
function pos(e){ const r=UI.canvas.getBoundingClientRect(); return {x:(e.clientX-r.left)/r.width*UI.canvas.width, y:(e.clientY-r.top)/r.height*UI.canvas.height}; }
function paint(p,last){ const x=mask.getContext('2d'); x.globalCompositeOperation=tool==='erase'?'destination-out':'source-over'; x.strokeStyle=x.fillStyle='#fff'; x.lineWidth=size*(mask.width/1000); x.lineCap='round'; x.lineJoin='round';
  x.beginPath(); x.moveTo(last?last.x:p.x,last?last.y:p.y); x.lineTo(p.x,p.y); x.stroke(); x.globalCompositeOperation='source-over'; }
function onDown(e){ if(busy||result) return; e.preventDefault(); UI.canvas.setPointerCapture(e.pointerId); const p=pos(e); pushUndo();
  if(tool==='wand'){ const px=Math.max(0,Math.min(base.width-1,Math.round(p.x))), py=Math.max(0,Math.min(base.height-1,Math.round(p.y))); union(mask,wandMask(base,px,py,size)); status('Everything in that colour is marked. Click again for another colour, Un-mark to fix mistakes.'); draw(); return; }
  if(tool==='rect'){ drag={x0:p.x,y0:p.y,x1:p.x,y1:p.y}; } else { drag={last:p}; paint(p,null); } draw(); }
function onMove(e){ if(!drag) return; const p=pos(e); if(tool==='rect'){ drag.x1=p.x; drag.y1=p.y; } else { paint(p,drag.last); drag.last=p; } draw(); }
function onUp(e){ if(!drag) return; if(tool==='rect'){ const x=mask.getContext('2d'); x.globalCompositeOperation='source-over'; x.fillStyle='#fff'; x.fillRect(Math.min(drag.x0,drag.x1),Math.min(drag.y0,drag.y1),Math.abs(drag.x1-drag.x0),Math.abs(drag.y1-drag.y0)); } drag=null; draw(); }
function setTool(t){ tool=t; ['brush','rect','erase','wand'].forEach(k=>q('cl_'+k).classList.toggle('on',k===t)); q('cl_sizelbl').textContent=t==='wand'?'Tolerance':'Size'; UI.canvas.style.cursor=t==='wand'?'cell':'crosshair'; }
function setBusy(b){ busy=b; UI.root.classList.toggle('busy',b); }
async function guarded(fn){ if(busy) return; setBusy(true); try{ await fn(); }catch(e){ console.error(e); status('Failed: '+(e&&e.message||e)); } finally{ setBusy(false); } }
function zone(kind){ pushUndo(); const x=mask.getContext('2d'); x.fillStyle='#fff'; const W=mask.width,H=mask.height;
  if(kind==='top') x.fillRect(0,0,W,H*0.22); if(kind==='bottom') x.fillRect(0,H*0.78,W,H*0.22); if(kind==='sides'){ x.fillRect(0,0,W*0.14,H); x.fillRect(W*0.86,0,W*0.14,H); } draw(); }
async function autoMark(){ await guarded(async()=>{ pushUndo();
  union(mask,await detectPeople(base,mask.width,mask.height,status)); draw();
  union(mask,await detectText(base,status)); draw();
  status('Fighters, text and logos marked. Giant side names? Pick the Wand and click a letter. Then Erase & rebuild.'); }); }

async function open(){
  cur=typeof currentPoster==='function'?currentPoster():null; if(!cur){ setStatus('Pick a poster first'); return; }
  const src=cur.clean||cur.src; const baseImg=await loadImg(src); if(!baseImg) return;
  base=mk(baseImg.width,baseImg.height); base.getContext('2d').drawImage(baseImg,0,0);
  mask=blank(baseImg.width,baseImg.height);
  result=null; undo=[]; UI.undo.disabled=true; UI.canvas.width=base.width; UI.canvas.height=base.height;
  UI.root.hidden=false; q('cl_title').textContent=cur.name+(cur.clean?'  ·  already cleaned — you can refine it further':'');
  status(cur.clean?'':'Press "Auto-mark everything", fix anything missed with Wand / Rectangle / Brush, then "Erase & rebuild".'); setTool('brush'); draw();
}
function close(){ UI.root.hidden=true; result=null; }
async function runClean(){ await guarded(async()=>{ const out=await cleanPoster(base,mask,q('cl_hq').checked,status); result=out; status('Done. Keep it, or go back and adjust the mask.'); draw(); }); }
function shrinkCanvas(c,max){ const r=Math.min(1,max/Math.max(c.width,c.height)); const t=mk(c.width*r,c.height*r); t.getContext('2d').drawImage(c,0,0,t.width,t.height); return t.toDataURL('image/jpeg',0.85); }
async function keep(){ if(!result) return; await guarded(async()=>{
  status('Saving…'); cur.clean=result.toDataURL('image/jpeg',0.92); cur.thumb=shrinkCanvas(result,240);
  if(typeof dbPut==='function') await dbPut(cur);
  if(typeof S!=='undefined'&&S.pfx){ S.pfx.coverTop=0; S.pfx.coverBottom=0; S.pfx.darken=Math.min(S.pfx.darken,0.12); S.pfx.blur=0; }
  if(typeof RANK!=='undefined') RANK=null;
  if(typeof rebuildUI==='function') rebuildUI(); else if(typeof queue==='function') queue();
  close(); setStatus('Cleaned poster saved — now place the KOTR fighters'); }); }
async function resetOriginal(){ if(!cur||!cur.clean) return; if(!confirm('Discard the cleaned version and go back to the original poster?')) return;
  cur.clean=null; const im=await loadImg(cur.src); if(im){ const c=mk(im.width,im.height); c.getContext('2d').drawImage(im,0,0); cur.thumb=shrinkCanvas(c,240); }
  if(typeof dbPut==='function') await dbPut(cur); close(); if(typeof rebuildUI==='function') rebuildUI(); }

function init(){
  UI.root=q('cleaner'); UI.canvas=q('cl_canvas'); UI.status=q('cl_status'); UI.prog=q('cl_prog'); UI.bar=UI.prog.firstElementChild; UI.undo=q('cl_undo'); UI.tools=q('cl_tools');
  UI.canvas.addEventListener('pointerdown',onDown); UI.canvas.addEventListener('pointermove',onMove); UI.canvas.addEventListener('pointerup',onUp); UI.canvas.addEventListener('pointercancel',onUp);
  ['brush','rect','erase','wand'].forEach(k=>q('cl_'+k).onclick=()=>setTool(k));
  q('cl_size').oninput=e=>size=+e.target.value;
  q('cl_undo').onclick=()=>{ const m=undo.pop(); if(m){ mask=m; UI.undo.disabled=!undo.length; draw(); } };
  q('cl_clear').onclick=()=>{ pushUndo(); mask=blank(mask.width,mask.height); draw(); };
  q('cl_auto').onclick=autoMark;
  q('cl_people').onclick=()=>guarded(async()=>{ pushUndo(); union(mask,await detectPeople(base,mask.width,mask.height,status)); status('Fighters marked.'); draw(); });
  q('cl_text').onclick=()=>guarded(async()=>{ pushUndo(); union(mask,await detectText(base,status)); status('Text and text logos marked. Giant display letters: use the Wand.'); draw(); });
  q('cl_ztop').onclick=()=>zone('top'); q('cl_zbottom').onclick=()=>zone('bottom'); q('cl_zsides').onclick=()=>zone('sides');
  q('cl_run').onclick=runClean; q('cl_keep').onclick=keep;
  q('cl_back').onclick=()=>{ result=null; status('Adjust the mask and run again.'); draw(); };
  q('cl_refine').onclick=()=>{ if(!result) return; base=result; result=null; pushUndo(); mask=blank(mask.width,mask.height); status('Mark any leftovers and run again.'); draw(); };
  q('cl_cancel').onclick=close; q('cl_reset').onclick=resetOriginal;
  window.addEventListener('keydown',e=>{ if(UI.root.hidden) return; if(e.key==='Escape') close(); if((e.ctrlKey||e.metaKey)&&e.key==='z'){ q('cl_undo').click(); } });
}
if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init); else init();
window.openCleaner=open;
})();
