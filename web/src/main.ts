import './style.css';
import {Viewer} from './viewer';
import {CameraDriver} from './tracking';
import {decodeAsset,identityPose,copyPose,deform,forwardKinematics,interpolatePose,sampleRecording} from './engine.mjs';
import {applyClip,solveTargets} from './controls.mjs';
import {encodePose,decodePose,PlaybackBuffer} from './protocol.mjs';
import {sampleMotion,rigKey,MAX_FBX_BYTES} from './mixamo.mjs';

const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const params=new URLSearchParams(location.search),remote=params.get('remote')==='1';
if(remote)document.body.classList.add('remote');
const viewer=new Viewer($('viewport')),camera=new CameraDriver($<HTMLVideoElement>('camera-preview'));
let asset:any,assetId='',base:any=identityPose(),lastPose:any=base,worker:Worker|undefined,busy=false,mode='pose',clip='none',playing=false,clock=0,speed=1,frameId=0,loadToken=0;
let deformMs=0,updateCount=0,updateHz=0,bytes=0,renderedFrames=0,lastFrame:any;
let selectedPhotos:File[]=[],selectedPhotoUrls:string[]=[],generationModels:any[]=[],generationPending=false;
let photoRestore:AbortController|undefined,photosIncomplete=false,generationSelectionRevision=0,photoSelectionRevision=0;
let deformSamples:number[]=[];
let deformationBackend='cpu',deformationReason='',deformationAdapter:any,verifying=false,gpuStressChecks:any[]=[];
let targets:any={},activeTarget='head',recording=false,recorded:any[]=[],recordStart=0,replayStart=0,replaying=false,verifyResults:any[]=[],jobPoll:number|undefined;
let socket:WebSocket|undefined,roomId=params.get('room')??'',sequence=0,lastSend=0;
let motions:any[]=[],motionId='',motionInPlace=true,motionLoop=true,motionCycle=0,teleportNext=false,importingMotion=false;
let motionReference:{name:string,buffer:ArrayBuffer}|undefined;
let replayCycle=0;
const importWorkers=new Set<Worker>();
const selectedMotion=()=>motions.find(m=>m.id===motionId);
const playback=new PlaybackBuffer(80);let latency=60,jitter=20,loss=0,connectionGeneration=0;
const controls:Record<string,{joint:number,axis:number}>= {'left-arm':{joint:16,axis:2},'right-arm':{joint:17,axis:2},'left-elbow':{joint:18,axis:1},'right-elbow':{joint:19,axis:1},'head-turn':{joint:15,axis:1},'jaw-open':{joint:22,axis:0}};
const labels:Record<string,string>={'left-arm':'Left shoulder','right-arm':'Right shoulder','left-elbow':'Left elbow','right-elbow':'Right elbow','head-turn':'Head turn','jaw-open':'Jaw open'};

function notice(message:string){$('notice').textContent=message;$('notice').hidden=!message;}
function loading(message:string){$('load-status').textContent=message;$('load-status').hidden=!message;}
function updateControlLabels(){for(const [id,{joint,axis}] of Object.entries(controls)){const input=$<HTMLInputElement>(id);input.value=String(base.angles[joint*3+axis]);$(`${id}-value`).textContent=`${Math.round(Number(input.value)*180/Math.PI)}°`;}}
function stopMotion(){playing=false;replaying=false;clip='none';$('play').textContent='▶';document.querySelectorAll('[data-clip]').forEach(el=>el.classList.toggle('selected',(el as HTMLElement).dataset.clip==='none'));}
function reset(){if(!asset||remote)return;setMode('pose');teleportNext=true;base=identityPose(asset.meta.nBones,asset.meta.nExpressions);base.angles[50]=-0.35;base.angles[53]=0.35;updateControlLabels();$<HTMLInputElement>('expression-strength').value='0';$('expression-value').textContent='0.00';initTargets();}

for(const [id,{joint,axis}] of Object.entries(controls)){
  const label=document.createElement('label');label.className='range-row';label.innerHTML=`<span>${labels[id]} <output id="${id}-value">0°</output></span><input id="${id}" type="range" min="${id==='jaw-open'?0:-1.8}" max="${id==='jaw-open'?0.6:1.8}" step="0.02" value="0">`;
  $('joint-controls').append(label);label.querySelector('input')!.addEventListener('input',e=>{if(remote)return;stopMotion();base.angles[joint*3+axis]=Number((e.target as HTMLInputElement).value);updateControlLabels();});
}
for(const [axis,label] of ['X · side','Y · height','Z · depth'].entries()){
  const row=document.createElement('label');row.className='range-row';row.innerHTML=`<span>${label}<output id="target-${axis}-value">0.00 m</output></span><input id="target-${axis}" type="range" min="-1.2" max="1.2" step="0.01" value="0">`;
  $('target-ranges').append(row);row.querySelector('input')!.addEventListener('input',e=>{if(!targets[activeTarget])return;targets[activeTarget][axis]=Number((e.target as HTMLInputElement).value);viewer.setTargets(targets);refreshTargets();});
}
for(const [axis,label] of ['Rotation X','Rotation Y','Rotation Z'].entries()){
  const row=document.createElement('label');row.className='range-row';row.innerHTML=`<span>${label}<output id="target-rotation-${axis}-value">0°</output></span><input id="target-rotation-${axis}" type="range" min="-1.8" max="1.8" step="0.02" value="0">`;
  $('target-rotations').append(row);row.querySelector('input')!.addEventListener('input',e=>{const key=`${activeTarget}Rotation`;targets[key]??=[0,0,0];targets[key][axis]=Number((e.target as HTMLInputElement).value);refreshTargets();});
}
function refreshTargets(){if(!targets[activeTarget])return;for(let i=0;i<3;i++){$<HTMLInputElement>(`target-${i}`).value=String(targets[activeTarget][i]);$(`target-${i}-value`).textContent=`${targets[activeTarget][i].toFixed(2)} m`;const rotation=targets[`${activeTarget}Rotation`]?.[i]??0;$<HTMLInputElement>(`target-rotation-${i}`).value=String(rotation);$(`target-rotation-${i}-value`).textContent=`${Math.round(rotation*180/Math.PI)}°`;}document.querySelectorAll('[data-target]').forEach(el=>el.classList.toggle('selected',(el as HTMLElement).dataset.target===activeTarget));}
function initTargets(){if(!asset)return;const {positions}=forwardKinematics(asset,base);targets={head:Array.from(positions.slice(45,48)),left:Array.from(positions.slice(60,63)),right:Array.from(positions.slice(63,66)),headRotation:[0,0,0],leftGrip:0,rightGrip:0};viewer.setTargets(targets);refreshTargets();}
viewer.onTargetChange=(name,position)=>{targets[name]=position;activeTarget=name;refreshTargets();};
document.querySelectorAll('[data-target]').forEach(el=>el.addEventListener('click',()=>{activeTarget=(el as HTMLElement).dataset.target!;viewer.gizmo.attach(viewer.targetObjects[activeTarget]);refreshTargets();}));
$<HTMLInputElement>('grip').addEventListener('input',e=>{targets.leftGrip=targets.rightGrip=Number((e.target as HTMLInputElement).value);$('grip-value').textContent=`${Math.round(targets.leftGrip*100)}%`;});

function setMode(value:string){
  if(remote)return;mode=value;stopMotion();
  if(mode!=='pose')document.querySelectorAll('[data-clip]').forEach(el=>el.classList.remove('selected'));
  document.querySelectorAll('[data-mode]').forEach(el=>el.classList.toggle('selected',(el as HTMLElement).dataset.mode===mode));
  $('pose-controls').hidden=mode!=='pose';$('target-controls').hidden=mode!=='targets';$('camera-controls').hidden=mode!=='camera';$('motion-controls').hidden=mode!=='motion';viewer.showTargets(mode==='targets');
  if(mode==='motion'){clock=0;motionCycle=0;teleportNext=true;clip=selectedMotion()?'mixamo':'none';}
  if(mode!=='camera'){camera.stop();$('camera-start').textContent='Enable camera';}
}
document.querySelectorAll('[data-mode]').forEach(el=>el.addEventListener('click',()=>setMode((el as HTMLElement).dataset.mode!)));
$('reset').onclick=reset;$('home').onclick=()=>viewer.home();
$<HTMLInputElement>('show-skeleton').onchange=e=>viewer.skeleton.visible=(e.target as HTMLInputElement).checked;
$<HTMLSelectElement>('speed').onchange=e=>speed=Number((e.target as HTMLSelectElement).value);
function setClip(value:string){if(remote)return;clip=value;playing=clip!=='none';replaying=false;clock=0;document.querySelectorAll('[data-clip]').forEach(el=>el.classList.toggle('selected',(el as HTMLElement).dataset.clip===clip));$('play').textContent=playing?'Ⅱ':'▶';}
document.querySelectorAll('[data-clip]').forEach(el=>el.addEventListener('click',()=>{setMode('pose');setClip((el as HTMLElement).dataset.clip!);}));
$('play').onclick=()=>{if(remote)return;if(mode==='motion'){
  const motion=selectedMotion();if(!motion){$('motion-status').textContent='Import an FBX or load the sample first.';return;}
  if(!motionLoop&&clock>=motion.duration){clock=0;motionCycle=0;teleportNext=true;}
  replaying=false;clip='mixamo';playing=!playing;$('play').textContent=playing?'Ⅱ':'▶';
}else if(clip==='none')setClip('wave');else{playing=!playing;$('play').textContent=playing?'Ⅱ':'▶';}};
$<HTMLInputElement>('expression-strength').oninput=e=>{if(remote)return;const index=Number($<HTMLSelectElement>('expression-index').value),strength=Number((e.target as HTMLInputElement).value);base.expression[index]=strength;$('expression-value').textContent=strength.toFixed(2);};
$<HTMLSelectElement>('expression-index').onchange=e=>{const v=base.expression[Number((e.target as HTMLSelectElement).value)]??0;$<HTMLInputElement>('expression-strength').value=String(v);$('expression-value').textContent=v.toFixed(2);};

camera.onStatus=message=>$('camera-status').textContent=message;
$('camera-start').onclick=async()=>{if(remote)return;if(camera.stream){camera.stop();$('camera-start').textContent='Enable camera';return;}try{await camera.start();$('camera-start').textContent='Stop camera';}catch(error){camera.stop();$('camera-status').textContent=`Camera unavailable: ${String(error)}`;}};
$('camera-calibrate').onclick=()=>camera.calibrate();

function refreshMotions(){
  const select=$<HTMLSelectElement>('motion-select');select.replaceChildren();
  if(!motions.length)select.add(new Option('No imported motions',''));
  for(const m of motions)select.add(new Option(m.name,m.id));
  select.value=motionId;select.disabled=remote||!motions.length;
  const motion=selectedMotion();
  $<HTMLInputElement>('motion-seek').disabled=remote||!motion;
  $<HTMLInputElement>('motion-seek').max=String(motion?.duration??1);
  $('motion-details').textContent=motion?`${motion.duration.toFixed(2)}s · ${motion.mapped.filter(Boolean).length}/52 body + finger joints · ${motion.count} samples · scale ${motion.scale.toFixed(5)}`:'Files stay in this browser. The generated avatar is not uploaded to Mixamo.';
  $('motion-warnings').textContent=motion?.warnings.join(' ')??'';
  $('motion-warnings').hidden=!motion?.warnings.length;
}
function chooseMotion(id:string,autoplay=true){
  if(remote||!motions.some(m=>m.id===id))return;
  motionId=id;setMode('motion');clip='mixamo';playing=autoplay;clock=0;motionCycle=0;teleportNext=true;
  $('play').textContent=playing?'Ⅱ':'▶';refreshMotions();
}
function convertMotion(file:File,buffer:ArrayBuffer,target:any,reference?:{name:string,buffer:ArrayBuffer}){
  return new Promise<any[]>((resolve,reject)=>{
    const w=new Worker(new URL('./mixamo.worker.mjs',import.meta.url),{type:'module'});importWorkers.add(w);
    const finish=()=>{clearTimeout(timeout);w.terminate();importWorkers.delete(w);};
    const timeout=window.setTimeout(()=>{finish();reject(new Error('FBX conversion exceeded 45 seconds. Try a smaller animation-only export.'));},45000);
    w.onmessage=({data})=>{
      if(data.type==='progress')$('motion-status').textContent=`Retargeting ${file.name} · ${Math.round(data.progress*100)}%`;
      else if(data.type==='complete'){finish();resolve(data.motions);}
      else if(data.type==='error'){finish();reject(new Error(data.message));}
    };
    w.onerror=e=>{finish();reject(new Error(`Motion worker: ${e.message}`));};
    const ref=reference?{name:reference.name,buffer:reference.buffer.slice(0)}:undefined;
    w.postMessage({buffer,name:file.name,asset:target,reference:ref},ref?[buffer,ref.buffer]:[buffer]);
  });
}
async function importMotionFiles(files:File[]){
  if(remote||importingMotion)return;
  if(!asset||!worker){$('motion-status').textContent='Wait for an avatar to finish loading first.';return;}
  if(files.length>12){$('motion-status').textContent='Import at most 12 FBX files at once.';return;}
  const token=loadToken,target={meta:{nBones:asset.meta.nBones,nExpressions:asset.meta.nExpressions},arrays:{joints:asset.arrays.joints.slice(),parents:asset.arrays.parents.slice()}};
  importingMotion=true;$<HTMLButtonElement>('motion-import').disabled=true;$<HTMLButtonElement>('motion-sample').disabled=true;
  let first='',added=0;const errors:string[]=[];
  try {
    for(const file of files){
      try {
        if(!/\.fbx$/i.test(file.name))throw new Error(`${file.name}: choose an FBX file.`);
        if(file.size>MAX_FBX_BYTES)throw new Error(`${file.name}: maximum file size is 64 MB.`);
        $('motion-status').textContent=`Reading ${file.name}…`;
        const converted=await convertMotion(file,await file.arrayBuffer(),target,motionReference);
        if(token!==loadToken)throw new Error('The avatar changed during conversion. Reimport for the current avatar.');
        for(const m of converted){m.id=crypto.randomUUID();motions.push(m);first||=m.id;added++;}
      }catch(error){errors.push(String(error));if(token!==loadToken)break;}
    }
    if(first)chooseMotion(first);else refreshMotions();
    $('motion-status').textContent=`${added?`Imported ${added} motion${added===1?'':'s'}. `:''}${errors.join(' ')||'Press play, scrub the timeline, or open a receiver.'}`;
  }finally{importingMotion=false;$<HTMLButtonElement>('motion-import').disabled=remote;$<HTMLButtonElement>('motion-sample').disabled=remote;}
}
$('motion-import').onclick=()=>{if(!remote)$<HTMLInputElement>('motion-files').click();};
$<HTMLInputElement>('motion-files').onchange=e=>{const el=e.target as HTMLInputElement;void importMotionFiles(Array.from(el.files??[]));el.value='';};
$('motion-sample').onclick=async()=>{
  if(remote||importingMotion)return;
  try{const response=await fetch('/motions/samba.fbx');if(!response.ok)throw new Error('Sample not installed. Run scripts/setup_mixamo_samples.ps1 and rebuild.');
    const file=new File([await response.blob()],'Mixamo Samba.fbx');await importMotionFiles([file]);
  }catch(error){$('motion-status').textContent=String(error);}
};
$<HTMLSelectElement>('motion-select').onchange=e=>chooseMotion((e.target as HTMLSelectElement).value);
$<HTMLInputElement>('motion-seek').oninput=e=>{if(remote)return;clock=Number((e.target as HTMLInputElement).value);playing=false;replaying=false;motionCycle=0;teleportNext=true;$('play').textContent='▶';};
$<HTMLInputElement>('motion-inplace').onchange=e=>{if(remote)return;motionInPlace=(e.target as HTMLInputElement).checked;teleportNext=true;};
$<HTMLInputElement>('motion-loop').onchange=e=>{if(remote)return;const motion=selectedMotion();if(motion)clock=Math.min(motion.duration,motionLoop?clock%motion.duration:clock);motionLoop=(e.target as HTMLInputElement).checked;motionCycle=0;teleportNext=true;};
$<HTMLInputElement>('motion-reference').onchange=async e=>{
  if(remote)return;const input=e.target as HTMLInputElement,file=input.files?.[0];input.value='';if(!file)return;
  if(!/\.fbx$/i.test(file.name)||file.size>MAX_FBX_BYTES){$('reference-name').textContent='Choose a matching T-pose FBX under 64 MB.';return;}
  motionReference={name:file.name,buffer:await file.arrayBuffer()};$('reference-name').textContent=`${file.name} · applies to future imports`;
};
$('reference-clear').onclick=()=>{if(!remote){motionReference=undefined;$('reference-name').textContent='Use the animation’s embedded rest pose (default).';}};

async function api(path:string,options?:RequestInit){const response=await fetch(path,options);if(!response.ok){let detail='Request failed';try{detail=(await response.json()).detail??detail;}catch{}throw new Error(`${response.status}: ${detail}`);}return response.json();}
async function refreshLibrary(){
  const avatars=await api('/api/avatars');$('avatar-count').textContent=String(avatars.length);$('avatar-list').replaceChildren();
  for(const item of avatars){const button=document.createElement('button');button.className=`avatar-item ${item.id===assetId?'selected':''}`;button.dataset.asset=item.id;const icon=document.createElement('span');icon.className='avatar-icon';icon.textContent='♙';const text=document.createElement('span'),title=document.createElement('strong'),small=document.createElement('small');title.textContent=item.label;small.textContent=`${(item.nGaussians/1000).toFixed(0)}k Gaussians · full body`;text.append(title,small);button.append(icon,text);button.onclick=()=>{if(!remote)void loadAvatar(item.id);};$('avatar-list').append(button);}
  return avatars;
}

async function loadAvatar(id:string){
  if(!/^[a-zA-Z0-9_-]{1,80}$/.test(id))return;
  const token=++loadToken;loading('Loading the avatar and its rig…');notice('');worker?.terminate();worker=undefined;busy=false;verifying=false;gpuStressChecks=[];deformationBackend='starting';deformationReason='';deformationAdapter=undefined;camera.stop();$('camera-start').textContent='Enable camera';recording=false;replaying=false;recorded=[];renderedFrames=0;deformSamples=[];stopMotion();
  $<HTMLButtonElement>('replay').disabled=true;$<HTMLButtonElement>('save-recording').disabled=true;$('record').classList.remove('recording');$('record').textContent='● Record';$('record-status').textContent='Record resolved poses, then replay the same motion.';
  if(!remote)void restoreAvatarPhotos(id,token);
  try{
    const meta=await api(`/api/avatars/${id}/avatar.json`),response=await fetch(`/api/avatars/${id}/data`);
    if(!response.ok)throw new Error('Avatar binary could not be loaded');
    const buffer=await response.arrayBuffer();
    if(buffer.byteLength!==meta.bytes)throw new Error('Avatar download is incomplete');
    if(crypto.subtle){const digest=await crypto.subtle.digest('SHA-256',buffer),hex=Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,'0')).join('');if(hex!==meta.sha256)throw new Error('Avatar checksum mismatch');}
    if(token!==loadToken)return;
    asset=decodeAsset(meta,buffer);assetId=id;base=identityPose(meta.nBones,meta.nExpressions);base.angles[50]=-0.35;base.angles[53]=0.35;lastPose=copyPose(base);
    const frame=deform(asset,base);lastFrame=frame;await viewer.load(asset,frame);if(token!==loadToken)return;
    if(viewer.mesh)viewer.mesh.visible=!remote;
    worker=new Worker(new URL('./deform.worker.mjs',import.meta.url),{type:'module'});
    worker.onmessage=({data})=>{
      if(token!==loadToken)return;
      if(data.type==='backend'){deformationBackend=data.backend;deformationReason=data.reason;deformationAdapter=data.adapter;$('deformer-status').textContent=data.backend==='webgpu'?'WebGPU deformation · full skin weights · timing includes readback':`CPU reference deformation${data.reason?' · GPU unavailable; fallback active':''}`;$('deformer-status').title=data.reason??'';}
      if(data.type==='frame'){busy=false;deformMs=data.ms;deformSamples.push(data.ms);if(deformSamples.length>300)deformSamples.shift();lastFrame=data;viewer.updateSplats(data);updateCount++;renderedFrames++;}
      if(data.type==='verified'){verifying=false;verifyResults=data.results;gpuStressChecks=data.stress??[];showVerification();if(gpuStressChecks.some(c=>!c.pass))notice('GPU/CPU stress comparison failed. Use ?deformer=cpu while investigating.');}
      if(data.type==='error'){busy=false;verifying=false;notice(data.message);}
    };
    worker.onerror=e=>{busy=false;notice(`Animation worker: ${e.message}`);};
    const workerBuffer=buffer.slice(0);worker.postMessage({type:'load',meta,buffer:workerBuffer,backend:params.get('deformer')??'auto'},[workerBuffer]);
    $('stage-empty').hidden=true;loading('');$('avatar-title').textContent=meta.label??(id==='example'?'Example / full body':'Your photo / full body');$('splat-count').textContent=`${meta.model} · ${meta.nGaussians.toLocaleString()} GAUSSIANS`;
    $('mode-badge').textContent=remote?'REMOTE AVATAR':'LOCAL AVATAR';
    const download=$<HTMLAnchorElement>('download');download.href=`/api/avatars/${id}/download`;download.hidden=false;
    const expr=$<HTMLSelectElement>('expression-index');expr.replaceChildren();for(let e=0;e<meta.nExpressions;e++){const option=new Option(`Expression ${e+1}`,String(e));expr.add(option);}
    const synthetic=meta.generation?.engine==='synthetic-demo';
    $<HTMLButtonElement>('verify').disabled=!meta.referencePoses?.length;
    $('verify').textContent=meta.referencePoses?.length?'Compare with Python reference':'No native reference in this asset';
    $<HTMLInputElement>('expression-strength').disabled=synthetic;expr.disabled=synthetic;
    updateControlLabels();initTargets();await refreshLibrary();
    motions=motions.filter(m=>m.rigKey===rigKey(asset));if(!motions.some(m=>m.id===motionId))motionId=motions[0]?.id??'';if(mode==='motion'&&selectedMotion())clip='mixamo';refreshMotions();
    if(socket?.readyState===WebSocket.OPEN&&!remote)socket.send(JSON.stringify({type:'asset',id}));
    playback.frames=[];playback.sequence=null;verifyResults=[];$('verify-results').hidden=true;
  }catch(error){if(token!==loadToken)return;choosePhotos([]);photoStatus('The avatar could not be opened. Its photo previews were cleared.');loading('');notice(String(error));}
}

function generationChoice(){return generationModels.find(m=>m.id===$<HTMLSelectElement>('generation-model').value);}
function updateGenerateButton(){
  const model=generationChoice(),valid=selectedPhotos.length>0&&selectedPhotos.length<=(model?.maxPhotos??1);
  $<HTMLButtonElement>('generate').disabled=!valid||photosIncomplete||!model?.available||remote||generationPending;
}
function updateGenerationControls(){
  const model=generationChoice(),shape=$<HTMLSelectElement>('generation-shape'),previous=shape.value,modes=model?.shapeModes??['zero','estimate'];
  shape.replaceChildren();for(const value of modes)shape.add(new Option(value==='predicted'?'Native LHM++ ShapeHead':value==='zero'?'Default shape · baseline':'Estimate from photo · experimental',value));
  shape.value=modes.includes(previous)?previous:modes[0];shape.disabled=modes.length===1;
  $('capture-tip').textContent=model?.maxPhotos>1?'Choose 1–8 full-body photos of the same person in the same outfit, with front, side, and back coverage. Use only photos you have permission to process.':'Face the camera. Keep your whole body in frame and your arms slightly away from your sides.';
  $('generation-help').textContent=model?.maxPhotos>1?'LHM++ runs locally in isolated Docker. Up to 16 MB per photo, 64 MB total. Predicted shape is built in; quality is experimental.':'One photo per LHM job. Estimated proportions and the larger checkpoint are research options, not validated quality upgrades.';
  if(!model?.available)$('generation-help').textContent=model?.reason??'Viewer mode. See docs/generation.md to configure photo generation.';
  if(selectedPhotos.length>(model?.maxPhotos??1))notice('Choose LHM++ for multiple photos, or remove photos until one remains.');
  else notice('');
  updateGenerateButton();
}
function photoStatus(message:string){$('photo-status').textContent=message;$('photo-status').hidden=!message;}
function choosePhotos(files:File[],source:'selected'|'saved'='selected'){
  if(files.length>8||files.some(f=>f.size>16*1024*1024)||files.reduce((sum,f)=>sum+f.size,0)>64*1024*1024){notice('Choose up to 8 photos, under 16 MB each and 64 MB combined.');return;}
  if(files.some(f=>!['image/jpeg','image/png','image/webp'].includes(f.type))){notice('Choose JPG, PNG, or WebP photos.');return;}
  photoRestore?.abort();photoRestore=undefined;photosIncomplete=false;photoSelectionRevision++;
  for(const url of selectedPhotoUrls)URL.revokeObjectURL(url);
  selectedPhotos=files;selectedPhotoUrls=files.map(file=>URL.createObjectURL(file));
  const preview=$<HTMLImageElement>('photo-preview');preview.hidden=files.length!==1;if(files.length===1)preview.src=selectedPhotoUrls[0];else preview.removeAttribute('src');
  $('drop-zone').querySelector('svg')!.style.display=files.length?'none':'';
  $('photo-label').textContent=files.length?`${files.length} ${source==='saved'?'saved source ':''}photo${files.length===1?'':'s'}${source==='selected'?' ready':''} · click to replace`:'Drop full-body photos';
  photoStatus(source==='saved'?'Saved inputs for the opened avatar. Replacing or removing photos only changes the next generation.':files.length?'Selected for a new generation; the saved avatar is unchanged.':'');
  const list=$('photo-list');list.replaceChildren();list.hidden=files.length<2;
  files.forEach((file,i)=>{const item=document.createElement('div'),img=document.createElement('img'),button=document.createElement('button');item.className='photo-item';img.src=selectedPhotoUrls[i];img.alt=`View ${i+1}: ${file.name}`;button.textContent='×';button.setAttribute('aria-label',`Remove photo ${i+1}`);button.onclick=()=>choosePhotos(selectedPhotos.filter((_,index)=>index!==i));item.append(img,button);list.append(item);});
  updateGenerationControls();
}
async function restoreAvatarPhotos(id:string,token:number){
  choosePhotos([]);photoStatus('Loading this avatar’s source photos…');
  const controller=new AbortController(),revision=generationSelectionRevision;photoRestore=controller;
  const current=()=>token===loadToken&&!controller.signal.aborted;
  try{
    const catalog=await api(`/api/avatars/${id}/source-photos`,{signal:controller.signal});
    if(!current())return;
    if(catalog.assetId===id&&catalog.synthetic===true&&catalog.total===0&&Array.isArray(catalog.photos)&&catalog.photos.length===0){photoRestore=undefined;photoStatus('Synthetic demo: original procedural geometry, with no source photos or model weights.');return;}
    if(catalog.assetId!==id||!Array.isArray(catalog.photos)||!Number.isInteger(catalog.total)||catalog.total<1||catalog.total>8||catalog.photos.length!==catalog.total)throw Error('Invalid source photo list');
    let shapeNote='';
    if(revision===generationSelectionRevision&&generationModels.some(model=>model.id===catalog.model)){
      $<HTMLSelectElement>('generation-model').value=catalog.model;updateGenerationControls();
      if(generationChoice()?.shapeModes.includes(catalog.shapeMode))$<HTMLSelectElement>('generation-shape').value=catalog.shapeMode;
      else shapeNote=' The saved proportions used an external setting; the next generation uses the selected body-proportion option.';
    }
    const loaded=await Promise.all(catalog.photos.map(async(photo:any)=>{
      if(!photo.available)return null;
      if(!Number.isInteger(photo.index)||photo.index<0||photo.index>=8||photo.url!==`/api/avatars/${id}/source-photos/${photo.index}`)throw Error('Invalid source photo URL');
      try{
        const response=await fetch(photo.url,{signal:controller.signal});if(!response.ok)throw Error('Photo unavailable');
        const blob=await response.blob();if(blob.type!=='image/png'||blob.size>16*1024*1024)throw Error('Invalid source photo');
        const bitmap=await createImageBitmap(blob);bitmap.close();
        return new File([blob],`view-${photo.index+1}.png`,{type:'image/png'});
      }catch(error){if(controller.signal.aborted)throw error;return null;}
    }));
    if(!current())return;
    const files=loaded.filter((photo:File|null):photo is File=>photo!==null);
    if(files.reduce((sum:number,file:File)=>sum+file.size,0)>64*1024*1024)throw Error('Saved photos exceed the upload limit');
    choosePhotos(files,'saved');photosIncomplete=files.length!==catalog.total;
    if(photosIncomplete)photoStatus(files.length?`Only ${files.length} of ${catalog.total} saved source photos are available. Choose a complete photo set before generating again.`:'Source photos are unavailable for this avatar. Choose new photos to generate another avatar.');
    else photoStatus($('photo-status').textContent+shapeNote);
    updateGenerateButton();
  }catch(error){
    if(!current())return;
    photoRestore=undefined;photosIncomplete=true;photoStatus('Could not load the source photos. Reopen the avatar to retry, or choose new photos.');updateGenerateButton();
  }
}
$('generation-model').onchange=()=>{generationSelectionRevision++;updateGenerationControls();};
$('generation-shape').onchange=()=>{generationSelectionRevision++;};
$<HTMLInputElement>('photo').onchange=e=>{const input=e.target as HTMLInputElement;choosePhotos(Array.from(input.files??[]));input.value='';};
$('drop-zone').ondragover=e=>{e.preventDefault();$('drop-zone').classList.add('dragging');};$('drop-zone').ondragleave=()=>$('drop-zone').classList.remove('dragging');
$('drop-zone').ondrop=e=>{e.preventDefault();$('drop-zone').classList.remove('dragging');choosePhotos(Array.from(e.dataTransfer?.files??[]));};
$('use-example').onclick=async()=>{photoRestore?.abort();photoRestore=undefined;const token=loadToken,revision=++photoSelectionRevision;try{const r=await fetch('/api/example-photo');if(!r.ok)throw new Error('Example unavailable');const blob=await r.blob();if(token===loadToken&&revision===photoSelectionRevision)choosePhotos([new File([blob],'example-full-body.png',{type:'image/png'})]);}catch(error){if(token===loadToken&&revision===photoSelectionRevision){photoStatus('The example photo could not be loaded. Try again or choose a photo.');notice(String(error));}}};
$('generate').onclick=async()=>{
  updateGenerateButton();if($<HTMLButtonElement>('generate').disabled)return;generationPending=true;updateGenerateButton();notice('');$('job-progress').hidden=false;
  try{
    const form=new FormData(),multi=(generationChoice()?.maxPhotos??1)>1;for(const file of selectedPhotos)form.append(multi?'photos':'photo',file);form.append('model',$<HTMLSelectElement>('generation-model').value);form.append('shape_mode',$<HTMLSelectElement>('generation-shape').value);const job=await api('/api/jobs',{method:'POST',body:form});localStorage.setItem('gsavatar-active-job',job.id);pollJob(job.id);
  }catch(error){generationPending=false;notice(String(error));updateGenerateButton();}
};
function pollJob(id:string){
  if(jobPoll)clearTimeout(jobPoll);
  const poll=async()=>{try{const job=await api(`/api/jobs/${id}`);$('job-progress').hidden=false;$('job-stage').textContent=job.stage;$('job-percent').textContent=`${Math.round(job.progress*100)}%`;$<HTMLProgressElement>('progress').value=job.progress;
    if(job.status==='complete'){generationPending=false;localStorage.removeItem('gsavatar-active-job');$('job-note').textContent=`Created in ${job.seconds}s. The GPU worker has exited.`;updateGenerateButton();await refreshLibrary();await loadAvatar(job.assetId);}
    else if(job.status==='failed'){generationPending=false;localStorage.removeItem('gsavatar-active-job');$('job-note').textContent='Check the photos and worker setup, then try again.';notice(job.error??'Generation failed');updateGenerateButton();}
    else jobPoll=window.setTimeout(poll,1000);
  }catch(error){generationPending=false;localStorage.removeItem('gsavatar-active-job');notice(String(error));updateGenerateButton();}};generationPending=true;updateGenerateButton();void poll();
}

$('record').onclick=()=>{if(!asset||remote)return;recording=!recording;if(recording){recorded=[];recordStart=performance.now();replaying=false;}else{$<HTMLButtonElement>('replay').disabled=recorded.length<2;$<HTMLButtonElement>('save-recording').disabled=recorded.length<2;}$('record').classList.toggle('recording',recording);$('record').textContent=recording?'■ Stop':'● Record';};
$('replay').onclick=()=>{if(recorded.length<2||remote)return;recording=false;replaying=true;playing=false;replayStart=performance.now();replayCycle=0;teleportNext=true;$('play').textContent='▶';$('record').classList.remove('recording');$('record').textContent='● Record';$('record-status').textContent='Replaying captured poses';};
$('save-recording').onclick=()=>{const payload={version:1,assetId,assetSha256:asset.meta.sha256,frames:recorded.map(f=>({time:f.time,pose:{...f.pose,angles:Array.from(f.pose.angles),expression:Array.from(f.pose.expression)}}))};const url=URL.createObjectURL(new Blob([JSON.stringify(payload)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`avatar-motion-${assetId}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};

function connectRoom(id:string){
  const gen=++connectionGeneration;socket?.close();roomId=id;playback.frames=[];playback.sequence=null;
  socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/${id}?role=${remote?'receiver':'sender'}`);socket.binaryType='arraybuffer';
  socket.onopen=()=>{if(gen!==connectionGeneration)return;$('connection').textContent=remote?'Receiving':'Sending';if(assetId&&!remote)socket!.send(JSON.stringify({type:'asset',id:assetId}));};
  socket.onclose=()=>{if(gen===connectionGeneration)$('connection').textContent='Disconnected';};
  socket.onerror=()=>notice('The local pose relay could not connect. Check the backend.');
  socket.onmessage=({data})=>{
    if(typeof data==='string'){const m=JSON.parse(data);if(m.type==='asset'&&remote&&m.id!==assetId)void loadAvatar(m.id);if(m.type==='error')notice(m.message);if(m.type==='sender-left')$('connection').textContent='Sender disconnected';return;}
    if(!remote||Math.random()<loss)return;
    try{const p=decodePose(data);bytes=data.byteLength;if(asset&&p.expression.length!==asset.meta.nExpressions)throw new Error('Remote rig does not match this avatar');const currentId=assetId;setTimeout(()=>{if(gen!==connectionGeneration||currentId!==assetId||!asset)return;if(playback.push(p,performance.now())&&viewer.mesh)viewer.mesh.visible=true;},Math.max(0,latency+(Math.random()*2-1)*jitter));}catch(error){notice(String(error));}
  };
}
$('share').onclick=()=>{if(!asset||remote)return;if(!roomId){roomId=crypto.randomUUID().slice(0,8);connectRoom(roomId);}const url=new URL(location.href);url.search='';url.searchParams.set('room',roomId);url.searchParams.set('remote','1');window.open(url,'_blank');$('network-note').textContent=`Room ${roomId} · send at 30 Hz. Adjust delay and loss in the receiver.`;};
function updateNetworkControls(){latency=Number($<HTMLInputElement>('latency').value);jitter=Number($<HTMLInputElement>('jitter').value);loss=Number($<HTMLSelectElement>('loss').value);$('latency-value').textContent=`${latency} ms`;$('jitter-value').textContent=`${jitter} ms / ${Math.round(loss*100)}%`;}
for(const id of ['latency','jitter','loss'])$(id).addEventListener('input',updateNetworkControls);
async function verifyAsset(){
  if(!worker||verifying)return;verifying=true;const token=loadToken,id=assetId,meta=asset.meta;
  $('verify-results').hidden=false;$('verify-results').textContent=`Comparing all ${meta.nGaussians.toLocaleString()} Gaussians against seven Python poses…`;
  try{
    if(meta.validation){
      const response=await fetch(`/api/avatars/${id}/validation`);if(!response.ok)throw new Error('Native validation bundle unavailable');
      const buffer=await response.arrayBuffer();if(buffer.byteLength!==meta.validation.bytes)throw new Error('Incomplete native validation bundle');
      const digest=await crypto.subtle.digest('SHA-256',buffer),hex=Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,'0')).join('');
      if(hex!==meta.validation.sha256)throw new Error('Native validation checksum mismatch');
      if(token!==loadToken)return;worker?.postMessage({type:'verify',buffer},[buffer]);
    }else worker.postMessage({type:'verify'});
  }catch(error){if(token===loadToken){verifying=false;notice(String(error));}}
}
$('verify').onclick=()=>void verifyAsset();
function showVerification(){
  const root=$('verify-results');root.hidden=false;root.replaceChildren();const title=document.createElement('strong');title.textContent=`${verifyResults.filter(x=>x.pass).length}/${verifyResults.length} reference poses pass`;root.append(title);const table=document.createElement('table');for(const result of verifyResults){const tr=document.createElement('tr');for(const value of [result.name,`${result.rmsMm.toFixed(3)} mm RMS`,result.pass?'✓':'FAIL']){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);}root.append(table);const note=document.createElement('small');note.textContent='All centers < 1 mm; orientations < 0.115°. Before renderer packing.';root.append(note);
}

let lastTick=performance.now();
setInterval(()=>{
  const now=performance.now(),dt=Math.min(0.1,(now-lastTick)/1000);lastTick=now;
  if(!asset||!worker)return;if(playing)clock+=dt*speed;
  let p:any;
  if(remote){p=playback.sample(now);if(!p)return;}
  else if(replaying&&recorded.length>1){const duration=recorded[recorded.length-1].time,elapsed=now-replayStart,cycle=Math.floor(elapsed/duration);if(cycle!==replayCycle){teleportNext=true;replayCycle=cycle;}p=sampleRecording(recorded,elapsed%duration);}
  else if(mode==='targets')p=solveTargets(asset,base,targets);
  else if(mode==='camera'){void camera.capture(now);p=camera.apply(asset,base);if(lastPose)p=interpolatePose(lastPose,p,0.35);}
  else if(mode==='motion'&&selectedMotion()){
    const motion=selectedMotion(),cycle=Math.floor(clock/motion.duration);
    if(motionLoop&&cycle!==motionCycle){teleportNext=true;motionCycle=cycle;}
    if(!motionLoop&&clock>=motion.duration){clock=motion.duration;playing=false;$('play').textContent='▶';}
    p=sampleMotion(motion,clock,base,{inPlace:motionInPlace,loop:motionLoop});
    const t=motionLoop?clock%motion.duration:clock;$<HTMLInputElement>('motion-seek').value=String(t);$('motion-time').textContent=`${t.toFixed(2)} / ${motion.duration.toFixed(2)}s`;
  }
  else p=applyClip(base,clip,clock);
  if(!remote&&teleportNext)p.teleport=true;
  ++frameId;if(!remote)p.tick=frameId;lastPose=p;
  if(recording&&!remote){recorded.push({time:now-recordStart,pose:copyPose(p)});$('record-status').textContent=`Recording · ${((now-recordStart)/1000).toFixed(1)}s · ${recorded.length} frames`;if(now-recordStart>60000)$('record').click();}
  if(!remote&&now-lastSend>=32){const blob=encodePose(p,sequence++);bytes=blob.byteLength;if(socket?.readyState===WebSocket.OPEN)socket.send(blob);lastSend=now;teleportNext=false;}
  if(!busy&&!verifying){busy=true;worker.postMessage({type:'pose',pose:p,id:frameId});}
},1000/30);
setInterval(()=>{updateHz=updateCount;updateCount=0;$('render-fps').textContent=viewer.fps?viewer.fps.toFixed(0):'—';$('deform-ms').textContent=deformMs?deformMs.toFixed(1):'—';$('pose-hz').textContent=asset?String(updateHz):'—';$('packet-size').textContent=bytes?String(bytes):'—';if(remote)$('network-note').textContent=`${playback.received} received · ${playback.stale} stale discarded · 80 ms interpolation buffer`;},1000);

// Inspectable diagnostics for research checks and browser automation.
(window as any).__lab={
  get generation(){return {model:asset?.meta.model,provenance:asset?.meta.generation,deformSamples:[...deformSamples],backend:deformationBackend,backendReason:deformationReason,adapter:deformationAdapter,assetBytes:asset?.meta.bytes,version:asset?.meta.version,gpuStressChecks};},
  simulateDeviceLoss(){worker?.postMessage({type:'simulate-device-loss'});},
  resetDeformSamples(){deformSamples=[];},
  setView(angle:number,distance=4.1,targetY=-0.2){viewer.controls.enableDamping=false;viewer.controls.target.set(0,targetY,0);viewer.camera.position.set(Math.sin(angle)*distance,targetY,Math.cos(angle)*distance);viewer.controls.update();},
  setAvatarVisible(visible:boolean){if(viewer.mesh)viewer.mesh.visible=visible;},
  get state(){return {assetId,loaded:!!asset&&!!worker,mode,clip,remote,roomId,renderedFrames,deformMs,updateHz,renderFps:viewer.fps,renderP95Ms:viewer.frameMs,bytes,received:playback.received,recorded:recorded.length,replaying,pose:copyPose(lastPose),probe:lastFrame?Array.from(lastFrame.positions.slice(0,18)):[],joints:lastFrame?Array.from(lastFrame.joints??[]):[],verification:verifyResults,cameraReady:camera.ready,cameraMs:camera.ms,motions:motions.map(m=>({id:m.id,name:m.name,duration:m.duration,mapped:m.mapped.filter(Boolean).length,scale:m.scale})),motionId,motionInPlace,motionLoop,playing,motionTime:clock,importingMotion};},
  setJoint(j:number,a:number,v:number){if(remote)return;stopMotion();base.angles[j*3+a]=v;updateControlLabels();},
  setRoot(x:number,y:number,z:number){if(!remote)base.rootPosition=[x,y,z];},
  setClip,connectRoom,loadAvatar,
  visiblePixels(){return viewer.visiblePixels();},
  verify:verifyAsset,stopCamera(){camera.stop();}
};

async function boot(){
  try{const health=await api('/api/health');const select=$<HTMLSelectElement>('generation-model');if(health.models){generationModels=health.models;select.replaceChildren();for(const model of health.models){const option=new Option(model.label+(model.available?'':' · unavailable'),model.id);option.disabled=!model.available;option.title=model.reason??'';select.add(option);}select.value=health.model;}updateGenerationControls();
    $('use-example').hidden=!health.examplePhotoAvailable;
    $('server-status').lastChild!.textContent=health.workerAvailable?'Generation worker ready':health.generationEnabled?'Viewer ready · check model setup':'Viewer ready · generation disabled';const avatars=await refreshLibrary();if(remote){$('avatar-title').textContent='Waiting for the sender';$('mode-badge').textContent='REMOTE AVATAR';if(roomId)connectRoom(roomId);}else{if(avatars.length)await loadAvatar(avatars.find((a:any)=>a.id===params.get('asset'))?.id??avatars.find((a:any)=>a.id==='example')?.id??avatars[0].id);if(roomId)connectRoom(roomId);const job=localStorage.getItem('gsavatar-active-job');if(job)pollJob(job);}}
  catch(error){$('server-status').lastChild!.textContent='Backend unavailable';notice(`Start the local backend on port 8765. ${String(error)}`);}
}
window.addEventListener('beforeunload',()=>{photoRestore?.abort();camera.stop();worker?.terminate();socket?.close();for(const w of importWorkers)w.terminate();for(const url of selectedPhotoUrls)URL.revokeObjectURL(url);});
void boot();
