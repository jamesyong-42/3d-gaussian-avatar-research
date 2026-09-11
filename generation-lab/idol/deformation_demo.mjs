import {loadDeformation} from './deformation_check.mjs';
import {CovarianceViewer} from './covariance_viewer.mjs';
import {ResidentViewer} from './resident_viewer.mjs';
const $=id=>document.getElementById(id);
const resident=new URLSearchParams(location.search).get('renderer')!=='bridge';
let lab,viewer,busy=false,dirty=true,playing=false,source=0,clock=0,last=performance.now(),preset='front';
const state={ready:false,frames:0,samples:[],playing:false,error:null,renderer:resident?'webgpu-resident':'webgl-bridge'};
function base(){return new Float32Array(lab.validation.cases.find(c=>c.name===`source-${source}-${preset}`).params);}
function controls(){
  const p=base();p[5]+=Number($('yaw').value);p[54]+=Number($('arms').value);p[57]-=Number($('arms').value);p[170]+=Number($('jaw').value);p[82]+=Number($('grip').value);p[127]+=Number($('grip').value);
  if(playing){p[54]+=.22*Math.sin(clock*1.2);p[57]-=.22*Math.sin(clock*1.2);p[5]+=.25*Math.sin(clock*.6);p[170]+=.12*(1+Math.sin(clock*2));p[82]+=.15*(1+Math.sin(clock*.9));p[1]+=.04*Math.sin(clock);}
  return p;
}
async function render(params,which=source){
  const start=performance.now(),result=resident?await lab.gpu.render(params,which,viewer):await lab.gpu.evaluate(params,which);
  const draw=resident?{}:viewer.draw(result.geometry,lab.assets[which]);
  if(window.__idol)window.__idol.lastGeometry=result.geometry;
  const timing={...(resident?{gpuLoopMs:result.totalMs}:{geometryMs:result.totalMs,...draw}),endToEndMs:performance.now()-start};state.frames++;state.samples.push(timing);if(state.samples.length>2000)state.samples.shift();
  $('status').textContent=`${lab.gpu.n.toLocaleString()} splats · ${state.renderer} · full frame ${timing.endToEndMs.toFixed(1)} ms · completed ${state.frames} frames`;
  return {...result,timing};
}
function setPlay(value){playing=value;state.playing=value;last=performance.now();$('play').textContent=value?'Pause motion':'Play signal-driven motion';dirty=true;}
function reset(){setPlay(false);clock=0;for(const id of ['yaw','arms','jaw','grip']){$(id).value=0;$(id+'-value').value='0.00';}dirty=true;}
for(const id of ['yaw','arms','jaw','grip'])$(id).addEventListener('input',()=>{setPlay(false);$(id+'-value').value=Number($(id).value).toFixed(2);dirty=true;});
$('source').addEventListener('change',()=>{source=Number($('source').value);reset();});$('preset').addEventListener('change',()=>{preset=$('preset').value;reset();});
$('play').addEventListener('click',()=>setPlay(!playing));$('reset').addEventListener('click',reset);
function fail(error){state.error=String(error);$('error').textContent=String(error);setPlay(false);$('controls').disabled=true;}
async function frame(now){
  if(playing)clock+=Math.min(.1,(now-last)/1000);last=now;
  if(lab&&!busy&&(dirty||playing)&&!state.error){busy=true;dirty=false;try{await render(controls());}catch(error){fail(error);}finally{busy=false;}}
  requestAnimationFrame(frame);
}
try{
  lab=await loadDeformation();viewer=resident?await ResidentViewer.create($('avatar'),lab.gpu):new CovarianceViewer($('avatar'),lab.gpu.n);
  window.__idol={state,lab,viewer,render,setPlay,pause:()=>setPlay(false),get busy(){return busy;},get source(){return source;},get parameters(){return controls();}};
  await render(controls());dirty=false;state.ready=true;$('controls').disabled=false;requestAnimationFrame(frame);
}catch(error){fail(error);}
addEventListener('pagehide',()=>{setPlay(false);viewer?.destroy();lab?.gpu.destroy();});
