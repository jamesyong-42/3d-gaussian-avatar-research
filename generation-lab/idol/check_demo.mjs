import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {chromium} from '../../web/node_modules/playwright-core/index.mjs';
import {startServer,validateFolder} from './lab_server.mjs';
if(process.argv.length<3||process.argv.length>4||!['resident','bridge'].includes(process.argv[3]??'resident'))throw Error('Supply a deformation run directory and optional resident|bridge');
const renderer=process.argv[3]??'resident';
const folder=validateFolder(process.argv[2]),here=path.dirname(fileURLToPath(import.meta.url));
if(execFileSync('C:/Program Files/Docker/Docker/resources/bin/docker.exe',['ps','--filter','label=research.task=idol','--format','{{.Names}}'],{encoding:'utf8',windowsHide:true}).trim())throw Error('Native IDOL container still running');
const record={date:new Date().toISOString(),renderer,scope:'Visible live deformation lab, not native CUDA rasterizer parity or main-app integration',errors:[],renders:[],
  imageLimits:{meanChannelError:.1,pixelsWithRgbL1Over30:100,minVisiblePixels:1000},
  sources:Object.fromEntries(['covariance_viewer.mjs','resident_viewer.mjs','deformation_demo.html','deformation_demo.mjs','deformation_check.mjs','deformation_gpu.mjs','neighbors_gpu.mjs','check_demo.mjs'].map(n=>[n,crypto.createHash('sha256').update(fs.readFileSync(path.join(here,n))).digest('hex')]))};
const prefix='demo-'+record.date.replace(/[:.]/g,'-'),destination=path.join(folder,prefix+'.json');
const lab=await startServer(folder,fs.readFileSync(path.join(here,'deformation_demo.html'),'utf8'));let browser;
try{
  browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,args:['--disable-background-timer-throttling','--disable-renderer-backgrounding']});
  const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
  page.on('pageerror',e=>record.errors.push(String(e)));await page.goto(lab.url+'?renderer='+renderer);
  await page.waitForFunction(()=>window.__idol?.state.ready||document.getElementById('error')?.textContent,{},{timeout:120000});
  const state=await page.evaluate(()=>window.__idol?.state);if(!state?.ready)throw Error(await page.locator('#error').textContent());
  record.rendererInfo=await page.evaluate(()=>window.__idol.viewer.info??{readbackInLiveLoop:true,renderSort:'CPU depth ordering'});
  const wait=()=>page.waitForFunction(()=>window.__idol.state.ready&&!window.__idol.busy&&!window.__idol.state.playing);
  for(const source of [0,1])for(const preset of ['front','arms']){
    const before=await page.evaluate(()=>window.__idol.state.frames);
    await page.selectOption('#source',String(source));await page.selectOption('#preset',preset);
    await page.waitForTimeout(150);
    await page.waitForFunction(n=>window.__idol.state.frames>n&&!window.__idol.busy,before);
    await wait();const filename=prefix+`-source-${source}-${preset}.png`;
    await page.locator('#avatar').screenshot({path:path.join(folder,filename)});
    const comparison=await page.evaluate(async({source,preset})=>{
      const app=window.__idol,browserPixels=await app.viewer.pixels();
      const {getBinary}=await import('/lab_io.mjs');const c=app.lab.validation.cases.find(x=>x.name===`source-${source}-${preset}`);
      const reference=new Float32Array(await getBinary('/'+c.file,c.bytes,c.sha256)),native=new Float32Array(app.lab.gpu.n*12);
      for(let i=0;i<app.lab.gpu.n;i++)native.set(reference.subarray(i*20,i*20+12),i*12);
      await app.viewer.draw(native,app.lab.assets[source]);const refPixels=await app.viewer.pixels();
      let absolute=0,maxChannel=0,changed=0,visible=0;
      for(let i=0;i<browserPixels.length;i+=4){let delta=0;for(let k=0;k<3;k++){const d=Math.abs(browserPixels[i+k]-refPixels[i+k]);absolute+=d;delta+=d;maxChannel=Math.max(maxChannel,d);}
        if(delta>30)changed++;if(765-browserPixels[i]-browserPixels[i+1]-browserPixels[i+2]>30)visible++;
      }
      await app.viewer.clear();const background=await app.viewer.pixels();let negativeControl=0;
      for(let i=0;i<background.length;i+=4)if(Math.abs(browserPixels[i]-background[i])+Math.abs(browserPixels[i+1]-background[i+1])+Math.abs(browserPixels[i+2]-background[i+2])>30)negativeControl++;
      await app.render(new Float32Array(c.params),source);
      return{visiblePixels:visible,negativeControlPixels:negativeControl,meanChannelError:absolute/(browserPixels.length*.75),maxChannelError:maxChannel,pixelsWithRgbL1Over30:changed,
        scope:'Native-reference and browser geometry drawn by the same diagnostic renderer. Not browser-vs-CUDA pixel parity.'};
    },{source,preset});
    record.renders.push({source,preset,file:filename,...comparison});console.log(JSON.stringify(record.renders.at(-1)));
    if(comparison.visiblePixels<1000||comparison.negativeControlPixels<1000)throw Error('Empty/near-empty avatar');
    if(comparison.meanChannelError>record.imageLimits.meanChannelError||comparison.pixelsWithRgbL1Over30>record.imageLimits.pixelsWithRgbL1Over30)throw Error('Same-renderer reference comparison failed');
  }
  // Four UI signals must change rendered pixels, not only update a label.
  await page.selectOption('#source','1');await page.selectOption('#preset','front');await page.waitForTimeout(200);await wait();
  record.controlProbes=[];
  for(const[id,value]of [['yaw','.45'],['arms','.25'],['jaw','.4'],['grip','.65']]){
    await page.click('#reset');await page.waitForTimeout(150);await wait();
    const before=await page.evaluate(async()=>{window.__idol.probePixels=await window.__idol.viewer.pixels();return window.__idol.state.frames;});
    await page.locator('#'+id).evaluate((input,value)=>{input.value=value;input.dispatchEvent(new Event('input',{bubbles:true}));},value);
    await page.waitForFunction(n=>window.__idol.state.frames>n&&!window.__idol.busy,before);
    const pixels=await page.evaluate(async()=>{const a=window.__idol.probePixels,b=await window.__idol.viewer.pixels();let changed=0;for(let i=0;i<a.length;i+=4)if(Math.abs(a[i]-b[i])+Math.abs(a[i+1]-b[i+1])+Math.abs(a[i+2]-b[i+2])>30)changed++;return changed;});
    record.controlProbes.push({signal:id,value:Number(value),pixelsWithRgbL1Over30:pixels,pass:pixels>0});
  }
  await page.click('#reset');await page.waitForTimeout(150);await wait();
  const baseline=await page.evaluate(async()=>{window.__idol.beforePixels=await window.__idol.viewer.pixels();return window.__idol.state.frames;});
  for(const[id,value]of [['yaw','.45'],['arms','.25'],['jaw','.25'],['grip','.45']])await page.locator('#'+id).evaluate((input,value)=>{input.value=value;input.dispatchEvent(new Event('input',{bubbles:true}));},value);
  await page.waitForFunction(n=>window.__idol.state.frames>n&&!window.__idol.busy,baseline);await page.waitForTimeout(150);await wait();
  record.controls=await page.evaluate(async()=>{const app=window.__idol,a=app.beforePixels,b=await app.viewer.pixels();let changed=0;for(let i=0;i<a.length;i+=4)if(Math.abs(a[i]-b[i])+Math.abs(a[i+1]-b[i+1])+Math.abs(a[i+2]-b[i+2])>30)changed++;return{changedPixels:changed,params:Array.from(app.parameters),pass:changed>1000};});
  await page.screenshot({path:path.join(folder,prefix+'-desktop.png'),fullPage:true});
  await page.click('#reset');await page.waitForTimeout(150);await wait();
  await page.click('#play');await page.waitForTimeout(2000);
  const start=await page.evaluate(async()=>{const app=window.__idol;app.motionPixels=await app.viewer.pixels();app.state.samples=[];return{time:performance.now(),frames:app.state.frames};});
  record.liveWindows=[];let previous=start;
  for(let window=0;window<3;window++){
    await page.waitForTimeout(10000);
    const sample=await page.evaluate(()=>({time:performance.now(),frames:window.__idol.state.frames}));
    record.liveWindows.push({window:window+1,seconds:(sample.time-previous.time)/1000,frames:sample.frames-previous.frames,deliveredFps:(sample.frames-previous.frames)*1000/(sample.time-previous.time)});
    previous=sample;console.log(JSON.stringify(record.liveWindows.at(-1)));
  }
  const end=await page.evaluate(()=>({time:performance.now(),frames:window.__idol.state.frames,samples:window.__idol.state.samples,error:window.__idol.state.error}));
  await page.click('#play');await wait();
  const motionChange=await page.evaluate(async()=>{const app=window.__idol,a=app.motionPixels,b=await app.viewer.pixels();let changed=0;for(let i=0;i<a.length;i+=4)if(Math.abs(a[i]-b[i])+Math.abs(a[i+1]-b[i+1])+Math.abs(a[i+2]-b[i+2])>30)changed++;return changed;});
  const percentile=(key,p)=>end.samples.map(s=>s[key]).sort((a,b)=>a-b)[Math.floor((end.samples.length-1)*p)];
  record.live={seconds:(end.time-start.time)/1000,frames:end.frames-start.frames,deliveredFps:(end.frames-start.frames)*1000/(end.time-start.time),motionChangedPixels:motionChange,
    geometryP50Ms:percentile('geometryMs',.5),sortDrawP50Ms:percentile('sortUploadDrawMs',.5),endToEndP50Ms:percentile('endToEndMs',.5),endToEndP95Ms:percentile('endToEndMs',.95),samples:end.samples,error:end.error,
    note:renderer==='resident'?'Includes RAF pacing, procedural signals, FK, full GPU geometry, GPU depth sort and GPU-completed rendering; no geometry or pixel readback inside the timed animation windows. Three windows on one desktop workstation.':'Includes RAF pacing, procedural signals, FK, full GPU geometry, readback, CPU depth sort/upload and synchronous WebGL completion. Three windows on one desktop workstation.'};
  await page.locator('#avatar').screenshot({path:path.join(folder,prefix+'-live.png')});
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(folder,prefix+'-mobile.png'),fullPage:true});
  record.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  record.pass=record.controlProbes.every(c=>c.pass)&&record.controls.pass&&record.live.frames>10&&motionChange>1000&&!end.error&&!record.errors.length&&!record.mobileOverflow;
  console.log(JSON.stringify({pass:record.pass,live:{...record.live,samples:undefined},mobileOverflow:record.mobileOverflow}));
}catch(error){record.pass=false;record.errors.push(String(error));console.error(String(error));}
finally{fs.writeFileSync(destination,JSON.stringify(record,null,2));if(browser)await browser.close();await new Promise(resolve=>lab.server.close(resolve));console.log(destination);}
if(!record.pass)process.exitCode=1;
