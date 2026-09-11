// Capture only completed, independently validated trials. No inference runs here.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {chromium} from '../../web/node_modules/playwright-core/index.mjs';

const lab=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const folders=process.argv.slice(2).map(p=>path.resolve(p));
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const trials=folders.map(folder=>({folder,run:read(path.join(folder,'run.json')),
  metrics:read(path.join(folder,'metrics.json')),published:read(path.join(folder,'published.json'))})).sort((a,b)=>a.metrics.views-b.metrics.views);
if(trials.map(t=>t.metrics.views).join(',')!=='1,4,8')throw new Error('Supply exactly the completed 1, 4, and 8 view trial directories.');
if(trials.some(t=>t.run.exitCode!==0||!t.run.finishedAt||!t.metrics.checkpointCoverageVerified||!t.published.validation.pass))throw new Error('An input has not passed generation and independent validation.');
const running=execFileSync('C:/Program Files/Docker/Docker/resources/bin/docker.exe',
  ['ps','--filter','label=research.task=lhmpp','--format','{{.Names}}'],{encoding:'utf8',windowsHide:true}).trim();
if(running)throw new Error('Stop measuring until experiment containers finish: '+running);
const id='comparison-'+new Date().toISOString().replace(/[:.]/g,'-');
const destination=path.join(lab,'reports/lhmpp',id);
fs.mkdirSync(destination);
const base=process.env.AVATAR_LAB_URL??'http://127.0.0.1:8765';
const evidence={id,createdAt:new Date().toISOString(),viewport:[1440,1000],camera:{distance:4.1,targetY:-.2},
  measurement:'Separate 5.5-second Samba playback windows after 2 seconds warm-up; displayed render FPS is the final reporting interval. No generation concurrent.',
  visibility:'Composited screenshot minus avatar-hidden control; central ROI; RGB L1 difference > 30; > 1000 pixels required.',results:[],errors:[]};
const save=()=>fs.writeFileSync(path.join(destination,'browser-results.json'),JSON.stringify(evidence,null,2));
const percentile=(values,f)=>[...values].sort((a,b)=>a-b)[Math.floor((values.length-1)*f)]??null;
const browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
  args:['--disable-background-timer-throttling','--disable-renderer-backgrounding']});
try{
  for(const trial of trials){
    const {assetId}=trial.published;
    console.log('CAPTURE '+assetId);
    const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
    page.on('pageerror',error=>evidence.errors.push(String(error)));
    const health=await (await page.request.get(base+'/api/health')).json();
    if(health.workerRunning)throw new Error('HTTP generation is active; do not measure contention.');
    const loadedAt=Date.now();
    await page.goto(base+'/?asset='+encodeURIComponent(assetId));
    await page.waitForFunction(id=>window.__lab?.state.assetId===id&&window.__lab.state.renderedFrames>15,assetId,{timeout:120000});
    const loadSeconds=(Date.now()-loadedAt)/1000, shots={};
    for(const [name,angle] of [['front',0],['side',Math.PI/2],['back',Math.PI]]){
      await page.evaluate(a=>{window.__lab.setClip('none');window.__lab.setView(a);},angle);
      await page.waitForTimeout(750);
      shots[name]=`${trial.metrics.views}v-${name}.png`;
      await page.locator('#viewport').screenshot({path:path.join(destination,shots[name])});
    }
    await page.evaluate(()=>{window.__lab.setView(0);window.__lab.verify();});
    await page.waitForFunction(()=>window.__lab.state.verification.length===7,{},{timeout:90000});
    await page.locator('[data-mode="motion"]').click();
    await page.locator('#motion-sample').click();
    await page.waitForFunction(()=>!window.__lab.state.importingMotion&&window.__lab.state.motions.length>0,{},{timeout:90000});
    await page.locator('#motion-seek').evaluate(input=>{input.value='4.2';input.dispatchEvent(new Event('input',{bubbles:true}));});
    await page.waitForTimeout(750);
    shots.dance=`${trial.metrics.views}v-dance.png`;
    await page.locator('#viewport').screenshot({path:path.join(destination,shots.dance)});
    await page.locator('#play').click();
    await page.waitForTimeout(2000);
    const start=await page.evaluate(()=>{window.__lab.resetDeformSamples();return {time:performance.now(),frames:window.__lab.state.renderedFrames};});
    await page.waitForTimeout(5500);
    const end=await page.evaluate(()=>({time:performance.now(),state:window.__lab.state,generation:window.__lab.generation}));
    // Freeze pose before visibility evidence, so the negative control differs only
    // in avatar visibility (not timeline overlays or animation).
    await page.locator('#play').click();
    await page.waitForTimeout(750);
    const motionEvidence=`${trial.metrics.views}v-live.png`,backgroundEvidence=`${trial.metrics.views}v-background.png`;
    const rendered=await page.locator('#viewport').screenshot({path:path.join(destination,motionEvidence)});
    let background;
    try{
      await page.evaluate(()=>window.__lab.setAvatarVisible(false));
      await page.waitForTimeout(400);
      background=await page.locator('#viewport').screenshot({path:path.join(destination,backgroundEvidence)});
    }finally{await page.evaluate(()=>window.__lab.setAvatarVisible(true));}
    const visiblePixels=await page.evaluate(async images=>{
      const decoded=await Promise.all(images.map(async data=>{
        const bitmap=await createImageBitmap(await (await fetch('data:image/png;base64,'+data)).blob());
        const canvas=new OffscreenCanvas(bitmap.width,bitmap.height),ctx=canvas.getContext('2d');
        ctx.drawImage(bitmap,0,0);const rgba=ctx.getImageData(0,0,bitmap.width,bitmap.height);bitmap.close();return rgba;
      }));
      const [a,b]=decoded;if(a.width!==b.width||a.height!==b.height)throw new Error('Screenshot dimensions differ.');
      let count=0;for(let y=Math.floor(a.height*.15);y<a.height*.95;y++)for(let x=Math.floor(a.width*.1);x<a.width*.9;x++){
        const i=(y*a.width+x)*4;
        if(Math.abs(a.data[i]-b.data[i])+Math.abs(a.data[i+1]-b.data[i+1])+Math.abs(a.data[i+2]-b.data[i+2])>30)count++;
      }return count;
    },[rendered.toString('base64'),background.toString('base64')]);
    const row={...trial,assetId,loadSeconds,shots,visiblePixels,motionEvidence,backgroundEvidence,
      model:end.generation.model,renderFps:end.state.renderFps,renderP95Ms:end.state.renderP95Ms,
      poseHzSnapshot:end.state.updateHz,poseHz:(end.state.renderedFrames-start.frames)/((end.time-start.time)/1000),
      measuredSeconds:(end.time-start.time)/1000,deformP50Ms:percentile(end.generation.deformSamples,.5),
      deformP95Ms:percentile(end.generation.deformSamples,.95),samples:end.generation.deformSamples,
      references:end.state.verification,motions:end.state.motions};
    row.pass=visiblePixels>1000&&row.references.length===7&&row.references.every(c=>c.pass)&&row.motions.some(m=>m.mapped===52);
    evidence.results.push(row);save();
    console.log(JSON.stringify({views:row.metrics.views,pass:row.pass,renderFps:row.renderFps,poseHz:row.poseHz,deformP95Ms:row.deformP95Ms}));
    await page.close();
    if(!row.pass)throw new Error('Browser validation failed: '+assetId);
  }
  evidence.pass=evidence.results.length===3&&evidence.results.every(r=>r.pass)&&!evidence.errors.length;
}catch(error){evidence.errors.push(String(error));evidence.pass=false;throw error;}
finally{save();await browser.close();console.log(path.join(destination,'browser-results.json'));}
if(!evidence.pass)process.exitCode=1;
