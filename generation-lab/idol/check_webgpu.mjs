// Offline corrective-stage validation, with no main-app changes or inference.
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {chromium} from '../../web/node_modules/playwright-core/index.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const reports=path.resolve(here,'../reports/idol');
if(process.argv.length!==3)throw Error('Supply one completed portability run directory');
const folder=fs.realpathSync(path.resolve(process.argv[2]));
if(path.dirname(folder)!==reports || !/^portability-\d{8}-\d{6}-[0-9a-f]{6}$/.test(path.basename(folder)))throw Error('Run outside IDOL reports');
const read=name=>JSON.parse(fs.readFileSync(path.join(folder,name),'utf8'));
const run=read('run.json'),native=read('metrics.json'),validation=read('validation.json');
if(run.exitCode!==0 || native.status!=='native-correctives-passed' || !native.cases.length || native.cases.some(x=>!x.passed))throw Error('Native corrective validation must pass first');
if(execFileSync('C:/Program Files/Docker/Docker/resources/bin/docker.exe',['ps','--filter','label=research.task=idol','--format','{{.Names}}'],{encoding:'utf8',windowsHide:true}).trim())throw Error('Wait for native IDOL containers to exit before measuring');
const allowed=new Set(['correctives.json','correctives.bin','validation.json',...validation.cases.map(x=>x.file)]);
const source=fs.readFileSync(path.join(here,'corrective_gpu.mjs'));
const evidence={createdAt:new Date().toISOString(),scope:'Corrective fields only; NOT full deformation, animation or rendering FPS',
  nativeRun:run.id,packageSha256:read('correctives.json').sha256,
  moduleSha256:crypto.createHash('sha256').update(source).digest('hex'),
  harnessSha256:crypto.createHash('sha256').update(fs.readFileSync(fileURLToPath(import.meta.url))).digest('hex'),cases:[],errors:[]};
const destination=path.join(folder,'webgpu-'+evidence.createdAt.replace(/[:.]/g,'-')+'.json');
const server=http.createServer((req,res)=>{
  try{
    const key=decodeURIComponent(new URL(req.url,'http://localhost').pathname).slice(1);
    res.setHeader('Cache-Control','no-store');
    res.setHeader('Vary','Range');
    if(key===''){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>IDOL corrective validation</title><h1>Corrective-stage lab</h1><p>No full avatar renderer.</p>');return;}
    if(key==='corrective_gpu.mjs'){res.setHeader('Content-Type','text/javascript');res.end(source);return;}
    if(!allowed.has(key)){res.writeHead(404);res.end();return;}
    const target=path.resolve(folder,key);
    if(!target.startsWith(folder+path.sep))throw Error('Path escaped run');
    res.setHeader('Content-Type',key.endsWith('.json')?'application/json':'application/octet-stream');
    const size=fs.statSync(target).size;
    const range=/^bytes=(\d+)-(\d+)$/.exec(req.headers.range??'');
    if(range){
      const begin=Number(range[1]),end=Number(range[2]);
      if(begin>end||end>=size||end-begin+1>4*1024**2){res.writeHead(416);res.end();return;}
      const payload=Buffer.alloc(end-begin+1),fd=fs.openSync(target,'r');
      try{if(fs.readSync(fd,payload,0,payload.length,begin)!==payload.length)throw Error('Short read');}finally{fs.closeSync(fd);}
      res.writeHead(206,{'Content-Range':`bytes ${begin}-${end}/${size}`,'Content-Length':payload.length});res.end(payload);
    }else{const payload=fs.readFileSync(target);res.setHeader('Content-Length',payload.length);res.end(payload);}
  }catch{res.writeHead(400);res.end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
try{
  browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
    args:['--disable-background-timer-throttling','--disable-renderer-backgrounding']});
  const page=await browser.newPage();
  page.on('pageerror',e=>evidence.errors.push(String(e)));
  await page.exposeFunction('progress',text=>console.log(text));
  await page.goto('http://127.0.0.1:'+server.address().port);
  const result=await page.evaluate(async()=>{
    const {decodePackage,CorrectiveGpu}=await import('/corrective_gpu.mjs');
    const sha=async binary=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',binary)),x=>x.toString(16).padStart(2,'0')).join('');
    const get=async url=>{const r=await fetch(url);if(!r.ok)throw Error('HTTP '+r.status+' '+url);return r;};
    // Bounded range requests tolerate local forwarding layers with response-size
    // limits. Verify the entire reassembled payload before decoding or GPU use.
    const getBinary=async(url,size)=>{
      if(!Number.isSafeInteger(size)||size<=0||size>256*1024**2)throw Error('Invalid payload size');
      const result=new Uint8Array(size),chunk=4*1024**2;
      for(let start=0;start<size;start+=chunk){
        const end=Math.min(size,start+chunk)-1;
        const response=await fetch(url+'?rangeStart='+start,{headers:{Range:`bytes=${start}-${end}`}});
        const part=new Uint8Array(await response.arrayBuffer());
        if(response.status!==206||part.length!==end-start+1||response.headers.get('content-range')!==`bytes ${start}-${end}/${size}`)
          throw Error('Invalid binary range: '+JSON.stringify({url,start,end,status:response.status,bytes:part.length}));
        result.set(part,start);
      }return result.buffer;
    };
    const start=performance.now();
    const meta=await (await get('/correctives.json')).json();
    const binary=await getBinary('/correctives.bin',meta.bytes);
    const actualHash=await sha(binary);
    if(actualHash!==meta.sha256)throw Error('Package hash mismatch: '+JSON.stringify({actualHash,expected:meta.sha256,bytes:binary.byteLength,expectedBytes:meta.bytes}));
    const downloaded=performance.now();
    const asset=decodePackage(meta,binary);const decoded=performance.now();
    const gpu=await CorrectiveGpu.create(asset);const uploaded=performance.now();
    const result={adapter:gpu.info,packageBytes:binary.byteLength,initializationMs:{downloadAndHash:downloaded-start,decodeAndValidate:decoded-downloaded,compileAndUpload:uploaded-decoded},
      cases:[],negativeChecks:[],benchmark:null};
    try{
      const validation=await (await get('/validation.json')).json();
      for(const invalid of [new Float32Array(505),new Float32Array(506).fill(NaN),new Float64Array(506)]){
        let rejected=false;try{await gpu.evaluate(invalid);}catch{rejected=true;}
        if(!rejected)throw Error('Invalid controls accepted');result.negativeChecks.push('invalid-controls-rejected');
      }
      const pending=gpu.evaluate(new Float32Array(506));
      let overlapRejected=false;try{await gpu.evaluate(new Float32Array(506));}catch{overlapRejected=true;}
      if(!overlapRejected)throw Error('Overlapping evaluation accepted');await pending;result.negativeChecks.push('concurrent-evaluation-rejected');
      for(const c of validation.cases){
        const bytes=await getBinary('/'+c.file,c.bytes);
        if(bytes.byteLength!==c.bytes || await sha(bytes)!==c.sha256)throw Error('Reference hash mismatch: '+c.name);
        const reference=new Float32Array(bytes),coefficients=new Float32Array(c.coefficients);
        const before=performance.now(),actual=await gpu.evaluate(coefficients),ms=performance.now()-before;
        if(actual.length!==reference.length)throw Error('Reference shape mismatch');
        let shapeMaxAbs=0,poseMaxAbs=0,square=0,count=0;
        for(let i=0;i<actual.length;i++){
          if(!Number.isFinite(actual[i])||!Number.isFinite(reference[i]))throw Error('Nonfinite field');
          if(i%4===3){if(actual[i]!==0||reference[i]!==0)throw Error('Invalid field padding');continue;}
          const delta=Math.abs(actual[i]-reference[i]);square+=delta*delta;count++;
          if(i%8<3)shapeMaxAbs=Math.max(shapeMaxAbs,delta);else poseMaxAbs=Math.max(poseMaxAbs,delta);
        }
        const row={name:c.name,shapeMaxAbs,poseMaxAbs,rms:Math.sqrt(square/count),dispatchReadbackCopyMs:ms,
          testedScalars:count,passed:Math.max(shapeMaxAbs,poseMaxAbs)<=validation.limits.correctiveMaxAbs};
        result.cases.push(row);
        if(result.cases.length%10===0)await window.progress('WEBGPU '+result.cases.length+'/'+validation.cases.length+' '+JSON.stringify(row));
        if(!row.passed)throw Error('GPU corrective parity failed: '+c.name);
      }
      const controls=validation.cases.slice(-8).map(c=>new Float32Array(c.coefficients));
      for(let i=0;i<30;i++)await gpu.evaluate(controls[i%controls.length]);
      const samples=[],windows=[];
      for(let window=0;window<3;window++){
        const first=samples.length,begin=performance.now();
        for(let i=0;i<300;i++){const t=performance.now();await gpu.evaluate(controls[i%controls.length]);samples.push(performance.now()-t);}
        const totalMs=performance.now()-begin,sorted=samples.slice(first).sort((a,b)=>a-b);
        windows.push({iterations:300,totalMs,stageHz:300000/totalMs,p50Ms:sorted[149],p95Ms:sorted[284]});
      }
      const total=windows.reduce((s,w)=>s+w.totalMs,0),sorted=[...samples].sort((a,b)=>a-b),at=q=>sorted[Math.floor(q*(sorted.length-1))];
      result.benchmark={scope:'Correction stage only, including coefficient upload, 3 dispatches, 6.43 MB readback and CPU copy. No FK, skinning, KNN, covariance or rendering.',
        warmupIterations:30,iterations:samples.length,totalMs:total,stageHz:samples.length*1000/total,p50Ms:at(.5),p95Ms:at(.95),minMs:sorted[0],maxMs:sorted.at(-1),windows,samples};
      gpu.destroy();let destroyedRejected=false;try{await gpu.evaluate(controls[0]);}catch{destroyedRejected=true;}
      if(!destroyedRejected)throw Error('Destroyed device accepted');result.negativeChecks.push('destroyed-instance-rejected');
      result.pass=true;return result;
    }finally{gpu.destroy();}
  });
  Object.assign(evidence,result);evidence.pass=result.pass&&!evidence.errors.length;
  console.log(JSON.stringify({pass:evidence.pass,cases:evidence.cases.length,adapter:evidence.adapter,benchmark:{...evidence.benchmark,samples:undefined}}));
}catch(error){evidence.pass=false;evidence.errors.push(String(error));process.exitCode=1;}
finally{
  fs.writeFileSync(destination,JSON.stringify(evidence,null,2),{flag:'wx'});
  if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));console.log(destination);
}
