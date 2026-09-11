import {getJson,getBinary,sha256} from './lab_io.mjs';
import {CorrectiveGpu,decodePackage} from './corrective_gpu.mjs';
import {compileRig,decodeGeometry} from './deformation_cpu.mjs';
import {DeformationGpu} from './deformation_gpu.mjs';
import {installNeighbors} from './neighbors_gpu.mjs';

export async function checkSmallNeighbors(device){
  const fixtures=[
    Array.from({length:4},()=>[0,0,0]),
    Array.from({length:65},(_,i)=>[i*.01,0,0]),
    Array.from({length:257},(_,i)=>[Math.sin(i*123.4)*.7,Math.cos(i*37.2)*.5,Math.sin(i*11.1)*.9]),
    Array.from({length:513},(_,i)=>[i<256?10:-10,(i%17)*1e-4,(i%31)*1e-4])];
  const results=[];
  for(const points of fixtures){
    const gpu={device,buffers:[],buffer:CorrectiveGpu.prototype.buffer},n=points.length;
    const pose=new Float32Array(n*16),radii=new Float32Array(n*4).fill(1);
    for(let i=0;i<n;i++){pose.set(points[i],i*16);pose[i*16+4]=pose[i*16+9]=pose[i*16+14]=1;}
    const p=gpu.buffer(pose,GPUBufferUsage.STORAGE),r=gpu.buffer(radii,GPUBufferUsage.STORAGE);
    try{
      const nearest=await installNeighbors(gpu,p,r,n),readback=gpu.buffer(n*48,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
      const encoder=device.createCommandEncoder();nearest.encode(encoder);encoder.copyBufferToBuffer(nearest.output,0,readback,0,n*48);
      device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
      const actual=new Float32Array(readback.getMappedRange().slice(0));readback.unmap();
      let maxRelative=0;
      for(let i=0;i<n;i++){
        const distances=[];for(let j=0;j<n;j++)if(i!==j){let d=0;for(let k=0;k<3;k++)d+=(pose[i*16+k]-pose[j*16+k])**2;distances.push(d);}
        distances.sort((a,b)=>a-b);const expected=(distances[0]+distances[1]+distances[2])/3;
        if(!Number.isFinite(actual[i*12+3]))throw Error('Nonfinite small 3NN result');
        maxRelative=Math.max(maxRelative,Math.abs(actual[i*12+3]-expected)/Math.max(expected,1e-7));
      }
      results.push({points:n,maxRelative,pass:maxRelative<=2e-6});
      if(!results.at(-1).pass)throw Error('Exact 3NN fixture failed '+JSON.stringify(results.at(-1)));
    }finally{for(const buffer of gpu.buffers)buffer.destroy();}
  }return results;
}

export async function loadDeformation(){
  const validation=await getJson('/validation.json'),rigResponse=await fetch('/rig.json');
  if(rigResponse.status!==200)throw Error('Rig HTTP failure');
  const rigBytes=await rigResponse.arrayBuffer();if(await sha256(rigBytes)!==validation.rigSha256)throw Error('Rig hash mismatch');
  const rigRaw=JSON.parse(new TextDecoder().decode(rigBytes)),meta=await getJson('/correctives.json');
  const correctives=decodePackage(meta,await getBinary('/'+meta.binary,meta.bytes,meta.sha256));
  const sourceCount=Math.max(...validation.cases.map(c=>c.source))+1,assets=[];
  for(let i=0;i<sourceCount;i++){const asset=await getJson('/asset-'+i+'.json');assets.push(decodeGeometry(asset,await getBinary('/'+asset.binary,asset.bytes,asset.sha256)));}
  const rig=compileRig(rigRaw),gpu=await DeformationGpu.create(correctives,rig,assets);
  return{gpu,validation,rig,assets,correctiveBytes:meta.bytes};
}
export async function runChecks(progress=()=>{}){
  const started=performance.now(),lab=await loadDeformation(),loaded=performance.now(),{gpu,validation,assets}=lab;
  const result={initializationMs:loaded-started,adapter:gpu.info,correctiveBytes:lab.correctiveBytes,assetBytes:assets.map(a=>a.meta.bytes),cases:[],smallNeighborTests:[],benchmark:null};
  try{
    result.smallNeighborTests=await checkSmallNeighbors(gpu.gpu.device);await progress({smallNeighborTests:result.smallNeighborTests});
    for(const c of validation.cases){
      const reference=new Float32Array(await getBinary('/'+c.file,c.bytes,c.sha256));
      const actual=await gpu.evaluate(new Float32Array(c.params),c.source),g=actual.geometry;
      let centerMaxDistance=0,covarianceMaxRelative=0,covarianceMaxAbsolute=0,neighborMeanMaxRelative=0,jointTransformMaxAbs=0,poseFeatureMaxAbs=0;
      for(let i=0;i<actual.pose.transforms.length;i++)jointTransformMaxAbs=Math.max(jointTransformMaxAbs,Math.abs(actual.pose.transforms[i]-c.jointTransforms[i]));
      for(let i=0;i<506;i++)poseFeatureMaxAbs=Math.max(poseFeatureMaxAbs,Math.abs(actual.pose.controls[i]-c.controls[i]));
      if(g.length!==validation.nGaussians*12||reference.length!==validation.nGaussians*20)throw Error('Geometry reference size mismatch');
      for(let i=0;i<validation.nGaussians;i++){
        const a=i*12,b=i*20;let d2=0,dCov=0,refCov=0;
        for(let k=0;k<10;k++)if(!Number.isFinite(g[a+k])||!Number.isFinite(reference[b+k]))throw Error('Nonfinite geometry');
        for(let k=0;k<3;k++)d2+=(g[a+k]-reference[b+k])**2;centerMaxDistance=Math.max(centerMaxDistance,Math.sqrt(d2));
        const actualMean=Math.max(g[a+3],1e-7),refMean=Math.max(reference[b+3],1e-7);
        neighborMeanMaxRelative=Math.max(neighborMeanMaxRelative,Math.abs(actualMean-refMean)/refMean);
        for(let k=4;k<10;k++){const delta=g[a+k]-reference[b+k];dCov+=delta*delta;refCov+=reference[b+k]**2;covarianceMaxAbsolute=Math.max(covarianceMaxAbsolute,Math.abs(delta));}
        covarianceMaxRelative=Math.max(covarianceMaxRelative,Math.sqrt(dCov)/Math.max(Math.sqrt(refCov),1e-12));
      }
      const row={name:c.name,source:c.source,jointTransformMaxAbs,poseFeatureMaxAbs,centerMaxDistance,covarianceMaxRelative,covarianceMaxAbsolute,neighborMeanMaxRelative,fkMs:actual.fkMs,totalMs:actual.totalMs};
      row.pass=poseFeatureMaxAbs<=validation.limits.correctiveMaxAbs&&['jointTransformMaxAbs','centerMaxDistance','covarianceMaxRelative','neighborMeanMaxRelative'].every(k=>row[k]<=validation.limits[k]);
      result.cases.push(row);await progress({case:row});
      // Keep all finite failures for diagnosis, do not raise tolerances after observing them.
    }
    const controls=validation.cases.slice(-8);for(let i=0;i<30;i++)await gpu.evaluate(new Float32Array(controls[i%8].params),controls[i%8].source);
    const samples=[],windows=[];
    for(let window=0;window<3;window++){
      const local=[];
      for(let i=0;i<300;i++){const c=controls[(window*300+i)%8],r=await gpu.evaluate(new Float32Array(c.params),c.source);local.push(r.totalMs);samples.push(r.totalMs);}
      const ordered=[...local].sort((a,b)=>a-b);
      const timing={window:window+1,iterations:300,p50Ms:ordered[149],p95Ms:ordered[284],totalMs:local.reduce((a,b)=>a+b,0)};
      windows.push(timing);await progress({benchmarkWindow:timing});
    }
    const sorted=[...samples].sort((a,b)=>a-b);
    result.benchmark={scope:'CPU FK + GPU correctives, full-influence skinning, fresh exact 3NN, covariance and readback. No renderer.',warmups:30,iterations:900,p50Ms:sorted[449],p95Ms:sorted[854],totalMs:samples.reduce((a,b)=>a+b,0),windows,samples};
    const reject=async(action)=>{try{await action();return false;}catch{return true;}};
    const c=controls[0],p=new Float32Array(c.params);
    result.boundaryChecks={invalidSource:await reject(()=>gpu.evaluate(p,-1)),nonfinite:await reject(()=>gpu.evaluate(new Float32Array(189).fill(NaN),c.source))};
    const inflight=gpu.evaluate(p,c.source);result.boundaryChecks.overlap=await reject(()=>gpu.evaluate(p,c.source));await inflight;
    gpu.destroy();gpu.destroy();result.boundaryChecks.destroyed=await reject(()=>gpu.evaluate(p,c.source));
    result.pass=result.cases.length===validation.cases.length&&result.cases.every(c=>c.pass)&&Object.values(result.boundaryChecks).every(Boolean);return result;
  }finally{gpu.destroy();}
}
