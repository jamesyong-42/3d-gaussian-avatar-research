import {decodeAsset,deform,checkReferences,attachValidation,compareFrame,identityPose,axisQuat} from './engine.mjs';
import {GpuDeformer} from './gpu-deform.mjs';
let asset,gpu,backend='cpu',reason='';
function status(){self.postMessage({type:'backend',backend,reason,adapter:gpu?.info});}
let queue=Promise.resolve();
self.onmessage=({data})=>{queue=queue.then(()=>handle(data)).catch(error=>self.postMessage({type:'error',message:String(error)}));};
async function handle(data){
  try {
    if(data.type==='load'){
      asset=decodeAsset(data.meta,data.buffer);
      if(data.backend!=='cpu'&&(data.backend==='webgpu'||asset.meta.nGaussians>=100000)){
        try{
          gpu=await GpuDeformer.create(asset);
          // Startup parity check needs no separately downloaded native fixtures.
          const p=identityPose(asset.meta.nBones,asset.meta.nExpressions);p.angles[50]=-.7;p.angles[4]=.4;p.expression[asset.meta.nExpressions-1]=.5;
          const cpu=deform(asset,p),out=await gpu.deform(p),checked=compareFrame(out,cpu.positions,cpu.rotations,asset.meta.nGaussians);
          if(!checked.pass)throw new Error('GPU startup parity failed: '+JSON.stringify(checked));
          backend='webgpu';
        }catch(error){reason=String(error);gpu?.destroy();gpu=undefined;}
      }
      status();self.postMessage({type:'loaded'});
    }
    else if(data.type==='pose') {
      const start=performance.now();let out;
      try{out=gpu?await gpu.deform(data.pose):deform(asset,data.pose);}
      catch(error){reason=String(error);gpu?.destroy();gpu=undefined;backend='cpu';status();out=deform(asset,data.pose);}
      self.postMessage({type:'frame',id:data.id,...out,backend,ms:performance.now()-start},[out.positions.buffer,out.rotations.buffer,out.joints.buffer]);
    }
    else if(data.type==='verify'){
      if(data.buffer)attachValidation(asset,data.buffer);
      const results=checkReferences(asset);
      if(gpu)for(let i=0;i<asset.meta.referencePoses.length;i++){
        const f=asset.meta.referencePoses[i],out=await gpu.deform(f);
        const check=compareFrame(out,asset.arrays[`reference_${f.name}_positions`],asset.arrays[`reference_${f.name}_rotations`],asset.meta.nGaussians);
        results[i]={...results[i],gpu:check,pass:results[i].pass&&check.pass};
      }
      const stress=[];
      if(gpu)for(let c=0;c<8;c++){
        const p=identityPose(asset.meta.nBones,asset.meta.nExpressions);
        for(let j=0;j<p.angles.length;j++)p.angles[j]=Math.sin((j+1)*(c+3)*1.618)*.55;
        for(let e=0;e<p.expression.length;e++)p.expression[e]=Math.cos((e+1)*(c+1))*.35;
        p.rootPosition=[.1*c,.03,-.1];p.rootRotation=axisQuat(.15,.12*c,-.08);
        const cpu=deform(asset,p),out=await gpu.deform(p);
        stress.push({name:`all-expressions-actor-root-${c}`,...compareFrame(out,cpu.positions,cpu.rotations,asset.meta.nGaussians)});
      }
      if(gpu&&([...results,...stress].some(check=>!check.pass))){reason='GPU parity verification failed; CPU reference restored';gpu.destroy();gpu=undefined;backend='cpu';status();}
      self.postMessage({type:'verified',results,stress});
    }
    else if(data.type==='simulate-device-loss')gpu?.device.destroy();
  }catch(error){self.postMessage({type:'error',message:String(error)});}
}
