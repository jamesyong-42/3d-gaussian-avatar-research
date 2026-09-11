import {parseMotionFbx, bakeMixamo, disposeMotionScene} from './mixamo.mjs';

self.onmessage=async({data})=>{
  let source,reference;
  try {
    source=parseMotionFbx(data.buffer,data.name);
    if(data.reference)reference=parseMotionFbx(data.reference.buffer,data.reference.name);
    const motions=await bakeMixamo(data.asset,source,{reference,
      onProgress:progress=>self.postMessage({type:'progress',progress}),
      yieldTask:()=>new Promise(resolve=>setTimeout(resolve,0)),
    });
    self.postMessage({type:'complete',motions},motions.flatMap(m=>[m.rotations.buffer,m.translations.buffer]));
  } catch(error) {self.postMessage({type:'error',message:error.message??String(error)});}
  finally {if(source)disposeMotionScene(source.scene);if(reference)disposeMotionScene(reference.scene);}
};
