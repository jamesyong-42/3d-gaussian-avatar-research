// Native IDOL parameter contract; independent of LHM's pose conventions.
const kind='idol-native-deformation-v1-lab',F=Math.fround;
function finiteArray(value,length,label){
  const flat=value?.flat?.(Infinity);
  if(!flat||flat.length!==length||flat.some(x=>typeof x!=='number'||!Number.isFinite(x)))throw Error('Invalid '+label);
  return new Float32Array(flat);
}
export function compileRig(rig){
  if(rig.deformation!==kind||rig.nBones!==55||rig.nExpressions!==10||rig.nShape!==10)throw Error('Invalid IDOL rig schema');
  if(!Array.isArray(rig.parents)||rig.parents.length!==55||rig.parents[0]!==-1||rig.parents.slice(1).some((p,i)=>!Number.isInteger(p)||p<0||p>i))throw Error('Invalid joint hierarchy');
  return {parents:new Int32Array(rig.parents),mean:finiteArray(rig.poseMean,165,'pose mean'),
    base:finiteArray(rig.jointBase,165,'joint base'),directions:finiteArray(rig.jointDirections,3300,'joint directions')};
}
export function validateParameters(p){
  if(!(p instanceof Float32Array)||p.length!==189||p.some(x=>!Number.isFinite(x)))throw Error('Expected 189 finite float32 native parameters');
  if(p[0]!==1)throw Error('Native scale is ignored upstream; this contract requires scale=1');
}
export function parameterOffset(j){
  if(!Number.isInteger(j)||j<0||j>=55)throw Error('Invalid joint index');
  return j===0?4:j<22?7+(j-1)*3:j<25?170+(j-22)*3:80+(j-25)*3;
}
export function rodrigues(x,y,z){
  // Match the native epsilon-in-angle convention, including zero rotations.
  const angle=Math.hypot(F(x+1e-8),F(y+1e-8),F(z+1e-8));
  const axis=[x/angle,y/angle,z/angle],s=Math.sin(angle),c=1-Math.cos(angle);
  const K=[0,-axis[2],axis[1],axis[2],0,-axis[0],-axis[1],axis[0],0],r=new Float32Array(9);
  for(let i=0;i<3;i++)for(let j=0;j<3;j++){
    let k2=0;for(let k=0;k<3;k++)k2+=K[i*3+k]*K[k*3+j];
    r[i*3+j]=F((i===j?1:0)+s*K[i*3+j]+c*k2);
  }return r;
}
export function resolveNativePose(rig,params){
  validateParameters(params);
  const controls=new Float32Array(506),transforms=new Float32Array(55*16),world=new Float32Array(55*12),joints=new Float32Array(165);
  controls.set(params.subarray(70,80),486);controls.set(params.subarray(179,189),496);
  for(let c=0;c<165;c++){
    let value=rig.base[c];for(let k=0;k<20;k++)value+=rig.directions[c*20+k]*controls[486+k];joints[c]=value;
  }
  for(let joint=0;joint<55;joint++){
    const offset=parameterOffset(joint),j=joint*3;
    const r=rodrigues(F(params[offset]+rig.mean[j]),F(params[offset+1]+rig.mean[j+1]),F(params[offset+2]+rig.mean[j+2]));
    if(joint)for(let k=0;k<9;k++)controls[(joint-1)*9+k]=F(r[k]-(k%4===0?1:0));
    const parent=rig.parents[joint],w=joint*12;
    if(parent<0){
      for(let row=0;row<3;row++){world.set(r.subarray(row*3,row*3+3),w+row*4);world[w+row*4+3]=joints[j+row];}
    }else{
      const p=parent*12,d=[0,1,2].map(k=>F(joints[j+k]-joints[parent*3+k]));
      for(let row=0;row<3;row++){
        for(let col=0;col<3;col++)world[w+row*4+col]=world[p+row*4]*r[col]+world[p+row*4+1]*r[3+col]+world[p+row*4+2]*r[6+col];
        world[w+row*4+3]=world[p+row*4]*d[0]+world[p+row*4+1]*d[1]+world[p+row*4+2]*d[2]+world[p+row*4+3];
      }
    }
    for(let row=0;row<3;row++){
      const k=w+row*4,out=joint*16+row*4;transforms.set(world.subarray(k,k+3),out);
      const rest=F(world[k]*joints[j]+world[k+1]*joints[j+1]+world[k+2]*joints[j+2]);
      transforms[out+3]=F(F(world[k+3]-rest)+params[1+row]);
    }
    transforms[joint*16+15]=1;
  }
  return {controls,transforms,joints};
}

export function decodeGeometry(meta,binary){
  const n=meta.nGaussians,b=meta.nInfluences;
  if(meta.deformation!==kind||meta.nBones!==55||!Number.isInteger(n)||n<4||n>1000000||!Number.isInteger(b)||b<n||b>n*55||!(binary instanceof ArrayBuffer))throw Error('Invalid geometry contract');
  const shapes={restPositions:['<f4',[n,3]],restOrientations:['<f4',[n,3,3]],radiusMultipliers:['<f4',[n,3]],
    colors:['<f4',[n,3]],opacities:['<f4',[n,1]],skinOffsets:['<u4',[n+1]],skinBones:['<u4',[b]],skinWeights:['<f4',[b]]};
  if(Object.keys(meta.arrays??{}).sort().join()!==Object.keys(shapes).sort().join())throw Error('Invalid geometry array set');
  let offset=0;const arrays={};
  for(const[name,[dtype,shape]]of Object.entries(shapes)){
    const spec=meta.arrays[name],bytes=shape.reduce((s,x)=>s*x,4);
    if(spec.offset!==offset||spec.dtype!==dtype||spec.bytes!==bytes||JSON.stringify(shape)!==JSON.stringify(spec.shape)||offset+bytes>binary.byteLength)throw Error('Invalid geometry layout: '+name);
    arrays[name]=new(dtype==='<f4'?Float32Array:Uint32Array)(binary,offset,bytes/4);offset+=bytes;
    if(dtype==='<f4'&&arrays[name].some(x=>!Number.isFinite(x)))throw Error('Nonfinite geometry');
  }
  if(offset!==binary.byteLength||offset!==meta.bytes)throw Error('Invalid geometry size');
  const a=arrays;
  if(a.skinOffsets[0]!==0||a.skinOffsets[n]!==b)throw Error('Invalid skin endpoints');
  for(let i=0;i<n;i++){
    const begin=a.skinOffsets[i],end=a.skinOffsets[i+1];if(end<=begin||end>b||end-begin>55)throw Error('Invalid skin range');
    let sum=0;
    for(let k=begin;k<end;k++){
      if(a.skinBones[k]>=55||(k>begin&&a.skinBones[k]<=a.skinBones[k-1])||a.skinWeights[k]<=0)throw Error('Invalid skin influence');sum+=a.skinWeights[k];
    }
    if(Math.abs(sum-1)>1e-5)throw Error('Invalid skin weight sum');
  }
  // The native saturated sigmoid intentionally allows RGB in [-.001, 1.001].
  // Preserve these values here; display clamping belongs to the renderer.
  if(a.radiusMultipliers.some(x=>x<0)||a.colors.some(x=>x<-.00101||x>1.00101)||a.opacities.some(x=>x<0||x>1))throw Error('Invalid appearance/radius');
  validateParameters(new Float32Array(meta.defaultParameters));
  return{meta,arrays};
}
