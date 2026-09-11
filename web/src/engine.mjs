// Portable, renderer-independent implementation of the exported LHM deformation.
// Matrices here are row-major; joint rotations are local axis-angle radians.
export function decodeArrays(specs, buffer) {
  const constructors = {'<f4': Float32Array, '<i4': Int32Array, '<u4': Uint32Array, '<u2': Uint16Array, '|u1': Uint8Array};
  const arrays = {};
  for (const [name, spec] of Object.entries(specs)) {
    const Ctor = constructors[spec.dtype];
    const count=spec.shape?.reduce((a,b)=>Number.isSafeInteger(b)&&b>=0?a*b:NaN,1);
    if (!Ctor || !Number.isSafeInteger(count) || !Number.isSafeInteger(spec.offset) || spec.offset < 0 || spec.offset%4 || spec.bytes!==count*Ctor.BYTES_PER_ELEMENT || spec.offset + spec.bytes > buffer.byteLength) throw new Error(`Invalid array ${name}`);
    arrays[name] = new Ctor(buffer, spec.offset, count);
  }
  return arrays;
}

export function decodeAsset(meta, buffer) {
  if (![1,2].includes(meta.version) || meta.deformation !== 'lhm-linear-blend-v1') throw new Error('Unsupported avatar package');
  const arrays=decodeArrays(meta.arrays,buffer);
  if(meta.version===2){
    if(meta.expressionStorage!=='sparse-vertices-v1'||!arrays.expressionIndices||meta.nExpressionVertices!==arrays.expressionIndices.length||arrays.expressionDirections.length!==meta.nExpressions*meta.nExpressionVertices*3)throw new Error('Invalid sparse expression layout');
    const lookup=new Int32Array(meta.nGaussians).fill(-1);
    arrays.expressionIndices.forEach((index,row)=>{if(index>=meta.nGaussians||(row&&index<=arrays.expressionIndices[row-1]))throw new Error('Invalid sparse expression indices');lookup[index]=row;});
    arrays.expressionLookup=lookup;
  }
  return {meta, arrays};
}

export function attachValidation(asset, buffer) {
  if(!asset.meta.validation||buffer.byteLength!==asset.meta.validation.bytes)throw new Error('Invalid validation bundle');
  const arrays=decodeArrays(asset.meta.validation.arrays,buffer);
  for(const name of Object.keys(arrays))if(!/^reference_\w+_(positions|rotations)$/.test(name))throw new Error('Unexpected validation array');
  Object.assign(asset.arrays,arrays);
}

export function identityPose(bones = 55, expressions = 100) {
  return {angles: new Float32Array(bones * 3), expression: new Float32Array(expressions), rootPosition: [0, 0, 0], rootRotation: [0, 0, 0, 1], tick: 0};
}
export function copyPose(p) {
  return {...p, angles: Float32Array.from(p.angles), expression: Float32Array.from(p.expression), rootPosition: [...p.rootPosition], rootRotation: [...p.rootRotation]};
}
export function axisQuat(x, y, z) {
  const angle = Math.hypot(x, y, z), s = angle < 1e-8 ? 0.5 : Math.sin(angle / 2) / angle;
  return [x * s, y * s, z * s, Math.cos(angle / 2)];
}
export function quatAxis(q) {
  if (q[3] < 0) q = q.map(x => -x);
  const s = Math.hypot(q[0], q[1], q[2]);
  const f = s < 1e-8 ? 2 : 2 * Math.atan2(s, q[3]) / s;
  return [q[0] * f, q[1] * f, q[2] * f];
}
export function quatMul(a, b) {
  return [a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1], a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0], a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3], a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]];
}
export function quatMatrix(q) {
  const [x,y,z,w] = q;
  return [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)];
}
export function slerp(a, b, t) {
  let dot = a.reduce((s, v, i) => s + v * b[i], 0);
  if (dot < 0) { b = b.map(x => -x); dot = -dot; }
  if (dot > 0.9995) { const q = a.map((x,i) => x+(b[i]-x)*t), n = Math.hypot(...q); return q.map(x=>x/n); }
  const theta = Math.acos(Math.min(1,dot)), s = Math.sin(theta);
  return a.map((x,i) => (Math.sin((1-t)*theta)*x + Math.sin(t*theta)*b[i])/s);
}
export function interpolatePose(a, b, t) {
  const p = copyPose(b);
  for (let j=0; j<p.angles.length; j+=3) p.angles.set(quatAxis(slerp(axisQuat(...a.angles.slice(j,j+3)), axisQuat(...b.angles.slice(j,j+3)), t)), j);
  p.expression = p.expression.map((x,i)=>a.expression[i]+(x-a.expression[i])*t);
  p.rootPosition = b.rootPosition.map((x,i)=>a.rootPosition[i]+(x-a.rootPosition[i])*t);
  p.rootRotation = slerp(a.rootRotation,b.rootRotation,t);
  return p;
}

// Seek/loop boundaries are discontinuities, not travel to interpolate through.
// Recordings can begin a timer tick after zero; hold that first sample until then.
export function sampleRecording(frames, time) {
  if (!frames.length) throw new Error('Recording has no frames');
  let i=0;
  while(i<frames.length-1 && frames[i+1].time<=time)i++;
  const a=frames[i],b=frames[i+1];
  if(!b || b.pose.teleport || time<=a.time)return copyPose(a.pose);
  const pose=interpolatePose(a.pose,b.pose,Math.max(0,Math.min(1,(time-a.time)/(b.time-a.time))));
  pose.teleport=!!a.pose.teleport;
  return pose;
}

// PyTorch3D's largest-candidate matrix_to_quaternion, followed by LHM normalization.
// This deliberately matches its treatment of a non-orthogonal LBS matrix.
function matrixQuat(m) {
  const candidates = [1+m[0]+m[4]+m[8], 1+m[0]-m[4]-m[8], 1-m[0]+m[4]-m[8], 1-m[0]-m[4]+m[8]];
  let k=0; for(let i=1;i<4;i++) if(candidates[i]>candidates[k]) k=i;
  const v = k===0 ? [m[7]-m[5],m[2]-m[6],m[3]-m[1],candidates[0]] : k===1 ? [candidates[1],m[1]+m[3],m[2]+m[6],m[7]-m[5]] : k===2 ? [m[1]+m[3],candidates[2],m[5]+m[7],m[2]-m[6]] : [m[2]+m[6],m[5]+m[7],candidates[3],m[3]-m[1]];
  const norm=Math.hypot(...v); return norm > 1e-12 ? v.map(x=>x/norm) : [0,0,0,1];
}

export function forwardKinematics(asset, pose) {
  const {joints,parents} = asset.arrays, b=asset.meta.nBones;
  const global=new Float64Array(b*12), skin=new Float64Array(b*12), positions=new Float32Array(b*3);
  for(let j=0;j<b;j++) {
    const k=j*12, p=parents[j], r=quatMatrix(axisQuat(...pose.angles.slice(j*3,j*3+3)));
    const d=[0,1,2].map(c=>joints[j*3+c]-(p<0?0:joints[p*3+c]));
    if(p<0) { for(let row=0;row<3;row++){ for(let col=0;col<3;col++)global[k+row*4+col]=r[row*3+col]; global[k+row*4+3]=d[row]; } }
    else { const pk=p*12; for(let row=0;row<3;row++) { for(let col=0;col<3;col++)global[k+row*4+col]=global[pk+row*4]*r[col]+global[pk+row*4+1]*r[3+col]+global[pk+row*4+2]*r[6+col]; global[k+row*4+3]=global[pk+row*4]*d[0]+global[pk+row*4+1]*d[1]+global[pk+row*4+2]*d[2]+global[pk+row*4+3]; } }
    for(let row=0;row<3;row++) { for(let col=0;col<3;col++)skin[k+row*4+col]=global[k+row*4+col]; skin[k+row*4+3]=global[k+row*4+3]-global[k+row*4]*joints[j*3]-global[k+row*4+1]*joints[j*3+1]-global[k+row*4+2]*joints[j*3+2]; positions[j*3+row]=global[k+row*4+3]; }
  }
  const root=quatMatrix(pose.rootRotation);
  for(let j=0;j<b;j++){ const v=positions.slice(j*3,j*3+3); for(let c=0;c<3;c++) positions[j*3+c]=root[c*3]*v[0]+root[c*3+1]*v[1]+root[c*3+2]*v[2]+pose.rootPosition[c]; }
  return {skin,positions,global};
}

export function deform(asset, pose) {
  const {meta,arrays:a}=asset, n=meta.nGaussians;
  // Keep lazy validation attachments from turning millions of skin-loop
  // property reads into dictionary lookups on the expanded arrays object.
  const {skinOffsets,skinBones,skinWeights,positions:restPositions,rotations:restRotations,
    expressionDirections,expressionLookup,rotationLocked,neutralLinear}=a;
  const {skin,positions:joints}=forwardKinematics(asset,pose);
  const positions=new Float32Array(n*3), rotations=new Float32Array(n*4), blend=new Float64Array(12), linear=new Float64Array(9);
  const active=[]; for(let e=0;e<pose.expression.length;e++) if(pose.expression[e]!==0)active.push(e);
  const exprCount=meta.version===2?meta.nExpressionVertices:n;
  const root=quatMatrix(pose.rootRotation), rootIdentity=pose.rootRotation[0]===0&&pose.rootRotation[1]===0&&pose.rootRotation[2]===0&&pose.rootRotation[3]===1;
  for(let i=0;i<n;i++) {
    // Scalar accumulators avoid the inner typed-array loop without changing
    // influence order or dropping weights. This is also the no-WebGPU fallback.
    let b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0,b7=0,b8=0,b9=0,b10=0,b11=0;
    for(let k=skinOffsets[i];k<skinOffsets[i+1];k++) { const s=skinBones[k]*12,w=skinWeights[k];
      b0+=w*skin[s];b1+=w*skin[s+1];b2+=w*skin[s+2];b3+=w*skin[s+3];b4+=w*skin[s+4];b5+=w*skin[s+5];
      b6+=w*skin[s+6];b7+=w*skin[s+7];b8+=w*skin[s+8];b9+=w*skin[s+9];b10+=w*skin[s+10];b11+=w*skin[s+11]; }
    blend[0]=b0;blend[1]=b1;blend[2]=b2;blend[3]=b3;blend[4]=b4;blend[5]=b5;blend[6]=b6;blend[7]=b7;blend[8]=b8;blend[9]=b9;blend[10]=b10;blend[11]=b11;
    let x=restPositions[i*3], y=restPositions[i*3+1], z=restPositions[i*3+2];
    const exprIndex=expressionLookup?expressionLookup[i]:i;
    if(exprIndex>=0)for(const e of active) { const k=(e*exprCount+exprIndex)*3, v=pose.expression[e]; x+=v*expressionDirections[k]; y+=v*expressionDirections[k+1]; z+=v*expressionDirections[k+2]; }
    const px=blend[0]*x+blend[1]*y+blend[2]*z+blend[3], py=blend[4]*x+blend[5]*y+blend[6]*z+blend[7], pz=blend[8]*x+blend[9]*y+blend[10]*z+blend[11];
    positions[i*3]=root[0]*px+root[1]*py+root[2]*pz+pose.rootPosition[0]; positions[i*3+1]=root[3]*px+root[4]*py+root[5]*pz+pose.rootPosition[1]; positions[i*3+2]=root[6]*px+root[7]*py+root[8]*pz+pose.rootPosition[2];
    let q;
    if(rotationLocked[i]) q=[restRotations[i*4],restRotations[i*4+1],restRotations[i*4+2],restRotations[i*4+3]];
    else { const k=i*9; for(let row=0;row<3;row++)for(let col=0;col<3;col++)linear[row*3+col]=blend[row*4]*neutralLinear[k+col]+blend[row*4+1]*neutralLinear[k+3+col]+blend[row*4+2]*neutralLinear[k+6+col]; q=quatMul(matrixQuat(linear),restRotations.subarray(i*4,i*4+4)); }
    if(!rootIdentity)q=quatMul(pose.rootRotation,q);
    rotations.set(q,i*4);
  }
  return {positions,rotations,joints};
}

export function compareFrame(out, expected, rq, n) {
    if(!expected||!rq||expected.length!==n*3||rq.length!==n*4||!out.positions.every(Number.isFinite)||!out.rotations.every(Number.isFinite))throw new Error('Missing or invalid validation data');
    let sum=0,max=0,rotationMax=0;
    for(let i=0;i<expected.length;i+=3) { const err=Math.hypot(out.positions[i]-expected[i],out.positions[i+1]-expected[i+1],out.positions[i+2]-expected[i+2]); sum+=err*err; max=Math.max(max,err); }
    for(let i=0;i<rq.length;i+=4){ let dot=0,an=0,bn=0; for(let c=0;c<4;c++){dot+=out.rotations[i+c]*rq[i+c];an+=out.rotations[i+c]**2;bn+=rq[i+c]**2;} rotationMax=Math.max(rotationMax,2*Math.acos(Math.min(1,Math.abs(dot)/Math.sqrt(an*bn)))); }
    return {rmsMm:1000*Math.sqrt(sum/n),maxMm:1000*max,rotationMaxDeg:rotationMax*180/Math.PI,pass:max<0.001&&rotationMax<0.002};
}

export function checkReferences(asset) {
  return asset.meta.referencePoses.map(f=> {
    const out=deform(asset,f), expected=asset.arrays[`reference_${f.name}_positions`], rq=asset.arrays[`reference_${f.name}_rotations`];
    return {name:f.name,...compareFrame(out,expected,rq,asset.meta.nGaussians)};
  });
}
