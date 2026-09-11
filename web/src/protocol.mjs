import {identityPose,interpolatePose} from './engine.mjs';

// GSW2: 28-byte header, bundled actor root, all local rotations, sparse expressions.
// Unlike the old prototype protocol this explicitly carries the expression dimension.
export function encodePose(pose, seq=0) {
  if(pose.angles.length!==165||pose.expression.length<1||pose.expression.length>256)throw new Error('Unsupported pose dimensions');
  const active=[]; for(let i=0;i<pose.expression.length;i++){if(!Number.isFinite(pose.expression[i]))throw new Error('Non-finite expression');if(Math.abs(pose.expression[i])>=0.0005)active.push(i);}
  const buffer=new ArrayBuffer(56+pose.angles.length*2+active.length*4),v=new DataView(buffer);
  v.setUint32(0,0x32575347,true);v.setUint8(4,2);v.setUint8(5,pose.teleport?1:0);
  v.setUint16(6,pose.angles.length/3,true);v.setUint16(8,pose.expression.length,true);v.setUint16(10,active.length,true);
  v.setUint32(12,seq>>>0,true);v.setUint32(16,pose.tick>>>0,true);v.setFloat64(20,Date.now(),true);
  let offset=28;
  for(const x of [...pose.rootPosition,...pose.rootRotation]){if(!Number.isFinite(x))throw new Error('Non-finite root');v.setFloat32(offset,x,true);offset+=4;}
  // Canonicalize axis-angle through its shortest equivalent quaternion representation.
  for(let j=0;j<pose.angles.length;j+=3){
    const raw=pose.angles.slice(j,j+3),norm=Math.hypot(...raw);
    if(!Number.isFinite(norm))throw new Error('Non-finite joint');
    const angle=((norm+Math.PI)%(2*Math.PI))-Math.PI,scale=norm>1e-8?angle/norm:1;
    for(const x of raw){v.setInt16(offset,Math.round(x*scale*32767/Math.PI),true);offset+=2;}
  }
  for(const i of active){const x=pose.expression[i];if(!Number.isFinite(x)||Math.abs(x)>32.767)throw new Error('Expression out of range');v.setUint16(offset,i,true);v.setInt16(offset+2,Math.round(x*1000),true);offset+=4;}
  return buffer;
}

export function decodePose(buffer) {
  const v=new DataView(buffer);
  if(v.byteLength<56||v.getUint32(0,true)!==0x32575347||v.getUint8(4)!==2)throw new Error('Invalid GSW2 snapshot');
  const bones=v.getUint16(6,true),expressions=v.getUint16(8,true),active=v.getUint16(10,true);
  if(bones!==55||expressions<1||expressions>256||active>expressions||v.byteLength!==56+bones*6+active*4)throw new Error('Snapshot dimensions or length mismatch');
  const p=identityPose(bones,expressions);p.seq=v.getUint32(12,true);p.tick=v.getUint32(16,true);p.teleport=!!(v.getUint8(5)&1);p.capturedAt=v.getFloat64(20,true);
  let o=28;
  for(let i=0;i<3;i++,o+=4)p.rootPosition[i]=v.getFloat32(o,true);
  for(let i=0;i<4;i++,o+=4)p.rootRotation[i]=v.getFloat32(o,true);
  if(![...p.rootPosition,...p.rootRotation,p.capturedAt].every(Number.isFinite))throw new Error('Non-finite snapshot');
  const norm=Math.hypot(...p.rootRotation);if(norm<1e-6)throw new Error('Invalid root quaternion');p.rootRotation=p.rootRotation.map(x=>x/norm);
  for(let i=0;i<p.angles.length;i++,o+=2)p.angles[i]=v.getInt16(o,true)*Math.PI/32767;
  const seen=new Set();
  for(let i=0;i<active;i++,o+=4){const index=v.getUint16(o,true);if(index>=expressions||seen.has(index))throw new Error('Invalid expression index');seen.add(index);p.expression[index]=v.getInt16(o+2,true)/1000;}
  return p;
}
export function newer(seq, previous){return previous===null||((seq-previous)>>>0)<0x80000000&&seq!==previous;}

export class PlaybackBuffer {
  constructor(delay=100){this.delay=delay;this.frames=[];this.sequence=null;this.received=0;this.stale=0;}
  push(pose,arrival){
    if(!newer(pose.seq,this.sequence)){this.stale++;return false;}
    this.sequence=pose.seq;this.received++;
    if(pose.teleport)this.frames=[];
    this.frames.push({pose,time:arrival});if(this.frames.length>120)this.frames.shift();return true;
  }
  sample(now){
    if(!this.frames.length)return null;
    const target=now-this.delay;
    while(this.frames.length>2&&this.frames[1].time<=target)this.frames.shift();
    const a=this.frames[0],b=this.frames[1];
    if(!b||a.pose.teleport||target<=a.time)return a.pose;
    return interpolatePose(a.pose,b.pose,Math.max(0,Math.min(1,(target-a.time)/(b.time-a.time))));
  }
}
