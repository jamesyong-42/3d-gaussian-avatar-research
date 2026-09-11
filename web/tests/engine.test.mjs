import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createSyntheticAsset} from '../src/synthetic.mjs';
import {decodeAsset,checkReferences,identityPose,interpolatePose,deform,forwardKinematics,axisQuat,quatMatrix} from '../src/engine.mjs';
import {encodePose,decodePose,PlaybackBuffer,newer} from '../src/protocol.mjs';
import {solveTargets} from '../src/controls.mjs';
const root=process.env.GSAVATAR_NATIVE_FIXTURE?pathToFileURL(path.resolve(process.env.GSAVATAR_NATIVE_FIXTURE)+path.sep):undefined;

test('real exported avatar matches original LHM centers and rotations for seven poses',{skip:!root&&'Set GSAVATAR_NATIVE_FIXTURE to a consented native export; not bundled.'},()=>{
  const meta=JSON.parse(fs.readFileSync(new URL('avatar.json',root))),b=fs.readFileSync(new URL('avatar.bin',root));
  const asset=decodeAsset(meta,b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
  const checks=checkReferences(asset);console.log(JSON.stringify(checks));
  assert.equal(checks.length,7);for(const c of checks)assert.ok(c.pass,JSON.stringify(c));
});
test('GSW2 roundtrip retains the 100th expression and bundled root',()=>{
  const p=identityPose();p.angles[54*3+1]=0.7;p.expression[99]=0.456;p.rootPosition=[0.1,0.2,0.3];p.tick=123;
  const b=encodePose(p,42),q=decodePose(b);assert.equal(q.seq,42);assert.equal(q.tick,123);assert.equal(q.expression.length,100);assert.ok(Math.abs(q.expression[99]-0.456)<0.001);assert.ok(Math.abs(q.angles[163]-0.7)<0.0001);assert.ok(b.byteLength<1200);
  assert.throws(()=>decodePose(b.slice(0,-1)));
});
test('root and pose interpolate together; rotations take the short arc',()=>{
  const a=identityPose(),b=identityPose();a.angles[0]=3.1;b.angles[0]=-3.1;b.rootPosition[2]=2;
  const p=interpolatePose(a,b,0.5);assert.ok(Math.abs(p.angles[0])>3);assert.equal(p.rootPosition[2],1);
});
test('playback drops stale snapshots, handles wraparound and teleport',()=>{
  const q=new PlaybackBuffer(50);const a=identityPose();a.seq=4;q.push(a,100);
  assert.equal(q.push({...a,seq:3},110),false);
  const b={...identityPose(),seq:5,rootPosition:[1,0,0]};q.push(b,200);assert.equal(q.sample(200).rootPosition[0],0.5);
  q.push({...b,seq:6,teleport:true,rootPosition:[5,0,0]},210);assert.equal(q.sample(210).rootPosition[0],5);assert.equal(newer(0,0xffffffff),true);
});
test('IK resolves world-space targets with a translated, rotated actor root',()=>{
  const asset=createSyntheticAsset();
  const base=identityPose();base.angles[50]=-.35;base.angles[53]=.35;base.rootPosition=[0.3,0.2,-0.1];base.rootRotation=axisQuat(0,0.6,0);
  const before=forwardKinematics(asset,base),target=Array.from(before.positions.slice(60,63));target[0]-=0.12;target[2]+=0.1;
  const p=solveTargets(asset,base,{left:target,leftRotation:[0,0.4,0],leftGrip:0.8});
  const after=forwardKinematics(asset,p),error=Math.hypot(...target.map((v,i)=>v-after.positions[60+i]));
  assert.ok(error<0.02,`Target error ${error}m`);assert.deepEqual(p.rootPosition,base.rootPosition);assert.deepEqual(p.rootRotation,base.rootRotation);
  const actor=after.global.subarray(20*12,20*12+12),r=quatMatrix(base.rootRotation),desired=quatMatrix(axisQuat(0,0.4,0));
  for(let row=0;row<3;row++)for(let col=0;col<3;col++){const world=r[row*3]*actor[col]+r[row*3+1]*actor[4+col]+r[row*3+2]*actor[8+col];assert.ok(Math.abs(world-desired[row*3+col])<1e-6);}
});
test('invalid signals are rejected and the maximum expression packet stays bounded',()=>{
  const p=identityPose();p.expression[2]=NaN;assert.throws(()=>encodePose(p),/expression/);
  p.expression.fill(0.5);const wire=encodePose(p);assert.equal(wire.byteLength,786);
  const q=decodePose(wire);assert.ok(q.expression.every(x=>x===0.5));
  p.angles[0]=Infinity;assert.throws(()=>encodePose(p),/joint/);
});
