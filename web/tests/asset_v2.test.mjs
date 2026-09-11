import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeAsset,deform,identityPose,axisQuat,attachValidation} from '../src/engine.mjs';
import {GpuDeformer} from '../src/gpu-deform.mjs';

function fixture(sparse){
 const directions=new Float32Array(sparse?6:18);directions[sparse?0:3]=.1;directions[sparse?4:13]=-.2;
 const data={positions:new Float32Array([0,0,0,1,0,0,0,1,0]),rotations:new Float32Array([0,0,0,1,0,0,0,1,0,0,0,1]),
  neutralLinear:new Float32Array(Array(3).fill([1,0,0,0,1,0,0,0,1]).flat()),rotationLocked:new Uint8Array(3),
  joints:new Float32Array([0,0,0,0,1,0]),parents:new Int32Array([-1,0]),skinOffsets:new Uint32Array([0,1,3,4]),
  skinBones:new Uint16Array([0,0,1,1]),skinWeights:new Float32Array([1,.25,.75,1]),expressionDirections:directions};
 if(sparse)data.expressionIndices=new Uint32Array([1]);
 const arrays={};let size=0;
 const dtype=new Map([[Float32Array,'<f4'],[Uint32Array,'<u4'],[Int32Array,'<i4'],[Uint16Array,'<u2'],[Uint8Array,'|u1']]);
 for(const [name,value] of Object.entries(data)){size=Math.ceil(size/4)*4;arrays[name]={offset:size,bytes:value.byteLength,shape:[value.length],dtype:dtype.get(value.constructor)};size+=value.byteLength;}
 const buffer=new ArrayBuffer(size);for(const [name,value] of Object.entries(data))new Uint8Array(buffer,arrays[name].offset,value.byteLength).set(new Uint8Array(value.buffer));
 const meta={version:sparse?2:1,deformation:'lhm-linear-blend-v1',nGaussians:3,nBones:2,nExpressions:2,nExpressionVertices:1,expressionStorage:'sparse-vertices-v1',arrays};
 return {meta,buffer};
}

test('sparse and dense expression layouts produce identical centers/rotations with actor root motion',()=>{
 const dense=fixture(false),sparse=fixture(true),a=decodeAsset(dense.meta,dense.buffer),b=decodeAsset(sparse.meta,sparse.buffer);
 for(let i=0;i<8;i++){
  const p=identityPose(2,2);p.angles[4]=i*.2;p.expression.set([.3*i,-.5]);p.rootPosition=[.2,-.3,.1];p.rootRotation=axisQuat(.1,.4,.2);
  assert.deepEqual(deform(a,p),deform(b,p));
 }
});

test('sparse package validates storage, counts, indices and array byte bounds',()=>{
 for(const change of [m=>m.version=3,m=>m.expressionStorage='unknown',m=>m.nExpressionVertices=2,m=>m.arrays.positions.bytes=1,m=>m.arrays.skinBones.offset=3,m=>m.arrays.joints.shape=[-1]]){
  const f=fixture(true);change(f.meta);assert.throws(()=>decodeAsset(f.meta,f.buffer));
 }
 const f=fixture(true);new Uint32Array(f.buffer,f.meta.arrays.expressionIndices.offset,1)[0]=3;
 assert.throws(()=>decodeAsset(f.meta,f.buffer),/indices/);
});

test('validation attachments cannot overwrite runtime arrays',()=>{
 const f=fixture(true),a=decodeAsset(f.meta,f.buffer);a.meta.validation={bytes:4,arrays:{positions:{offset:0,bytes:4,dtype:'<f4',shape:[1]}}};
 assert.throws(()=>attachValidation(a,new ArrayBuffer(4)),/Unexpected/);
});

test('WebGPU absence is an explicit initialization failure for the CPU fallback',async()=>{
 if(globalThis.navigator?.gpu)return;
 await assert.rejects(()=>GpuDeformer.create({}),/WebGPU is unavailable/);
});
