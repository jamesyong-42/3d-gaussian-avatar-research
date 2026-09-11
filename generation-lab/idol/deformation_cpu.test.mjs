import test from 'node:test';
import assert from 'node:assert/strict';
import {compileRig,resolveNativePose,parameterOffset,rodrigues,validateParameters,decodeGeometry} from './deformation_cpu.mjs';
function fixture(){
  const raw={deformation:'idol-native-deformation-v1-lab',nBones:55,nExpressions:10,nShape:10,
    parents:Array.from({length:55},(_,i)=>i-1),poseMean:Array(165).fill(0),jointBase:Array.from({length:165},(_,i)=>i%3===1?Math.floor(i/3)*.01:0),jointDirections:Array(3300).fill(0)};
  const params=new Float32Array(189);params[0]=1;return{raw,params};
}
test('neutral joints cancel to identity transforms; translation applied once',()=>{
  const{raw,params}=fixture();params.set([.1,-.2,.3],1);const {transforms}=resolveNativePose(compileRig(raw),params);
  for(let j=0;j<55;j++)for(let i=0;i<16;i++)assert.ok(Math.abs(transforms[j*16+i]-(i%5===0?1:i%4===3&&i<12?params[1+Math.floor(i/4)]:0))<1e-6);
});
test('hand/jaw/eye offsets map each native rotation slot exactly once',()=>{
  const used=Array.from({length:55},(_,i)=>parameterOffset(i));assert.equal(new Set(used).size,55);
  assert.equal(used[22],170);assert.equal(used[24],176);assert.equal(used[25],80);assert.equal(used[54],167);
  assert.throws(()=>parameterOffset(55));assert.throws(()=>parameterOffset(-1));
});
test('hand mean added exactly once and root excluded from corrective features',()=>{
  const{raw,params}=fixture();raw.poseMean[75]=.2;params[80]=.3;params[4]=.8;
  const result=resolveNativePose(compileRig(raw),params),expected=rodrigues(Math.fround(.3+.2),0,0);
  for(let k=0;k<9;k++)assert.ok(Math.abs(result.controls[24*9+k]-(expected[k]-(k%4===0?1:0)))<1e-7);
  assert.equal(result.controls[0],0);
});
test('shape and expression can both affect rest joints',()=>{
  const{raw,params}=fixture();raw.jointDirections[0]=.1;raw.jointDirections[10]=.2;params[70]=2;params[179]=1;
  const result=resolveNativePose(compileRig(raw),params);assert.ok(Math.abs(result.joints[0]-.4)<1e-7);assert.equal(result.controls[486],2);assert.equal(result.controls[496],1);
});
test('reject schema, cyclic hierarchy, nonfinite controls and unsupported scale',()=>{
  const{raw,params}=fixture();raw.parents[5]=5;assert.throws(()=>compileRig(raw));
  for(const p of [new Float32Array(188),new Float64Array(189),new Float32Array(189).fill(NaN)])assert.throws(()=>validateParameters(p));
  params[0]=2;assert.throws(()=>validateParameters(params));
});

function geometryFixture(){
  const shapes={restPositions:['<f4',[4,3]],restOrientations:['<f4',[4,3,3]],radiusMultipliers:['<f4',[4,3]],colors:['<f4',[4,3]],opacities:['<f4',[4,1]],skinOffsets:['<u4',[5]],skinBones:['<u4',[8]],skinWeights:['<f4',[8]]};
  const meta={deformation:'idol-native-deformation-v1-lab',nBones:55,nGaussians:4,nInfluences:8,arrays:{},defaultParameters:Array.from(fixture().params)};
  let offset=0;for(const[name,[dtype,shape]]of Object.entries(shapes)){const bytes=shape.reduce((a,b)=>a*b,4);meta.arrays[name]={dtype,shape,bytes,offset};offset+=bytes;}
  meta.bytes=offset;const binary=new ArrayBuffer(offset),arrays={};
  for(const[name,s]of Object.entries(meta.arrays))arrays[name]=new(s.dtype==='<f4'?Float32Array:Uint32Array)(binary,s.offset,s.bytes/4);
  arrays.radiusMultipliers.fill(1);arrays.colors.fill(.5);arrays.opacities.fill(1);arrays.skinOffsets.set([0,2,4,6,8]);
  arrays.skinBones.set([0,1,0,1,0,1,0,1]);arrays.skinWeights.set([.6,.4,.6,.4,.6,.4,.6,.4]);
  for(let i=0;i<4;i++)for(let k=0;k<3;k++)arrays.restOrientations[i*9+k*4]=1;
  return{meta,binary,arrays};
}
test('geometry preserves native RGB saturation and all sorted skin influences',()=>{
  const{meta,binary,arrays}=geometryFixture();arrays.colors[0]=-.001;arrays.colors[1]=1.001;
  const decoded=decodeGeometry(meta,binary);assert.equal(decoded.arrays.skinWeights.length,8);assert.equal(decoded.arrays.colors[0],Math.fround(-.001));
});
test('geometry rejects missing, misaligned, overlapping and truncated array layouts',()=>{
  for(const mutate of [f=>delete f.meta.arrays.colors,f=>f.meta.arrays.colors.offset++,f=>f.meta.arrays.colors.offset=0,f=>f.meta.arrays.colors.dtype='<u4',f=>f.meta.bytes--,f=>f.binary=f.binary.slice(0,-4)]){
    const f=geometryFixture();mutate(f);assert.throws(()=>decodeGeometry(f.meta,f.binary));
  }
});
test('geometry rejects invalid skin CSR, bones, weights and normalization',()=>{
  for(const mutate of [a=>a.skinOffsets[0]=1,a=>a.skinOffsets[4]=7,a=>a.skinOffsets[2]=1,a=>a.skinBones[0]=55,a=>a.skinBones[1]=0,a=>a.skinWeights[0]=0,a=>a.skinWeights[0]=.2]){
    const f=geometryFixture();mutate(f.arrays);assert.throws(()=>decodeGeometry(f.meta,f.binary));
  }
});
test('geometry rejects nonfinite attributes, unsupported appearance and scale',()=>{
  for(const mutate of [f=>f.arrays.restPositions[0]=NaN,f=>f.arrays.radiusMultipliers[0]=-1,f=>f.arrays.colors[0]=1.1,f=>f.arrays.opacities[0]=2,f=>f.meta.defaultParameters[0]=2]){
    const f=geometryFixture();mutate(f);assert.throws(()=>decodeGeometry(f.meta,f.binary));
  }
});
