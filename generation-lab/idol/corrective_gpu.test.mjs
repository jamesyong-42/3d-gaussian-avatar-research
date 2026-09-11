import test from 'node:test';
import assert from 'node:assert/strict';
import {decodePackage,validateControls} from './corrective_gpu.mjs';

function fixture(){
  const arrays={poseOffsets:new Uint32Array([0,1,1,2]),poseBones:new Uint32Array([0,53]),poseValues:new Float32Array(54),
    shapeValues:new Float32Array(180),midpointParents:new Uint32Array([0,1,1,2,0,2]),gaussianFaces:new Uint32Array([0,3,5,3,1,4,5,4,2,3,4,5])};
  const shapes={poseOffsets:[4],poseBones:[2],poseValues:[2,9,3],shapeValues:[3,3,20],midpointParents:[3,2],gaussianFaces:[4,3]};
  const bytes=Object.values(arrays).reduce((s,a)=>s+a.byteLength,0),binary=new ArrayBuffer(bytes),specs={};let offset=0;
  for(const [name,a]of Object.entries(arrays)){
    new Uint8Array(binary,offset,a.byteLength).set(new Uint8Array(a.buffer));
    specs[name]={offset,bytes:a.byteLength,dtype:a instanceof Float32Array?'<f4':'<u4',shape:shapes[name]};offset+=a.byteLength;
  }
  const meta={deformation:'idol-corrective-field-subdivision-v1-lab',nCoarse:3,nVertices:6,nGaussians:4,nonzeroBlocks:2,
    nPoseFeatures:486,nShapeFeatures:20,bytes,runtimeBytes:bytes,arrays:specs};return{meta,binary};
}
test('decode checks complete package layout',()=>{const {meta,binary}=fixture();assert.equal(decodePackage(meta,binary).arrays.poseBones[1],53);});
test('reject wrong layout, sizes, overlaps and extra arrays',()=>{
  for(const mutation of ['type','counts','dimensions','offset','dtype','truncated','extra']){
    const{meta,binary}=fixture();let payload=binary;
    if(mutation==='type')meta.deformation='avatar';if(mutation==='counts')meta.nVertices=3;
    if(mutation==='dimensions')meta.arrays.shapeValues.shape=[3,20,3];if(mutation==='offset')meta.arrays.poseBones.offset=0;
    if(mutation==='dtype')meta.arrays.poseBones.dtype='<f4';if(mutation==='truncated')payload=binary.slice(0,-4);
    if(mutation==='extra')meta.arrays.unexpected={};assert.throws(()=>decodePackage(meta,payload),undefined,mutation);
  }
});
test('reject invalid CSR, topology and nonfinite basis',()=>{
  for(const [name,index,value]of [['poseOffsets',0,1],['poseOffsets',2,99],['poseBones',1,54],['midpointParents',0,3],['gaussianFaces',0,6],['poseValues',0,NaN],['shapeValues',0,Infinity]]){
    const{meta,binary}=fixture();const{arrays}=decodePackage(meta,binary);arrays[name][index]=value;assert.throws(()=>decodePackage(meta,binary));
  }
});
test('control boundary rejects length, precision and nonfinite inputs',()=>{
  validateControls(new Float32Array(506));
  for(const x of [[],new Float32Array(505),new Float64Array(506),new Float32Array(506).fill(NaN),new Float32Array(506).fill(Infinity)])assert.throws(()=>validateControls(x));
});
