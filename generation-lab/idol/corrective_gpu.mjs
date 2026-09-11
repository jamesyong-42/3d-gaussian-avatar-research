// Research spike: corrective fields only, not a complete IDOL avatar deformer.
const type = 'idol-corrective-field-subdivision-v1-lab';
export function decodePackage(meta, binary) {
  if (meta.deformation !== type || !(binary instanceof ArrayBuffer)) throw Error('Wrong corrective package');
  const {nCoarse:c,nVertices:v,nGaussians:n,nonzeroBlocks:b}=meta;
  if (![c,v,n,b].every(x=>Number.isSafeInteger(x)&&x>0) || c>=v || v>500000 || n>1000000 || b>c*54 ||
      meta.nPoseFeatures!==486 || meta.nShapeFeatures!==20) throw Error('Invalid corrective counts');
  const layout={poseOffsets:['<u4',[c+1]],poseBones:['<u4',[b]],poseValues:['<f4',[b,9,3]],
    shapeValues:['<f4',[c,3,20]],midpointParents:['<u4',[v-c,2]],gaussianFaces:['<u4',[n,3]]};
  if (Object.keys(meta.arrays??{}).sort().join()!==Object.keys(layout).sort().join()) throw Error('Invalid array set');
  let offset=0;const arrays={};
  for (const [name,[dtype,shape]] of Object.entries(layout)) {
    const spec=meta.arrays[name],bytes=shape.reduce((a,b)=>a*b,4);
    if (spec.dtype!==dtype || JSON.stringify(spec.shape)!==JSON.stringify(shape) || spec.offset!==offset || spec.bytes!==bytes ||
        offset+bytes>binary.byteLength) throw Error('Invalid array layout: '+name);
    arrays[name]=new (dtype==='<u4'?Uint32Array:Float32Array)(binary,offset,bytes/4);offset+=bytes;
  }
  if (offset!==binary.byteLength || meta.bytes!==offset || meta.runtimeBytes!==offset) throw Error('Invalid payload length');
  const a=arrays;
  if (a.poseOffsets[0]!==0 || a.poseOffsets[c]!==b) throw Error('Invalid sparse endpoints');
  for (let i=0;i<c;i++) {
    const begin=a.poseOffsets[i],end=a.poseOffsets[i+1];
    if (end<begin || end>b || end-begin>54) throw Error('Invalid sparse offsets');
    for (let j=begin;j<end;j++) if (a.poseBones[j]>=54 || (j>begin&&a.poseBones[j]<=a.poseBones[j-1])) throw Error('Invalid sparse bones');
  }
  if (a.midpointParents.some(x=>x>=c) || a.gaussianFaces.some(x=>x>=v)) throw Error('Invalid topology index');
  if (a.poseValues.some(x=>!Number.isFinite(x)) || a.shapeValues.some(x=>!Number.isFinite(x))) throw Error('Nonfinite coefficient');
  return {meta,arrays};
}

export function validateControls(coefficients) {
  if (!(coefficients instanceof Float32Array) || coefficients.length!==506 || coefficients.some(x=>!Number.isFinite(x)))
    throw Error('Expected 506 finite float32 controls: 486 pose features + 20 shape/expression coefficients');
}

const header=`struct Field { shape:vec4f, pose:vec4f }
@group(0) @binding(6) var<uniform> counts:vec4u;`;
const coarseShader=header+`
@group(0) @binding(0) var<storage,read> offsets:array<u32>;
@group(0) @binding(1) var<storage,read> bones:array<u32>;
@group(0) @binding(2) var<storage,read> values:array<f32>;
@group(0) @binding(3) var<storage,read> shapes:array<f32>;
@group(0) @binding(4) var<storage,read> controls:array<f32>;
@group(0) @binding(5) var<storage,read_write> fields:array<Field>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x;if(i>=counts.x){return;}
  var pose=vec3f(0);var shape=vec3f(0);
  for(var block=offsets[i];block<offsets[i+1u];block++) {
    let feature=bones[block]*9u;
    for(var entry=0u;entry<9u;entry++) {
      let k=block*27u+entry*3u;
      pose+=controls[feature+entry]*vec3f(values[k],values[k+1u],values[k+2u]);
    }
  }
  for(var k=0u;k<20u;k++) {
    let s=i*60u+k;
    shape+=controls[486u+k]*vec3f(shapes[s],shapes[s+20u],shapes[s+40u]);
  }
  fields[i]=Field(vec4f(shape,0),vec4f(pose,0));
}`;
const midpointShader=header+`
@group(0) @binding(0) var<storage,read> parents:array<vec2u>;
@group(0) @binding(1) var<storage,read_write> fields:array<Field>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x;if(i>=counts.y-counts.x){return;}
  let pair=parents[i];let a=fields[pair.x];let b=fields[pair.y];
  fields[counts.x+i]=Field((a.shape+b.shape)*.5,(a.pose+b.pose)*.5);
}`;
const poolShader=header+`
@group(0) @binding(0) var<storage,read> faces:array<u32>;
@group(0) @binding(1) var<storage,read> fields:array<Field>;
@group(0) @binding(2) var<storage,read_write> result:array<Field>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x;if(i>=counts.z){return;}
  let a=fields[faces[i*3u]];let b=fields[faces[i*3u+1u]];let c=fields[faces[i*3u+2u]];
  result[i]=Field((a.shape+b.shape+c.shape)/3.,(a.pose+b.pose+c.pose)/3.);
}`;

export class CorrectiveGpu {
  static async create(asset) {
    if (!globalThis.navigator?.gpu) throw Error('WebGPU unavailable');
    const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
    if (!adapter) throw Error('No WebGPU adapter');
    const {meta:m,arrays:a}=asset;
    const largest=Math.max(...Object.values(a).map(x=>x.byteLength),m.nVertices*32,m.nGaussians*32);
    if (largest>adapter.limits.maxBufferSize || largest>adapter.limits.maxStorageBufferBindingSize ||
        Math.ceil(Math.max(m.nVertices,m.nGaussians)/64)>adapter.limits.maxComputeWorkgroupsPerDimension ||
        adapter.limits.maxStorageBuffersPerShaderStage<6) throw Error('Corrective package exceeds GPU limits');
    const device=await adapter.requestDevice();
    const self=new CorrectiveGpu();Object.assign(self,{device,meta:m,buffers:[],busy:false,destroyed:false});
    self.info={vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,device:adapter.info?.device,
      description:adapter.info?.description,isFallbackAdapter:adapter.info?.isFallbackAdapter,
      maxBufferSize:device.limits.maxBufferSize,maxStorageBufferBindingSize:device.limits.maxStorageBufferBindingSize};
    device.lost.then(info=>{self.failure=info.message||'GPU device lost';});
    device.addEventListener('uncapturederror',e=>{self.failure=e.error.message;});
    try {
      device.pushErrorScope('validation');
      const S=GPUBufferUsage.STORAGE;
      const source=Object.fromEntries(Object.entries(a).map(([k,v])=>[k,self.buffer(v,S)]));
      const counts=self.buffer(new Uint32Array([m.nCoarse,m.nVertices,m.nGaussians,m.nonzeroBlocks]),GPUBufferUsage.UNIFORM);
      self.controls=self.buffer(506*4,S|GPUBufferUsage.COPY_DST);
      self.vertices=self.buffer(m.nVertices*32,S);
      self.output=self.buffer(m.nGaussians*32,S|GPUBufferUsage.COPY_SRC);
      self.readback=self.buffer(m.nGaussians*32,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
      self.passes=[];
      for (const [label,code,bindings,count] of [
        ['Sparse coarse fields',coarseShader,[source.poseOffsets,source.poseBones,source.poseValues,source.shapeValues,self.controls,self.vertices],m.nCoarse],
        ['Midpoint reconstruction',midpointShader,[source.midpointParents,self.vertices],m.nVertices-m.nCoarse],
        ['Gaussian face pooling',poolShader,[source.gaussianFaces,self.vertices,self.output],m.nGaussians]]) {
        const module=device.createShaderModule({code,label});
        const messages=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');
        if (messages.length) throw Error(messages.map(x=>x.message).join('\n'));
        const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'},label});
        const entries=bindings.map((buffer,binding)=>({binding,resource:{buffer}}));entries.push({binding:6,resource:{buffer:counts}});
        const bindGroup=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries});
        self.passes.push({pipeline,bindGroup,count});
      }
      const error=await device.popErrorScope();if(error) throw Error(error.message);
      self.info.ownedBufferBytes=self.buffers.reduce((s,b)=>s+b.size,0);
      return self;
    } catch(error) {self.destroy();throw error;}
  }
  buffer(data,usage) {
    const size=typeof data==='number'?data:data.byteLength;
    const buffer=this.device.createBuffer({size:Math.max(4,size),usage,mappedAtCreation:typeof data!=='number'});
    this.buffers.push(buffer);
    if(typeof data!=='number') {new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));buffer.unmap();}
    return buffer;
  }
  async evaluate(coefficients) {
    validateControls(coefficients);
    if(this.destroyed||this.failure) throw Error(this.failure||'Corrective GPU destroyed');
    if(this.busy) throw Error('Corrective evaluation already in flight');
    this.busy=true;let mapped=false;
    try {
      const encoder=this.device.createCommandEncoder();
      this.encodeFields(encoder,coefficients);
      encoder.copyBufferToBuffer(this.output,0,this.readback,0,this.meta.nGaussians*32);
      this.device.queue.submit([encoder.finish()]);
      await this.readback.mapAsync(GPUMapMode.READ);mapped=true;
      if(this.failure) throw Error(this.failure);
      return new Float32Array(this.readback.getMappedRange().slice(0));
    } finally {if(mapped)this.readback.unmap();this.busy=false;}
  }
  encodeFields(encoder,coefficients) {
    validateControls(coefficients);
    if(this.destroyed||this.failure)throw Error(this.failure||'Corrective GPU destroyed');
    this.device.queue.writeBuffer(this.controls,0,coefficients);
    for(const {pipeline,bindGroup,count} of this.passes) {
      const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
    }
  }
  destroy() {if(this.destroyed)return;this.destroyed=true;for(const b of this.buffers)b.destroy();this.device.destroy();}
}
