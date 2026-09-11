// WebGPU computes the exact full-influence model. Readback feeds the existing
// WebGL renderer; reported wall time includes FK, dispatch, readback, and unpack.
import {forwardKinematics,quatMatrix} from './engine.mjs';

const shader=`
struct Rest { position:vec4f, rotation:vec4f, n0:vec4f, n1:vec4f, n2:vec4f, begin:u32, end:u32, expr:i32, pad:u32 }
struct Influence { bone:u32, weight:f32 }
struct Output { position:vec4f, rotation:vec4f }
@group(0) @binding(0) var<storage,read> rest:array<Rest>;
@group(0) @binding(1) var<storage,read> influences:array<Influence>;
@group(0) @binding(2) var<storage,read> expressions:array<f32>;
@group(0) @binding(3) var<storage,read> pose:array<f32>;
@group(0) @binding(4) var<storage,read_write> result:array<Output>;
@group(0) @binding(5) var<uniform> counts:vec4u;
fn qmul(a:vec4f,b:vec4f)->vec4f { return vec4f(a.w*b.xyz+b.w*a.xyz+cross(a.xyz,b.xyz),a.w*b.w-dot(a.xyz,b.xyz)); }
fn matrixQuat(a:vec3f,b:vec3f,c:vec3f)->vec4f {
  let candidates=vec4f(1+a.x+b.y+c.z,1+a.x-b.y-c.z,1-a.x+b.y-c.z,1-a.x-b.y+c.z);
  var k=0u;for(var j=1u;j<4u;j++){if(candidates[j]>candidates[k]){k=j;}}
  var q=vec4f(c.y-b.z,a.z-c.x,b.x-a.y,candidates.x);
  if(k==1u){q=vec4f(candidates.y,a.y+b.x,a.z+c.x,c.y-b.z);}
  if(k==2u){q=vec4f(a.y+b.x,candidates.z,b.z+c.y,a.z-c.x);}
  if(k==3u){q=vec4f(a.z+c.x,b.z+c.y,candidates.w,b.x-a.y);}
  if(length(q)>1e-12){return normalize(q);}return vec4f(0,0,0,1);
}
fn four(i:u32)->vec4f {return vec4f(pose[i],pose[i+1u],pose[i+2u],pose[i+3u]);}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x;if(i>=counts.x){return;}
  let r=rest[i];var b0=vec4f(0);var b1=vec4f(0);var b2=vec4f(0);
  for(var k=r.begin;k<r.end;k++){let w=influences[k];let s=w.bone*12u;b0+=w.weight*four(s);b1+=w.weight*four(s+4u);b2+=w.weight*four(s+8u);}
  let eb=counts.w*12u;let control=eb+2u*counts.y;
  var center=r.position.xyz;
  if(r.expr>=0){for(var j=0u;j<u32(pose[control]);j++){
    let e=u32(pose[eb+counts.y+j]);let index=(e*counts.z+u32(r.expr))*3u;
    center+=pose[eb+e]*vec3f(expressions[index],expressions[index+1u],expressions[index+2u]);
  }}
  let v=vec4f(center,1);let posed=vec3f(dot(b0,v),dot(b1,v),dot(b2,v));
  let root0=vec3f(pose[control+1u],pose[control+2u],pose[control+3u]);
  let root1=vec3f(pose[control+4u],pose[control+5u],pose[control+6u]);
  let root2=vec3f(pose[control+7u],pose[control+8u],pose[control+9u]);
  let translate=vec3f(pose[control+10u],pose[control+11u],pose[control+12u]);
  result[i].position=vec4f(vec3f(dot(root0,posed),dot(root1,posed),dot(root2,posed))+translate,1);
  var q=r.rotation;
  if(r.position.w==0){
    let c0=vec3f(r.n0.x,r.n1.x,r.n2.x);let c1=vec3f(r.n0.y,r.n1.y,r.n2.y);let c2=vec3f(r.n0.z,r.n1.z,r.n2.z);
    let a=vec3f(dot(b0.xyz,c0),dot(b0.xyz,c1),dot(b0.xyz,c2));
    let b=vec3f(dot(b1.xyz,c0),dot(b1.xyz,c1),dot(b1.xyz,c2));
    let c=vec3f(dot(b2.xyz,c0),dot(b2.xyz,c1),dot(b2.xyz,c2));q=qmul(matrixQuat(a,b,c),q);
  }
  result[i].rotation=qmul(four(control+13u),q);
}`;

export class GpuDeformer {
  static async create(asset){
    if(!globalThis.navigator?.gpu)throw new Error('WebGPU is unavailable in this browser/context');
    const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
    if(!adapter)throw new Error('No WebGPU adapter');
    const {meta,arrays:a}=asset,n=meta.nGaussians;
    const sizes=[n*96,a.skinBones.length*8,Math.max(4,a.expressionDirections.byteLength),n*32];
    const largest=Math.max(...sizes);
    if(largest>adapter.limits.maxStorageBufferBindingSize||largest>adapter.limits.maxBufferSize||Math.ceil(n/64)>adapter.limits.maxComputeWorkgroupsPerDimension)throw new Error('Avatar exceeds WebGPU device limits');
    const device=await adapter.requestDevice({requiredLimits:{maxStorageBufferBindingSize:Math.max(128*1024**2,largest),maxBufferSize:Math.max(256*1024**2,largest)}});
    const instance=new GpuDeformer();instance.device=device;instance.asset=asset;instance.buffers=[];
    instance.info={vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,device:adapter.info?.device,description:adapter.info?.description};
    device.lost.then(info=>{instance.lost=info.message||'WebGPU device lost';});
    device.addEventListener('uncapturederror',event=>{instance.lost=event.error.message;});
    try{
      device.pushErrorScope('validation');
      const module=device.createShaderModule({code:shader,label:'Full-influence Gaussian deformation'});
      const compilation=await module.getCompilationInfo();
      const errors=compilation.messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(m=>m.message).join('\n'));
      instance.pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
      const rest=new ArrayBuffer(n*96),f=new Float32Array(rest),u=new Uint32Array(rest),ints=new Int32Array(rest);
      for(let i=0;i<n;i++){
        const k=i*24;f.set(a.positions.subarray(i*3,i*3+3),k);f[k+3]=a.rotationLocked[i];f.set(a.rotations.subarray(i*4,i*4+4),k+4);
        for(let row=0;row<3;row++)f.set(a.neutralLinear.subarray(i*9+row*3,i*9+row*3+3),k+8+row*4);
        u[k+20]=a.skinOffsets[i];u[k+21]=a.skinOffsets[i+1];ints[k+22]=a.expressionLookup?a.expressionLookup[i]:i;
      }
      const weights=new ArrayBuffer(a.skinBones.length*8),wb=new Uint32Array(weights),wf=new Float32Array(weights);
      for(let i=0;i<a.skinBones.length;i++){wb[i*2]=a.skinBones[i];wf[i*2+1]=a.skinWeights[i];}
      instance.poseData=new Float32Array(meta.nBones*12+meta.nExpressions*2+17);
      const storage=GPUBufferUsage.STORAGE;
      const bindings=[instance.buffer(rest,storage),instance.buffer(weights,storage),instance.buffer(a.expressionDirections,storage),
        instance.buffer(instance.poseData,storage|GPUBufferUsage.COPY_DST),instance.buffer(n*32,storage|GPUBufferUsage.COPY_SRC),
        instance.buffer(new Uint32Array([n,meta.nExpressions,meta.version===2?meta.nExpressionVertices:n,meta.nBones]),GPUBufferUsage.UNIFORM)];
      instance.poseBuffer=bindings[3];instance.output=bindings[4];
      instance.readback=instance.buffer(n*32,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
      instance.bindGroup=device.createBindGroup({layout:instance.pipeline.getBindGroupLayout(0),entries:bindings.map((buffer,binding)=>({binding,resource:{buffer}}))});
      const error=await device.popErrorScope();if(error)throw new Error(error.message);
      return instance;
    }catch(error){instance.destroy();throw error;}
  }
  buffer(data,usage){
    const bytes=typeof data==='number'?data:data.byteLength;
    const buffer=this.device.createBuffer({size:Math.max(4,Math.ceil(bytes/4)*4),usage,mappedAtCreation:typeof data!=='number'});
    this.buffers.push(buffer);
    if(typeof data!=='number'){
      const source=data instanceof ArrayBuffer?new Uint8Array(data):new Uint8Array(data.buffer,data.byteOffset,data.byteLength);
      new Uint8Array(buffer.getMappedRange()).set(source);buffer.unmap();
    }return buffer;
  }
  async deform(pose){
    if(this.lost)throw new Error(this.lost);
    const {meta}=this.asset,{skin,positions:joints}=forwardKinematics(this.asset,pose),d=this.poseData;
    d.set(skin);d.set(pose.expression,skin.length);
    const active=skin.length+meta.nExpressions,control=active+meta.nExpressions;
    let count=0;for(let e=0;e<pose.expression.length;e++)if(pose.expression[e]!==0)d[active+count++]=e;
    d[control]=count;d.set(quatMatrix(pose.rootRotation),control+1);d.set(pose.rootPosition,control+10);d.set(pose.rootRotation,control+13);
    const device=this.device;device.queue.writeBuffer(this.poseBuffer,0,d);
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
    pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.bindGroup);pass.dispatchWorkgroups(Math.ceil(meta.nGaussians/64));pass.end();
    encoder.copyBufferToBuffer(this.output,0,this.readback,0,meta.nGaussians*32);device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    try{
      const raw=new Float32Array(this.readback.getMappedRange()),positions=new Float32Array(meta.nGaussians*3),rotations=new Float32Array(meta.nGaussians*4);
      for(let i=0;i<meta.nGaussians;i++){
        const k=i*8;positions[i*3]=raw[k];positions[i*3+1]=raw[k+1];positions[i*3+2]=raw[k+2];
        rotations[i*4]=raw[k+4];rotations[i*4+1]=raw[k+5];rotations[i*4+2]=raw[k+6];rotations[i*4+3]=raw[k+7];
      }return {positions,rotations,joints};
    }finally{this.readback.unmap();}
  }
  destroy(){for(const buffer of this.buffers??[])buffer.destroy();this.device?.destroy();}
}
