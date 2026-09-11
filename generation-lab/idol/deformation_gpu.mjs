import {CorrectiveGpu} from './corrective_gpu.mjs';
import {resolveNativePose} from './deformation_cpu.mjs';
import {makePass,encodePass,installNeighbors} from './neighbors_gpu.mjs';

const skin=`
struct Rest { position:vec4f, r0:vec4f, r1:vec4f, r2:vec4f, range:vec4u }
struct Influence { bone:u32, weight:f32 }
struct Field { shape:vec4f, pose:vec4f }
struct Posed { center:vec4f, r0:vec4f, r1:vec4f, r2:vec4f }
@group(0) @binding(0) var<storage,read> rest:array<Rest>;
@group(0) @binding(1) var<storage,read> influences:array<Influence>;
@group(0) @binding(2) var<storage,read> transforms:array<vec4f>;
@group(0) @binding(3) var<storage,read> fields:array<Field>;
@group(0) @binding(4) var<storage,read_write> posed:array<Posed>;
@group(0) @binding(5) var<uniform> counts:vec4u;
fn rotate(row:vec3f,a:vec3f,b:vec3f,c:vec3f)->vec4f {
  return vec4f(row.x*a+row.y*b+row.z*c,0);
}
fn nativeDot4(a:vec4f,b:vec4f)->f32 {
  // Native skinning materializes the four products then reduces pairwise.
  // A 100,000-row CUDA probe matched (p0+p1)+(p2+p3) exactly.
  let p=a*b;return (p.x+p.y)+(p.z+p.w);
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  let i=id.x;if(i>=counts.x){return;}let r=rest[i];var a=vec4f(0);var b=vec4f(0);var c=vec4f(0);
  for(var k=r.range.x;k<r.range.y;k++){let w=influences[k];let j=w.bone*4u;a=fma(vec4f(w.weight),transforms[j],a);b=fma(vec4f(w.weight),transforms[j+1u],b);c=fma(vec4f(w.weight),transforms[j+2u],c);}
  let point=vec4f((r.position.xyz+fields[i].shape.xyz)+fields[i].pose.xyz,1);
  posed[i]=Posed(vec4f(nativeDot4(a,point),nativeDot4(b,point),nativeDot4(c,point),1),
    rotate(a.xyz,r.r0.xyz,r.r1.xyz,r.r2.xyz),rotate(b.xyz,r.r0.xyz,r.r1.xyz,r.r2.xyz),rotate(c.xyz,r.r0.xyz,r.r1.xyz,r.r2.xyz));
}`;

export class DeformationGpu {
  static async create(correctives,rig,assets){
    if(!assets.length||assets.some(a=>a.meta.nGaussians!==correctives.meta.nGaussians))throw Error('Geometry/corrective counts differ');
    const gpu=await CorrectiveGpu.create(correctives),self=new DeformationGpu();
    Object.assign(self,{gpu,rig,assets,states:[],busy:false,destroyed:false,n:correctives.meta.nGaussians});
    try{
      const device=gpu.device,S=GPUBufferUsage.STORAGE,U=GPUBufferUsage.UNIFORM,n=self.n;
      device.pushErrorScope('validation');
      self.transforms=gpu.buffer(55*16*4,S|GPUBufferUsage.COPY_DST);
      self.posed=gpu.buffer(n*64,S|GPUBufferUsage.COPY_SRC);
      const counts=gpu.buffer(new Uint32Array([n,55,0,0]),U);
      const b=(buffer,binding)=>({binding,resource:{buffer}});
      for(const asset of assets){
        const a=asset.arrays,rest=new ArrayBuffer(n*80),f=new Float32Array(rest),u=new Uint32Array(rest),radii=new Float32Array(n*4);
        for(let i=0;i<n;i++){
          const k=i*20;f.set(a.restPositions.subarray(i*3,i*3+3),k);f[k+3]=1;
          for(let row=0;row<3;row++)f.set(a.restOrientations.subarray(i*9+row*3,i*9+row*3+3),k+4+row*4);
          u[k+16]=a.skinOffsets[i];u[k+17]=a.skinOffsets[i+1];radii.set(a.radiusMultipliers.subarray(i*3,i*3+3),i*4);
        }
        const weights=new ArrayBuffer(a.skinBones.length*8),wb=new Uint32Array(weights),wf=new Float32Array(weights);
        for(let i=0;i<a.skinBones.length;i++){wb[i*2]=a.skinBones[i];wf[i*2+1]=a.skinWeights[i];}
        const restBuffer=gpu.buffer(new Uint8Array(rest),S),weightBuffer=gpu.buffer(new Uint8Array(weights),S),radiusBuffer=gpu.buffer(radii,S);
        const skinPass=await makePass(gpu,skin,[restBuffer,weightBuffer,self.transforms,gpu.output,self.posed,counts].map(b),Math.ceil(n/64),'Full-influence IDOL skinning');
        const neighbors=await installNeighbors(gpu,self.posed,radiusBuffer,n);
        self.states.push({skinPass,neighbors});
      }
      self.readback=gpu.buffer(n*48,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
      const error=await device.popErrorScope();if(error)throw Error(error.message);
      self.info={...gpu.info,ownedBufferBytes:gpu.buffers.reduce((s,b)=>s+b.size,0),sortPasses:self.states[0].neighbors.sortPasses};
      return self;
    }catch(error){self.destroy();throw error;}
  }
  async evaluate(params,source=0){
    if(this.destroyed||this.gpu.failure)throw Error(this.gpu.failure||'Deformation GPU destroyed');
    if(this.busy)throw Error('Deformation already in flight');
    if(!Number.isInteger(source)||!this.states[source])throw Error('Invalid avatar source');
    const start=performance.now(),pose=resolveNativePose(this.rig,params),fkDone=performance.now();
    this.busy=true;let mapped=false;
    try{
      const gpu=this.gpu,device=gpu.device,state=this.states[source];
      device.queue.writeBuffer(this.transforms,0,pose.transforms);
      const encoder=device.createCommandEncoder();gpu.encodeFields(encoder,pose.controls);
      encodePass(encoder,state.skinPass);state.neighbors.encode(encoder);
      encoder.copyBufferToBuffer(state.neighbors.output,0,this.readback,0,this.n*48);
      device.queue.submit([encoder.finish()]);await this.readback.mapAsync(GPUMapMode.READ);mapped=true;
      if(gpu.failure)throw Error(gpu.failure);
      const geometry=new Float32Array(this.readback.getMappedRange().slice(0));
      return {geometry,pose,fkMs:fkDone-start,totalMs:performance.now()-start};
    }finally{if(mapped)this.readback.unmap();this.busy=false;}
  }
  async render(params,source,viewer){
    if(this.destroyed||this.gpu.failure)throw Error(this.gpu.failure||'Deformation GPU destroyed');
    if(this.busy)throw Error('Deformation already in flight');
    if(!Number.isInteger(source)||!this.states[source])throw Error('Invalid avatar source');
    const start=performance.now(),pose=resolveNativePose(this.rig,params);this.busy=true;
    try{
      const gpu=this.gpu,device=gpu.device,state=this.states[source],encoder=device.createCommandEncoder();
      device.queue.writeBuffer(this.transforms,0,pose.transforms);gpu.encodeFields(encoder,pose.controls);
      encodePass(encoder,state.skinPass);state.neighbors.encode(encoder);viewer.encode(encoder,source);
      device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
      if(gpu.failure)throw Error(gpu.failure);return{totalMs:performance.now()-start,pose};
    }finally{this.busy=false;}
  }
  destroy(){if(this.destroyed)return;this.destroyed=true;this.gpu.destroy();}
}
