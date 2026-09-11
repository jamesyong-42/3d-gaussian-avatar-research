// A WebGPU-resident diagnostic renderer: full covariance and fresh depth sorting.
// Compute output remains on-device. Still not the native CUDA rasterizer.
import {makePass,encodePass} from './neighbors_gpu.mjs';
const header=`
struct Geometry { centerMean:vec4f, covariance0:vec4f, covariance1:vec4f }
struct Camera { row0:vec4f, row1:vec4f, row2:vec4f, projection:vec4f, counts:vec4u }
@group(0) @binding(0) var<storage,read> geometry:array<Geometry>;
@group(0) @binding(3) var<uniform> camera:Camera;
`;
const depth=header+`
@group(0) @binding(1) var<storage,read_write> order:array<vec2u>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  let i=id.x;if(i>=camera.counts.y){return;}if(i>=camera.counts.x){order[i]=vec2u(0xffffffffu,0xffffffffu);return;}
  let z=max(0.,dot(camera.row2,vec4f(geometry[i].centerMean.xyz,1)));
  order[i]=vec2u(bitcast<u32>(z),i);
}`;
const render=header+`
@group(0) @binding(1) var<storage,read> order:array<vec2u>;
@group(0) @binding(2) var<storage,read> colors:array<vec4f>;
struct VertexOutput { @builtin(position) position:vec4f, @location(0) gaussian:vec2f, @location(1) color:vec4f }
fn sigma(g:Geometry,v:vec3f)->vec3f {
  let a=g.covariance0;let b=g.covariance1;
  return vec3f(a.x*v.x+a.y*v.y+a.z*v.z,a.y*v.x+a.w*v.y+b.x*v.z,a.z*v.x+b.x*v.y+b.y*v.z);
}
@vertex fn vertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->VertexOutput {
  let index=order[camera.counts.x-1u-instance].y;let g=geometry[index];let p=vec4f(g.centerMean.xyz,1);
  let v=vec3f(dot(camera.row0,p),dot(camera.row1,p),dot(camera.row2,p));
  let corners=array<vec2f,6>(vec2f(-1,-1),vec2f(1,-1),vec2f(1,1),vec2f(-1,-1),vec2f(1,1),vec2f(-1,1));
  let gaussian=corners[vertex]*3.33;var color=colors[index];
  if(v.z<=.2){return VertexOutput(vec4f(2,2,1,1),gaussian,vec4f(0));}
  let focal=camera.projection.z;
  let jx=focal/v.z*camera.row0.xyz-focal*v.x/(v.z*v.z)*camera.row2.xyz;
  let jy=focal/v.z*camera.row1.xyz-focal*v.y/(v.z*v.z)*camera.row2.xyz;
  var a=dot(jx,sigma(g,jx));let b=dot(jx,sigma(g,jy));var c=dot(jy,sigma(g,jy));
  let before=max(0.,a*c-b*b);a+=.3;c+=.3;
  color.w*=sqrt(clamp(before/max(a*c-b*b,1e-12),0.,1.));
  let halfTrace=(a+c)*.5;let delta=sqrt(max(0.,(a-c)*(a-c)*.25+b*b));
  let eigen=max(vec2f(halfTrace+delta,halfTrace-delta),vec2f(1e-8));
  var axis=select(vec2f(0,1),vec2f(1,0),a>=c);if(abs(b)>1e-10){axis=normalize(vec2f(b,eigen.x-a));}
  let offset=axis*sqrt(eigen.x)*gaussian.x+vec2f(-axis.y,axis.x)*sqrt(eigen.y)*gaussian.y;
  let pixel=focal*v.xy/v.z+offset;
  return VertexOutput(vec4f(2.*pixel.x/camera.projection.x,-2.*pixel.y/camera.projection.y,0,1),gaussian,color);
}
@fragment fn fragment(in:VertexOutput)->@location(0) vec4f {
  let alpha=min(.99,in.color.w*exp(-.5*dot(in.gaussian,in.gaussian)));if(alpha<1./255.){discard;}
  return vec4f(clamp(in.color.xyz,vec3f(0),vec3f(1))*alpha,alpha);
}`;
export class ResidentViewer{
  static async create(canvas,deformer){
    const self=new ResidentViewer(),gpu=deformer.gpu,device=gpu.device;
    Object.assign(self,{canvas,deformer,gpu,states:[],format:'rgba8unorm'});
    self.context=canvas.getContext('webgpu');if(!self.context)throw Error('WebGPU canvas unavailable');
    canvas.width=640;canvas.height=896;
    self.context.configure({device,format:self.format,alphaMode:'opaque',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_DST});
    // Persistent render target: a presented canvas texture need not survive until
    // a later diagnostic readback. The live loop only performs a GPU-to-GPU copy.
    self.target=device.createTexture({size:[640,896],format:self.format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
    device.pushErrorScope('validation');
    const module=device.createShaderModule({code:render,label:'Full-covariance GPU splats'});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(e=>e.message).join('\n'));
    self.pipeline=await device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vertex'},fragment:{module,entryPoint:'fragment',targets:[{format:self.format,
      blend:{color:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'}}}]},primitive:{topology:'triangle-list'}});
    const entry=(binding,buffer)=>({binding,resource:{buffer}});
    for(let source=0;source<deformer.assets.length;source++){
      const asset=deformer.assets[source],neighbors=deformer.states[source].neighbors;
      const uniforms=new Float32Array(20),counts=new Uint32Array(uniforms.buffer),color=new Float32Array(deformer.n*4);
      uniforms.set(asset.meta.camera.packed.slice(4,16));uniforms.set([640,896,asset.meta.camera.packed[0],0],12);counts.set([deformer.n,neighbors.power,0,0],16);
      for(let i=0;i<deformer.n;i++){color.set(asset.arrays.colors.subarray(i*3,i*3+3),i*4);color[i*4+3]=asset.arrays.opacities[i];}
      const cameraBuffer=gpu.buffer(uniforms,GPUBufferUsage.UNIFORM),colorBuffer=gpu.buffer(color,GPUBufferUsage.STORAGE);
      const keyPass=await makePass(gpu,depth,[entry(0,neighbors.output),entry(1,neighbors.order),entry(3,cameraBuffer)],Math.ceil(neighbors.power/64),'GPU depth keys');
      const bindGroup=device.createBindGroup({layout:self.pipeline.getBindGroupLayout(0),entries:[entry(0,neighbors.output),entry(1,neighbors.order),entry(2,colorBuffer),entry(3,cameraBuffer)]});
      self.states.push({keyPass,neighbors,bindGroup});
    }
    const error=await device.popErrorScope();if(error)throw Error(error.message);
    self.info={ownedGpuBufferBytes:gpu.buffers.reduce((s,b)=>s+b.size,0),ownedRenderTextureBytes:640*896*4,renderFormat:self.format,readbackInLiveLoop:false,renderSort:'Fresh GPU bitonic depth ordering, far to near'};
    return self;
  }
  encode(encoder,source){
    const state=this.states[source];encodePass(encoder,state.keyPass);for(const pass of state.neighbors.sorting)encodePass(encoder,pass);
    const pass=encoder.beginRenderPass({colorAttachments:[{view:this.target.createView(),clearValue:{r:1,g:1,b:1,a:1},loadOp:'clear',storeOp:'store'}]});
    pass.setPipeline(this.pipeline);pass.setBindGroup(0,state.bindGroup);pass.draw(6,this.deformer.n);pass.end();
    this.present(encoder);
  }
  present(encoder){encoder.copyTextureToTexture({texture:this.target},{texture:this.context.getCurrentTexture()},[this.canvas.width,this.canvas.height]);}
  async pixels(){
    // Diagnostic-only pixel readback, never used during animation.
    const device=this.gpu.device,bytes=this.canvas.width*this.canvas.height*4;
    const buffer=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{
      const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture:this.target},{buffer,bytesPerRow:this.canvas.width*4},[this.canvas.width,this.canvas.height]);device.queue.submit([encoder.finish()]);
      await buffer.mapAsync(GPUMapMode.READ);const result=new Uint8Array(buffer.getMappedRange().slice(0));buffer.unmap();return result;
    }finally{buffer.destroy();}
  }
  async draw(geometry,asset){
    const source=this.deformer.assets.indexOf(asset);if(source<0)throw Error('Unknown source');
    // Only native-reference image comparison uses this upload path.
    this.gpu.device.queue.writeBuffer(this.states[source].neighbors.output,0,geometry);
    const encoder=this.gpu.device.createCommandEncoder();this.encode(encoder,source);this.gpu.device.queue.submit([encoder.finish()]);await this.gpu.device.queue.onSubmittedWorkDone();
  }
  async clear(){
    const device=this.gpu.device,encoder=device.createCommandEncoder();const pass=encoder.beginRenderPass({colorAttachments:[{view:this.target.createView(),clearValue:{r:1,g:1,b:1,a:1},loadOp:'clear',storeOp:'store'}]});pass.end();this.present(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
  }
  destroy(){this.target.destroy();this.context.unconfigure();}
}
