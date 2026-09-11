// Exact 3-neighbor search: dynamic Morton ordering + conservative block bounds.
// No candidate cap, fixed-neighbor table, or approximate radius cutoff.
const structs=`
struct Posed { center:vec4f, r0:vec4f, r1:vec4f, r2:vec4f }
struct Box { lo:vec4f, hi:vec4f }
struct Geometry { centerMean:vec4f, covariance0:vec4f, covariance1:vec4f }
@group(0) @binding(6) var<uniform> counts:vec4u;
`;
const boxes=sorted=>structs+`
@group(0) @binding(0) var<storage,read> posed:array<Posed>;
${sorted?'@group(0) @binding(1) var<storage,read> order:array<vec2u>;':''}
@group(0) @binding(2) var<storage,read_write> boxes:array<Box>;
var<workgroup> low:array<vec3f,64>;
var<workgroup> high:array<vec3f,64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_id) local:vec3u) {
  let t=local.x;var lo=vec3f(3.402823e38);var hi=-lo;
  for(var k=0u;k<4u;k++){let index=group.x*256u+t+k*64u;if(index<counts.x){
    let p=posed[${sorted?'order[index].y':'index'}].center.xyz;lo=min(lo,p);hi=max(hi,p);
  }}
  low[t]=lo;high[t]=hi;workgroupBarrier();
  for(var stride=32u;stride>0u;stride=stride/2u){if(t<stride){low[t]=min(low[t],low[t+stride]);high[t]=max(high[t],high[t+stride]);}workgroupBarrier();}
  if(t==0u){boxes[group.x]=Box(vec4f(low[0],0),vec4f(high[0],0));}
}`;
const globalBox=structs+`
@group(0) @binding(2) var<storage,read> boxes:array<Box>;
@group(0) @binding(3) var<storage,read_write> total:array<Box>;
var<workgroup> low:array<vec3f,64>;var<workgroup> high:array<vec3f,64>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) local:vec3u){
  let t=local.x;var lo=vec3f(3.402823e38);var hi=-lo;
  for(var i=t;i<counts.z;i+=64u){lo=min(lo,boxes[i].lo.xyz);hi=max(hi,boxes[i].hi.xyz);}
  low[t]=lo;high[t]=hi;workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){if(t<stride){low[t]=min(low[t],low[t+stride]);high[t]=max(high[t],high[t+stride]);}workgroupBarrier();}
  if(t==0u){total[0]=Box(vec4f(low[0],0),vec4f(high[0],0));}
}`;
const keys=structs+`
@group(0) @binding(0) var<storage,read> posed:array<Posed>;
@group(0) @binding(1) var<storage,read_write> order:array<vec2u>;
@group(0) @binding(3) var<storage,read> total:array<Box>;
fn spread(value:u32)->u32 {
  var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;
  x=(x|(x<<4u))&0x030c30c3u;return (x|(x<<2u))&0x09249249u;
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  let i=id.x;if(i>=counts.y){return;}if(i>=counts.x){order[i]=vec2u(0xffffffffu,0xffffffffu);return;}
  let box=total[0];let unit=clamp((posed[i].center.xyz-box.lo.xyz)/max(box.hi.xyz-box.lo.xyz,vec3f(1e-12)),vec3f(0),vec3f(1));
  let grid=vec3u(unit*1023.);let key=spread(grid.x)|(spread(grid.y)<<1u)|(spread(grid.z)<<2u);
  order[i]=vec2u(key,i);
}`;
const sort=`
@group(0) @binding(0) var<storage,read_write> order:array<vec2u>;
@group(0) @binding(1) var<uniform> params:vec4u;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  let i=id.x;if(i>=params.z){return;}let j=i^params.y;if(j<=i){return;}
  let a=order[i];let b=order[j];let greater=a.x>b.x||(a.x==b.x&&a.y>b.y);
  let ascending=(i&params.x)==0u;
  if(greater==ascending){order[i]=b;order[j]=a;}
}`;
const nearest=structs+`
@group(0) @binding(0) var<storage,read> posed:array<Posed>;
@group(0) @binding(1) var<storage,read> order:array<vec2u>;
@group(0) @binding(2) var<storage,read> boxes:array<Box>;
@group(0) @binding(4) var<storage,read_write> result:array<Geometry>;
@group(0) @binding(5) var<storage,read> radii:array<vec4f>;
fn insert(best:vec3f,distance:f32)->vec3f{
  if(distance<best.x){return vec3f(distance,best.x,best.y);}
  if(distance<best.y){return vec3f(best.x,distance,best.y);}
  return vec3f(best.xy,min(best.z,distance));
}
fn distance(a:vec3f,b:vec3f)->f32{let d=a-b;return dot(d,d);}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  let rank=id.x;if(rank>=counts.x){return;}let index=order[rank].y;let entry=posed[index];let p=entry.center.xyz;
  var best=vec3f(3.402823e38);
  for(var j=max(0,i32(rank)-3);j<min(i32(counts.x),i32(rank)+4);j++){
    if(u32(j)!=rank){best=insert(best,distance(p,posed[order[u32(j)].y].center.xyz));}
  }
  let initialRadius=best.z;best=vec3f(3.402823e38);
  for(var block=0u;block<counts.z;block++){
    let box=boxes[block];
    // Expand, never shrink, the box to protect a conservative floating bound.
    let guard=1e-6*max(vec3f(1),max(abs(box.lo.xyz),abs(box.hi.xyz)));
    let delta=max(vec3f(0),max(box.lo.xyz-guard-p,p-box.hi.xyz-guard));
    if(dot(delta,delta)>min(initialRadius,best.z)){continue;}
    for(var j=block*256u;j<min((block+1u)*256u,counts.x);j++){
      let other=order[j].y;if(other!=index){best=insert(best,distance(p,posed[other].center.xyz));}
    }
  }
  let mean=(best.x+best.y+best.z)/3.;
  let scale=sqrt(max(mean,1e-7))*radii[index].xyz;let variance=scale*scale;
  let a=entry.r0.xyz;let b=entry.r1.xyz;let c=entry.r2.xyz;
  result[index]=Geometry(vec4f(p,mean),vec4f(dot(a*variance,a),dot(a*variance,b),dot(a*variance,c),dot(b*variance,b)),
    vec4f(dot(b*variance,c),dot(c*variance,c),0,0));
}`;

export async function makePass(gpu,code,entries,groups,label){
  const device=gpu.device,module=device.createShaderModule({code,label});
  const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');
  if(errors.length)throw Error(label+': '+errors.map(x=>x.message).join('\n'));
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'},label});
  const bindGroup=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries});
  return {pipeline,bindGroup,groups};
}
export function encodePass(encoder,{pipeline,bindGroup,groups}){
  const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bindGroup);pass.dispatchWorkgroups(groups);pass.end();
}
export async function installNeighbors(gpu,posed,radii,n){
  if(!Number.isInteger(n)||n<4||n>1000000)throw Error('3NN requires 4–1,000,000 points');
  const S=GPUBufferUsage.STORAGE,U=GPUBufferUsage.UNIFORM,power=2**Math.ceil(Math.log2(n)),blocks=Math.ceil(n/256);
  const order=gpu.buffer(power*8,S),bounds=gpu.buffer(blocks*32,S),total=gpu.buffer(32,S),output=gpu.buffer(n*48,S|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  const counts=gpu.buffer(new Uint32Array([n,power,blocks,0]),U);
  const bind=(binding,buffer)=>({binding,resource:{buffer}}),c=bind(6,counts),passes=[];
  passes.push(await makePass(gpu,boxes(false),[bind(0,posed),bind(2,bounds),c],blocks,'Unsorted point bounds'));
  passes.push(await makePass(gpu,globalBox,[bind(2,bounds),bind(3,total),c],1,'Global point bounds'));
  passes.push(await makePass(gpu,keys,[bind(0,posed),bind(1,order),bind(3,total),c],Math.ceil(power/64),'Morton keys'));
  const stages=[];for(let step=2;step<=power;step*=2)for(let stride=step/2;stride>=1;stride/=2)stages.push([step,stride,power,0]);
  const parameters=new Uint32Array(stages.length*64);stages.forEach((s,i)=>parameters.set(s,i*64));
  const params=gpu.buffer(parameters,U),first=await makePass(gpu,sort,[bind(0,order),{binding:1,resource:{buffer:params,offset:0,size:16}}],Math.ceil(power/64),'Bitonic Morton ordering');
  for(let i=0;i<stages.length;i++)passes.push({...first,bindGroup:gpu.device.createBindGroup({layout:first.pipeline.getBindGroupLayout(0),entries:[bind(0,order),{binding:1,resource:{buffer:params,offset:i*256,size:16}}]})});
  passes.push(await makePass(gpu,boxes(true),[bind(0,posed),bind(1,order),bind(2,bounds),c],blocks,'Sorted block bounds'));
  passes.push(await makePass(gpu,nearest,[bind(0,posed),bind(1,order),bind(2,bounds),bind(4,output),bind(5,radii),c],Math.ceil(n/64),'Exact 3NN and full covariance'));
  return{output,order,power,passes,sorting:passes.slice(3,3+stages.length),sortPasses:stages.length,encode(encoder){for(const pass of passes)encodePass(encoder,pass);}};
}
