// Isolated WebGL2 diagnostic renderer: full covariance, no CUDA pixel-parity claim.
const vertex=`#version 300 es
precision highp float;
layout(location=0) in vec3 center;
layout(location=1) in vec4 cov0;
layout(location=2) in vec2 cov1;
layout(location=3) in vec4 rgba;
uniform vec4 viewRows[3];uniform vec2 viewport;uniform float focal;
out vec2 gaussian;out vec4 color;
vec3 sigma(vec3 v){return vec3(cov0.x*v.x+cov0.y*v.y+cov0.z*v.z,cov0.y*v.x+cov0.w*v.y+cov1.x*v.z,cov0.z*v.x+cov1.x*v.y+cov1.y*v.z);}
void main(){
  vec4 p=vec4(center,1);vec3 camera=vec3(dot(viewRows[0],p),dot(viewRows[1],p),dot(viewRows[2],p));
  vec2 corners[6]=vec2[](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
  gaussian=corners[gl_VertexID]*3.33;color=rgba;
  if(camera.z<=.2){gl_Position=vec4(2,2,2,1);color.a=0.;return;}
  float z=camera.z;vec3 jx=focal/z*viewRows[0].xyz-focal*camera.x/(z*z)*viewRows[2].xyz;
  vec3 jy=focal/z*viewRows[1].xyz-focal*camera.y/(z*z)*viewRows[2].xyz;
  float a=dot(jx,sigma(jx)),b=dot(jx,sigma(jy)),c=dot(jy,sigma(jy));
  float before=max(0.,a*c-b*b);a+=.3;c+=.3;
  color.a*=sqrt(clamp(before/max(a*c-b*b,1e-12),0.,1.));
  float halfTrace=(a+c)*.5,delta=sqrt(max(0.,(a-c)*(a-c)*.25+b*b));
  vec2 eigen=max(vec2(halfTrace+delta,halfTrace-delta),vec2(1e-8));
  vec2 axis=abs(b)>1e-10?normalize(vec2(b,eigen.x-a)):(a>=c?vec2(1,0):vec2(0,1));
  vec2 offset=axis*sqrt(eigen.x)*gaussian.x+vec2(-axis.y,axis.x)*sqrt(eigen.y)*gaussian.y;
  vec2 pixel=focal*camera.xy/z+offset;
  gl_Position=vec4(2.*pixel.x/viewport.x,-2.*pixel.y/viewport.y,0,1);
}`;
const fragment=`#version 300 es
precision highp float;in vec2 gaussian;in vec4 color;out vec4 outColor;
void main(){float alpha=min(.99,color.a*exp(-.5*dot(gaussian,gaussian)));if(alpha<1./255.)discard;outColor=vec4(clamp(color.rgb,0.,1.)*alpha,alpha);}`;
export class CovarianceViewer{
  constructor(canvas,n){
    this.canvas=canvas;this.n=n;this.gl=canvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:true,powerPreference:'high-performance'});
    if(!this.gl)throw Error('WebGL2 unavailable');const gl=this.gl;
    const shader=(type,code)=>{const s=gl.createShader(type);gl.shaderSource(s,code);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
    const v=shader(gl.VERTEX_SHADER,vertex),f=shader(gl.FRAGMENT_SHADER,fragment);this.program=gl.createProgram();gl.attachShader(this.program,v);gl.attachShader(this.program,f);gl.linkProgram(this.program);
    if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));gl.deleteShader(v);gl.deleteShader(f);
    this.vao=gl.createVertexArray();gl.bindVertexArray(this.vao);this.buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,n*64,gl.DYNAMIC_DRAW);
    for(const[location,size,offset]of [[0,3,0],[1,4,16],[2,2,32],[3,4,48]]){gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,size,gl.FLOAT,false,64,offset);gl.vertexAttribDivisor(location,1);}
    this.data=new Float32Array(n*16);this.order=Uint32Array.from({length:n},(_,i)=>i);this.depth=new Float32Array(n);
    gl.useProgram(this.program);this.viewLocation=gl.getUniformLocation(this.program,'viewRows');this.sizeLocation=gl.getUniformLocation(this.program,'viewport');this.focalLocation=gl.getUniformLocation(this.program,'focal');
    gl.disable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.clearColor(1,1,1,1);
    canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();this.lost=true;});
  }
  draw(geometry,asset){
    if(this.lost||this.destroyed)throw Error('Covariance renderer unavailable');
    const start=performance.now(),gl=this.gl,n=this.n;
    if(!(geometry instanceof Float32Array)||geometry.length!==n*12||geometry.some(x=>!Number.isFinite(x)))throw Error('Invalid render geometry');
    const camera=asset.meta.camera.packed,view=new Float32Array(camera.slice(4,16)),width=asset.meta.camera.imageSize[0],height=asset.meta.camera.imageSize[1];
    if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    for(let i=0;i<n;i++)this.depth[i]=view[8]*geometry[i*12]+view[9]*geometry[i*12+1]+view[10]*geometry[i*12+2]+view[11];
    this.order.sort((a,b)=>this.depth[b]-this.depth[a]);const sorted=performance.now(),a=asset.arrays;
    for(let rank=0;rank<n;rank++){
      const i=this.order[rank],k=rank*16;this.data.set(geometry.subarray(i*12,i*12+12),k);this.data.set(a.colors.subarray(i*3,i*3+3),k+12);this.data[k+15]=a.opacities[i];
    }
    gl.viewport(0,0,width,height);gl.useProgram(this.program);gl.uniform4fv(this.viewLocation,view);gl.uniform2f(this.sizeLocation,width,height);gl.uniform1f(this.focalLocation,camera[0]);
    gl.bindVertexArray(this.vao);gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferSubData(gl.ARRAY_BUFFER,0,this.data);gl.clear(gl.COLOR_BUFFER_BIT);gl.drawArraysInstanced(gl.TRIANGLES,0,6,n);gl.finish();
    if(gl.getError()!==gl.NO_ERROR)throw Error('WebGL draw failed');
    return {sortMs:sorted-start,sortUploadDrawMs:performance.now()-start};
  }
  pixels(){const gl=this.gl,result=new Uint8Array(this.canvas.width*this.canvas.height*4);gl.readPixels(0,0,this.canvas.width,this.canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,result);return result;}
  clear(){this.gl.clear(this.gl.COLOR_BUFFER_BIT);this.gl.finish();}
  destroy(){if(this.destroyed)return;this.destroyed=true;const gl=this.gl;gl.deleteBuffer(this.buffer);gl.deleteVertexArray(this.vao);gl.deleteProgram(this.program);}
}
