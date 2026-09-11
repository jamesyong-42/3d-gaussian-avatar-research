import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {TransformControls} from 'three/addons/controls/TransformControls.js';
import {ExtSplats,SplatMesh,SparkRenderer,utils} from '@sparkjsdev/spark';

export class Viewer {
  scene=new THREE.Scene(); camera=new THREE.PerspectiveCamera(37,1,0.01,100);
  renderer:THREE.WebGLRenderer; controls:OrbitControls; spark:SparkRenderer;
  mesh?:SplatMesh; splats?:ExtSplats; asset:any; skeleton:THREE.LineSegments;
  ground=new THREE.Group();private homeTarget=new THREE.Vector3();private homeDistance=3.9;
  gizmo:TransformControls; targets=new THREE.Group(); targetObjects:Record<string,THREE.Mesh>={};
  onTargetChange?: (name:string,position:number[])=>void;
  frameCount=0; fps=0; frameMs=0; private times:number[]=[]; private last=performance.now(); private tick=performance.now();
  constructor(public element:HTMLElement){
    this.renderer=new THREE.WebGLRenderer({antialias:false,alpha:false,powerPreference:'high-performance'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));this.renderer.setClearColor('#10151b');
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.element.append(this.renderer.domElement);
    this.spark=new SparkRenderer({renderer:this.renderer});this.scene.add(this.spark);
    this.camera.position.set(0,0.55,3.9);this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.target.set(0,0,0);this.controls.enableDamping=true;
    const grid=new THREE.GridHelper(10,50,0x33434a,0x202b32);this.ground.add(grid);
    const floor=new THREE.Mesh(new THREE.CircleGeometry(1.1,96),new THREE.MeshBasicMaterial({color:0x1b2930,transparent:true,opacity:0.35,depthWrite:false}));floor.rotation.x=-Math.PI/2;floor.position.y=0.001;this.ground.add(floor);this.ground.position.y=-1.35;this.scene.add(this.ground);
    this.skeleton=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x87efcf,depthTest:false,transparent:true,opacity:0.8}));this.skeleton.renderOrder=10;this.skeleton.visible=false;this.scene.add(this.skeleton);
    this.gizmo=new TransformControls(this.camera,this.renderer.domElement);this.gizmo.setSize(0.65);this.scene.add(this.gizmo.getHelper());
    this.gizmo.addEventListener('dragging-changed',e=>{this.controls.enabled=!e.value;});
    this.gizmo.addEventListener('objectChange',()=>{const object=this.gizmo.object;if(object)this.onTargetChange?.(object.name,object.position.toArray());});
    for(const [name,color] of [['head',0xebc58b],['left',0x7fe0c4],['right',0x8aa9ff]] as const){const object=new THREE.Mesh(new THREE.SphereGeometry(0.028,16,12),new THREE.MeshBasicMaterial({color,depthTest:false}));object.name=name;object.renderOrder=20;this.targets.add(object);this.targetObjects[name]=object;}
    this.targets.visible=false;this.scene.add(this.targets);
    this.renderer.domElement.addEventListener('pointerdown',e=>{if(!this.targets.visible||this.gizmo.dragging)return;const rect=this.renderer.domElement.getBoundingClientRect(),ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),this.camera);const hit=ray.intersectObjects(Object.values(this.targetObjects))[0];if(hit)this.gizmo.attach(hit.object);});
    new ResizeObserver(()=>this.resize()).observe(element);this.resize();
    this.renderer.setAnimationLoop(()=>{const now=performance.now();this.times.push(now-this.last);if(this.times.length>120)this.times.shift();this.last=now;this.controls.update();this.renderer.render(this.scene,this.camera);this.frameCount++;if(now-this.tick>1000){this.fps=this.frameCount*1000/(now-this.tick);this.frameMs=[...this.times].sort((a,b)=>a-b)[Math.floor(this.times.length*0.95)]||0;this.frameCount=0;this.tick=now;}});
  }
  resize(){const {width,height}=this.element.getBoundingClientRect();if(width<1||height<1)return;this.renderer.setSize(width,height);this.camera.aspect=width/height;this.camera.updateProjectionMatrix();}
  async load(asset:any,frame:any){
    if(this.mesh){this.scene.remove(this.mesh);this.mesh.dispose();this.mesh=undefined;}
    this.asset=asset;
    // ExtSplats.setSplat grows CPU arrays but does not create the GPU textures.
    // Supplying texture-aligned arrays at construction initializes both sides.
    const {maxSplats}=utils.getTextureSize(asset.meta.nGaussians);
    this.splats=new ExtSplats({extArrays:[new Uint32Array(maxSplats*4),new Uint32Array(maxSplats*4)],numSplats:asset.meta.nGaussians});
    await this.splats.initialized;
    this.updateSplats(frame);
    this.mesh=new SplatMesh({extSplats:this.splats,lod:false});await this.mesh.initialized;this.scene.add(this.mesh);
    let min=Infinity,max=-Infinity;for(let i=1;i<frame.positions.length;i+=3){min=Math.min(min,frame.positions[i]);max=Math.max(max,frame.positions[i]);}
    this.ground.position.y=min-0.015;this.homeTarget.set(0,(min+max)/2,0);this.homeDistance=Math.max(2.8,(max-min)*2.2);this.home();
  }
  updateSplats(frame:any){
    if(!this.splats)return;
    const a=this.asset.arrays,p=new THREE.Vector3(),s=new THREE.Vector3(),q=new THREE.Quaternion(),color=new THREE.Color();
    for(let i=0;i<this.asset.meta.nGaussians;i++){
      p.fromArray(frame.positions,i*3);s.fromArray(a.scales,i*3);q.fromArray(frame.rotations,i*4);color.setRGB(a.colors[i*3],a.colors[i*3+1],a.colors[i*3+2]);
      this.splats.setSplat(i,p,s,q,a.opacities[i],color);
    }
    this.splats.numSplats=this.asset.meta.nGaussians;
    for(const texture of this.splats.textures)if(texture)texture.needsUpdate=true;
    this.mesh?.updateVersion();
    if(frame.joints){const vertices:number[]=[];for(let j=1;j<this.asset.meta.nBones;j++){const parent=a.parents[j];vertices.push(...Array.from(frame.joints.slice(parent*3,parent*3+3)) as number[],...Array.from(frame.joints.slice(j*3,j*3+3)) as number[]);}this.skeleton.geometry.dispose();this.skeleton.geometry=new THREE.BufferGeometry();this.skeleton.geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));}
  }
  showTargets(show:boolean){this.targets.visible=show;if(!show)this.gizmo.detach();}
  visiblePixels(){
    this.renderer.render(this.scene,this.camera);
    const gl=this.renderer.getContext(),w=gl.drawingBufferWidth,h=gl.drawingBufferHeight,pixels=new Uint8Array(w*h*4);
    gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
    let count=0;for(let y=Math.floor(h*0.35);y<h*0.9;y++)for(let x=Math.floor(w*0.2);x<w*0.8;x++){const i=(y*w+x)*4;if(Math.max(pixels[i],pixels[i+1],pixels[i+2])>85)count++;}
    return count;
  }
  setTargets(targets:Record<string,number[]>){for(const name of ['head','left','right'])if(targets[name])this.targetObjects[name].position.fromArray(targets[name]);}
  home(){this.controls.target.copy(this.homeTarget);this.camera.position.copy(this.homeTarget).add(new THREE.Vector3(0,0.12,this.homeDistance));this.controls.update();}
}
