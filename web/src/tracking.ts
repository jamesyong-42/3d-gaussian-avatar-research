import * as THREE from 'three';
import {copyPose,forwardKinematics,quatAxis} from './engine.mjs';

export class CameraDriver{
  worker?:Worker;stream?:MediaStream;ready=false;busy=false;starting=false;generation=0;last=0;signals:any;ms=0;calibration:any;
  onStatus:(message:string)=>void=()=>{};
  constructor(public video:HTMLVideoElement){}
  async start(){
    if(this.starting||this.stream)return;const generation=++this.generation;this.starting=true;
    try{
    this.onStatus('Requesting camera access…');
    const stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'user'},audio:false});
    if(generation!==this.generation){stream.getTracks().forEach(track=>track.stop());return;}this.stream=stream;
    this.video.srcObject=this.stream;await this.video.play();
    if(generation!==this.generation)return;
    this.onStatus('Loading local tracking models…');
    this.worker=new Worker(new URL('./tracking.worker.ts',import.meta.url),{type:'module'});
    this.worker.onmessage=({data})=>{this.busy=false;if(data.type==='ready'){this.ready=true;this.onStatus('Tracking active · stand back to show your body');}else if(data.type==='signals'){this.signals=data;this.last=performance.now();this.ms=data.ms;this.onStatus(`Body ${data.body.length?'✓':'—'} · Face ${data.face.length?'✓':'—'} · ${data.hands.length} hands · ${Math.round(data.ms)} ms${this.calibration?' · heading calibrated':''}`);}else{this.onStatus(data.message);this.ready=false;}};
    this.worker.onerror=e=>{this.busy=false;this.ready=false;this.onStatus(`Tracking worker failed: ${e.message}`);};
    this.worker.postMessage({type:'init'});
    }finally{if(generation===this.generation)this.starting=false;}
  }
  stop(){++this.generation;this.starting=false;this.stream?.getTracks().forEach(track=>track.stop());this.worker?.terminate();this.worker=undefined;this.stream=undefined;this.video.srcObject=null;this.ready=false;this.busy=false;this.signals=null;this.calibration=null;this.ms=0;}
  calibrate(){if(this.signals?.body?.length){this.calibration=structuredClone(this.signals);this.onStatus('Neutral shoulder heading captured · translation stays fixed');}else this.onStatus('Show shoulders and hips to calibrate.');}
  async capture(now:number){
    if(!this.ready||this.busy||!this.video.videoWidth||now-this.last<85)return;
    this.busy=true;
    try{const bitmap=await createImageBitmap(this.video);if(this.worker)this.worker.postMessage({type:'frame',bitmap,time:now},[bitmap]);else bitmap.close();}catch{this.busy=false;}
  }
  apply(asset:any,base:any){
    const p=copyPose(base),s=this.signals;
    // Confidence dropouts decay to the base pose instead of extrapolating a limb.
    if(!s||performance.now()-this.last>500)return p;
    const points=s.body;
    if(points.length===33){
      const convert=(i:number)=>new THREE.Vector3(points[i].x,-points[i].y,-points[i].z);
      const pairs=[[1,4,23,25],[2,5,24,26],[4,7,25,27],[5,8,26,28],[7,10,27,31],[8,11,28,32],[16,18,11,13],[17,19,12,14],[18,20,13,15],[19,21,14,16]];
      const rawReference=this.calibration?.body;
      for(const [j,child,from,to] of pairs){
        if((points[from].visibility??1)<0.55||(points[to].visibility??1)<0.55)continue;
        const fk=forwardKinematics(asset,p),parent=asset.arrays.parents[j];
        const desired=convert(to).sub(convert(from)).normalize(),m=fk.global.subarray(parent*12,parent*12+12);
        const inv=new THREE.Matrix3().set(m[0],m[4],m[8],m[1],m[5],m[9],m[2],m[6],m[10]);desired.applyMatrix3(inv).normalize();
        const joints=asset.arrays.joints,rest=new THREE.Vector3().fromArray(joints,child*3).sub(new THREE.Vector3().fromArray(joints,j*3)).normalize();
        const q=new THREE.Quaternion().setFromUnitVectors(rest,desired);
        p.angles.set(quatAxis(q.toArray()),j*3);
      }
      // Keep hip-relative webcam observations separate from actor world translation.
      if(rawReference){const shoulders=convert(12).sub(convert(11));const ref=new THREE.Vector3(rawReference[12].x-rawReference[11].x,0,-rawReference[12].z+rawReference[11].z);p.angles[1]=Math.atan2(shoulders.z,shoulders.x)-Math.atan2(ref.z,ref.x);}
    }
    const face:Record<string,number>=Object.fromEntries((s.face??[]).map((v:any)=>[v.categoryName,v.score]));
    if(face.jawOpen!==undefined)p.angles[22*3]=face.jawOpen*0.5;
    // Eye rotation channels have anatomical meaning; learned expression basis
    // coefficients are not assumed to be ARKit blendshape indices.
    p.angles[23*3+1]=((face.eyeLookOutLeft??0)-(face.eyeLookInLeft??0))*0.25;
    p.angles[24*3+1]=((face.eyeLookInRight??0)-(face.eyeLookOutRight??0))*0.25;
    if(s.faceMatrix){const m=new THREE.Matrix4().fromArray(s.faceMatrix),q=new THREE.Quaternion();m.decompose(new THREE.Vector3(),q,new THREE.Vector3());const e=new THREE.Euler().setFromQuaternion(q,'YXZ');p.angles[45]=-e.x;p.angles[46]=-e.y;p.angles[47]=e.z;}
    for(let h=0;h<(s.hands??[]).length;h++){
      const marks=s.hands[h],side=s.handedness[h]?.[0]?.categoryName==='Left'?'left':'right',start=side==='left'?25:40,sign=side==='left'?1:-1;
      const fingerStarts=[5,9,17,13,1];
      for(let f=0;f<5;f++)for(let j=0;j<3;j++){
        const b=fingerStarts[f]+j,a=j===0?0:b-1,c=b+1;
        const u=new THREE.Vector3(marks[b].x-marks[a].x,marks[b].y-marks[a].y,marks[b].z-marks[a].z),v=new THREE.Vector3(marks[c].x-marks[b].x,marks[c].y-marks[b].y,marks[c].z-marks[b].z);
        p.angles[(start+f*3+j)*3+2]=sign*Math.min(1.4,u.angleTo(v));
      }
    }
    return p;
  }
}
