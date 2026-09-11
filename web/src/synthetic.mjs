// Original, deterministic geometry: no photos, scans, body-model templates,
// trained weights or third-party motion data are used by this demo.
import {decodeAsset} from './engine.mjs';

export const JOINT_NAMES = [
  'pelvis','left_hip','right_hip','spine1','left_knee','right_knee',
  'spine2','left_ankle','right_ankle','spine3','left_foot','right_foot',
  'neck','left_collar','right_collar','head','left_shoulder','right_shoulder',
  'left_elbow','right_elbow','left_wrist','right_wrist','jaw','left_eye','right_eye',
  ...['left','right'].flatMap(side => ['index','middle','pinky','ring','thumb'].flatMap(f => [1,2,3].map(j => `${side}_${f}${j}`))),
];

export function createSyntheticPackage({rings = 12, segments = 16} = {}) {
  if (!Number.isInteger(rings) || rings < 3 || rings > 40 || !Number.isInteger(segments) || segments < 6 || segments > 64) throw Error('Invalid synthetic resolution');
  const joints = [
    [0,0,0],[.09,-.08,0],[-.09,-.08,0],[0,.12,0],[.10,-.46,.025],[-.10,-.46,.025],
    [0,.25,0],[.10,-.87,0],[-.10,-.87,0],[0,.39,0],[.10,-.93,.13],[-.10,-.93,.13],
    [0,.54,0],[.07,.43,0],[-.07,.43,0],[0,.68,0],[.19,.43,0],[-.19,.43,0],
    [.42,.22,0],[-.42,.22,0],[.61,.02,0],[-.61,.02,0],[0,.635,.06],[.035,.705,.08],[-.035,.705,.08],
  ];
  const parents = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,15,15,15];
  for (const side of [1,-1]) for (let finger=0; finger<5; finger++) {
    const wrist=side===1?20:21, start=joints.length;
    for (let part=0; part<3; part++) {
      joints.push([side*(.63+part*.026),.02+(finger-2)*.019,.015+(finger===4?.012:0)]);
      parents.push(part===0?wrist:start+part-1);
    }
  }
  const positions=[], colors=[], scales=[], bones=[];
  function ellipsoid(center, radius, bone, color, count=rings, around=segments) {
    for (let r=0;r<count;r++) {
      const theta=Math.PI*(r+.5)/count;
      for (let s=0;s<around;s++) {
        const phi=2*Math.PI*(s+.5*(r%2))/around;
        positions.push(center[0]+radius[0]*Math.sin(theta)*Math.cos(phi), center[1]+radius[1]*Math.cos(theta), center[2]+radius[2]*Math.sin(theta)*Math.sin(phi));
        colors.push(...color);bones.push(bone);
        const size=Math.max(.006,Math.min(.025,Math.max(...radius)*1.8/count));
        scales.push(size,size,size);
      }
    }
  }
  const mint=[.25,.8,.63], blue=[.27,.49,.88], amber=[.95,.67,.29];
  ellipsoid([0,.22,0],[.17,.30,.095],3,mint,20,28);
  ellipsoid([0,-.04,0],[.16,.14,.10],0,blue,12,22);
  ellipsoid([0,.69,0],[.105,.14,.105],15,amber,16,24);
  ellipsoid([0,.52,0],[.055,.075,.055],12,mint,8,12);
  for (const end of [4,5,7,8,10,11,18,19,20,21]) {
    const start=parents[end], a=joints[start], b=joints[end], leg=end<=11;
    const steps=leg?9:7, radius=leg?.047:.035;
    for (let k=0;k<steps;k++) {
      const t=(k+.5)/steps, center=a.map((v,i)=>v*(1-t)+b[i]*t);
      ellipsoid(center,[radius,radius,radius],start,leg?blue:mint,4,8);
    }
  }
  for (const wrist of [20,21]) ellipsoid(joints[wrist],[.047,.039,.025],wrist,amber,7,12);
  for(let j=25;j<55;j++) ellipsoid(joints[j],[.014,.009,.009],j,amber,3,6);
  const n=bones.length, rotations=new Float32Array(n*4), neutral=new Float32Array(n*9);
  for(let i=0;i<n;i++){rotations[i*4+3]=1;neutral.set([1,0,0,0,1,0,0,0,1],i*9);}
  const data={positions:new Float32Array(positions),rotations,scales:new Float32Array(scales),colors:new Float32Array(colors),
    opacities:new Float32Array(n).fill(.9),neutralLinear:neutral,rotationLocked:new Uint8Array(n),regions:new Uint8Array(n),
    joints:new Float32Array(joints.flat()),parents:new Int32Array(parents),betas:new Float32Array(10),
    skinOffsets:Uint32Array.from({length:n+1},(_,i)=>i),skinBones:new Uint16Array(bones),skinWeights:new Float32Array(n).fill(1),
    expressionIndices:new Uint32Array(0),expressionDirections:new Float32Array(0)};
  const types=new Map([[Float32Array,'<f4'],[Uint32Array,'<u4'],[Int32Array,'<i4'],[Uint16Array,'<u2'],[Uint8Array,'|u1']]);
  const arrays={};let bytes=0;
  for(const [name,value] of Object.entries(data)){bytes=Math.ceil(bytes/4)*4;arrays[name]={offset:bytes,bytes:value.byteLength,shape:[value.length],dtype:types.get(value.constructor)};bytes+=value.byteLength;}
  const buffer=new ArrayBuffer(bytes);
  for(const [name,value] of Object.entries(data))new Uint8Array(buffer,arrays[name].offset,value.byteLength).set(new Uint8Array(value.buffer,value.byteOffset,value.byteLength));
  const meta={version:2,model:'Synthetic demo · not photo-generated',label:'Synthetic demo / no source photos',rig:'synthetic-55',
    deformation:'lhm-linear-blend-v1',skinning:'all-nonzero-float32',nGaussians:n,nBones:55,nExpressions:100,nExpressionVertices:0,
    expressionStorage:'sparse-vertices-v1',jointNames:JOINT_NAMES,arrays,bytes,binary:'avatar.bin',influences:n,
    createdAt:'2026-09-11T00:00:00Z',referencePoses:[],
    coordinates:'Right-handed, Y-up, meters; local axis-angle radians; xyzw quaternions',
    generation:{engine:'synthetic-demo',shape:{mode:'synthetic'},description:'Original analytic shapes. No photos, model weights, templates or native reference data.'}};
  return {meta,buffer};
}

export function createSyntheticAsset(options) {
  const {meta,buffer}=createSyntheticPackage(options);
  return decodeAsset(meta,buffer);
}
