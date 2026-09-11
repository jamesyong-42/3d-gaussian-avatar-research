import {AnimationMixer, LoadingManager, LoopOnce, Matrix4, Quaternion, Texture, Vector3} from 'three';
import {FBXLoader} from 'three/addons/loaders/FBXLoader.js';
import {copyPose, identityPose, quatAxis, slerp, axisQuat, quatMatrix} from './engine.mjs';

export const MIXAMO_BONES = [
  'Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg', 'RightLeg',
  'Spine1', 'LeftFoot', 'RightFoot', 'Spine2', 'LeftToeBase', 'RightToeBase',
  'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm',
  'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand', null, null, null,
  ...['Left', 'Right'].flatMap(side => ['Index', 'Middle', 'Pinky', 'Ring', 'Thumb']
    .flatMap(finger => [1, 2, 3].map(n => `${side}Hand${finger}${n}`))),
];
export const AIM_CHILD = {0:3, 1:4, 2:5, 3:6, 4:7, 5:8, 6:9, 7:10, 8:11,
  9:12, 12:15, 13:16, 14:17, 16:18, 17:19, 18:20, 19:21, 20:28, 21:43};
for (let j=25; j<55; j++) if ((j-25)%3!==2) AIM_CHILD[j]=j+1;
const REQUIRED = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 18, 19, 20, 21];
export const MAX_FBX_BYTES = 64*1024*1024;

export function boneKey(name) {
  return name.split(/[|:]/).pop().replace(/^mixamorig\d*[_-]?/i, '').toLowerCase();
}

// A compatibility key, not a security checksum. Includes actual target rest
// positions and hierarchy, so clips cannot silently bind to different proportions.
export function rigKey(asset) {
  return JSON.stringify([asset.meta.nBones, Array.from(asset.arrays.parents), Array.from(asset.arrays.joints)]);
}

export function parseMotionFbx(buffer, name='animation.fbx') {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength<24 || buffer.byteLength>MAX_FBX_BYTES)
    throw new Error('Choose a valid FBX file smaller than 64 MB.');
  // Only motion is used. Ignore every texture, including absolute paths in FBX.
  // This also allows FBX parsing inside a worker without an Image/document API.
  const manager = new LoadingManager();
  manager.addHandler(/.*/, {path:'', setPath(path){this.path=path;return this;}, load(){return new Texture();}});
  manager.setURLModifier(() => {throw new Error('External resources are disabled for motion imports.');});
  const originalWindow = globalThis.window;
  if (typeof originalWindow==='undefined') globalThis.window={URL:{createObjectURL:()=> 'data:image/png;base64,'}};
  let scene;
  try {scene = new FBXLoader(manager).parse(buffer, '');}
  catch (error) {throw new Error(`Could not read ${name}: ${error.message}`);}
  finally {if (typeof originalWindow==='undefined') delete globalThis.window;}
  scene.updateMatrixWorld(true);
  const bones = new Map(), transforms = [];
  scene.traverse(node => {
    transforms.push({node, position:node.position.clone(), rotation:node.quaternion.clone(), scale:node.scale.clone()});
    if (!node.isBone) return;
    const key=boneKey(node.name);
    if (bones.has(key)) {
      const original=bones.get(key);
      // FBXLoader can attach an identity clone for a second skin cluster. It
      // shares the FBX ID and follows the animated parent; it is not a new rig.
      if(node.parent===original && node.ID!==undefined && node.ID===original.ID
        && node.children.length===0 && node.position.length()<1e-6
        && node.quaternion.angleTo(new Quaternion())<1e-6
        && node.scale.distanceTo(new Vector3(1,1,1))<1e-6)return;
      throw new Error(`Ambiguous skeleton: duplicate bone ${node.name}. Import one character per FBX.`);
    }
    bones.set(key,node);
  });
  if (bones.size>512) {disposeMotionScene(scene);throw new Error('FBX contains too many bones for a humanoid motion.');}
  const missing=REQUIRED.filter(j=>!bones.has(boneKey(MIXAMO_BONES[j]))).map(j=>MIXAMO_BONES[j]);
  if (missing.length) {disposeMotionScene(scene);throw new Error(`Not a supported Mixamo skeleton. Missing: ${missing.join(', ')}.`);}
  const source={scene,bones,transforms,name,clips:(scene.animations??[]).filter(c=>c.tracks.length&&c.duration!==0)};
  source.bind=captureBind(source);
  source.hasSkin=false;scene.traverse(n=>{if(n.isSkinnedMesh)source.hasSkin=true;});
  return source;
}

export function captureBind(source) {
  source.scene.updateMatrixWorld(true);
  const bind=new Map();
  for (const [name,node] of source.bones) {
    const scale=new Vector3(),position=new Vector3(),rotation=new Quaternion();
    node.matrixWorld.decompose(position,rotation,scale);
    if(![...position.toArray(),...rotation.toArray()].every(Number.isFinite)||rotation.length()<1e-8)throw new Error('Non-finite or invalid FBX bind transform.');
    if ([scale.x,scale.y,scale.z].some(x=>x<=0 || !Number.isFinite(x))) throw new Error('Mirrored or invalid FBX skeleton scale is unsupported.');
    if (Math.max(scale.x,scale.y,scale.z)/Math.min(scale.x,scale.y,scale.z)>1.01) throw new Error('Non-uniformly scaled skeletons must be baked before import.');
    bind.set(name,{position,rotation:rotation.normalize()});
  }
  return bind;
}

export function resetSource(source) {
  for(const t of source.transforms) {t.node.position.copy(t.position);t.node.quaternion.copy(t.rotation);t.node.scale.copy(t.scale);}
  source.scene.updateMatrixWorld(true);
}

export function disposeMotionScene(scene) {
  scene.traverse(node=>{
    node.geometry?.dispose();
    for(const material of (Array.isArray(node.material)?node.material:[node.material])) if(material) {
      for(const value of Object.values(material)) if(value?.isTexture)value.dispose();
      material.dispose();
    }
    node.skeleton?.dispose();
  });
}

const keyFor=j=>MIXAMO_BONES[j]&&boneKey(MIXAMO_BONES[j]);
const point=(asset,j)=>new Vector3().fromArray(asset.arrays.joints,j*3);
const distance=(bind,a,b)=>bind.get(keyFor(a)).position.distanceTo(bind.get(keyFor(b)).position);

export function makeRetargeter(asset, source, reference) {
  if(asset.meta.nBones!==55)throw new Error('Mixamo conversion currently targets the 55-joint SMPL-X rig.');
  const bind=reference?.bind??source.bind;
  for(const j of REQUIRED)if(!bind.has(keyFor(j)))throw new Error(`Reference is missing ${MIXAMO_BONES[j]}.`);
  // Define source anatomical axes from its bind stance, independent of FBX's
  // bone local axes, units, container rotation, and Y-up/Z-up file convention.
  const x=bind.get('leftupleg').position.clone().sub(bind.get('rightupleg').position).normalize();
  const y=bind.get('head').position.clone().sub(bind.get('hips').position);
  y.addScaledVector(x,-y.dot(x)).normalize();
  const z=new Vector3().crossVectors(x,y).normalize();
  if(x.length()<0.9 || y.length()<0.9 || z.length()<0.9)throw new Error('Source bind stance is degenerate; provide a matching T-pose FBX.');
  const basis=new Matrix4().makeBasis(x,y,z),alignment=new Quaternion().setFromRotationMatrix(basis).invert();
  const alignmentInverse=alignment.clone().invert();
  const sourceLeg=(distance(bind,1,4)+distance(bind,4,7)+distance(bind,2,5)+distance(bind,5,8))/2;
  const targetLeg=(point(asset,1).distanceTo(point(asset,4))+point(asset,4).distanceTo(point(asset,7))
    +point(asset,2).distanceTo(point(asset,5))+point(asset,5).distanceTo(point(asset,8)))/2;
  const scale=targetLeg/sourceLeg;
  if(!Number.isFinite(scale)||scale<1e-6||scale>1e3)throw new Error('Invalid source skeleton proportions.');
  const mapped=MIXAMO_BONES.map((_,j)=>source.bones.has(keyFor(j))&&bind.has(keyFor(j)));
  const stance=Array.from({length:55},()=>new Quaternion());
  for(let j=0;j<55;j++) {
    const child=AIM_CHILD[j];
    if(!mapped[j])continue;
    if(child!==undefined && mapped[child]) {
      const from=point(asset,child).sub(point(asset,j)).normalize();
      const to=bind.get(keyFor(child)).position.clone().sub(bind.get(keyFor(j)).position).applyQuaternion(alignment).normalize();
      if(from.length()>0.9&&to.length()>0.9)stance[j].setFromUnitVectors(from,to);
    } else if(asset.arrays.parents[j]>=0)stance[j].copy(stance[asset.arrays.parents[j]]);
  }
  const globals=Array.from({length:55},()=>new Quaternion());
  const referenceHip=bind.get('hips').position.clone();
  const sourceHip=source.bones.get('hips');
  let firstHip;
  return {
    scale,alignment,stance,mapped,source,bind,
    sample() {
      source.scene.updateMatrixWorld(true);
      const pose=identityPose(55,asset.meta.nExpressions);
      for(let j=0;j<55;j++) {
        const parent=asset.arrays.parents[j],parentQ=parent<0?new Quaternion():globals[parent];
        if(mapped[j]) {
          const current=source.bones.get(keyFor(j)).getWorldQuaternion(new Quaternion());
          // Target world rotation = aligned source world delta × target stance.
          globals[j].copy(alignment).multiply(current).multiply(bind.get(keyFor(j)).rotation.clone().invert())
            .multiply(alignmentInverse).multiply(stance[j]).normalize();
          const local=parentQ.clone().invert().multiply(globals[j]).normalize();
          pose.angles.set(quatAxis(local.toArray()),j*3);
        } else globals[j].copy(parentQ);
      }
      const hip=sourceHip.getWorldPosition(new Vector3()).sub(referenceHip).applyQuaternion(alignment).multiplyScalar(scale);
      firstHip??=hip.clone();
      // Center horizontal travel at the first frame; preserve vertical motion
      // relative to bind (otherwise crouching clips start artificially raised).
      pose.rootPosition=[hip.x-firstHip.x,hip.y,hip.z-firstHip.z];
      return pose;
    },
  };
}

export async function bakeMixamo(asset,source,{reference,fps=30,onProgress=()=>{},yieldTask=()=>Promise.resolve()}={}) {
  if(!Number.isFinite(fps)||fps<1||fps>60)throw new Error('Bake rate must be between 1 and 60 frames per second.');
  if(!source.clips.length)throw new Error('This FBX has a skeleton but no animation. Use it as a T-pose reference, then import a motion FBX.');
  if(source.clips.length>16 || source.clips.reduce((sum,c)=>sum+c.duration,0)>300)throw new Error('Import at most 16 clips / 5 minutes of motion at once.');
  const results=[];
  for(let c=0;c<source.clips.length;c++) {
    const original=source.clips[c];
    if(!Number.isFinite(original.duration)||original.duration<=0)throw new Error('Animation has an invalid duration.');
    for(const track of original.tracks) {
      if(!track.values.every(Number.isFinite)||!track.times.every(Number.isFinite))throw new Error('Animation contains non-finite keyframes.');
      for(let i=1;i<track.times.length;i++)if(track.times[i]<track.times[i-1])throw new Error('Animation keyframes are not ordered.');
      const size=track.getValueSize(),varies=track.values.some((v,i)=>Math.abs(v-track.values[i%size])>1e-5);
      if(varies&&track.name.endsWith('.scale'))throw new Error('Animated scale is unsupported. Bake a rotation-only skeleton with hips translation.');
      if(varies&&track.name.endsWith('.position')) {
        const name=track.name.slice(0,-9),node=source.scene.getObjectByName(name);
        const ancestors=new Set();for(let n=source.bones.get('hips');n;n=n.parent)ancestors.add(n);
        if(!node||!ancestors.has(node))throw new Error(`Animated translation on ${name} is unsupported. Only hips / root translation can be retargeted.`);
      }
    }
    resetSource(source);
    const converter=makeRetargeter(asset,source,reference),mixer=new AnimationMixer(source.scene);
    const action=mixer.clipAction(original);action.setLoop(LoopOnce,1);action.clampWhenFinished=true;action.play();
    const count=Math.ceil(original.duration*fps)+1,step=original.duration/(count-1);
    const rotations=new Float32Array(count*165),translations=new Float32Array(count*3);
    try {
      for(let i=0;i<count;i++) {
        mixer.setTime(Math.min(original.duration,i*step));
        const pose=converter.sample();rotations.set(pose.angles,i*165);translations.set(pose.rootPosition,i*3);
        if(i%90===0){onProgress((c+i/count)/source.clips.length);await yieldTask();}
      }
    } finally {mixer.stopAllAction();mixer.uncacheRoot(source.scene);resetSource(source);}
    const filename=source.name.replace(/\.fbx$/i,''),name=source.clips.length===1?filename:`${filename} / ${original.name||c+1}`;
    const warnings=[];
    const missing=MIXAMO_BONES.filter((name,j)=>name&&!converter.mapped[j]);
    if(missing.length)warnings.push(`Unmapped bones keep the base pose: ${missing.join(', ')}`);
    if(!source.hasSkin&&!reference)warnings.push('Using the FBX node rest pose. If limbs look wrong, supply a T-pose FBX from the same Mixamo character and reimport.');
    if(reference)warnings.push(`Reference: ${reference.name}. It must be from the same source character.`);
    results.push({version:1,kind:'smplx-motion',name,duration:original.duration,fps,count,step,
      rotations,translations,mapped:converter.mapped,scale:converter.scale,warnings,rigKey:rigKey(asset),
      reference:reference?.name??(source.hasSkin?'FBX skin bind pose':'FBX node rest pose')});
  }
  onProgress(1);return results;
}

export function sampleMotion(motion,time,base,{inPlace=true,loop=true}={}) {
  const t=loop?((time%motion.duration)+motion.duration)%motion.duration:Math.max(0,Math.min(motion.duration,time));
  const f=t/motion.step,a=Math.min(motion.count-1,Math.floor(f)),b=Math.min(a+1,motion.count-1),u=Math.min(1,f-a);
  const pose=copyPose(base);
  for(let j=0;j<55;j++)if(motion.mapped[j]) {
    const qa=axisQuat(...motion.rotations.subarray(a*165+j*3,a*165+j*3+3));
    const qb=axisQuat(...motion.rotations.subarray(b*165+j*3,b*165+j*3+3));
    pose.angles.set(quatAxis(slerp(qa,qb,u)),j*3);
  }
  const displacement=[0,1,2].map(i=>motion.translations[a*3+i]+(motion.translations[b*3+i]-motion.translations[a*3+i])*u);
  if(inPlace){displacement[0]=0;displacement[2]=0;}
  const r=quatMatrix(base.rootRotation);
  pose.rootPosition=base.rootPosition.map((v,i)=>v+r[i*3]*displacement[0]+r[i*3+1]*displacement[1]+r[i*3+2]*displacement[2]);
  return pose;
}
