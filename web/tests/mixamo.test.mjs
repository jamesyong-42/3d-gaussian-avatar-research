import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createSyntheticAsset} from '../src/synthetic.mjs';
import {AnimationClip, AnimationMixer, Bone, Group, LoopOnce, Quaternion, QuaternionKeyframeTrack, Vector3, VectorKeyframeTrack} from 'three';
import {decodeAsset, identityPose, forwardKinematics, axisQuat, sampleRecording} from '../src/engine.mjs';
import {MIXAMO_BONES, AIM_CHILD, boneKey, parseMotionFbx, captureBind, makeRetargeter, bakeMixamo, sampleMotion, disposeMotionScene, resetSource} from '../src/mixamo.mjs';

const asset=createSyntheticAsset();
const sampleUrl=new URL('../public/motions/samba.fbx',import.meta.url);
const bufferOf=b=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);

// Independent FK comparison: target joint-to-child directions versus positions
// evaluated by Three.js on the source skeleton, not the retargeter's quaternions.
function directionError(source, pose, alignment) {
  const target=forwardKinematics(asset,pose).positions;
  let max=0,checked=0;
  for(const [key,child] of Object.entries(AIM_CHILD)) {
    const j=Number(key),a=source.bones.get(boneKey(MIXAMO_BONES[j])),b=source.bones.get(boneKey(MIXAMO_BONES[child]));
    if(!a||!b)continue;
    const expected=b.getWorldPosition(new Vector3()).sub(a.getWorldPosition(new Vector3())).applyQuaternion(alignment).normalize();
    const actual=new Vector3().fromArray(target,child*3).sub(new Vector3().fromArray(target,j*3)).normalize();
    max=Math.max(max,expected.angleTo(actual)*180/Math.PI);checked++;
  }
  return {max,checked};
}

function syntheticSource({units=100,rotated=true}={}) {
  const scene=new Group(),bones=new Map(),nodes=[],globalAxes=[];
  scene.scale.setScalar(units);
  if(rotated)scene.quaternion.setFromAxisAngle(new Vector3(1,0,0),Math.PI/2);
  // Give every source bone an unrelated local axis system while preserving
  // the same physical rest-joint positions. Blindly copying local quats fails.
  for(let j=0;j<55;j++) {
    if(!MIXAMO_BONES[j])continue;
    const node=new Bone(),parent=asset.arrays.parents[j];node.name=`mixamorig${MIXAMO_BONES[j]}`;
    const axes=new Quaternion().setFromAxisAngle(new Vector3(1,j%3+1,2).normalize(),0.17*(j%7));
    globalAxes[j]=axes;
    const pos=new Vector3().fromArray(asset.arrays.joints,j*3);
    if(parent>=0){pos.sub(new Vector3().fromArray(asset.arrays.joints,parent*3)).applyQuaternion(globalAxes[parent].clone().invert());node.quaternion.copy(globalAxes[parent]).invert().multiply(axes);nodes[parent].add(node);}
    else {node.quaternion.copy(axes);scene.add(node);}
    node.position.copy(pos);nodes[j]=node;bones.set(boneKey(node.name),node);
  }
  scene.updateMatrixWorld(true);
  const transforms=[];scene.traverse(node=>transforms.push({node,position:node.position.clone(),rotation:node.quaternion.clone(),scale:node.scale.clone()}));
  const source={scene,bones,transforms,name:'synthetic.fbx',hasSkin:false,clips:[]};source.bind=captureBind(source);
  const hip=nodes[0],forearm=nodes[18];
  source.clips=[new AnimationClip('test',1,[
    new VectorKeyframeTrack(`${hip.name}.position`,[0,1],[...hip.position.toArray(),...hip.position.clone().add(new Vector3(.12,.3,.4)).toArray()]),
    new QuaternionKeyframeTrack(`${forearm.name}.quaternion`,[0,1],[...forearm.quaternion.toArray(),...forearm.quaternion.clone().multiply(new Quaternion().setFromAxisAngle(new Vector3(0,1,0),.7)).toArray()]),
  ])];
  return source;
}

test('real Mixamo FBX maps 52 joints and matches independent source FK throughout the clip', {skip:!fs.existsSync(sampleUrl)&&'Run scripts/setup_mixamo_samples.ps1'}, async()=>{
  const source=parseMotionFbx(bufferOf(fs.readFileSync(sampleUrl)),'Mixamo Samba.fbx');
  try {
    assert.equal(source.hasSkin,true);assert.equal(source.clips.length,1); // Empty Take001 is ignored.
    const [motion]=await bakeMixamo(asset,source);
    assert.equal(motion.mapped.filter(Boolean).length,52);assert.equal(motion.rotations.length,motion.count*165);
    assert.ok(motion.rotations.every(Number.isFinite));assert.ok(motion.translations.every(Number.isFinite));
    assert.ok(motion.duration>18&&motion.duration<19);assert.deepEqual(motion.warnings,[]);
    const converter=makeRetargeter(asset,source),mixer=new AnimationMixer(source.scene);
    const action=mixer.clipAction(source.clips[0]);action.setLoop(LoopOnce,1);action.clampWhenFinished=true;action.play();
    let max=0,comparisons=0;
    for(let frame=0;frame<motion.count;frame+=11) {
      const time=frame*motion.step;mixer.setTime(time);source.scene.updateMatrixWorld(true);
      const pose=sampleMotion(motion,time,identityPose(),{loop:false});
      const error=directionError(source,pose,converter.alignment);max=Math.max(max,error.max);comparisons+=error.checked;
    }
    mixer.stopAllAction();mixer.uncacheRoot(source.scene);resetSource(source);
    console.log('MIXAMO_FK',JSON.stringify({clip:motion.name,duration:motion.duration,samples:motion.count,mapped:52,scale:motion.scale,comparisons,maxDirectionErrorDeg:max}));
    assert.ok(max<0.02,`Source/target direction mismatch: ${max} degrees`);
  } finally {disposeMotionScene(source.scene);}
});

test('retargeting corrects unrelated bone axes, Z-up containers, and centimetre units',async()=>{
  const source=syntheticSource(),converter=makeRetargeter(asset,source),[motion]=await bakeMixamo(asset,source);
  assert.ok(Math.abs(motion.scale-.01)<1e-9);
  const mixer=new AnimationMixer(source.scene),action=mixer.clipAction(source.clips[0]);action.setLoop(LoopOnce,1);action.clampWhenFinished=true;action.play();
  for(const time of [0,.2,.5,1]) {
    mixer.setTime(time);source.scene.updateMatrixWorld(true);
    const pose=sampleMotion(motion,time,identityPose(),{loop:false,inPlace:false});
    assert.ok(directionError(source,pose,converter.alignment).max<.02);
    const delta=source.bones.get('hips').getWorldPosition(new Vector3()).sub(source.bind.get('hips').position).applyQuaternion(converter.alignment).multiplyScalar(converter.scale);
    pose.rootPosition.forEach((x,i)=>assert.ok(Math.abs(x-delta.toArray()[i])<1e-6));
  }
  mixer.stopAllAction();mixer.uncacheRoot(source.scene);
});

test('matching reference recovers motion when an animation-only node rest pose is wrong',async()=>{
  const reference=syntheticSource(),source=syntheticSource();
  // A file may store its first animated pose as its initial node transforms.
  source.bones.get('leftforearm').rotateX(.8);source.bind=captureBind(source);
  source.transforms.find(t=>t.node===source.bones.get('leftforearm')).rotation.copy(source.bones.get('leftforearm').quaternion);
  const [expected]=await bakeMixamo(asset,reference),[recovered]=await bakeMixamo(asset,source,{reference});
  assert.ok(recovered.warnings.some(x=>x.includes('same source character')));
  expected.rotations.forEach((x,i)=>assert.ok(Math.abs(x-recovered.rotations[i])<1e-6));
});

test('root travel can be disabled; actor orientation and all face channels stay independent',async()=>{
  const [motion]=await bakeMixamo(asset,syntheticSource());
  const base=identityPose();base.rootPosition=[2,3,4];base.rootRotation=axisQuat(0,.8,0);
  base.expression[99]=.67;base.angles[66]=.25;base.angles[70]=.1;base.angles[73]=-.1;
  const inPlace=sampleMotion(motion,.6,base),travel=sampleMotion(motion,.6,base,{inPlace:false});
  assert.equal(inPlace.rootPosition[0],2);assert.equal(inPlace.rootPosition[2],4);assert.notEqual(inPlace.rootPosition[1],3);
  assert.ok(Math.abs(travel.rootPosition[0]-2)+Math.abs(travel.rootPosition[2]-4)>.1);
  const displacement=new Vector3().fromArray(motion.translations,18*3).applyQuaternion(new Quaternion().fromArray(base.rootRotation)).add(new Vector3().fromArray(base.rootPosition));
  travel.rootPosition.forEach((v,i)=>assert.ok(Math.abs(v-displacement.toArray()[i])<1e-6));
  assert.deepEqual(travel.rootRotation,base.rootRotation);assert.deepEqual(travel.expression,base.expression);
  assert.deepEqual(travel.angles.slice(66,75),base.angles.slice(66,75));
  assert.deepEqual(base.rootPosition,[2,3,4]);
});

test('motion sampling clamps, loops, and interpolates rotations along the short arc',()=>{
  const rotations=new Float32Array(330);rotations[0]=3.1;rotations[165]=-3.1;
  const motion={duration:1,count:2,step:1,rotations,translations:new Float32Array([0,0,0,1,2,3]),mapped:Array.from({length:55},(_,j)=>j===0)};
  assert.ok(Math.abs(sampleMotion(motion,.5,identityPose()).angles[0])>3);
  assert.deepEqual(sampleMotion(motion,2,identityPose(),{loop:false,inPlace:false}).rootPosition,[1,2,3]);
  assert.deepEqual(sampleMotion(motion,1,identityPose(),{loop:true,inPlace:false}).rootPosition,[0,0,0]);
  assert.deepEqual(sampleMotion(motion,-1,identityPose(),{loop:false,inPlace:false}).rootPosition,[0,0,0]);
});

test('motion import rejects invalid input and unsupported deformations with clear errors',async()=>{
  assert.equal(boneKey('Armature|mixamorig:LeftHandIndex1'),'lefthandindex1');assert.equal(boneKey('mixamorig1LeftArm'),'leftarm');
  assert.throws(()=>parseMotionFbx(new ArrayBuffer(8)),/valid FBX/);
  assert.throws(()=>parseMotionFbx(new ArrayBuffer(100)),/Could not read/);
  const missing=syntheticSource();missing.bind.delete('head');assert.throws(()=>makeRetargeter(asset,missing),/missing Head/);
  const scaled=syntheticSource();scaled.scene.scale.y*=2;assert.throws(()=>captureBind(scaled),/Non-uniform/);
  const invalidBind=syntheticSource();invalidBind.bones.get('lefthandthumb3').position.x=NaN;assert.throws(()=>captureBind(invalidBind),/invalid FBX bind/);
  const empty=syntheticSource();empty.clips=[];await assert.rejects(bakeMixamo(asset,empty),/no animation/);
  const bad=syntheticSource();bad.clips[0].tracks[0].values[0]=NaN;await assert.rejects(bakeMixamo(asset,bad),/non-finite keyframes/);
  const unordered=syntheticSource();unordered.clips[0].tracks[0].times.set([1,0]);await assert.rejects(bakeMixamo(asset,unordered),/not ordered/);
  const tooLong=syntheticSource();tooLong.clips[0].duration=301;await assert.rejects(bakeMixamo(asset,tooLong),/5 minutes/);
  await assert.rejects(bakeMixamo(asset,syntheticSource(),{fps:Infinity}),/Bake rate/);
  const stretchy=syntheticSource();stretchy.clips[0].tracks.push(new VectorKeyframeTrack('mixamorigLeftArm.scale',[0,1],[1,1,1,2,2,2]));
  await assert.rejects(bakeMixamo(asset,stretchy),/Animated scale/);
  const offset=syntheticSource();offset.clips[0].tracks.push(new VectorKeyframeTrack('mixamorigLeftArm.position',[0,1],[0,0,0,1,0,0]));
  await assert.rejects(bakeMixamo(asset,offset),/Only hips/);
});

test('recording replay holds discontinuities and never extrapolates before its first sample',()=>{
  const a=identityPose(),b=identityPose(),c=identityPose();a.rootPosition[0]=1;b.rootPosition[0]=10;b.teleport=true;c.rootPosition[0]=12;
  const frames=[{time:30,pose:a},{time:100,pose:b},{time:200,pose:c}];
  assert.equal(sampleRecording(frames,0).rootPosition[0],1);
  assert.equal(sampleRecording(frames,99).rootPosition[0],1);
  assert.equal(sampleRecording(frames,100).rootPosition[0],10);
  assert.equal(sampleRecording(frames,150).rootPosition[0],11);
  assert.equal(sampleRecording(frames,300).rootPosition[0],12);
});
