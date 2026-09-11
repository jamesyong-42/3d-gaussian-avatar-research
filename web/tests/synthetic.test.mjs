import test from 'node:test';
import assert from 'node:assert/strict';
import {createSyntheticPackage,createSyntheticAsset,JOINT_NAMES} from '../src/synthetic.mjs';
import {deform,identityPose} from '../src/engine.mjs';
import {applyClip} from '../src/controls.mjs';

test('demo is deterministic, explicitly synthetic and carries no native references',()=>{
  const a=createSyntheticPackage(),b=createSyntheticPackage();
  assert.deepEqual(a,b);assert.equal(a.meta.generation.engine,'synthetic-demo');
  assert.equal(a.meta.referencePoses.length,0);assert.equal(JOINT_NAMES.length,55);
  assert.ok(a.meta.nGaussians>1000);assert.ok(a.meta.bytes<2_000_000);
  assert.equal(a.meta.nExpressionVertices,0);
});

test('synthetic rig is finite and responds to the same animation contract',()=>{
  const asset=createSyntheticAsset(),pose=identityPose();
  const rest=deform(asset,pose);pose.angles[16*3+2]=.8;
  const moved=deform(asset,pose);
  assert.ok(moved.positions.every(Number.isFinite));
  assert.ok(moved.rotations.every(Number.isFinite));
  assert.notDeepEqual(moved.positions,rest.positions);
  assert.throws(()=>createSyntheticPackage({rings:100000}),/resolution/);
});
