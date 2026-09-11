# Mixamo → Gaussian avatar

Mixamo FBX motion now drives the existing LHM Gaussian avatar in the browser.
The avatar does not need to be uploaded to Mixamo or generated again. The source
FBX's character mesh and textures are not rendered; only its skeleton and motion
are used.

## Try it

1. Open http://127.0.0.1:8765 and select the example or your saved avatar.
2. Select **Mixamo** in the control desk.
3. Click **Try the real Mixamo Samba sample**, or **Import FBX motions** and
   select one or more of your own Mixamo downloads.
4. Choose a motion. Playback starts automatically. The play button and speed
   selector are below the viewport; dragging the timeline pauses at that frame.
5. **In place** suppresses horizontal travel but preserves vertical motion.
   Uncheck it to apply source hips/root travel, scaled to the avatar's leg length.
   **Loop** repeats the clip; disabling it holds the final pose at the end.
6. Record, replay, download the resolved poses, or open a receiver as before.

The sample is **Samba**, not a relabeled procedural clip. The Wave / Walk / Squat /
Turn buttons below the viewport still select the original procedural motions.
Actual Mixamo idle, walk, and wave files can be imported through the same path;
those particular files have not yet been supplied or verified.

For a fresh checkout, install the optional sample before building:

```sh
# From the repository root:
node scripts/setup-optional-assets.mjs --mixamo
npm --prefix web run build
```

The setup script downloads and verifies the SHA-256 of the sample pinned to
Three.js **r185**. The [official FBX example](https://threejs.org/examples/webgl_loader_fbx.html)
attributes the character and animation to Mixamo. The source is
[Samba Dancing.fbx in Three.js r185](https://github.com/mrdoob/three.js/blob/r185/examples/models/fbx/Samba%20Dancing.fbx).
The downloaded FBX is excluded from version control. This prototype does not
relicense third-party assets; review the applicable terms before distribution.

## Export and rest-pose requirements

Use Mixamo's FBX export, preferably at 30 fps with no keyframe reduction for the
initial comparison. A download **with skin** is a convenient first test because
it carries the source skeleton's bind information. Animation-only files are
supported when their stored rest transforms or bind-pose data are usable.

The loader uses FBX bind/rest data, not an assumption that the first animated
frame is a T-pose. If an animation-only file has incorrect rest information,
open **Optional T-pose reference**, select a T-pose FBX from the **same source
character**, then reimport the motion. The reference is applied to future imports
only; it does not alter previously baked clips. A reference from a differently
proportioned or oriented character is not a valid substitute.

The adapter recognizes standard Mixamo names with or without a `mixamorig`
namespace. It maps 52 body/finger joints onto the 55-joint SMPL-X hierarchy.
Missing required body joints cause an error. Missing optional joints retain the
base pose and are listed in a warning. Jaw, eyes, and all 100 learned expression
coefficients remain independent; Mixamo is not supplying facial animation here.

## How the motion becomes control signals

```text
Local FBX → parsing / 30 Hz retarget bake in a worker → motion library
                                                         ↓
             selected clip + time + speed + loop + in-place setting
                                                         ↓
                55 local rotations + actor root + face/base channels
                               ├→ Gaussian deformation + rendering
                               ├→ resolved-pose recording / replay
                               └→ existing GSW2 sender → receiver
```

The retargeter preserves the target avatar's rest-joint positions and bone
lengths. It does not copy source joint positions into the avatar. In outline:

- Derive an anatomical coordinate alignment `C` from source hips and head.
- For each joint, calculate a rest-stance correction `A` aligning the target's
  primary bone direction with the aligned source bind direction. This handles
  different A/T stances. Terminal bones inherit their parent's stance correction.
- Evaluate source animation using Three.js's animation mixer. If its world
  rotation is `Q(t)` and bind rotation is `B`, the desired target world rotation
  is `C · Q(t) · inverse(B) · inverse(C) · A`.
- Convert desired world rotations to target-local rotations using the target
  hierarchy. Store local axis-angle samples and interpolate via quaternion slerp.
- Scale hips displacement by target/source mean leg length. Center X/Z travel
  at the first frame; retain Y displacement relative to the source bind pose.
  Apply displacement in the existing actor-root orientation.

This transfers source rotation changes and handles differing bone-local axes;
it is not a direct copy of Mixamo Euler angles. Stance roll uses a minimum-swing
convention, not a learned anatomical twist calibration.

The receiver gets the same **386-byte** zero-expression GSW2 pose snapshots as
other adapters. It does not need an FBX, a Mixamo account, the motion library, or
an animation mixer. Seeking, clip changes, and loop wraps mark discontinuities so
the pose buffer can reset. Recording replay also holds across captured jumps
instead of interpolating a slide between unrelated timeline positions.

## Boundaries

- Files are parsed locally in a dedicated worker. FBX texture loads are ignored;
  no motion files are uploaded. This is resource-bounded local research parsing,
  not a hardened service for hostile files.
- Limits: 64 MiB per file, 12 files per import batch, 16 takes / five minutes of
  motion per file, and a 45-second conversion timeout. Worker resources are
  released after conversion. Repeated imports accumulate in memory until reload.
- The motion library and optional reference are in-memory, not persisted. Clips
  remain available when switching to an avatar with an identical rest hierarchy
  and joint positions; incompatible clips are removed from the current library.
- Mirrored/non-uniform skeleton scale, animated scale, and animated translations
  below the hips are rejected. This is a Mixamo adapter, not a universal FBX rig
  retargeter. Only one source character per file is supported.
- Horizontal travel is not a locomotion controller. There is no foot planting,
  floor/contact IK, collision handling, root-motion accumulation across loops,
  crossfade, or animation state machine. Different proportions can cause foot
  sliding or floor penetration. Loose clothing/hair do not get secondary physics.
- Hands, extreme poses, and unseen appearance remain limited by reconstruction
  and skinning quality. Numerical skeleton agreement is not proof of photorealism.
- No Unity/Meta code changed, and this does not establish Quest performance.

## Verification

`tests/mixamo.test.mjs` checks a real sample against independent Three.js source
forward kinematics, plus synthetic local-axis, coordinate-system, units, root,
reference-pose, facial-channel, interpolation, and malformed-input cases.

The public `tests/browser/synthetic.spec.ts` verifies the no-model workflow and receiver. The broader private-workspace native/Mixamo browser captures are not bundled. See [evidence notes](../docs/evidence.md); the real-FBX unit check skips until its optional sample is installed.
