# Portable avatar and animation contracts

## Assets v1 and v2

`avatar.json` identifies `version: 1`, `deformation: lhm-linear-blend-v1`, model,
joint names, counts, creation time, byte length, SHA-256, and typed-array layouts.
`avatar.bin` contains 4-byte-aligned little-endian arrays in stable Gaussian order.
The browser verifies its full byte length and SHA-256 before use.

Version 1 remains supported unchanged. Version 2 keeps the same Gaussian order,
rig, joint controls, and `lhm-linear-blend-v1` math, while introducing:

- `expressionStorage: sparse-vertices-v1`, sorted unique uint32
  `expressionIndices[M]`, and float32 `expressionDirections[E, M, 3]`. Omitted
  vertices have numerically zero directions for every expression. No nonzero
  expression values or skinning weights are pruned, normalized, or quantized.
- `nExpressionVertices: M`. The CPU/GPU runtime maps Gaussian indices to sparse
  rows; it does not expand this back into the dense 160K × 100 basis.
- `validation: {binary, bytes, sha256, arrays}` describes a separate
  `validation.bin` containing the seven native center/orientation fixtures.
  `referencePoses` remains in metadata. Normal playback downloads only the
  runtime binary. The reference-check action downloads and SHA-256 verifies
  validation data on demand. Explicit ZIP export includes both binaries.

The measured eight-view LHM++ asset shrinks from 267,728,556 to 92,528,556 runtime
bytes. Validation adds 31,360,000 bytes only when requested. The sparse basis
retains 40,000 vertices and all 12,000,000 nonzero expression values. Signed zeros
may become positive zero; numeric behavior, not original byte layout, is preserved.
Original dense assets and their hashes remain untouched.

| Array | Meaning |
| --- | --- |
| positions, rotations | Shape-adjusted zero-pose centers and predicted local Gaussian quaternions |
| scales, colors, opacities | Activated anisotropic scales, RGB, opacity; not PLY logits/log-scales |
| neutralLinear | Original neutral-to-zero linear transform per Gaussian |
| skinOffsets, skinBones, skinWeights | Exact CSR storage of every nonzero skin influence; no top-4 truncation |
| expressionDirections | 100 × Gaussian count × 3 in v1; 100 × sparse-vertex count × 3 in v2, transformed into zero-pose space |
| joints, parents, jointNames | 55-joint SMPL-X rest hierarchy and names |
| rotationLocked, regions | LHM constrained-body orientation behavior and face/hand masks |
| reference_* | Full Python-generated centers/quaternions for seven test poses |

These arrays are exported by intercepting the actual LHM inference call without
editing the third-party source. The default remains LHM-500M-HF with ten zero
body-shape betas. The generation experiment also supports the pinned LHM-1B-HF
checkpoint and optional native Multi-HMR shape estimation. Estimated betas are
applied to both Gaussian export and rest joints; estimator failure does not fall
back silently to zero. Predicted Gaussian offsets carry reconstructed appearance
and geometry in either mode.

New manifests include optional `generation` provenance: selected model, checkpoint
revision/content hash, source/exporter hashes, input hash, seed, shape parameters,
and runtime settings. Existing v1 assets remain readable. The deformation/wire
identifiers and expression semantics have not changed. A different reconstruction
family still requires its own validated exporter/deformer.

The LHM++ adapter has now passed that gate: PixelShuffle predicts native shape,
bakes its diffused volumetric weights at the predicted Gaussian offsets, and
preserves native fixed hand/face bindings. Static baking is allowed only when
expression directions outside those fixed regions are zero. Unlike original LHM,
all LHM++ Gaussian rotation locks are zero. Model identity remains explicit in
`model` and `generation.engine`; the shared deformation identifier names math,
not a claim that the reconstruction checkpoints are identical.

## Resolved pose

One snapshot contains actor root position (meters), actor root quaternion (xyzw),
55 local joint axis-angle rotations (radians), 100 learned expression coefficients,
and one simulation tick. Coordinate space is right-handed, Y-up SMPL-X space.
The actor root is separate from the pelvis joint's local rotation. A Unity
adapter must explicitly convert handedness, vectors, quaternions, and units.

All input adapters produce this same structure. The receiver does not rerun IK,
Mixamo retargeting, procedural animation, tracking, or local root movement. Root and articulation are
interpolated together using one buffered snapshot pair; rotations use shortest-arc
quaternion slerp, not component-wise axis-angle lerp.

## Deformation

For each Gaussian, form a weighted posed joint matrix `P` from the full sparse
skin weights. Let `N` be its exported neutral linear transform, `x0` its exported
zero-pose center, `E` its expression directions, and `e` the expression vector.

```text
center = actorRoot × (P × [x0 + E·e, 1])
rotation = actorRoot.rotation × normalize(matrixToQuaternion(P.linear × N)) × q0
```

For LHM's constrained body points, omit the middle quaternion term, matching
the original renderer. `matrixToQuaternion` follows PyTorch3D's largest-candidate
algorithm even for the non-orthogonal blended matrix. Scales/colors/opacities stay
as exported. This reproduces **LHM's model**, not an alternative physical
covariance-transport rule. Numerical parity is checked before Spark's packing and
rasterization; it is not a claim of pixel-identical Python/WebGL images.

### WebGPU execution and fallback

Avatars with at least 100K splats try WebGPU compute automatically; the original
40K baseline stays on the CPU path. `?deformer=cpu` selects the reference path;
`?deformer=webgpu` explicitly requests GPU execution. If GPU initialization,
startup parity, limits, device state, or later verification fails, the worker
reports the reason and resumes CPU deformation. No different skinning
approximation is substituted. A simulated device-loss browser test covers this.

FK still resolves the 55 joints on the CPU. The worker uploads joint matrices,
active expression indices/values, and actor-root transform to WebGPU. One compute
invocation per Gaussian sums every stored influence, applies sparse expressions,
and reproduces the same largest-candidate quaternion rule. Float32 GPU output is
read back to the existing WebGL/Spark renderer; this is **not** a zero-copy
GPU-rendering integration. Reported deformation wall time includes FK, upload,
dispatch, readback, and unpacking; it is not a GPU timestamp-query kernel time.

The existing seven full native fixtures now check both CPU and GPU paths. Eight
additional deterministic CPU/GPU stress poses activate all expressions and vary
joint and actor-root rotations/translations. These are implementation checks,
not native-quality ground truth or physical/anatomical constraints. They use the
same <1 mm center / <0.002 rad orientation thresholds, before renderer packing.

## Multi-photo generation API

`POST /api/jobs` accepts either the legacy `photo` field or repeated `photos`
fields, never both. LHM 500M/1B still accept exactly one photo, with `zero` or
`estimate` shape. `LHMPP-700M-PixelShuffle` accepts 1–8 photos and only `predicted`
shape (native ShapeHead); the default model and old request defaults stay unchanged.

Every file must decode as JPG/PNG/WebP, be at least 128 pixels per side and at most
24 MP. Limits are 16 MB per file and 64 MB combined. The server applies EXIF
orientation, converts RGB, fits inside 2048 × 2048, and saves ordered normalized
`view-000.png` etc.; client filenames never become filesystem paths. Invalid
batches create no job. Queue capacity is rechecked after asynchronous reads.

LHM++ jobs use the dedicated Linux CUDA image/volume with networking disabled,
read-only model/input mounts, a shared GPU lock, 360-second container timeout,
120-second lock wait, and 15 GiB reserve on the configured Docker storage filesystem. Exact image IDs, source hashes, input
hashes, native logs, and failed evidence are retained. Compact manifests are staged
as `avatar.pending.json` until the independent Node/native fixture checks pass;
only then is `avatar.json` published. Shutdown/recovery stops only a container
whose exact job ID and research labels match this service's recorded job.

Readiness probes are read-only and never install or start Docker. This remains a
local research service, not an Internet-hardened, multi-tenant image-processing API.

## GSW2 wire protocol

All multi-byte fields are little-endian. This is **not compatible** with the
earlier 50-expression prototype packet; it explicitly carries expression count.

| Offset | Bytes | Value |
| --- | ---: | --- |
| 0 | 4 | ASCII `GSW2` |
| 4 | 1 | Version 2 |
| 5 | 1 | Flags: bit 0 = teleport/reset interpolation |
| 6 | 2 | Bone count, currently exactly 55 |
| 8 | 2 | Expression dimension, 100 for this asset |
| 10 | 2 | Number of nonzero expression entries |
| 12 | 4 | Unsigned sequence, wraparound-aware stale rejection |
| 16 | 4 | Sender simulation tick, never regenerated by receiver |
| 20 | 8 | Float64 sender capture time, milliseconds since epoch |
| 28 | 28 | 3 float32 root-position + 4 float32 xyzw root-quaternion values |
| 56 | 330 | 165 signed int16 axis-angle components, scaled by `32767/π` |
| 386 | 4 × active count | uint16 expression index + int16 coefficient × 1000 |

Joint rotations are canonicalized to the shortest equivalent axis-angle before
quantization. Root quaternions are normalized on decode. Truncation, dimension
mismatch, duplicate expression indices, and nonfinite root/joint/expression input
are rejected. The normal packet is **386 bytes**; all 100 expressions active
produce **786 bytes**. At 30 Hz, zero-expression payload is 11,580 B/s (~92.6 kbps),
excluding WebSocket/TCP/IP overhead and asset transfer.

Asset identity is announced once via a reliable room message before poses are
used. Each receiver independently downloads and checksum-validates the matching
asset. The room allows one sender. No pixels or per-Gaussian positions traverse
the pose relay.

The receiver applies optional synthetic latency/jitter/loss, rejects stale
sequences, then uses an 80 ms **arrival-time** interpolation buffer. Capture time
is carried for inspection but no cross-machine clock synchronization is claimed.
It holds the newest complete pose during a gap and resets the buffer on teleport.
There is no extrapolation, clock-drift correction, reconnect/resend protocol, or
production transport tuning in this MVP.

## Files to carry toward Unity / Meta

- `backend/export_lhm.py`: the asset-producing boundary.
- `../generation-lab/lhmpp/export_portable.py`: the distinct native LHM++ adapter.
- `backend/pack_asset.py`: numeric-lossless compact v2 packaging and separate validation.
- `src/engine.mjs`: authoritative portable deformation and FK reference.
- `src/gpu-deform.mjs`: full-influence WebGPU execution to validate against that reference.
- `src/controls.mjs`: illustrative sparse-target and procedural adapters.
- `src/mixamo.mjs`: FBX-to-SMPL-X retargeting and motion sampling; see [MIXAMO.md](MIXAMO.md).
- `src/tracking.ts`: experimental webcam adapter, not a Meta implementation.
- `src/protocol.mjs`: resolved pose packet and interpolation behavior.
- `tests/engine.test.mjs`: numeric parity and contract tests to reproduce in C#.
- `tests/mixamo.test.mjs`: independent source-skeleton FK and retargeting edge cases.

Port and validate the asset/pose/deformation contracts before integrating real
device signals. A normal static PLY export cannot replace this rigged asset.
