# Third-party research and dependencies

This project integrates published research and separately installed software. No model weights, human body templates, author example photographs, generated human assets, motion files or vendor source trees are bundled. A repository license does not by itself establish rights for every checkpoint, dataset, template or output.

| Component | Role / provenance | Distribution boundary |
| --- | --- | --- |
| [LHM](https://github.com/aigc3d/LHM), [paper](https://arxiv.org/abs/2503.10625) | Native single-photo reconstruction and rigged Gaussian fields | Source/checkpoint pins in generation guide; install separately |
| [LHM++](https://github.com/aigc3d/LHM-plusplus), [paper](https://arxiv.org/abs/2506.13766v2) | Multi-photo reconstruction and native shape/binding behavior | Separate opt-in lab image, weights and priors |
| [IDOL](https://github.com/yiyuzhuang/IDOL) | Isolated alternative reconstruction/corrective/covariance track | Separate source, licensed assets and lab contracts |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/), [FLAME](https://flame.is.tue.mpg.de/) | Native body/face models and template conventions | Review applicable terms independently; no templates bundled |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | Native utilities; matrix-to-quaternion algorithm reproduced to match native numerical behavior | Native dependency installed separately; algorithm provenance identified in engine comments |
| [Three.js](https://github.com/mrdoob/three.js), [Spark](https://github.com/sparkjsdev/spark) | Browser scene, FBX parsing, Gaussian rendering | Installed from pinned npm dependencies with their own licenses |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Optional browser tracking runtime/task models | npm runtime; three task files downloaded only on request |
| [Mixamo](https://www.mixamo.com/) / [Three.js FBX example](https://threejs.org/examples/webgl_loader_fbx.html) | Optional source motion for retargeting tests | No motion bundled; do not relicense example FBX as project code |
| [DINOv2](https://github.com/facebookresearch/dinov2), upstream native dependencies | Reconstruction priors and native feature extraction | Separate verified assets; review each component's terms |

FastAPI, Uvicorn, Pillow, NumPy, Requests, TypeScript, Vite, Playwright and GitHub Actions retain their respective package licenses and notices. Consult the installed package metadata and pinned upstream source for authoritative license text before redistributing a built application or container.

The synthetic demo is original analytic geometry generated from source, not a scan, a learned model output or a redistributed SMPL-X template. Its semantic joint names and array contract are deliberately compatible with the browser control interface. This attribution does not claim that we invented linear blend skinning, Gaussian splatting, IK, quaternion interpolation or FBX retargeting.

The full paper-by-paper attribution and our adoption boundary are in the [architecture review](https://jamesyong-42.github.io/gaussian-avatar-lab/#papers).
