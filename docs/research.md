# Research decisions and next gates

The [architecture website](https://jamesyong-42.github.io/3d-gaussian-avatar-research/#papers) is the full, cited 21-paper review: objective, landscape, per-paper lessons, design provenance, generation, animation, transport, evidence and Unity/Meta roadmap. It is a dated research account, not a claim to continuously track the newest paper.

## Current choices

LHM provides the single-photo baseline and its learned rigged Gaussian representation. LHM++ adds multi-photo reconstruction through a separately validated exporter. These models produce the avatar, rig-related fields and learned priors; we do not claim to have invented them.

Our system work is the explicit asset/pose boundary, model-family adaptation, complete-influence portable deformation, sparse packaging, independent verification, signal/FBX adapters, local capture and receiver semantics. IDOL's corrective/covariance path is retained as an isolated alternative, not forced into an incompatible LHM formula.

The previously discussed MON3TR and VRGA papers inform reconstruction/animation design and evaluation questions. They are literature inputs, not secretly implemented generation backends and not rejected solely because LHM runs. See their individual entries and sources in the architecture review for the specific adoption boundary.

## Exploration plan

1. Freeze an authorized, diverse subject/outfit dataset with calibrated held-out views. Keep capture rights and retention policy explicit.
2. Preserve LHM 500M/zero-shape as a measured baseline; compare checkpoint/shape changes on identical inputs with repeated cold and warm trials.
3. Evaluate LHM++ view subsets with nested coverage, fixed methodology and separate native/portable timings. Do not compare different subjects as model-quality evidence.
4. Finish IDOL integration gates: deformation contract, asset budget, renderer equivalence, controls, recording and receiver behavior. A successful isolated viewer is necessary but not sufficient.
5. Triage other paper families by available source/weights, rights, capture burden, native quality and the ability to export reusable animation state. Run feasibility gates before investing in full UI integration.
6. Establish semantic face calibration, contacts/locomotion, drift/tracking-loss policy, transport reconnection and sustained browser/device performance.
7. Port proven asset/pose contracts into Unity; validate handedness, native numerical fixtures, stereo rendering and actual Quest/Meta device signals separately.

Each candidate must pass rights/readiness → native reconstruction → portable geometry → visible browser controls → recording/receiver → resource budget. Failure records are research results; a bigger model or newer paper is not automatically a better product choice.
