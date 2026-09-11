# Evidence and reproduction boundaries

The architecture distinguishes upstream paper claims, historical private-workspace observations and tests that can run from this public repository. They are not interchangeable.

## Public, asset-free checks

The public release's setup generates original procedural geometry, then exercises the same browser pose, skinning, recording and relay contracts. Default tests use small synthetic arrays and mocked native workers. Browser smoke tests cover loading, animation, recording/replay and a second receiver. The setup and browser smoke tests were executed on Windows for this release; GitHub Actions provides the published cross-platform CI record.

Native-reference and real-FBX tests skip when their optional inputs are absent. Windows may also skip a filesystem-symlink test if the OS disallows symlink creation; a separate mocked containment test covers the path boundary. None of these skips is native model validation. This release did not retrain a model or rerun the expensive native research sweep.

## Reported research observations, September 7–10, 2026

| Track | Reported observation | What it does not prove |
| --- | --- | --- |
| LHM 500M / 1B | 12 configurations across three author inputs and two shape modes; 84 native pose checks. Whole-process jobs around 31–33 s / 41–43 s. | Held-out identity quality, repeated p95, cross-family ranking |
| LHM++ | 1/4/8-view native/export/browser checks; 160K splats; one compact runtime about 92.5 MB plus 31.4 MB optional validation. Around 30 pose updates/s and 55 display FPS on the measured desktop. | Sustained sessions, arbitrary hardware, full neural-refinement image quality |
| IDOL lab | 99/99 full geometry cases on two frozen native assets; isolated GPU-resident path around 55 FPS in short desktop windows. | Main-app integration, pixel-identical CUDA/WebGPU rendering, Quest readiness |
| Controls / transport | Procedural and FBX retargeting, resolved-pose capture and delayed/loss-injected receiver experiments. | Physical foot contacts, calibrated semantic face controls, Internet congestion behavior |
| Protected preview | Gateway authentication and same-origin HTTP/WebSocket checks. | Multi-user isolation, penetration testing or current public availability |

Measurements above originated in the research workspace. The underlying captures, generated human assets and machine-specific reports are not redistributed, so readers cannot independently audit those raw historical runs from this clone. Links that previously targeted private reports now point here explicitly. Do not cite the public synthetic demo as a reproduction of those measurements.

## Reproduce responsibly

Use your own consented inputs and licensed priors; install pinned model sources/checkpoints in separate environments. Retain input/artifact hashes, exact source revisions, environment versions, hardware, seeds, failures and timing methodology. Compare native versus portable geometry before renderer packing. For image-quality claims, obtain calibrated held-out views and a broader subject/outfit set before reporting metrics or selecting a winner.

The main LHM-family contract uses <1 mm center / <0.002 rad orientation acceptance before Spark packing. IDOL uses a different geometry/covariance contract and separate thresholds. Identical seeds do not promise bitwise CUDA determinism. Display FPS, pose update rate, GPU-completed work and whole-job time must be reported separately.
