# IDOL isolated research track

This code is **not a main-app generation backend** and does not use the main LHM deformation contract unchanged. It preserves the IDOL native geometry, learned corrective and covariance investigations for further research.

[Author implementation](https://github.com/yiyuzhuang/IDOL), source commit `9fd9296c28e8f8f9ed5f5c594f3df1574b8ec82d`. Checkpoint revision `3c80ec956f9bede01d9fbb3917ca229f8d2673b1`; selected file sizes and hashes live in `prepare_assets.py`. Models, caches, body templates, native exports, participant photos and evidence binaries are **not bundled**.

## Pipeline map

1. `prepare_assets.py`, `prepare_input.py`: verified assets and explicitly selected inputs.
2. `container.py`, `native_trial.py`: bounded native feasibility experiment in a distinct labeled Python environment.
3. `portability_trial.py`, `corrective_codec.py`: deformation-family audit and compact corrective fields.
4. `deformation_trial.py`, `deformation_cpu.mjs`, `deformation_gpu.mjs`, `neighbors_gpu.mjs`: full-influence geometry, fresh neighbor-derived sizes and covariance updates.
5. `resident_viewer.mjs`, `deformation_demo.html`: isolated GPU-resident browser path, separate from the main Spark viewer.
6. `check_*` / `verify_portability.py`: retained numerical and rendered-evidence checks against local run records.

The native wrapper reuses the pinned LHM++ base image but has its own environment volume, source checkout and dependencies. Complete the [LHM++ image setup](../lhmpp/README.md), then inspect `container.py --help`, `install_environment.sh` and `prepare_assets.py` before executing this track. Check out IDOL into `generation-lab/vendors/IDOL` at the exact commit above. Do not run author or downloaded pickle-based model code on untrusted inputs or in the API environment.

Native `portability` / `deformation` actions require one or two completed, locally retained `--native-run` IDs. The browser reports require those exported artifacts. A clean clone intentionally cannot open those historical reports or regenerate them without the separate setup. Only synthetic math/codec tests run in default CI.

Historical research reported 99/99 full geometry checks and about 55 FPS in short desktop windows for an isolated 200,800-splat path. These are **reported private-workspace observations**, not public-CI output, a quality score, Quest performance or a production-ready integration. See [evidence and limits](../../docs/evidence.md) and [the next gates](../../docs/research.md).
