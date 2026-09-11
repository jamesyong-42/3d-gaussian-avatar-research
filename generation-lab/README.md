# Generation research lab

Retained research code, separated from the lightweight browser quick start. Native trials are opt-in, can download many gigabytes, and require independently obtained model/body-template rights. Outputs, vendor checkouts, model files and reports are ignored and are **not included** in the public repository.

- [Main generation setup](../docs/generation.md): LHM 500M baseline and 1B comparison.
- [LHM++](lhmpp/README.md): separate multi-photo inference and portable export.
- [IDOL](idol/README.md): distinct native deformation and browser-geometry experiment.
- [Research plan](../docs/research.md): evaluation gates and unresolved questions.
- [Evidence boundary](../docs/evidence.md): historical measurements versus public CI.

`lab.py --help` lists baseline experiment commands. `fetch-lhm` downloads the selected pinned checkpoint using `huggingface_hub`. `freeze` and the original sweep are **historical engineering fixtures**: they expect an existing native `example` export and the author's example inputs. They are not a consented evaluation dataset, are not part of setup, and must not be run as an identity-quality benchmark. Custom research should freeze its own authorized inputs and calibrated holdouts.

`validate_asset.mjs` checks native exported fixtures. LHM++ and IDOL adapters retain failures and run records locally. Never upload their report directories without a separate rights/privacy audit. A compatible skeleton alone does not establish compatible deformation.

## Build context boundary

The lab `.dockerignore` allows only the LHM++ Dockerfile and its explicitly checked-out source into the image context. It excludes the avatar library, checkpoints, scratch space, reports and credentials. Do not build a research image from the whole parent workspace.
