# LHM++ experimental adapter

Integrated into the main web API, but installed separately from the baseline LHM worker. Historical 1/4/8-view runs passed native export and browser deformation checks; that does not establish identity quality, arbitrary-hardware support or a turnkey CUDA install. See [public evidence notes](../../docs/evidence.md).

## Pins and scope

- [Paper](https://arxiv.org/abs/2506.13766v2) and [author source](https://github.com/aigc3d/LHM-plusplus), source commit `906b5d9fb967ab42efb92f6fa55bf22cac86b653`.
- PixelShuffle checkpoint revision `5f1c4274068e11b93721219618d36594b6087cb6`; SHA-256 `aa0750e7632352c50421c0e041f1e543277ce73a3af7ff26e446399394cac21e`.
- [Native priors](https://huggingface.co/3DAIGC/LHMPP-Prior/tree/b683c8f68bede4f318b0bb539730b8e6711d30a0), revision `b683c8f68bede4f318b0bb539730b8e6711d30a0`.
- The Dockerfile pins its base image; the installer targets Python 3.10 / Torch 2.3 / CUDA 12.1. `TORCH_CUDA_ARCH_LIST=8.9` reflects the original RTX 4090 experiment. Other GPU architectures need deliberate build configuration and revalidation.
- Gaussian RGB only. No optional neural refinement pass.

## Explicit installation

First review all upstream model/template terms and install Docker with NVIDIA GPU support. Run from the repository root:

```sh
git clone https://github.com/aigc3d/LHM-plusplus.git generation-lab/vendors/LHM-plusplus
git -C generation-lab/vendors/LHM-plusplus checkout 906b5d9fb967ab42efb92f6fa55bf22cac86b653
docker build -f generation-lab/lhmpp/Dockerfile -t gaussian-lhmpp:cu121-source906b5d9 generation-lab
```

Use the lightweight `.venv` interpreter (`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on Linux) for these orchestration commands:

```sh
# Downloads the pinned checkpoint and selected priors; review terms first.
.venv/bin/python generation-lab/lhmpp/prepare_assets.py
# Point at an independently obtained, checksum-matching DINOv2 checkpoint.
.venv/bin/python generation-lab/lhmpp/prepare_encoder.py --source /absolute/path/to/dinov2_vitl14_reg4_pretrain.pth
.venv/bin/python generation-lab/lhmpp/container.py install
.venv/bin/python generation-lab/lhmpp/container.py check
```

`prepare_encoder.py` also checks the conventional Torch cache when `--source` is omitted; it does not fetch or replace a different file. The official download URL is included in that script. Installation has network access; inference does not. The environment volume is created with `research.task=lhmpp` and is reused only when its label matches. The main API separately validates the source/image labels and available storage.

Set `GSAVATAR_DOCKER_RESERVE_PATH` to the filesystem actually holding Docker storage. A 15 GiB free reserve is a stop threshold, **not** a total installation-space estimate. Enable generation as described in [the main guide](../../docs/generation.md), then test one authorized photo before expanding to eight.

## Export design

`native_trial.py` records native attributes, predictions and reference poses. `export_portable.py` bakes compatible bindings into the main `lhm-linear-blend-v1` contract only after checking its assumptions. All nonzero skinning influences are preserved; orientation-lock behavior follows this model, not the LHM baseline. Sparse-v2 packaging moves native validation into a separate bundle. The web job publishes the ready manifest only after independent validation.

`container.py native --input-dir <folder> --views <1..8>` is a lab entry point. Provide exactly that many normalized PNGs, with authorization to process them. Omitting the input directory selects author examples for the historical smoke experiment; it is not a cleared evaluation dataset. All generated data remains ignored.
