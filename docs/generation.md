# Photo generation: separate, opt-in GPU setup

The public quick start is an asset-free synthetic demonstration. To reconstruct a person, install the author model **outside the API environment**, obtain the applicable model/body-template rights, then explicitly enable the adapter. No weights, priors, participant photos, upstream example media or native environments are included in this repository.

The original research used an RTX 4090. The public release verifies the lightweight viewer/API without those installations; it does **not** claim a fresh CUDA install on every platform has passed. Budget significant disk space, native build time and downloads. A browser GPU alone is not a reconstruction worker.

## 1. LHM baseline: one photo

Our adapter targets the [author implementation](https://github.com/aigc3d/LHM) at commit `4f88aaeb3629249fbbddb4d0784a06962d9e1338` and intercepts native inference without editing the upstream checkout.

```sh
git clone https://github.com/aigc3d/LHM.git third_party/LHM
git -C third_party/LHM checkout 4f88aaeb3629249fbbddb4d0784a06962d9e1338
```

Use Python 3.10 for this native environment. Follow the pinned checkout's `INSTALL.md` and platform install scripts for compatible PyTorch, CUDA, PyTorch3D and rasterization extensions. The upstream README provides Windows `install_cu121.bat` and Linux `install_cu121.sh` / `install_cu118.sh`; review them before execution. These scripts install native dependencies and may download assets. Do **not** run them inside this repository's API `.venv`.

Obtain the required native priors into `third_party/LHM/pretrained_models` following the author instructions. These include body templates and other dependencies with their own terms; a downloadable archive is not proof of redistribution or commercial permission. Do not download upstream motion videos just to run the browser's procedural clips or FBX importer.

Install the pinned reconstruction checkpoint using the **native environment's Python**, which must contain `huggingface_hub`, Pillow and the upstream dependencies:

```sh
# Example on Linux; use lhm_env/Scripts/python.exe on Windows.
third_party/LHM/lhm_env/bin/python generation-lab/lab.py fetch-lhm LHM-500M-HF
```

Run `python generation-lab/lab.py --help` if using a different setup; the model fetch is explicit and writes only to configured checkpoint storage. Set `GSAVATAR_GENERATION_PYTHON` to the absolute interpreter path if your environment lives elsewhere. The baseline checkpoint requires `config.json` and `model.safetensors` at `local/models/LHM-500M-HF`; the loader does not silently substitute a different model.

| Model | Checkpoint revision | Default location |
| --- | --- | --- |
| [LHM 500M-HF](https://huggingface.co/3DAIGC/LHM-500M-HF) | `dd6392905187a91fd67b3f6962aa74481e943764` | `local/models/LHM-500M-HF` |
| [LHM 1B-HF](https://huggingface.co/3DAIGC/LHM-1B-HF) | `92372582f660066b9f1b9513860744357265b3d5` | `generation-lab/checkpoints/LHM-1B-HF` |

The 1B file size and expected SHA-256 are pinned in [generation_config.py](../web/backend/generation_config.py). For 500M, the repository revision is pinned and the actual checkpoint digest is recorded by export; it does not currently enforce an independently supplied expected digest. Do not describe those two verification levels as identical.

## 2. Configure and verify the connection

Copy `.env.example` to `.env`, then set your actual paths:

```dotenv
GSAVATAR_ENABLE_GENERATION=1
GSAVATAR_LHM_ROOT=third_party/LHM
# Set this to your real native environment interpreter, not API .venv:
# GSAVATAR_GENERATION_PYTHON=/absolute/path/to/lhm_env/bin/python
# Set only if the native worker needs a toolkit override:
# GSAVATAR_CUDA_HOME=/usr/local/cuda
```

```sh
python scripts/doctor.py
python scripts/serve.py
```

Readiness checks files and configuration; it is not a GPU smoke test. Open the app, choose **LHM 500M**, upload an authorized full-body image, and submit one job. Inspect any failure in your local job/asset logs. Native preprocessing may need additional upstream caches on its first run. Export records provenance and seven reference poses; use **Compare with Python reference** after loading the result.

The native process terminates after export. Later animation, recording and receiver playback do not need reconstruction to run again. Keep the initial `zero` shape baseline; `estimate` invokes a native body-proportion estimator and is an experimental comparison, not a proven quality upgrade.

## 3. LHM++: one to eight photos

This is a separate Linux/NVIDIA Docker track, not a package installed into the LHM or API environments. Its source, priors and checkpoint pins differ from LHM. Use the [LHM++ lab setup](../generation-lab/lhmpp/README.md). It requires Docker GPU access, substantial local storage, a labeled environment volume and the pinned PixelShuffle weights.

Once `doctor.py` reports the configured LHM++ files/environment ready, select it in the web model menu. Upload 1–8 views of the **same person and outfit**; the adapter preserves native ShapeHead prediction, full skinning influences and native binding behavior. Inference containers have networking disabled. Uploads are normalized before the container sees them.

Main-app LHM++ rendering uses exported Gaussian RGB, not the paper's optional neural refinement renderer. Do not equate browser screenshots with the full paper's image-quality result.

## 4. IDOL and other research families

[IDOL](../generation-lab/idol/README.md) has its own corrective fields and covariance behavior. It is not a drop-in LHM checkpoint and is not a selectable main-app backend. The retained lab code explores native geometry parity and a separate browser renderer. Other papers are research candidates, not installed generators; see [the roadmap](research.md).

## Common failures

| Symptom | Check |
| --- | --- |
| Generation button disabled | `.env` flag, interpreter path, author source, model files; restart after changes |
| Python import / CUDA extension error | You selected the native interpreter, compatible Torch/CUDA/build tools and complete priors |
| Docker unavailable | Start Docker yourself; verify NVIDIA container support and the lab image/volume labels |
| Missing source photos after import | Copy authorized original input files with their metadata, or accept playback-only mode; never substitute a random image |
| Port occupied | Stop the intended old instance or choose `--port 8875`; do not kill unrelated Python processes |
| Native reference unavailable | Synthetic assets deliberately have none; v2 assets need their separate validation bundle |

Use [configuration](configuration.md) for paths, [contracts](../web/CONTRACTS.md) for formats, and [evidence notes](evidence.md) for what has actually been measured.
