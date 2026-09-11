# Configuration

Defaults are safe for the synthetic local viewer. A root `.env` is optional; existing environment variables take precedence. Relative paths resolve against the repository root, not the shell's working directory. Restart the server after configuration changes.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GSAVATAR_DATA_DIR` | `local/data` | Avatar library, normalized inputs, job records and GPU lock |
| `GSAVATAR_ENABLE_GENERATION` | `0` | Set exactly `1` only after model setup |
| `GSAVATAR_LHM_ROOT` | `third_party/LHM` | Separate author checkout |
| `GSAVATAR_GENERATION_PYTHON` | LHM's `lhm_env` Python | Worker with CUDA/native dependencies; **not** the API `.venv` |
| `GSAVATAR_CUDA_HOME` | Unset | Optional CUDA toolkit override for the native worker |
| `GSAVATAR_LHM_500M_DIR` | `local/models/LHM-500M-HF` | 500M checkpoint folder |
| `GSAVATAR_LHM_1B_DIR` | `generation-lab/checkpoints/LHM-1B-HF` | 1B checkpoint folder |
| `GSAVATAR_LHMPP_DIR` | `generation-lab/checkpoints/LHMPP-700M-PixelShuffle` | LHM++ checkpoint folder |
| `GSAVATAR_DOCKER` | Docker found on PATH | Docker CLI executable |
| `GSAVATAR_DOCKER_RESERVE_PATH` | Repository drive/filesystem root | Filesystem checked for the Docker free-space reserve; point to the actual Docker storage filesystem |
| `GSAVATAR_EXAMPLE_PHOTO` | `local/consented-example.png` | Optional photo you are authorized to process |

No environment setting downloads models, starts Docker, accepts a license or configures a public tunnel. `doctor.py` reports file/configuration readiness, not inference success.

## Data and retention

`local/data/avatars/<id>/` holds the manifest, binary and any source photos, validation data or generation logs. `local/data/jobs/` stores job state. Startup marks interrupted jobs as failed rather than silently rerunning them. There is no automated retention/deletion policy or per-user access boundary.

To use an existing library, point `GSAVATAR_DATA_DIR` at a deliberate data directory. Back it up first. Do not run two servers against the same library: job recovery and process ownership assume one application instance. A public fork must not point at another user's research directory by default.

Changing selected photos in the UI does not modify an existing avatar. Saved source photos are restored on opening an avatar where available; incomplete input sets are reported and cannot silently regenerate as a partial set. The synthetic avatar correctly has no source photos.

## Browser overrides

`?deformer=cpu` selects the CPU reference; `?deformer=webgpu` requests GPU compute. Large avatars try WebGPU automatically; unsupported devices fall back with an explanation. Rendering still uses Spark/WebGL2, including after a WebGPU deformation pass.

`AVATAR_LAB_URL` configures browser-test access; `PLAYWRIGHT_CHROMIUM_EXECUTABLE` optionally selects an installed browser. `GSAVATAR_NATIVE_FIXTURE` selects a local native export for the optional Node parity test. None of these values belong in committed test reports.
