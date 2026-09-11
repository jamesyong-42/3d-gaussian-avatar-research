#!/usr/bin/env bash
set -euo pipefail
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5
mkdir -p /scratch/tmp
if [ ! -x /opt/lhmpp-env/bin/python ]; then
    /opt/conda/bin/python -m venv --copies --system-site-packages /opt/lhmpp-env
fi
source /opt/lhmpp-env/bin/activate
python -m pip install --no-cache-dir -c /experiment/constraints.txt numpy setuptools packaging wheel ninja
python -m pip install --no-cache-dir --no-deps \
    'https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp310-cp310-linux_x86_64.whl' \
    'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl' \
    'https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt23cu121-cp310-cp310-linux_x86_64.whl' \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1%2Bcu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -m pip install --no-cache-dir -c /experiment/constraints.txt -r /opt/lhmpp/requirements.txt \
    spconv-cu121==2.3.8 onnxruntime fvcore iopath ninja
python -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/lhmpp/lib/pointops
python -m pip install --no-cache-dir --no-build-isolation --no-deps \
    'git+https://github.com/ashawkey/diff-gaussian-rasterization.git@8829d14f814fccdaf840b7b0f3021a616583c0a1' \
    'git+https://github.com/camenduru/simple-knn.git@60f461f4a56b7967e5d8045bf92f8c33f36976d0'
python -m pip freeze > /evidence/environment-freeze.txt
python -m pip check
