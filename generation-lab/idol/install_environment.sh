#!/usr/bin/env bash
set -euo pipefail
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=3
mkdir -p /scratch/tmp
if [ ! -x /opt/idol-env/bin/python ]; then
    /opt/conda/bin/python -m venv --copies --system-site-packages /opt/idol-env
fi
source /opt/idol-env/bin/activate
python -m pip install --no-cache-dir -c /experiment/constraints.txt \
    numpy setuptools wheel ninja packaging einops==0.8.0 omegaconf==2.3.0 \
    transformers timm pytorch-lightning torchmetrics huggingface-hub \
    matplotlib==3.8.4 scipy==1.13.0 opencv-python-headless==4.9.0.80 \
    pillow==10.3.0 fvcore iopath plyfile imageio imageio-ffmpeg
# Compatible prebuilt PyTorch3D; record the deviation from upstream 0.7.7.
python -m pip install --no-cache-dir --no-deps \
    'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl'
python -m pip install --no-cache-dir --no-build-isolation --no-deps \
    'git+https://github.com/camenduru/simple-knn.git@60f461f4a56b7967e5d8045bf92f8c33f36976d0'
# IDOL requires the official dr_aa API, not the older main branch. Both have
# package version 0.0.0, so explicitly reinstall this pinned implementation.
python -m pip install --no-cache-dir --no-build-isolation --no-deps --force-reinstall \
    'git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0'
# Compile our three required extensions from a scratch copy of pinned source.
python -m pip install --no-cache-dir --no-build-isolation --no-deps /scratch/build-source
python -m pip freeze > /evidence/environment-freeze.txt
python -m pip check
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings as S; assert 'antialiasing' in S._fields; print('IDOL antialiasing API verified')"
