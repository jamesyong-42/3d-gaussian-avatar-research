FROM pytorch/pytorch@sha256:1bb28822b361bdb2d8cde5a58f08337490bf6e73fc96b0aa1035268c295f3d00
LABEL research.task="lhmpp" research.source="906b5d9fb967ab42efb92f6fa55bf22cac86b653"
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git ca-certificates build-essential ninja-build libgl1 libglib2.0-0 libegl1 libopengl0 ffmpeg
COPY vendors/LHM-plusplus /opt/lhmpp
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=6 \
    MPLCONFIGDIR=/scratch/matplotlib XDG_CACHE_HOME=/scratch/cache TORCH_HOME=/scratch/torch \
    HF_HOME=/scratch/huggingface TMPDIR=/scratch/tmp
WORKDIR /opt/lhmpp
