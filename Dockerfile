
FROM rayproject/ray:nightly-py313-gpu as build
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON=3.13
ARG HOSTTYPE
ARG MIMALLOC_VERSION
ENV HOSTTYPE=${HOSTTYPE:-x86_64}
ENV MIMALLOC_VERSION=${MIMALLOC_VERSION:-v3.5.3}
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    autoconf \
    cmake \
    wget \
    git
# Install MiMalloc drop-in replacement for 
WORKDIR /tmp
RUN wget https://github.com/microsoft/mimalloc/releases/download/${MIMALLOC_VERSION}/mimalloc-${MIMALLOC_VERSION}-source.tar.gz && \
    mkdir mimalloc-${MIMALLOC_VERSION} && \
    tar -xzf mimalloc-${MIMALLOC_VERSION}-source.tar.gz && \
    cd mimalloc-${MIMALLOC_VERSION} && \
    mkdir -p out/release && \
    cd out/release && \
    cmake ../.. && \
    make && \
    make install 

FROM rayproject/ray:nightly-py313-gpu as stage
# Set args for Python version
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON=3.13
ARG HOSTTYPE=${HOSTTYPE:-x86_64}
ARG RAY_UID=1000
ARG RAY_GID=100

FROM stage

COPY --from=build /usr/local/lib/* /usr/local/lib/.
COPY --from=build /usr/local/include/* /usr/local/include/.

# Mount the entire build context (including '.git/') just for this step
# NOTE:
#  - mounting '.git/' allows the Python project to build with 'setuptools-scm'
#  - no 'COPY .' because we don't want to copy extra files (especially '.git/')
#  - using '/tmp/pip-cache' allows pip to cache
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --upgrade "pip>=25" "setuptools>=80" "wheel>=0.45"
USER root
RUN --mount=type=bind,source=.,target=/home/app/src,rw \
    --mount=type=cache,target=/tmp/pip-cache \
    pip install /home/app/src && \
    chown root:root `which py-spy` && \
    chmod u+s `which py-spy`
# Anaconda pip packages are not added to LD_LIBRARY_PATH, rebuild cache with new packages
ENV ANACONDA_SITE_PACKAGES=/home/ray/anaconda3/lib/python3.13/site-packages
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ANACONDA_SITE_PACKAGES/nvidia/cuda_runtime/lib:$ANACONDA_SITE_PACKAGES/nvidia/cu13/lib:$ANACONDA_SITE_PACKAGES/nvidia/cudnn/lib:/home/ray/anaconda3/lib
RUN ldconfig

RUN apt-get update && apt-get install -y \
    gdb

COPY i3_ray_server/* .
