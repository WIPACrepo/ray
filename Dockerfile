FROM rayproject/ray:2.52.0-gpu as stage
# Set args for Python version
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON=3.13
ARG HOSTTYPE=${HOSTTYPE:-x86_64}
ARG RAY_UID=1000
ARG RAY_GID=100

FROM stage

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
    pip install /home/app/src

WORKDIR /serve_app