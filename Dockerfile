FROM python:3.13

RUN useradd -m -U app
WORKDIR /home/app
USER app

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
USER app

ENV PYTHONPATH=/home/app
CMD ["python", "-m", "i3_ray_server"]
