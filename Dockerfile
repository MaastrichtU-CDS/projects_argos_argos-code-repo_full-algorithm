# basic python3 image as base
#FROM ananyac/argos_base
FROM ananyac/argos_v6_gpu_base

RUN apt-get clean
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --upgrade pip
RUN pip install setuptools --upgrade

# This is a placeholder that should be overloaded by invoking
# docker build with '--build-arg PKG_NAME=...'
ARG PKG_NAME="argosfeddeep"

# install federated algorithm
COPY . /app
RUN pip install /app

ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `docker_wrapper()` when the image is run.
CMD python -c "from vantage6.tools.docker_wrapper import docker_wrapper; docker_wrapper('${PKG_NAME}')"