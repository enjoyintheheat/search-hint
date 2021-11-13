FROM nvidia/cuda:11.4.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Moscow"

RUN apt-get update && apt-get install -y --no-install-recommends make \
  build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
  libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev \
  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN . ~/.bashrc

ENV HOME="/app"
WORKDIR ${HOME}

ENV PYTHON_VERSION=3.9.0
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

RUN pip install --upgrade pip

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .