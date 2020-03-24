FROM ubuntu:18.04

ARG CUDA_VERSION=10.0
ARG CUDA_FULL_VERSION=10.0.130_410.48
ARG PYTHON_VERSION=3.7

ENV LANG=C.UTF-8

# WORKDIR /

RUN apt-get update && apt install --no-install-recommends -y \
    ca-certificates \
    wget \
    pkg-config \
    curl \
    libcurl4-openssl-dev \
    vim

# Conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:${PATH}

RUN conda install -y -c conda-forge \
                  python=${PYTHON_VERSION} \
                  cudatoolkit \
                  pygobject \
                  python-annoy

RUN conda clean -a -y

RUN pip install \
                pillow \
                numpy \
                tensorflow \
                grpcio-tools \
                keras

COPY . /opt/inf-ser

RUN pip install /opt/inf-ser/whl/tensorrtserver-1.4.0-py2.py3-none-linux_x86_64.whl

CMD ["python","/opt/inf-ser/inference_server.py"]
