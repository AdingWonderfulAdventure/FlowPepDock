# basic image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# workplace
WORKDIR /FlowPepDock
COPY . /FlowPepDock

RUN apt-get update && apt-get upgrade -y && apt-get install -y wget bzip2 pkg-config build-essential python3-dev python3-pip libatlas-base-dev gfortran libfreetype6-dev

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.11.1-0-Linux-x86_64.sh && \
	bash Miniconda3-py39_24.11.1-0-Linux-x86_64.sh -b -f -p /opt/miniconda && \
    rm Miniconda3-py39_24.11.1-0-Linux-x86_64.sh

ENV PATH=/opt/miniconda/bin:$PATH

RUN conda init bash

RUN conda env create -f flowpepdock_env.yaml -n FlowPepDock && echo "conda activate FlowPepDock" >> ~/.bashrc

ENV PATH=/opt/miniconda/envs/FlowPepDock/bin:$PATH

RUN /opt/miniconda/envs/FlowPepDock/bin/pip install --no-cache-dir -r requirement.txt

RUN /opt/miniconda/envs/FlowPepDock/bin/python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'

RUN /opt/miniconda/envs/FlowPepDock/bin/python -c "import torch, yaml, MDAnalysis, esm"
