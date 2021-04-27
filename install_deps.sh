#!/bin/bash

conda create -y -n coordinpaint python=3.7
source activate coordinpaint

pip install \
    numpy \
    matplotlib \
    torch \
    torchvision \
    scikit-image

conda install -y opencv
