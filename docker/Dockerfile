# CUDA 10.1 requires driver version >= 418.39
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && \
    apt-get -y install apt-utils libopencv-dev cmake git sudo vim software-properties-common nano wget tree

RUN pip install matplotlib tqdm yacs Pillow tensorboardx wandb scikit-learn opencv-python coloredlogs termcolor dill
RUN pip install six==1.13.0

RUN pip install pandas numpy

# Avoid showing "I have no name!" in a container
RUN rm -f /etc/bash.bashrc && \
    wget -O /etc/bash.bashrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/dockerfiles/bashrc && \
    sed -i "s/tf-docker/bitrap/g" /etc/bash.bashrc && \
    sed -i "25,35d" /etc/bash.bashrc

RUN export PYTHONPATH=$PWD:PYTHONPATH
