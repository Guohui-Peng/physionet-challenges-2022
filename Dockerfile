FROM tensorflow/tensorflow:2.7.1-gpu
# FROM tensorflow/tensorflow:2.7.1

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER pguohui2004@163.com

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.
RUN apt update && \
    apt install -y git && \
    apt-get install -y libsndfile1 && \
    /usr/bin/python3 -m pip install --upgrade pip

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
