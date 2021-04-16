FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update -y && apt-get install -y git build-essential

COPY requirements.txt /tmp/requirements.txt

ENV CUDA_HOME=/usr/local/cuda-11.0
RUN pip install -U pip && \
    pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt