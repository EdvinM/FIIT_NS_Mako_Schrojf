FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN apt-get update && apt-get install -y \
     libsm6 \
     libxext6 \
     libxrender-dev

COPY requirements.txt .
