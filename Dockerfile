FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y \
     libsm6 \
     libxext6 \
     libxrender-dev

COPY requirements.txt .
RUN pip install -r requirements.txt

