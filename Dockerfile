FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

COPY requirements.txt .
RUN pip install -r requirements.txt
