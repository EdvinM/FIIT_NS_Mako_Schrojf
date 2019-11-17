FROM ubuntu:16.04

COPY data /data
COPY models /models
COPY notebooks /notebooks
COPY src /src
COPY data /data
COPY requirements.txt /

RUN pip install -r requirements.txt
RUN jupyter serverextension enable --py jupyter_http_over_ws
