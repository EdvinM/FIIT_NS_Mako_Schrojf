#!/bin/bash

pip --no-cache-dir install --upgrade \
    pip \
    setuptools

TF_PACKAGE="tensorflow"
TF_PACKAGE_VERSION="2.0.0"
pip install ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

pip install jupyter matplotlib
pip install jupyter_http_over_ws
pip install ipython==7.9.0
jupyter serverextension enable --py jupyter_http_over_ws

pip install pyyaml \
    numpy scipy matplotlib pandas sympy nose \
    opencv-python opencv-contrib-python \
    imutils

echo "Done"
