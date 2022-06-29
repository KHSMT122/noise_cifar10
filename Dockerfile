FROM tensorflow/tensorflow:latest-devel-gpu

WORKDIR /root

SHELL ["/bin/bash", "-c"]

RUN pip install keras && \
    pip install tensorflow-gpu && \
    pip install opencv-python
