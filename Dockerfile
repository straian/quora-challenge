FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y pciutils

RUN pip install cython
RUN pip install pycocotools
RUN pip install scikit-image
RUN pip install matplotlib
RUN pip install psutil
RUN pip install nltk
RUN python -m nltk.downloader all
RUN pip install tensorflow-hub
RUN pip install keras-bert
RUN pip install tensorboard

COPY python python

