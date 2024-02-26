FROM ubuntu:22.04

ARG OUTSIDE_GID
ARG OUTSIDE_UID
ARG OUTSIDE_USER
ARG OUTSIDE_GROUP

RUN groupadd --gid ${OUTSIDE_GID} ${OUTSIDE_GROUP}
RUN useradd --create-home --uid $OUTSIDE_UID --gid $OUTSIDE_GID $OUTSIDE_USER

ENV SHELL=/bin/bash

WORKDIR /work/

# Build with some basic utilities
RUN apt-get update 

RUN apt-get install -y \
    python3-pip \
    apt-utils \
    vim \
    git \
    unzip 

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo
RUN apt-get install -y python3-tk

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install python packages
RUN pip3 install --upgrade pip

# Install python packages
RUN pip install numpy \
    matplotlib \
    pandas==2.0.3 \
    numpy \
    seaborn \
    tensorflow \
    chardet \
    scikit-learn==1.4.0 \
    optuna \
    lightgbm==4.3.0 \
    aif360 \
    fairgbm \
    scikit-lego[cvxpy] \
    fairlearn

# Install jupyter
RUN pip install jupyter \
    jupyterlab  \
    notebook

# Install credit_pipeline inside the container
# If is already cloned
# RUN pip install -e /work/$OUTSIDE_USER/credit_pipeline (need to test this)

# docker build -t credit:$USER -f Dockerfile --build-arg OUTSIDE_GROUP=`/usr/bin/id -ng $USER` --build-arg OUTSIDE_GID=`/usr/bin/id -g $USER` --build-arg OUTSIDE_USER=$USER --build-arg OUTSIDE_UID=$UID .

# Without jupyter:
# docker run -it --userns=host --name credit -v /work/$USER:/work/$USER credit:$USER  /bin/bash

# With jupyter:
# docker run -it --userns=host --name credit -v /work/$USER:/work/$USER -p 30001:30001 credit:$USER  /bin/bash

# To enter the container as non-root:
# docker exec -ti -u $USER credit bash

# To run jupyter:
# docker exec -ti -u $USER credit bash -c "jupyter-lab --port 30001 --ip 0.0.0.0"