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
    python3-tk \
    apt-utils \
    vim \
    git \
    unzip 

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install python packages
RUN pip3 install --upgrade pip

# Install python packages
RUN pip install numpy \
    matplotlib \
    pandas \
    numpy \
    seaborn \
    tensorflow \
    chardet \
    scikit-learn \
    optuna \
    lightgbm \
    aif360 \
    fairgbm \
    scikit-lego[cvxpy] \
    fairlearn



# docker build -t credit:$USER -f Dockerfile --build-arg OUTSIDE_GROUP=`/usr/bin/id -ng $USER` --build-arg OUTSIDE_GID=`/usr/bin/id -g $USER` --build-arg OUTSIDE_USER=$USER --build-arg OUTSIDE_UID=$UID .


# docker run -it --userns=host --name credit -v /work/$USER:/work/$USER credit:$USER  /bin/bash

# docker exec -ti -u $USER credit bash

