ARG FBUILD_VERSION=12.6.3
ARG FBUILD_SYSTEM=ubuntu
ARG FBUILD_SYSTEM_VERSION=22.04

FROM nvidia/cuda:${FBUILD_VERSION}-cudnn-devel-${FBUILD_SYSTEM}${FBUILD_SYSTEM_VERSION}

RUN apt-get update -y \ 
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y wget \
		libffi-dev  \
		gcc \
		git \
		build-essential \
		curl \
		tcl-dev \
		uuid-dev \
		liblzma-dev \
		libssl-dev \
		libsqlite3-dev \
		libbz2-dev \
		ffmpeg \
		libsm6 \
		libxext6

ARG FBUILD_USERNAME=fapannen
ARG FBUILD_PYTHON_MAJOR_VERSION=3
ARG FBUILD_PYTHON_MINOR_VERSION=10
ARG FBUILD_PYTHON_REV_VERSION=0
ARG FBUILD_PYTHON_VERSION=${FBUILD_PYTHON_MAJOR_VERSION}.${FBUILD_PYTHON_MINOR_VERSION}.${FBUILD_PYTHON_REV_VERSION}

RUN mkdir /home/${FBUILD_USERNAME}

RUN mkdir /opt/python${FBUILD_PYTHON_VERSION}

RUN wget https://www.python.org/ftp/python/${FBUILD_PYTHON_VERSION}/Python-${FBUILD_PYTHON_VERSION}.tgz
RUN tar -zxvf Python-${FBUILD_PYTHON_VERSION}.tgz
RUN cd Python-${FBUILD_PYTHON_VERSION} \
	&& ./configure --prefix=/usr --with-ensurepip=install \
	&& make \
	&& make install

# Delete the python source code and temp files
RUN rm Python-${FBUILD_PYTHON_VERSION}.tgz
RUN rm -r Python-${FBUILD_PYTHON_VERSION}/

# Now link it so that $python works
RUN ln -s /usr/bin/python${FBUILD_PYTHON_MAJOR_VERSION}.${FBUILD_PYTHON_MINOR_VERSION} /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN python -m pip install --upgrade pip

# Now prepare the python libraries
COPY requirements.txt /opt/base_python_requirements.txt
RUN pip install -r /opt/base_python_requirements.txt

# Setup Huggingface cache - make sure this variable
# points to your directory that is designated to be the cache.
# See README for more details.
ENV HF_HOME=/home/${FBUILD_USERNAME}/huggingface
