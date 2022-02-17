# Adapted from https://github.com/hogepodge/tvm-docker/blob/main/Dockerfile
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

#USER root

# Make apt non-interactive to keep installation from being blocked
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}"

# Bring the image up to date
RUN apt-get dist-upgrade -y \
 && apt update -y \
 && apt upgrade -y \
 && apt install -y git

ADD config.sh config.sh
ADD apt.txt apt.txt
ADD pip.txt pip.txt

# install sudo and don't fail
RUN yes O | apt install -y sudo

# Create the tvm user
#RUN groupadd -r tvm -g 3604 \
# && useradd -u 3604 -r -g tvm -m -c "TVM user" -s /bin/bash tvm \
# && usermod -aG sudo tvm \
# && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN ./config.sh

#USER tvm

ENV THREADS=8
