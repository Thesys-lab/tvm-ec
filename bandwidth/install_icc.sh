#!/bin/bash

# install Intel oneapi in Ubuntu under directory /opt/intel/oneapi
# reference: https://estuarine.jp/2021/03/install-oneapi/?lang=en

CURRDIR=$PWD
# Add package repository
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo apt-key add -
sudo apt-add-repository 'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'

# Install run-time packages
sudo apt-get update
sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero intel-media-va-driver-non-free libmfx1

# OPTIONAL: Install developer packages
sudo apt-get install -y libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev level-zero-dev

cd /tmp
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update

sudo apt install -y intel-basekit
sudo apt install -y intel-hpckit


source /opt/intel/oneapi/setvars.sh
# adding this line to ~/.bashrc
echo -e '\n# enable Intel OneApi\n/opt/intel/oneapi/setvars.sh' >> ~/.bashrc

cd $CURRDIR