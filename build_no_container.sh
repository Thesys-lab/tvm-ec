#!/bin/bash
# works on Ubuntu 22.04 (follow https://tvm.apache.org/docs/install/from_source.html)

BUILDDIR=build

# install dependencies
sudo apt update
sudo apt install -y cmake llvm ninja-build python3 python3-dev python3-setuptools python3-pip gcc libtinfo-dev \
    zlib1g-dev build-essential cmake libedit-dev libxml2-dev

mkdir -p $BUILDDIR

build_cfg=$BUILDDIR/config.cmake
new_cfg=cmake/config.cmake
need_cmake=0
if [ ! -f $build_cfg ]; then
  cp $new_cfg $BUILDDIR
  need_cmake=1
else
  DIFF=$(diff $new_cfg $build_cfg)
  if [ "$DIFF" != "" ]; then
    cp $new_cfg $BUILDDIR
    need_cmake=1
  fi
fi

echo "NEED CMAKE = $need_cmake"
cd $BUILDDIR
if [ $need_cmake -eq 1 ]; then
  cmake ../.
fi
make -j$(nproc)

# install python dependencies
pip3 install numpy decorator attrs tornado psutil 'xgboost==1.1.0' cloudpickle

cd ../python
python3 setup.py develop --user