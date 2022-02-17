#!/bin/bash

set -e

BUILDDIR=build

# Build TVM
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

cd ../python
python3 setup.py develop --user
