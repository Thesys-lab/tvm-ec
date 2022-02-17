#!/bin/bash

IMAGE_NAME=ec-tvm

build_container() {
  cd .devcontainer
  docker build . -f Dockerfile-interactive -t $IMAGE_NAME
}

if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  build_container
fi

docker run -it --rm -v $(pwd):/home/tvm tvm
