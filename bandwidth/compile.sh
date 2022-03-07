#!/bin/bash
# please use source to run this file


ICC="icc" 

if [[ $1 == "icc" ]]; then
    if ! command -v $ICC &> /dev/null; then
        echo "$ICC could not be found, calling install_icc.sh to install oneapi"
        source install_icc.sh
    fi

    make stream-icc
elif [[ $1 = "openmp" ]]; then
    make stream-openmp
elif [[ $1 = "stream" ]]; then
    make stream
elif [[ $1 = "clean" ]]; then
    make clean
else
    echo "argument unrecognized, choose from icc, openmp, stream, clean"
    exit 1
fi

export OMP_NUM_THREADS=$(($(cat /proc/cpuinfo | grep "cpu cores" | head -1 | tr -dc '0-9')))
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"