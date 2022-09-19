#!/bin/bash

while getopts p:d: flag
do 
    case "${flag}" in
        p) ecParity=${OPTARG};;
        d) ecData=${OPTARG};;
    esac
done

if [ -z "$ecParity" ] || [ -z "$ecData" ];
then
    echo "Please set the command line arguments!!!"
    echo "-p ecParity [2-4]"
    echo "-d ecData [8-10]"
    exit 1
fi

tgt_file=platform/P_${ecParity}_n_128000_D_${ecData}.json
if [ ! -f "${tgt_file}" ];
then
    tgt_file=log/ecP${ecParity}/P_${ecParity}_n_128000_D_${ecData}.json
    if [ ! -f "${tgt_file}" ];
    then
        echo "Cannot find a corresponding TVM log!!!"
        exit 1
    fi
fi

echo "getting best schedule from ${tgt_file}"
echo "format: (runtime (s), bandwidth (MB/s), bandwidth std)"
python3 ../ec/benchmark.py \
--read_log_file ${tgt_file} \
-ecParity $ecParity -N 128000 -ecData $ecData -v 0