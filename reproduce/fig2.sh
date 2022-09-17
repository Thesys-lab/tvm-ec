#!/bin/bash

N=128000
echo "format: (runtime (s), bandwidth (MB/s), bandwidth std)"
echo "N = $N"
for ecParity in 2 3 4
do
    for ecData in 8 9 10
    do
        echo "Parity=$ecParity Data=$ecData"
        python3 ../ec/benchmark.py \
        --read_log_file log/ecP$ecParity/P_${ecParity}_n_${N}_D_${ecData}.json \
        -ecParity $ecParity -N $N -ecData $ecData -v 0
    done
done