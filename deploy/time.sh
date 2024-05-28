#!/bin/bash


N=128000
echo "format: (runtime (s), bandwidth (MB/s), bandwidth std)"
echo "N = $N"
for ecParity in 2 3 4
do
    for ecData in 8 9 10
    do
        echo "Parity=$ecParity Data=$ecData"
        /usr/bin/time -v ./cpp_deploy_test $ecParity $ecData 16000 >> time.log 2>&1
    done
done