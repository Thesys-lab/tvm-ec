#!/bin/bash

echo "autotuning in progress, this might take 1-2 days or longer"
echo "please dir platform/ for autotuning progress"
echo "should contain 6 files with 20000 lines each file when done"
echo ""
echo "in progress..."

bak_PYTHONPATH=$PYTHONPATH
export PYTHONPATH=../ec

for ecParity in 2 3 4
do
    for ecData in 8 9 10
    do
        echo -n "Autotune Parity=$ecParity Data=$ecData..."
        python3 ../ec/benchmark.py \
        -ecParity $ecParity -ecData $ecData -N 128000 \
        --log_dir platform/ --result_file tmp.json --tune_num_trials_total 20000 >/dev/null 2>&1
        echo "Done"
    done
done

rm tmp.json

echo "Autotune Done!!! Getting benchmark results:"

N=128000
echo "format: (runtime (s), bandwidth (MB/s), bandwidth std)"
echo "N = $N"
for ecParity in 2 3 4
do
    for ecData in 8 9 10
    do
        echo "Parity=$ecParity Data=$ecData"
        python3 ../ec/benchmark.py \
        --read_log_file platform/P_${ecParity}_n_${N}_D_${ecData}.json \
        -ecParity $ecParity -N $N -ecData $ecData -v 0
    done
done

export PYTHONPATH=$bak_PYTHONPATH