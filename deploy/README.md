# deploy in C++
To execute the deploy exmaple
```bash
make
./cpp_deploy_test <P> <D> <n>
```
`P`, `D`, `n`: erasure code parameters. Supports range `P`: [2, 4], `N`: [8, 10], `n`: 16000 on the hardware platform specified in the paper

## To export to custom platform
Need to first autotune custom TVM schedule for the hardware platform, with `-e` specified to the output `.so` file
```bash
cd ec
python3 ../ec/benchmark.py \
        -ecParity <ecParity> -ecData <ecData> -N <ecN> \
        --log_dir <log_dir> --result_file <result_file> --tune_num_trials_total <num_trials> \
        -e <export_file>
```
Then, modify `cpp_deploy.cc` to read from the exported `.so`