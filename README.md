Erasure Coding on TVM
==============================================
This repository contains a private fork of the [TVM](https://github.com/apache/tvm) project.
`release` branch is the functional branch for external use

## Setting up your local repository
Run the following commands to set up your local repository
```bash
git clone https://github.com/jackkosaian/ec-tvm.git
cd ec-tvm
git checkout release
git submodule update --init --recursive
```

<!-- We next set up the local repository to fetch changes to the upstream TVM:
```bash
git remote add upstream https://github.com/apache/tvm.git
git remote set-url --push upstream DISABLE
```

To pull in changes from the original TVM project, perform:
```bash
git fetch upstream
git merge upstream/main
``` -->

## Running in Docker container
We provide a container environment with all dependencies installed
To build and run the Docker container, execute:
```bash
./start_docker.sh
```
This will create a running container that has the current directory mounted at `/home/tvm`

To build TVM from within the container, run:
```bash
./build.sh
```
## Running in local environment
We also provide a script to build TVM outside of the container. The script is tested on Ubuntu 22.04:
```bash
./build_no_container.sh
```

## Example
To run an example of tuning a GEMM with meta schedule, run:
```bash
cd ec
python3 gemm.py
```
To tune an ec schedule, run
```bash
cd ec
python3 ../ec/benchmark.py \
        -ecParity <ecParity> -ecData <ecData> -N <ecN> \
        --log_dir <log_dir> --result_file <result_file> --tune_num_trials_total <num_trials>
```
`ecParity`, `ecData`: matrix parameter of erasure codes. `k = ecParity*w`, `m = ecData*w`\
`ecN`: parameter `n` of erasure code\
`log_dir`: directory to store log files, should end with `/`\
`result_file`: file to store the statistics\
`num_trials`: number of trials to autotune

See `reproduce/` for example usage. `reproduce/` also contains instruction to reproduce the result in the paper.

## Exporting to C++
TVM-EC supports exporting the tuned schedule to C++ shared library. See `deploy/` for details.