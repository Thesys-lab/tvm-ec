# Reproducing results

This directory is for the purpose of reproducing the benchmark results from figure 2 and figure 3 of our paper.

## Reproducing **TVM-EC** results

### **Preparing environments**
**Requirement**: [Apache-TVM](https://tvm.apache.org/) <br>
We provide several scripts that automates the installation of TVM into a docker container with all the required packages installed. <br>
<br>
To create and start the container: `./start_docker.sh` at project root. This generally runs without problem on Intel CPUs, but is not tested thoroughly on other CPUs. <br>
<br>
To use **CloudLab**: Note that the container is pretty large (13+ GB), which might not fit into system partition storage. We recommend attach more storage space to `/mydata` and run `./setup_cloudlab.sh` in the project root directory to change docker default directory there.<br>
<br>
To build **TVM**: After the container is running, execute `./build.sh` from inside the container to build **TVM** from scratch.

### **Benchmark**
If you are using exact the same hardware setup as described in the paper section **6.1**, directly execute `./fig2.sh` to get the benchmark result. This scripts loads pre-autotuned log files from `log/` directory so that you do not need to generate autotuned schedule yourself. Note that since encoding and decoding follows the same algorithm in our implementation, this is also the decoding benchmark for figure 3.

If not, we recommend autotune new schedules that are tailored for your platform. The pre-tuned schedules inside `log/` might not be optimal on other platforms. Executing `./fig2_platform.sh` generates the autotuned schedule for the specific hardware platform inside `platform/`. Note that it might take 1-2 days or longer to autotune all the configurations, we recommend running the script inside a **tmux** session.

## Reproducing **Xorslp-EC** and **ISA-L** results
Please follow [`xorslp/reproducing/README.md`](https://github.com/JiyuuuHuuu/xorslp_ec/blob/e8e5924f0229be6e99d524a83e49a4914c579edb/reproducing/README.md) to get benchmark results of implementations from other developers. We executed `throughput_sec75_1024blk.sh` to get the results of figure 2 and figure 3.