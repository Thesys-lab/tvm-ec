"""
Simple example of tuning and evaluating a GEMM with Meta Schedule
"""

import argparse
from enum import auto
from functools import partial
import numpy as np
import os
import shutil
import tvm
from tvm import te
from tvm.script import tir as T
from tvm.ir.module import IRModule
from tvm import tir
import tvm.meta_schedule as ms

from common import get_tvm_target_string


@tvm.script.ir_module
class GemmModule:
    """
    Define a GEMM with M=32, N=128, K=64 
    """
    @T.prim_func
    # def main(a: T.handle, b: T.handle, c: T.handle, m: T.int32, n: T.int32, k: T.int32):
    def main(a: T.handle, b: T.handle, c: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (32, 64), dtype="float32")
        B = T.match_buffer(b, (64, 128), dtype="float32")
        C = T.match_buffer(c, (32, 128), dtype="float32")

        for i, j, k in T.grid(32, 128, 64):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vk, vj])

def autotune(args, sch, tgt_string):
    workdir = args.outdir
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)

    sch_new: tir.Schedule = ms.tune_tir(
        mod=sch.mod,
        target=tvm.target.Target(tgt_string),
        config=ms.ReplayTraceConfig(
            num_trials_per_iter=args.tune_num_trials_per_iter,
            num_trials_total=args.tune_num_trials_total,
        ),
        work_dir=workdir,
        runner=ms.runner.LocalRunner(),
        task_name='simple',
    )

    return sch_new


def eval(args, sch, name, tgt_string, arg_list_fn):
    mod = tvm.build(sch.mod, target=tgt_string)
    tgt = tvm.target.Target(target=tgt_string, host=tgt_string)
    dev = tvm.device(tgt.kind.name, 0)
    evaluator = mod.time_evaluator(mod.entry_name, dev, number=args.num_it)
    mod_args = arg_list_fn()
    mean_time = evaluator(*mod_args).mean
    print('%s: %f' % (name, mean_time))


def gen_mod_args(M, N, K):
    arg_list = [
        tvm.nd.array(np.arange(M * K).reshape(M, K).astype("float32")),
        tvm.nd.array(np.arange(K * N).reshape(K, N).astype("float32")),
        tvm.nd.array(np.zeros(M * N).reshape(M, N).astype("float32"))
    ]
    return arg_list


def add_common_args(parser):
    parser.add_argument('--tune', action='store_true',
                    help='Whether to tune the module.')
    parser.add_argument('--num_it', type=int, default=100,
                    help='Number of iterations of evaluation to run')
    parser.add_argument('--pdb', action='store_true',
                    help='Whether to pause in certain portions of the script')
    parser.add_argument('--debug', action='store_true',
                    help='Whether to print debug information')
    parser.add_argument('--outdir', default='/tmp/ms-mm-tir',
                    help='Directory to save to')
    parser.add_argument('--tune_num_trials_per_iter', type=int, default=64)
    parser.add_argument('--tune_num_trials_total', type=int, default=10)
    parser.add_argument('-M', type=int, default=32)
    parser.add_argument('-N', type=int, default=128)
    parser.add_argument('-K', type=int, default=64)


def main():
    @T.prim_func
    def gemm(a: T.handle, b: T.handle, c: T.handle, m: T.int32, n: T.int32, k: T.int32):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (m, k), dtype="float32")
        B = T.match_buffer(b, (k, n), dtype="float32")
        C = T.match_buffer(c, (m, n), dtype="float32")

        for i, j, k in T.grid(m, n, k):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vk, vj])


    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    M = args.M
    N = args.N
    K = args.K
    bitwidth = 8

    arg_gen_fn = partial(gen_mod_args, M, N, K)
    target_string = get_tvm_target_string()

    # func = GemmModule.main.specialize({m: M, n: N, k: K})
    ir_mod = GemmModule
    _, _, _, m, n, k = gemm.params
    gemm = gemm.specialize({m: M, n: N, k: K})
    # sch = tvm.tir.Schedule(ir_mod)
    sch = tvm.tir.Schedule(gemm)
    sch_new = autotune(args, sch, target_string)

    if args.debug:
        print(sch.mod.script())
        print(sch_new.mod.script())

    for name, sched in [('og', sch), ('tuned', sch_new)]:
        eval(args, sched, name, target_string, arg_gen_fn)


if __name__ == '__main__':
    main()