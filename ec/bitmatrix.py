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

"""
NOTE: The correct functionality of this script requires adding the following lines
to python/tvm/script/tir/__init__.pyi.

+def bitwise_xor(a: PrimExpr, b: PrimExpr) -> PrimExpr: ...
+def bitwise_and(a: PrimExpr, b: PrimExpr) -> PrimExpr: ...
"""


@tvm.script.ir_module
class ECModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (4, 8), dtype="uint8")
        B = T.match_buffer(b, (8, 65536), dtype="uint8")
        C = T.match_buffer(c, (4, 65536), dtype="uint8")

        for i, j, k in T.grid(4, 65536, 8):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.uint8(0)
                C[vi, vj] = T.bitwise_xor(C[vi, vj], T.bitwise_and(A[vi, vk], B[vk, vj], dtype="uint8"), dtype="uint8")
    

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
        tvm.nd.array(np.arange(M * K).reshape(M, K).astype("uint8")),
        tvm.nd.array(np.arange(K * N).reshape(K, N).astype("uint8")),
        tvm.nd.array(np.zeros(M * N).reshape(M, N).astype("uint8"))
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


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    M = 4
    N = 65536
    K = 8

    arg_gen_fn = partial(gen_mod_args, M, N, K)
    target_string = get_tvm_target_string()

    ir_mod = ECModule
    sch = tvm.tir.Schedule(ir_mod)
    sch_new = autotune(args, sch, target_string)

    if args.debug:
        print(sch.mod.script())
        print(sch_new.mod.script())

    for name, sched in [('og', sch), ('tuned', sch_new)]:
        eval(args, sched, name, target_string, arg_gen_fn)


if __name__ == '__main__':
    main()