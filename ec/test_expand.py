import argparse
from common import get_tvm_target_string
import numpy as np
import tvm
from tvm import te, auto_scheduler
from tvm import tir


def map(idx, ecW, A):
    interm = A & (1 << (ecW - idx - 1))
    return tir.Select(interm > 0, tvm.tir.const(0xff, dtype='uint8'), tvm.tir.const(0, dtype='uint8'))

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def expand(M, K, ecW, dtype):
    ecData = K//ecW
    A = te.placeholder((M, ecData), name="A", dtype=dtype)
    out = te.compute((M, K), lambda i, j: map(j%ecW, ecW, A[i, te.floordiv(j, ecW)]), name="expand")

    return [A, out]

def add_common_args(parser):
    parser.add_argument('-M', type=int, default=1)
    parser.add_argument('-ecData', type=int, default=1)
    parser.add_argument('--tune_num_trials_total', type=int, default=10)
    parser.add_argument('--log_dir', default='log/expand.json',
                help='Directory to save log to')
    parser.add_argument('--tune_verbose', type=int, default=1)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    argv = parser.parse_args()

    target = get_tvm_target_string()

    M = argv.M
    ecData = argv.ecData
    ecW = 8 # for uint8
    K = ecData * ecW

    print("M=" + str(M) + " ecData=" + str(ecData) + " K=" + str(K))

    task = tvm.auto_scheduler.SearchTask(func=expand, args=(M, K, ecW, "uint8"), target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = argv.log_dir
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=argv.tune_num_trials_total,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

    func = tvm.build(sch, args, target)

    a_np = np.random.randint(np.iinfo(np.uint8).max, size=(M, ecData)).astype(np.uint8)

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=dev)
    out_tvm = tvm.nd.empty((M, K), dtype="uint8", device=dev)
    func(a_tvm, out_tvm)

    # Check results
    print(a_np)
    print(out_tvm.numpy())

    evaluator = func.time_evaluator(func.entry_name, dev, number=50)
    ex_time = np.median(evaluator(a_tvm, out_tvm).results)
    print(
        "Execution time of this operator: %.6f s"
        % (ex_time)
    )


if __name__ == '__main__':
    main()