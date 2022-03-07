import argparse
from ctypes import sizeof
from common import get_tvm_target_string
import numpy as np
import tvm
from tvm import te, auto_scheduler


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    C = te.placeholder((M, N), name="C", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    matmul = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((M, N), lambda i, j: matmul[i, j] + C[i, j], name="out")

    return [A, B, C, out]


def add_common_args(parser):
    parser.add_argument('-M', type=int, default=32)
    parser.add_argument('-N', type=int, default=128)
    parser.add_argument('-K', type=int, default=64)
    parser.add_argument('--log_dir', default='log/matmul.json',
                    help='Directory to save log to')
    parser.add_argument('--tune_num_trials_total', type=int, default=10)
    parser.add_argument('--tune_verbose', type=int, default=1)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    argv = parser.parse_args()

    target = get_tvm_target_string()

    M = argv.M
    N = argv.N
    K = argv.K

    task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(M, N, K, "float32"), target=target)

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
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, N)).astype(np.float32)
    c_np = np.random.uniform(size=(M, N)).astype(np.float32)
    out_np = a_np.dot(b_np) + c_np

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev)
    func(a_tvm, b_tvm, c_tvm, out_tvm)

    # Check results
    np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    ex_time = np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results)
    print(
        "Execution time of this operator: %.6f s"
        % (ex_time)
    )

    # bandwidth = ((M*N)+(N*K)+(K*M))*4/(1024**2)/ex_time
    bandwidth = (a_np.size+b_np.size+c_np.size+out_np.size)*out_np.itemsize/(1024**2)/ex_time
    print("Bandwidth: %.3f MB/s"% (bandwidth))


if __name__ == '__main__':
    main()