import argparse

from matplotlib.pyplot import axis
from common import get_tvm_target_string
import numpy as np
import tvm
from tvm import te, auto_scheduler

xor = te.comm_reducer(lambda x, y: x ^ y, lambda t: tvm.tir.const(0, dtype=t), name="xor")

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def bitmatrix(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    bitmul = te.compute(
        (M, N),
        lambda i, j: xor(A[i, k] & B[k, j], axis = k),
        name="bitmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((M, N), lambda i, j: te.popcount(bitmul[i, j])&0b1, name="out")

    return [A, B, out]

def add_common_args(parser):
    parser.add_argument('-M', type=int, default=32)
    parser.add_argument('-N', type=int, default=128)
    parser.add_argument('-K', type=int, default=64)
    parser.add_argument('--log_dir', default='log/matmul.json',
                    help='Directory to save log to')
    parser.add_argument('--tune_num_trials_total', type=int, default=10)
    parser.add_argument('--tune_verbose', type=int, default=1)


def np_bitmatrix(M, N, K, A, B):
    out = np.zeros([M, N], dtype=np.uint8)
    for i in range(0, M):
        for j in range(0, N):
            for k in range (0, K):
                out[i, j] = np.bitwise_xor(np.bitwise_and(A[i, k], B[k, j]), out[i, j])
            out[i, j] = np.bitwise_and(out[i, j].bit_count(), 0b1)
    return out


def benchmark(argv):
    target = get_tvm_target_string()

    M = argv['M']
    N = argv['N']
    K = argv['K']

    task = tvm.auto_scheduler.SearchTask(func=bitmatrix, args=(M, N, K, "uint8"), target=target)

    log_file = argv['log_file']
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=argv['tune_num_trials_total'],
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )

    task.tune(tune_option)
    sch, args = task.apply_best(log_file)

    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=(M, K)).astype(np.uint8)
    b_np = np.random.uniform(size=(K, N)).astype(np.uint8)
    # out_np = np_bitmatrix(M, N, K, a_np, b_np)
    out_np = np.random.uniform(size=(M, N)).astype(np.uint8)

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev, dtype="uint8")
    func(a_tvm, b_tvm, out_tvm)

    # Check results
    # np.testing.assert_equal(out_np, out_tvm.numpy())

    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    ex_time = np.median(evaluator(a_tvm, b_tvm, out_tvm).results)

    # bandwidth = ((M*N)+(N*K)+(K*M))*4/(1024**2)/ex_time
    bandwidth = (a_np.size+b_np.size+out_np.size)*out_np.itemsize/(1024**2)/ex_time
    return (ex_time, bandwidth)


def get_best_benchmark(argv):
    target = get_tvm_target_string()

    M = argv['M']
    N = argv['N']
    K = argv['K']

    task = tvm.auto_scheduler.SearchTask(func=bitmatrix, args=(M, N, K, "uint8"), target=target)

    log_file = argv['log_file']
    
    sch, args = task.apply_best(log_file)

    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=(M, K)).astype(np.uint8)
    b_np = np.random.uniform(size=(K, N)).astype(np.uint8)
    # out_np = np_bitmatrix(M, N, K, a_np, b_np)
    out_np = np.random.uniform(size=(M, N)).astype(np.uint8)

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    out_tvm = tvm.nd.empty((M, N), device=dev, dtype='uint8')
    func(a_tvm, b_tvm, out_tvm)

    # Check results
    # np.testing.assert_equal(out_np, out_tvm.numpy())

    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    ex_time = np.median(evaluator(a_tvm, b_tvm, out_tvm).results)

    bandwidth = (a_np.size+b_np.size+out_np.size)*b_np.itemsize/(1024**2)/ex_time
    return (ex_time, bandwidth)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    argv = parser.parse_args()

    target = get_tvm_target_string()

    M = argv.M
    N = argv.N
    K = argv.K

    task = tvm.auto_scheduler.SearchTask(func=bitmatrix, args=(M, N, K, "uint8"), target=target)

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

    a_np = np.random.randint(np.iinfo(np.uint8).max, size=(M, K)).astype(np.uint8)
    b_np = np.random.randint(np.iinfo(np.uint8).max, size=(K, N)).astype(np.uint8)
    out_np = np_bitmatrix(M, N, K, a_np, b_np)

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, dtype="uint8", device=dev)
    func(a_tvm, b_tvm, out_tvm)

    # Check results
    np.testing.assert_equal(out_np, out_tvm.numpy())

    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    ex_time = np.median(evaluator(a_tvm, b_tvm, out_tvm).results)
    print(
        "Execution time of this operator: %.6f s"
        % (ex_time)
    )

    # bandwidth = ((M*N)+(N*K)+(K*M))*4/(1024**2)/ex_time
    bandwidth = (a_np.size+b_np.size+out_np.size)*out_np.itemsize/(1024**2)/ex_time
    print("Bandwidth: %.3f MB/s"% (bandwidth))


if __name__ == '__main__':
    main()