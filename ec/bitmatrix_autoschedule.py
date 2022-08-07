import argparse
from common import get_tvm_target_string, export_lib
import numpy as np
from statistics import mean
import tvm
from tvm import te, auto_scheduler
from common import np_bitmatrix, np_expand_bitmatrix

xor = te.comm_reducer(
    lambda x,
    y: x ^ y,
    lambda t: tvm.tir.const(
        0,
        dtype=t),
    name="xor")

# def map(idx, ecW, A):
#     interm = A & (1 << (ecW - idx - 1))
# return tir.Select(interm > 0, tvm.tir.const(0xff, dtype='uint8'),
# tvm.tir.const(0, dtype='uint8'))

# @auto_scheduler.register_workload  # Note the auto_scheduler decorator
# def bitmatrix(M, N, K, ecW, dtype):
#     ecData = K//ecW
#     A = te.placeholder((M, ecData), name="A", dtype=dtype)
#     B = te.placeholder((K, N), name="B", dtype=dtype)

#     A_expand = te.compute((M, K), lambda i, j: map(j%ecW, ecW, A[i, te.floordiv(j, ecW)]), name="expand")

#     k = te.reduce_axis((0, K), name="k")
#     bitmul = te.compute(
#         (M, N),
#         lambda i, j: xor(A_expand[i, k] & B[k, j], axis = k),
#         name="bitmul",
#         attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
#     )

#     return [A, B, bitmul]


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def bitmatrix(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    bitmul = te.compute(
        (M, N),
        lambda i, j: xor(A[i, k] & B[k, j], axis=k),
        name="bitmul",
        # enable automatic layout transform for tensor B
        attrs={"layout_free_placeholders": [B]},
    )
    # out = te.compute((M, N), lambda i, j: te.popcount(bitmul[i, j])&0b1, name="out")

    return [A, B, bitmul]


def print_basic_schedule(M, N, K, dtype):
    # declare a matrix element-wise multiply
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    bitmul = te.compute(
        (M, N),
        lambda i, j: xor(A[i, k] & B[k, j], axis=k),
        name="bitmul",
        # enable automatic layout transform for tensor B
        attrs={"layout_free_placeholders": [B]},
    )
    # out = te.compute((M, N), lambda i, j: te.popcount(bitmul[i, j])&0b1, name="out")

    s = te.create_schedule([bitmul.op])
    # lower will transform the computation from definition to the real
    # callable function. With argument `simple_mode=True`, it will
    # return you a readable C like statement, we use it here to print the
    # schedule result.
    print(tvm.lower(s, [A, B, bitmul], simple_mode=True))


def add_common_args(parser):
    parser.add_argument('-ecParity', '-P', type=int, default=4)
    parser.add_argument('-N', type=int, default=128)
    parser.add_argument('-ecData', '-D', type=int, default=8)
    parser.add_argument('-ecW', type=int, default=8)
    parser.add_argument('--log_dir', default='log/bitmatrix.json',
                        help='Directory to save log to')
    parser.add_argument('--tune_num_trials_total', type=int, default=10)
    parser.add_argument('--tune_verbose', type=int, default=1)
    parser.add_argument('--bandwidth_size', default='h')
    parser.add_argument(
        '--print_basic',
        '-p',
        default=False,
        action='store_true')


def benchmark(argv):
    target = get_tvm_target_string()

    ecParity = argv['ecParity']
    ecData = argv['ecData']
    ecW = argv['ecW']
    N = argv['N']
    M = ecParity * ecW
    K = ecData * ecW

    task = tvm.auto_scheduler.SearchTask(
        func=bitmatrix, args=(
            M, N, K, "uint8"), target=target)

    log_file = argv['log_file']
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=argv['tune_num_trials_total'],
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )

    task.tune(tune_option)
    sch, args = task.apply_best(log_file)

    func = tvm.build(sch, args, target)
    a_np = np.random.randint(
        np.iinfo(
            np.uint8).max,
        size=(
            M,
            ecData)).astype(
                np.uint8)
    a_np_expanded = np_expand_bitmatrix(a_np)
    b_np = np.random.randint(
        np.iinfo(
            np.uint8).max,
        size=(
            K,
            N)).astype(
                np.uint8)
    # out_np = np_bitmatrix(M, N, K, a_np, b_np)
    out_np = np.random.uniform(size=(M, N)).astype(np.uint8)

    if argv["export"]:
        export_lib(func, argv["export"])

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np_expanded, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev, dtype="uint8")
    func(a_tvm, b_tvm, out_tvm)

    # Check results
    # np.testing.assert_equal(out_np, out_tvm.numpy())

    evaluator = func.time_evaluator(
        func.entry_name, dev, number=100, repeat=10)
    ex_time = np.median(evaluator(a_tvm, b_tvm, out_tvm).results)

    bandwidth = 0
    if argv['bandwidth_size'] == 'f':
        bandwidth = (a_np.size + b_np.size + out_np.size) * \
            out_np.itemsize / (1024**2) / ex_time
    else:
        bandwidth = (b_np.size) * out_np.itemsize / (1024**2) / ex_time
    return (ex_time, bandwidth)


def get_best_benchmark(argv):
    target = get_tvm_target_string()

    ecParity = argv['ecParity']
    ecData = argv['ecData']
    ecW = argv['ecW']
    N = argv['N']
    M = ecParity * ecW
    K = ecData * ecW

    task = tvm.auto_scheduler.SearchTask(
        func=bitmatrix, args=(
            M, N, K, "uint8"), target=target)
    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = argv['log_file']

    sch, args = task.apply_best(log_file)

    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

    func = tvm.build(sch, args, target)
    a_np = np.random.randint(
        np.iinfo(
            np.uint8).max,
        size=(
            M,
            ecData)).astype(
                np.uint8)
    a_np_expanded = np_expand_bitmatrix(a_np)
    b_np = np.random.randint(
        np.iinfo(
            np.uint8).max,
        size=(
            K,
            N)).astype(
                np.uint8)
    # out_np = np_bitmatrix(M, N, K, a_np, b_np)
    out_np = np.random.uniform(size=(M, N)).astype(np.uint8)

    if argv["export"]:
        export_lib(func, argv["export"])

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np_expanded, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, dtype="uint8", device=dev)
    func(a_tvm, b_tvm, out_tvm)

    # Check results
    # np.testing.assert_equal(out_np, out_tvm.numpy())

    ex_time = list()
    for _ in range(50):
        evaluator = func.time_evaluator(
            func.entry_name, dev, number=100, repeat=10)
        ex_time.append(np.mean(evaluator(a_tvm, b_tvm, out_tvm).results))

    ex_time = mean(ex_time)
    bandwidth = 0
    if argv['bandwidth_size'] == 'f':
        bandwidth = (a_np.size + b_np.size + out_np.size) * \
            out_np.itemsize / (1024**2) / ex_time
    else:
        bandwidth = (b_np.size) * out_np.itemsize / (1024**2) / ex_time
    return (ex_time, bandwidth)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    argv = parser.parse_args()

    target = get_tvm_target_string()

    ecParity = argv.ecParity
    ecData = argv.ecData
    ecW = argv.ecW
    N = argv.N
    M = ecParity * ecW
    K = ecData * ecW

    if argv.print_basic:
        print_basic_schedule(M, N, K, "uint8")
        exit(0)

    task = tvm.auto_scheduler.SearchTask(
        func=bitmatrix, args=(
            M, N, K, "uint8"), target=target)

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

    a_np = np.random.randint(
        np.iinfo(
            np.uint8).max,
        size=(
            M,
            ecData)).astype(
                np.uint8)
    a_np_expanded = np_expand_bitmatrix(a_np)
    b_np = np.random.randint(
        np.iinfo(
            np.uint8).max,
        size=(
            K,
            N)).astype(
                np.uint8)
    out_np = np_bitmatrix(M, N, K, a_np, b_np)

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np_expanded, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, dtype="uint8", device=dev)
    func(a_tvm, b_tvm, out_tvm)

    # Check results
    np.testing.assert_equal(out_np, out_tvm.numpy())

    ex_time = list()
    for _ in range(50):
        evaluator = func.time_evaluator(
            func.entry_name, dev, number=100, repeat=10)
        ex_time.append(np.mean(evaluator(a_tvm, b_tvm, out_tvm).results))

    ex_time = mean(ex_time)
    print(
        "Execution time of this operator: %.6f s"
        % (ex_time)
    )

    bandwidth = (a_np.size + b_np.size + out_np.size) * \
        out_np.itemsize / (1024**2) / ex_time
    print("Bandwidth: %.3f MB/s" % (bandwidth))


if __name__ == '__main__':
    main()
