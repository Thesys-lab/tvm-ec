from gemm_autoschedule import benchmark, get_best_benchmark
from copy import deepcopy
import json
import argparse


def add_common_args(parser):
    parser.add_argument('-M', type=int, default=32)
    parser.add_argument('-N', type=int, default=128)
    parser.add_argument('-K', type=int, default=64)
    parser.add_argument('--read_log_file', default=None,
                    help='Run the benchmark with tuned schedule in log file')
    parser.add_argument('--log_dir', default='log/default/',
                    help='Specify the dir to store the log file')
    parser.add_argument('--result_file', default='result.json',
                    help='Specify the location to store the benchmark result')
    parser.add_argument('--tune_num_trials_total', type=int, default=500)
    parser.add_argument("--square_matrix_sizes", nargs="+", default=[32,], type=int,
                    help='Specify the size of square matrix to tune on seperated by space')


def run_benchmark(argv):
    result = []
    for m in argv.square_matrix_sizes:
        a = {
            'M': m,
            'N': m,
            'K': m,
            'log_file': argv.log_dir+'m_'+str(m)+'.json',
            'tune_num_trials_total': argv.tune_num_trials_total
        }

        out = benchmark(a)
        a['execution_time(s)'] = out[0]
        a['bandwidth(MB/s)'] = out[1]

        result.append(deepcopy(a))

    with open(argv.result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def best_benchmark(argv):
    a = {
            'M': argv.M,
            'N': argv.N,
            'K': argv.K,
            'log_file': argv.read_log_file,
    }
    print(get_best_benchmark(a))


if __name__ == '__main__':
    # run_benchmark((2048,))
    # run_benchmark((32,))
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    argv = parser.parse_args()

    if argv.read_log_file:
        best_benchmark(argv)
    else:
        run_benchmark(argv)
