from bitmatrix_autoschedule import benchmark as b_benchmark
from bitmatrix_autoschedule import benchmark_decode as b_benchmark_decode
from bitmatrix_autoschedule import get_best_benchmark as b_best_benchmark
from bitmatrix_autoschedule import get_best_benchmark_decode as b_best_benchmark_decode
from gemm_autoschedule import benchmark as g_benchmark
from gemm_autoschedule import get_best_benchmark as g_best_benchmark
from copy import deepcopy
import json
import argparse


def add_common_args(parser):
    parser.add_argument('-M', type=int, default=32)
    parser.add_argument('-N', type=int, default=128)
    parser.add_argument('-K', type=int, default=64)
    parser.add_argument('-ecParity', type=int, default=4)
    parser.add_argument('-ecData', type=int, default=8)
    parser.add_argument('-ecW', type=int, default=8)
    parser.add_argument('--read_log_file', default=None,
                        help='Run the benchmark with tuned schedule in log file')
    parser.add_argument('--log_dir', default='log/default/',
                        help='Specify the dir to store the log file')
    parser.add_argument('--result_file', default='result.json',
                        help='Specify the location to store the benchmark result')
    parser.add_argument('--tune_num_trials_total', type=int, default=500)
    parser.add_argument("--square_matrix_sizes", nargs="+", default=[32, ], type=int,
                        help='Specify the size of square matrix to tune on seperated by space')
    parser.add_argument("--config_file", default=None,
                        help='The configuration file (JSON) to run the experiment. \
                        should consists of a list of benchmark configs')
    parser.add_argument("--computation", default='b',
                        help='Specify the computation type: "b" for bitmatrix, "g" for gemm')
    parser.add_argument('--export', '-e', default=None, type=str)
    parser.add_argument('--decode', '-d', action='store_true')


def run_benchmark(argv):
    result = []
    if argv.computation == 'b':
        if argv.decode:
            benchmark = b_benchmark_decode
        else:
            benchmark = b_benchmark
    else:
        benchmark = g_benchmark
    a = {}
    if argv.config_file:
        with open(argv.config_file) as f:
            experiments = json.load(f)

        for exp in experiments:
            if argv.computation == 'b':
                a = {
                    'ecParity': exp.get('ecParity', None),
                    'N': exp['N'],
                    'ecData': exp['ecData'],
                    'ecW': exp['ecW'],
                    'log_file': argv.log_dir + 'P_' + str(exp.get('ecParity', 'X')) + '_n_' + str(exp['N']) + '_D_' + str(exp['ecData']) + '.json',
                    'tune_num_trials_total': exp['tune_num_trials_total'],
                    'bandwidth_size': 'h',
                    'export': argv.export
                }
            else:
                a = {
                    'M': exp.get('M', None),
                    'N': exp['N'],
                    'K': exp['K'],
                    'log_file': argv.log_dir + 'm_' + str(exp['M']) + '_n_' + str(exp['N']) + '_k_' + str(exp['K']) + '.json',
                    'tune_num_trials_total': exp['tune_num_trials_total'],
                    'export': argv.export
                }

            out = benchmark(a)
            a['execution_time(s)'] = out[0]
            a['bandwidth(MB/s)'] = out[1]
            
            try:
                a['std'] = out[2]
            except:
                pass

            result.append(deepcopy(a))
            with open(argv.result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

    else:
        for m in argv.square_matrix_sizes:
            if argv.computation == 'b':
                a = {
                    'ecParity': m,
                    'N': m,
                    'ecData': m,
                    'ecW': argv.ecW,
                    'log_file': argv.log_dir + 'P_n_d_' + str(m) + '.json',
                    'tune_num_trials_total': argv.tune_num_trials_total,
                    'bandwidth_size': 'h',
                    'export': argv.export
                }
            else:
                a = {
                    'M': m,
                    'N': m,
                    'K': m,
                    'log_file': argv.log_dir + 'm_' + str(m) + '.json',
                    'tune_num_trials_total': argv.tune_num_trials_total,
                    'export': argv.export
                }

            out = benchmark(a)
            a['execution_time(s)'] = out[0]
            a['bandwidth(MB/s)'] = out[1]

            result.append(deepcopy(a))

    with open(argv.result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def best_benchmark(argv):
    a = {}
    if argv.computation == 'b':
        if argv.decode:
            get_best_benchmark = b_best_benchmark_decode
        else:
            get_best_benchmark = b_best_benchmark
        a = {
            'ecParity': argv.ecParity,
            'N': argv.N,
            'ecData': argv.ecData,
            'ecW': argv.ecW,
            'log_file': argv.read_log_file,
            'bandwidth_size': 'h',
            'export': argv.export
        }
    else:
        get_best_benchmark = g_best_benchmark
        a = {
            'M': argv.M,
            'N': argv.N,
            'K': argv.K,
            'log_file': argv.read_log_file,
            'export': argv.export
        }
    print(get_best_benchmark(a))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    argv = parser.parse_args()

    if argv.read_log_file:
        best_benchmark(argv)
    else:
        run_benchmark(argv)
