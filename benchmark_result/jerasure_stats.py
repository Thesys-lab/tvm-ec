import numpy as np
import argparse
import json

def generate_trace_data_size(d, p, N):
    return [
        [d*p*4, d*p*8*8*4, d*N*8*1, d*N*8*1], #[encoding matrix, encoding bitmatrix, data matrix, survivor matrix]
        [d*p*8*8*4, d*N*8*1], #[decoding bitmatrix, survivor matrix]
        [d*p*8*8*4, d*N*8*1, d*N*8*1, p*N*8*1] #[preprocess bitmatrix, data movement data matrix, data matrix, data movement parity matrix]
    ]

trace_data_annotation = [
    ["make encoding matrix", "matrix to bitmatrix conversion", "encoding function", "decoding function"],
    ["make decoding matrix", "bitmatrix multiplication function"],
    ["preprocess bitmatrix", "data movement for data matrix", "bitmatrix multiplication", "data movement for output matrix"]
]

def add_common_args(parser):
    parser.add_argument('--input_file', '-f', required=True)
    parser.add_argument('--timing_trace', '-t', required=True, type=int,
                            help="trace indicates which file is timed, range [0, 2]")
    parser.add_argument('--output_file', '-o', default=None, type=str)
    parser.add_argument('-N', type=int, default=128)
    parser.add_argument('--ecParity', '-P', type=int, default=4)
    parser.add_argument('--ecData', '-D', type=int, default=10)

def print_result(time_avg, time_std, throuput_avg, throuput_std, argv):
    header = ""
    for a in trace_data_annotation[argv.timing_trace]:
        header += a + " | "
    time_result = ""
    for time, std in zip(time_avg, time_std):
        time_result += str(time) + "/" + str(std) + " | "

    throughput_result = ""
    for throughput, std in zip(throuput_avg, throuput_std):
        throughput_result += str(throughput) + "/" + str(std) + " | "

    print(header)
    print("execution time average (microseconds)/std:")
    print(time_result)
    print("throughput average (MB/s)/std:")
    print(throughput_result)

def to_json(time_avg, time_std, throuput_avg, throuput_std, argv):
    out = {}
    for annotation, time, t_std, thru, thru_std in \
            zip(trace_data_annotation[argv.timing_trace], time_avg,
                time_std, throuput_avg, throuput_std):
        out[annotation] = {"time(microseconds)": time,
                           "time std": t_std,
                           "throughput(MB/s)": thru,
                           "throughput std": thru_std}

    with open(argv.output_file, "w") as f:
        json.dump(out, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    argv = parser.parse_args()

    ecParity = argv.ecParity
    ecData = argv.ecData
    N = argv.N

    trace_data_size = generate_trace_data_size(ecData, ecParity, N)

    with open(argv.input_file) as f:
        data = f.read().splitlines()

    time = np.array([[float(a) for a in line.split()] for line in data])
    
    throuput = trace_data_size[argv.timing_trace]/time

    time_avg = np.mean(time, axis=0)
    time_std = np.std(time, axis=0)
    throuput_avg = np.mean(throuput, axis=0)
    throuput_std = np.std(throuput, axis=0)

    print_result(time_avg, time_std, throuput_avg, throuput_std, argv)
    if (argv.output_file != None):
        to_json(time_avg, time_std, throuput_avg, throuput_std, argv)
