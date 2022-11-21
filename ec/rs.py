from bitmatrix_autoschedule import bitmatrix_encode
from pyfinite import RSCode
import numpy as np
from common import np_fill_bitmatrix

def add_common_args(parser):
    parser.add_argument('-ecParity', '-P', type=int, default=4)
    parser.add_argument('-N', type=int, default=128000)
    parser.add_argument('-ecData', '-D', type=int, default=10)
    parser.add_argument('-ecW', type=int, default=8)
    parser.add_argument('--read_log_file', '-l', required=True,
                        help='Run the benchmark with tuned schedule in log file')
    parser.add_argument('--decode', '-d', action='store_true')

def encode(argv):
    K = argv.ecData * argv.ecW
    args = {
        'ecParity': argv.ecParity,
        'N': argv.N,
        'ecData': argv.ecData,
        'ecW': argv.ecW,
        'log_file': argv.read_log_file,
        'verbose': 0
    }
    encoder = RSCode(10, 4).CreateEncoderBitMatrix(8)
    bitmatrix = np.array(b).astype(np.uint8)
    bitmatrix = np_fill_bitmatrix(bitmatrix)
    print(bitmatrix)
    data = np.random.randint(np.iinfo(np.uint8).max,
                             size=(K, N)).astype(np.uint8)
    parity = bitmatrix_encode(argv, encoder, data)
    print(parity)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    argv = parser.parse_args()
    if argv.decode:
        print("Not implemented")
    else:
        encode(argv)