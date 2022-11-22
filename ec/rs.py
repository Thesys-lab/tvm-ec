from bitmatrix_autoschedule import bitmatrix_multiply
from pyfinite.rs_code import RSCode
import numpy as np
from common import np_fill_bitmatrix
import argparse

def add_common_args(parser):
    parser.add_argument('-ecParity', '-P', type=int, default=4)
    parser.add_argument('-N', type=int, default=128000)
    parser.add_argument('-ecData', '-D', type=int, default=10)
    parser.add_argument('-ecW', type=int, default=8)
    parser.add_argument('--read_log_file', '-l', required=True,
                        help='Run the benchmark with tuned schedule in log file')
    parser.add_argument('--encode_only', '-e', action='store_true')

class ReedSolomon:
    def __init__(self, argv):
        self.argv = argv

    def encode(self, data):
        argv = self.argv
        K = argv.ecData * argv.ecW
        M = argv.ecParity * argv.ecW
        args = {
            'ecParity': argv.ecParity,
            'N': argv.N,
            'ecData': argv.ecData,
            'ecW': argv.ecW,
            'log_file': argv.read_log_file,
            'verbose': 0
        }
        self.encoder = RSCode(argv.ecData, argv.ecParity)
        encoder = np.array(self.encoder.CreateEncoderBitMatrix(argv.ecW)).astype(np.uint8)
        encoder = np_fill_bitmatrix(encoder)
        print(encoder.shape)
        parity = bitmatrix_multiply(args, encoder, data)
        print(parity)
        return np.concatenate((data, parity), axis=0)

    def decode(self, remain, drop):
        argv = self.argv
        K = argv.ecData * argv.ecW
        M = argv.ecParity * argv.ecW
        args = {
            'ecParity': argv.ecParity,
            'N': argv.N,
            'ecData': argv.ecData,
            'ecW': argv.ecW,
            'log_file': argv.read_log_file,
            'verbose': 0
        }
        decoder = np.array(self.encoder.CreateDecoderBitMatrix(argv.ecW, drop)).astype(np.uint8)
        decoder = np_fill_bitmatrix(decoder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    argv = parser.parse_args()

    K = argv.ecData * argv.ecW
    rs = ReedSolomon(argv)
    data = np.random.randint(np.iinfo(np.uint8).max,
                                size=(K, argv.N)).astype(np.uint8)
    msg = rs.encode(data)
    drop = (1, 2, 3, 4)
    # remain = np.delete(msg, drop, axis=0)
    print(remain.shape)
    recover = rs.decode(remain, drop)

    print(np.array_equal(data, recover))