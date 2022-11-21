"""
Utility scripts
"""

import psutil
import subprocess
import numpy as np
from pathlib import Path


def get_tvm_target_string():
    """
    Get the target string to be used in TVM
    """
    llc_process = subprocess.Popen(('llc', '--version'), stdout=subprocess.PIPE)
    output = subprocess.check_output(('grep', 'Host CPU'), stdin=llc_process.stdout, universal_newlines=True)
    llc_process.wait()
    cpu_str = output.split()[-1]
    ncore = psutil.cpu_count() // 2
    tgt_string = 'llvm -mcpu=' + cpu_str + ' -num-cores ' + str(ncore)
    return tgt_string


def np_expand_bitmatrix(A):
    ecW = A.itemsize*8
    A_expand = np.zeros((A.shape[0], A.shape[1]*ecW)).astype(A.dtype)
    
    for i in range(A_expand.shape[0]):
        for j in range(A_expand.shape[1]):
            tar = A[i, j//ecW]
            idx = j%ecW
            interm = tar & (1 << (ecW - idx - 1))
            if interm > 0:
                A_expand[i, j] = ~0
            else:
                A_expand[i, j] = 0
    
    return A_expand

# convert the 1s in a bitmatrix to ~0
def np_fill_bitmatrix(A):
    ecW = A.itemsize*8
    A_expand = np.zeros((A.shape[0], A.shape[1])).astype(A.dtype)
    
    for i in range(A_expand.shape[0]):
        for j in range(A_expand.shape[1]):
            if A[i, j] > 0:
                A_expand[i, j] = ~0
    
    return A_expand

def np_bitmatrix(M, N, K, A, B):
    A = np_expand_bitmatrix(A)
    out = np.zeros([M, N], dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                out[i, j] = np.bitwise_xor(np.bitwise_and(A[i, k], B[k, j]), out[i, j])
    return out


def np_bitmatrix_popcount(M, N, K, A, B):
    out = np.zeros([M, N], dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                out[i, j] = np.bitwise_xor(np.bitwise_and(A[i, k], B[k, j]), out[i, j])
            out[i, j] = np.bitwise_and(out[i, j].bit_count(), 0b1)
    return out

def export_lib(func, lib_name):
    output_file = Path(lib_name)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    # export library
    func.export_library(lib_name)

if __name__ == '__main__':
    A = np.array(
        [[1,2,3],
         [4,5,6],
         [7,8,9]]
    ).astype(np.uint8)
    B = np.array(
        [[1,2,3],
         [4,5,6],
         [7,8,9],
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [1,2,3],
         [4,5,6],
         [7,8,9]]
    ).astype(np.uint8)
    out = np_bitmatrix(3, 3, 24, A, B)
    print(out)
