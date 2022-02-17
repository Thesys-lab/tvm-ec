"""
Utility scripts
"""

import psutil
import subprocess


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