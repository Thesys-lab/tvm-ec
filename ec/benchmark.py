from gemm_autoschedule import benchmark
from copy import deepcopy
import json
from progressbar import ProgressBar

def run_benchmark(m_size):
    result = []
    for m in m_size:
        a = {
            'M': m,
            'N': m,
            'K': m,
            'log_dir': 'log/trial_500/m_'+str(m)+'.json',
            'tune_num_trials_total': 500
        }

        out = benchmark(a)
        a['execution_time(s)'] = out[0]
        a['bandwidth(MB/s)'] = out[1]

        result.append(deepcopy(a))

    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    run_benchmark((32, 64, 128, 256, 512, 1024, 2048))
    # run_benchmark((32,))