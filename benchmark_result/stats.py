import pandas as pd
import json


with open("log_cloudlab/5_10_22/ecData_comp.json") as f:
    data = json.load(f)
a = pd.DataFrame(data)
a.drop('log_file', axis=1, inplace=True)
a.drop('tune_num_trials_total', axis=1, inplace=True)
a.drop('execution_time(s)', axis=1, inplace=True)
print(a)

a.to_csv('log_cloudlab/5_10_22/stats.csv')