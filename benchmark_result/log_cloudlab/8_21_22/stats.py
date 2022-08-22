import pandas as pd
import json
import matplotlib.pyplot as plt


with open("log_cloudlab/8_21_22/decode.json") as f:
    data = json.load(f)
a = pd.DataFrame(data)
a.drop('log_file', axis=1, inplace=True)
a.drop('tune_num_trials_total', axis=1, inplace=True)
a.drop('execution_time(s)', axis=1, inplace=True)
a.drop('ecParity', axis=1, inplace=True)
print(a)

a.to_csv('log_cloudlab/8_21_22/stats.csv')

# x = df_2['ecData'].tolist()
# y_2 = df_2['bandwidth(MB/s)'].tolist()
# y_3 = df_3['bandwidth(MB/s)'].tolist()

x = a['ecData'].tolist()
y = a['bandwidth(MB/s)'].tolist()
std = a['std'].tolist()

# plt.plot(x, y_2, label = 'ecParity=2')
# plt.plot(x, y_3, label = 'ecParity=3')
# plt.legend()
# plt.savefig('log_cloudlab/8_6_22/stats.png')

# from benchmark_result/log_cloudlab/8_6_22/std.py
ecP_3_mean = [15798.03603839614, 16147.651402719292, 11023.844099314276, 9197.340634458993, 8773.740885869449, 7821.220755796575, 7046.363918296148, 6674.2877487462065]
ecP_3_std = [1121.259468251426, 922.8241167918006, 435.9292191843391, 360.51522788643916, 252.67102603440495, 180.31390437616494, 143.72563679125028, 123.05200799532709]

with open("log_cloudlab/8_21_22/xorslp.json") as f:
    data = json.load(f)
b = pd.DataFrame(data)
b.drop('ecParity', axis=1, inplace=True)

x_xor = b['ecData'].tolist()
y_xor = b['bandwidth(MB/s)'].tolist()
std_xor = b['std'].tolist()

plt.errorbar(x, y, std, fmt='-o', label = 'decode')
plt.errorbar(x, ecP_3_mean, ecP_3_std, fmt='-o', label = 'encode')
plt.errorbar(x_xor, y_xor, std_xor, fmt='-o', label = 'xorslp_dec')
plt.legend()
plt.ylim([0, max(y+ecP_3_mean)*1.1])
plt.savefig('log_cloudlab/8_21_22/stats.png')