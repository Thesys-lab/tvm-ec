import pandas as pd
import json
import matplotlib.pyplot as plt

ecP_2_mean = [10470.992837924592, 8188.68411211635, 6181.980512048654]
ecP_2_std = [399.5444952031626, 223.01551589367006, 61.73826273445333]

ecP_3_mean = [7821.220755796575, 7046.363918296148, 6674.2877487462065]
ecP_3_std = [180.31390437616494, 143.72563679125028, 123.05200799532709]


with open("ecP4.json") as f:
    data = json.load(f)
a = pd.DataFrame(data)
a.drop('log_file', axis=1, inplace=True)
a.drop('tune_num_trials_total', axis=1, inplace=True)
a.drop('execution_time(s)', axis=1, inplace=True)
print(a)

a.to_csv('stats.csv')

x = a['ecData'].tolist()
y = a['bandwidth(MB/s)'].tolist()
std = a['std'].tolist()

with open("xorslp.json") as f:
    data = json.load(f)
b = pd.DataFrame(data)
df_2 = b.loc[b['ecParity'] == 2]
df_3 = b.loc[b['ecParity'] == 3]
df_4 = b.loc[b['ecParity'] == 4]

y_2 = df_2['bandwidth(MB/s)'].tolist()
std_2 = df_2['std'].tolist()
y_3 = df_3['bandwidth(MB/s)'].tolist()
std_3 = df_3['std'].tolist()
y_4 = df_4['bandwidth(MB/s)'].tolist()
std_4 = df_4['std'].tolist()

plt.errorbar(x, y, std, fmt='-o', label = 'tvm Parity=4')
plt.errorbar(x, ecP_3_mean, ecP_3_std, fmt='-o', label = 'tvm Parity=3')
plt.errorbar(x, ecP_2_mean, ecP_2_std, fmt='-o', label = 'tvm Parity=2')
plt.errorbar(x, y_2, std_2, fmt='-o', label = 'xorslp Parity=2')
plt.errorbar(x, y_3, std_3, fmt='-o', label = 'xorslp Parity=3')
plt.errorbar(x, y_4, std_4, fmt='-o', label = 'xorslp Parity=4')
plt.legend()
plt.ylim([0, max(y+ecP_3_mean+ecP_2_mean)*1.1])
plt.savefig('stats.png')