import pandas as pd
import json
import matplotlib.pyplot as plt


with open("log_cloudlab/5_10_22/ecData_comp.json") as f:
    data = json.load(f)
with open("log_cloudlab/5_10_22/ecData_8.json") as f:
    data_8 = json.load(f)
a = pd.DataFrame(data)
a.drop('log_file', axis=1, inplace=True)
a.drop('tune_num_trials_total', axis=1, inplace=True)
a.drop('execution_time(s)', axis=1, inplace=True)
a.at[5, 'bandwidth(MB/s)'] = data_8[0]['bandwidth(MB/s)']
print(a)

a.to_csv('log_cloudlab/5_10_22/stats.csv')

df_2 = a.loc[a['ecParity'] == 2]
df_3 = a.loc[a['ecParity'] == 3]

print(df_2)

x = df_2['ecData'].tolist()
y_2 = df_2['bandwidth(MB/s)'].tolist()
y_3 = df_3['bandwidth(MB/s)'].tolist()

plt.plot(x, y_2, label = 'ecParity=2')
plt.plot(x, y_3, label = 'ecParity=3')
plt.legend()
plt.savefig('log_cloudlab/5_10_22/stats.png')