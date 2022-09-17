ecP_2_mean = [20968.65559167231, 21031.14076501528, 15792.014160846657, 12553.284432765007, 10612.566974481726, 10470.992837924592, 8188.68411211635, 6181.980512048654]
ecP_2_std = [1544.8933378895915, 1562.3511974095286, 824.782093374023, 545.3569332144192, 324.8457046658973, 399.5444952031626, 223.01551589367006, 61.73826273445333]

ecP_3_mean = [15798.03603839614, 16147.651402719292, 11023.844099314276, 9197.340634458993, 8773.740885869449, 7821.220755796575, 7046.363918296148, 6674.2877487462065]
ecP_3_std = [1121.259468251426, 922.8241167918006, 435.9292191843391, 360.51522788643916, 252.67102603440495, 180.31390437616494, 143.72563679125028, 123.05200799532709]

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(3, 11, 1)

plt.errorbar(x, ecP_2_mean, ecP_2_std, fmt='-o', label = 'ecParity=2')
plt.errorbar(x, ecP_3_mean, ecP_3_std, fmt='-o', label = 'ecParity=3')
plt.legend()
plt.ylim([0, max(ecP_2_mean+ecP_3_mean)*1.1])
plt.savefig('stats_w_std.png')