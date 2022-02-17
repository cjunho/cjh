"Maximum evolution of eta against time"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" ________________ Recall maximum and time ________________ """
df = pd.read_csv("data/max.csv")
max1 = df.to_numpy()

df2 = pd.read_csv("data/time.csv")
tt = df2.to_numpy()

""" ________________ Plot evolution of maximum ________________ """
plt.title("Maximum", fontsize=20) 
plt.plot(tt, max1, linewidth=5, color='black')
plt.xticks(fontsize=13, weight='bold')
plt.yticks(fontsize=13, weight='bold')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,1))
plt.xlabel("time", fontsize=15, weight='bold')
plt.grid()

plt.savefig('data/maximum.eps')
plt.show()
