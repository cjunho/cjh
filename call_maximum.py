"Maximum evolution of eta against time"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" ________________ Time variable ________________ """
t0 = -200                       # initial time t0 
dt = 0.005                      # time step
T = 0.1                         # duration time
tt = np.arange(t0,t0+T,dt)      # time variable
""" ________________ Recall energy ________________ """
df = pd.read_csv("data123/max.csv")
max1 = df.to_numpy()

""" ________________ Plot evolution ________________ """
plt.title("Maximum", fontsize=20) 
plt.plot(tt, max1, linewidth=5, color='black')
plt.xticks(fontsize=13, weight='bold')
plt.yticks(fontsize=13, weight='bold')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,1))
plt.xlabel("time", fontsize=15, weight='bold')
plt.grid()

plt.savefig('data123/maximum.eps')
plt.show()
