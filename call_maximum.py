"Maximum evolution of eta against time"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

soliton_number="SP3"        # choices=["SP2", "SP3"]
#SP2: Two solitons interaction;
#SP3: Three solitons interaction

if soliton_number=="SP3":   # initial time t0 
    t0 = -200
else:
    t0 = 0    
    
""" ________________ Time variable ________________ """
dt = 0.005                      # time step
T = 0.1                         # duration time
tt = np.arange(t0,t0+0.1,dt)    # time variable

""" ________________ Recall energy ________________ """
df = pd.read_csv(r"data123/max.csv")
max1 = df.to_numpy()

""" ________________ Plot evolution ________________ """
plt.title(r"Maximum", fontsize=20) 
plt.plot(tt, max1, linewidth=5, color='black')
plt.xticks(fontsize=13, weight='bold')
plt.yticks(fontsize=13, weight='bold')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,1))
plt.xlabel("time", fontsize=15, weight='bold')
plt.grid()

plt.savefig('data123/maximum.eps')
plt.show()
