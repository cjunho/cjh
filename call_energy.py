"Energy evolution of BLE against time"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
""" ________________ Recall energy and time ________________ """
df = pd.read_csv("data/energy.csv")
energy = df.to_numpy()
energy = energy - energy[0]     # E(t)-E(t_0)

df2 = pd.read_csv("data/time.csv")
tt = df2.to_numpy()

""" ________________ Plot energy evolution ________________ """
plt.title("E(t)-E(%i)" %tt[0], fontsize=20) 
plt.plot(tt, energy, linewidth=5, color='black')
plt.xticks(fontsize=13, weight='bold')
plt.yticks(fontsize=13, weight='bold')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,1))
plt.xlabel("time", fontsize=15, weight='bold')
plt.grid()

plt.savefig('data/energy.eps')
plt.show()
