import numpy as np
import pandas as pd
import os.path
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



df = pd.read_csv("C:/Users/amtjch/Desktop/pfe_sp2/pfe_sp2_short/maxA13000.csv")
t_short=df.to_numpy()[:,0]
maxA_short = df.to_numpy()[:,1]



df = pd.read_csv("C:/Users/amtjch/Desktop/pfe_sp2/pfe_sp2/maxA.csv")
t=df.to_numpy()[:,0]
maxA = df.to_numpy()[:,1]


df = pd.read_csv("C:/Users/amtjch/Desktop/pfe_sp2/pfe_sp2_long/maxA13000.csv")
t_long=df.to_numpy()[:,0]
maxA_long = df.to_numpy()[:,1]

'cg4'
df = pd.read_csv("C:/Users/amtjch/Desktop/pfe_sp2/pfe_sp2_cg4/maxA13000.csv")
t_cg4=df.to_numpy()[:,0]
maxA_cg4 = df.to_numpy()[:,1]


plt.figure(figsize=(10,6))
# 
plt.plot(t_short,maxA_short,label=r'CG2/$\Delta t$', linewidth='4')
plt.plot(t,maxA,label=r'CG2/$\frac{\Delta t}{2}$', linewidth='4')
plt.plot(t_long,maxA_long,'--',label=r'CG2/$\frac{\Delta t}{4}$', linewidth='4')
plt.plot(t_cg4,maxA_cg4,':',label=r'CG4/$\Delta t$', linewidth='4')



# plt.plot(tt4,max4/AA4,'--',label='Sx/CG3/St', linewidth='4')
plt.xlabel(' $t (s)$ ',size=16)
# plt.ylabel(' $E_{kin}, E_{pot}, E_{tot}$ ',size=16)
plt.legend(fontsize="16")
plt.ylabel( '$A$ ',size=16)
plt.grid()
# plt.axes([0,10,0.001,0.002])
#plt.yticks([0.0010082,0.0010083,0.0010084,0.0010085,0.0010086])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # [0 10 0.0010082 0.0010086])
plt.savefig('maxA_sp2.eps')
plt.show(block=True)
# plt.pause(0.001)
plt.gcf().clear()
