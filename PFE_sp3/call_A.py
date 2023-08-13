import numpy as np
import pandas as pd
import os.path
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



fileE = 'potflow3dperenergy.txt'
outputE = open(fileE,'r')
tijd = []
A_data = []

for line in outputE:
    words = line.split()
    tijd.append(float(words[0]))
    A_data.append(float(words[5]))



tt10=np.array(tijd)
A10=np.array(A_data)
outputE.close()


qq=int(50)
lqq=int(len(A10)/qq)
max1=np.zeros((lqq+1,))
A1=np.zeros((lqq+1,))
A1[0]=A10[0]

for jj in range(lqq):
    A1[jj+1]=sum(A10[jj*qq:(jj+1)*qq])/qq
t1=np.linspace(tt10[0],tt10[-1],lqq+1)





plt.figure(figsize=(10,6))

plt.plot(t1,A1,label=r'CG2', linewidth='4')
plt.xlabel(' $t (s)$ ',size=16)
plt.legend(loc='lower right',fontsize="14")
plt.ylabel( '$A(m)$ ',size=16)
plt.grid()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # [0 10 0.0010082 0.0010086])
plt.savefig('A_sp3.png')
plt.show(block=True)
# plt.pause(0.001)
plt.gcf().clear()

# print("Finished program!")