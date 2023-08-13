import numpy as np
import pandas as pd
import os.path
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


fileE = 'potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
max_data = []
A_data = []

for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    
    max_data.append(float(words[4]))
    A_data.append(float(words[5]))



tt10=np.array(tijd)
max10=np.array(max_data)
A10=np.array(A_data)
outputE.close()


qq=int(50)
lqq=int(len(max10)/qq)
max1=np.zeros((lqq+1,))
A1=np.zeros((lqq+1,))
max1[0]=max10[0]
A1[0]=A10[0]

for jj in range(lqq):
    max1[jj+1]=sum(max10[jj*qq:(jj+1)*qq])/qq
    A1[jj+1]=sum(A10[jj*qq:(jj+1)*qq])/qq
t1=np.linspace(tt10[0],tt10[-1],lqq+1)





plt.figure(figsize=(10,6))

plt.plot(t1,max1/A1,label=r'CG2', linewidth='4')
plt.xlabel(' $t (s)$ ',size=16)
plt.legend(loc='lower right',fontsize="14")
plt.ylabel( '$max/A$ ',size=16)
plt.grid()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
plt.savefig('amplification_sp3.png')
plt.show(block=True)
# plt.pause(0.001)
plt.gcf().clear()

# print("Finished program!")
