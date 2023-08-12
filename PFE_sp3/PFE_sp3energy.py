import numpy as np
import pandas as pd
import os.path
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



fileE = 'C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_short/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
ene_data = []
AA_data=[]
for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[1]))
    # EKin.append(float(words[2]))
    ene_data.append(float(words[3]))
    # AA_data.append(float(words[2]))


tt1=np.array(tijd)
ene1=np.array(ene_data)
# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()

with open('C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_short/potflow3dperenergy.txt', 'r') as f2:
    qwe = f2.read().split()
   
E0=float(qwe[1])+float(qwe[2])


plt.figure(figsize=(10,6))
plt.plot(tt1,abs(ene1/E0),label=r'CG2', linewidth='4')

plt.xlabel(' $t (s)$ ',size=16)
plt.legend(fontsize="16")
plt.ylabel( r'$\left|(E(t)-E(0))/E(0)\right|$ ',size=16)
plt.grid()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # [0 10 0.0010082 0.0010086])
plt.savefig('energy_sp3.png')
plt.show(block=True)
# plt.pause(0.001)
plt.gcf().clear()


# print("Finished program!")
