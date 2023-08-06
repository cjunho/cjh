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


fileE = 'C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3/energy.txt'
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


tt2=np.array(tijd)
ene2=np.array(ene_data)
# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()

"long_start"
fileE = 'C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_long/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
ene_data = []

for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[1]))
    # EKin.append(float(words[2]))
    ene_data.append(float(words[3]))
    # AA_data.append(float(words[2]))


tt3=np.array(tijd)
ene3=np.array(ene_data)
# AA2=np.array(AA_data)*(4/3)**(1/3)
outputE.close()
"long_end"


fileE = 'C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_cg4/energy.txt'
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


tt4=tijd
ene4=np.array(ene_data)
outputE.close()

plt.figure(figsize=(10,6))
plt.plot(tt1,abs(ene1/2580391452963.693),label=r'CG2/$\Delta t$', linewidth='4')
plt.plot(tt2,4*abs(ene2/2580391452963.693),label=r'$4\times CG2$/$\frac{\Delta t}{2}$', linewidth='4')
plt.plot(tt3,16*abs(ene3/2580391452963.693),'--',label=r'$16\times CG2$/$\frac{\Delta t}{4}$', linewidth='4')
plt.plot(tt4,abs(ene4/2580391402673.916),':',label=r'CG4/$\Delta t$', linewidth='4')


plt.xlabel(' $t (s)$ ',size=16)
# plt.ylabel(' $E_{kin}, E_{pot}, E_{tot}$ ',size=16)
plt.legend(fontsize="16")
plt.ylabel( r'$\left|(E(t)-E(0))/E(0)\right|$ ',size=16)
plt.grid()
# plt.axes([0,10,0.001,0.002])
#plt.yticks([0.0010082,0.0010083,0.0010084,0.0010085,0.0010086])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # [0 10 0.0010082 0.0010086])
plt.savefig('energy_sp3.png')
plt.show(block=True)
# plt.pause(0.001)
plt.gcf().clear()


# print("Finished program!")
