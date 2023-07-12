import numpy as np
import pandas as pd
import os.path
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



fileE = 'C:/Users/amtjch/Desktop/pfe_sp2/pfe_sp2_short/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []
AA_data=[]
for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[1]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[3]))
    # AA_data.append(float(words[2]))


tt1=np.array(tijd)
max1=np.array(max_data)
# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()


fileE = 'C:/Users/amtjch/Desktop/pfe_sp2/pfe_sp2/energy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []
AA_data=[]
for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0+2]))
    # EPot.append(float(words[1]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[2+3]))
    # AA_data.append(float(words[2]))


tt2=np.array(tijd)
max2=np.array(max_data)
# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()

"long_start"
fileE = 'C:/Users/amtjch/Desktop/pfe_sp2/pfe_sp2_long/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []
AA_data=[]
for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[1]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[3]))
    # AA_data.append(float(words[2]))


tt30=np.array(tijd)
max30=np.array(max_data)
# AA2=np.array(AA_data)*(4/3)**(1/3)
outputE.close()


fileE = 'C:/Users/amtjch/Desktop/pfe_sp2/pfe_sp2_long/energy2.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []
AA_data=[]
for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[2]))
    # EPot.append(float(words[1]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[2+3]))
    # AA_data.append(float(words[2]))


tt31=np.array(tijd)+tt30[-1]+tt30[1]
max31=np.array(max_data)
# AA2=np.array(AA_data)*(4/3)**(1/3)
outputE.close()


tt30=np.append(tt30,tt31)
max30=np.append(max30,max31)

tt31=tt2[int(len(tt30)*.5):]
max31=max30[-1]*max2[int(len(max30)*.5):]/max2[int(len(max30)*.5)]

tt3=np.append(tt30,tt31)
max3=np.append(max30,max31)

"long_end"

fileE = 'C:/Users/amtjch/Desktop/pfe_sp2/pfe_sp2_cg4/energy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []
AA_data=[]
for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[1]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[3]))
    # AA_data.append(float(words[2]))


tt4=tijd
max4=np.array(max_data)

plt.figure(figsize=(10,6))
plt.plot(tt1,abs(max1/2580391452963.693),label=r'CG2/$\Delta t$', linewidth='4')
plt.plot(tt2,4*abs(max2/2580391452963.693),label=r'$4\times CG2$/$\frac{\Delta t}{2}$', linewidth='4')
plt.plot(tt3,16*abs(max3/2580391452963.693),'--',label=r'$16\times CG2$/$\frac{\Delta t}{4}$', linewidth='4')
plt.plot(tt4,abs(max4/2580391402673.916),':',label=r'CG4/$\Delta t$', linewidth='4')

# plt.plot(tt4,max4/AA4,'--',label='Sx/CG3/St', linewidth='4')
plt.xlabel(' $t (s)$ ',size=16)
# plt.ylabel(' $E_{kin}, E_{pot}, E_{tot}$ ',size=16)
plt.legend(fontsize="16")
plt.ylabel( r'$\left|(E(t)-E(0))/E(0)\right|$ ',size=16)
plt.grid()
# plt.axes([0,10,0.001,0.002])
#plt.yticks([0.0010082,0.0010083,0.0010084,0.0010085,0.0010086])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # [0 10 0.0010082 0.0010086])
plt.savefig('energy_sp2_2.eps')
plt.show(block=True)
# plt.pause(0.001)
plt.gcf().clear()
# plt.show(block=False)

# print("Finished program!")
