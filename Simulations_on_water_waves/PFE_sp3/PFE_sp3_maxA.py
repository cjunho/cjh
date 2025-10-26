import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt

nchoice = 1

if nchoice>0: # 
   
    fileE = 'C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_short/potflow3dperenergy.txt'
    
    outputE = open(fileE,'r')
    #  infile.readline() # skip the first line not first line here (yet)
    tijd = []
    max_data=[]
    A_data = []

    for line in outputE:
        words = line.split()
        tijd.append(float(words[0]))        
        max_data.append(float(words[4]))
        A_data.append(float(words[5]))

    tt10=np.array(tijd)
    max10=np.array(max_data)
    A10=np.array(A_data)
    outputE.close()
        
    qq=int(25)
    lqq=int(len(A10)/qq)
    max1=np.zeros((lqq+1,))
    max1[0]=max10[0]
    A1=np.zeros((lqq+1,))
    A1[0]=A10[0]

    for jj in range(lqq):
        max1[jj+1]=sum(max10[jj*qq:(jj+1)*qq])/qq
        A1[jj+1]=sum(A10[jj*qq:(jj+1)*qq])/qq
    t1=np.linspace(tt10[0],tt10[-1],lqq+1)

if nchoice>1: #
   
    fileE = 'C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3/energy.txt'
    outputE = open(fileE,'r')       
    tijd = []
    max_data=[]
    A_data = []

    for line in outputE:
        words = line.split()
        tijd.append(float(words[0]))        
        max_data.append(float(words[4]))
        A_data.append(float(words[5]))

    tt20=np.array(tijd)
    max20=np.array(max_data)
    A20=np.array(A_data)
    outputE.close()
    
    qq=int(50)
    lqq=int(len(A20)/qq)
    max2=np.zeros((lqq+1,))
    max2[0]=max20[0]
    A2=np.zeros((lqq+1,))
    A2[0]=A20[0]
    
    for jj in range(lqq):
        max2[jj+1]=sum(max20[jj*qq:(jj+1)*qq])/qq
        A2[jj+1]=sum(A20[jj*qq:(jj+1)*qq])/qq
    t2=np.linspace(tt20[0],tt20[-1],lqq+1)

if nchoice>2: # 1, 2
    fileE = 'C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_long/potflow3dperenergy.txt'
    outputE = open(fileE,'r')
    tijd = []
    max_data=[]
    A_data = []

    for line in outputE:
        words = line.split()
        tijd.append(float(words[0]))    
        max_data.append(float(words[4]))
        A_data.append(float(words[5]))

    tt30=np.array(tijd)
    max30=np.array(max_data)
    A30=np.array(A_data)
    outputE.close()
    qq=int(100)
    lqq=int(len(A30)/qq)
    max3=np.zeros((lqq+1,))
    max3[0]=max30[0]
    A3=np.zeros((lqq+1,))
    A3[0]=A30[0]

    for jj in range(lqq):
        A3[jj+1]=sum(A30[jj*qq:(jj+1)*qq])/qq
        max3[jj+1]=sum(max30[jj*qq:(jj+1)*qq])/qq
    t3=np.linspace(tt30[0],tt30[-1],lqq+1)
    
if nchoice>3:
    fileE = 'C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_cg4/energy.txt'
    outputE = open(fileE,'r')
    tijd = []
    A_data = []
    max_data=[]
    
    for line in outputE:
        words = line.split()
        tijd.append(float(words[0]))
        max_data.append(float(words[4]))
        A_data.append(float(words[5]))

    tt40=np.array(tijd)
    max40=np.array(max_data)
    A40=np.array(A_data)
    outputE.close()
    
    qq=int(50)
    lqq=int(len(A40)/qq)
    max4=np.zeros((lqq+1,))
    max4[0]=max40[0]
    A4=np.zeros((lqq+1,))
    A4[0]=A40[0]

    for jj in range(lqq):
        max4[jj+1]=sum(max40[jj*qq:(jj+1)*qq])/qq
        A4[jj+1]=sum(A40[jj*qq:(jj+1)*qq])/qq
    t4=np.linspace(tt40[0],tt40[-1],lqq+1)

plt.figure(figsize=(10,6))

if nchoice>0:
    plt.plot(t1,max1/A1,label=r'CG2/$\Delta t$', linewidth='4')
if nchoice>1:
    plt.plot(t2,max2/A2,label=r'CG2/$\frac{\Delta t}{2}$', linewidth='4')
if nchoice>2:
    plt.plot(t3,max3/A3,'--',label=r'CG2/$\frac{\Delta t}{4}$', linewidth='4')
if nchoice>3:
    plt.plot(t4,max4/A4,':',label=r'CG4/$\Delta t$', linewidth='4')


plt.xlabel(' $t (s)$ ',size=16)
plt.legend(loc='lower right',fontsize="14")
plt.ylabel( '$max/A$ ',size=16)
plt.grid()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('MaxA_sp3.png')
plt.show(block=True)
plt.gcf().clear()

print("Finished program!")
