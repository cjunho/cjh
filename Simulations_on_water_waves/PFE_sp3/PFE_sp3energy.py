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
    ene_data = []

    for line in outputE:
        words = line.split()
        tijd.append(float(words[0]))        
        ene_data.append(float(words[3]))

    t1=np.array(tijd)
    ene1=np.array(ene_data)
    outputE.close()
    with open('C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_short/potflow3dperenergy.txt', 'r') as f2:
        qwe = f2.read().split()
   
    E01=float(qwe[1])+float(qwe[2])
   

if nchoice>1: #
   
    fileE = 'C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3/energy.txt'
    outputE = open(fileE,'r')       
    tijd = []
    ene_data = []

    for line in outputE:
        words = line.split()            
        tijd.append(float(words[0]))            
        ene_data.append(float(words[3]))

    t2=np.array(tijd)
    ene2=np.array(ene_data)
    outputE.close()
    with open('C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3/energy.txt', 'r') as f2:
        qwe = f2.read().split()
   
    E02=float(qwe[1])+float(qwe[2])
    

if nchoice>2: # 1, 2
    fileE = 'C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_long/potflow3dperenergy.txt'
    outputE = open(fileE,'r')
    tijd = []
    ene_data = []

    for line in outputE:
        words = line.split()
        tijd.append(float(words[0]))    
        ene_data.append(float(words[3]))

    t3=np.array(tijd)
    ene3=np.array(ene_data)
    outputE.close()
    with open('C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_long/potflow3dperenergy.txt', 'r') as f2:
        qwe = f2.read().split()
   
    E03=float(qwe[1])+float(qwe[2])
    
if nchoice>3:
    fileE = 'C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_cg4/energy.txt'
    outputE = open(fileE,'r')
    tijd = []
    ene_data = []
    AA_data=[]
    for line in outputE:
        words = line.split()
        tijd.append(float(words[0]))
        ene_data.append(float(words[3]))

    t4=np.array(tijd)
    ene4=np.array(ene_data)
    outputE.close()
    with open('C:/Users/amtjch/Desktop/pfe_sp3/pfe_sp3_cg4/energy.txt', 'r') as f2:
        qwe = f2.read().split()
   
    E04=float(qwe[1])+float(qwe[2])
    

plt.figure(figsize=(10,6))

if nchoice>0:
    plt.plot(t1,abs(ene1/E01),label=r'CG2/$\Delta t$', linewidth='4')
if nchoice>1:
    plt.plot(t2,abs(ene2/E02),label=r'CG2/$\frac{\Delta t}{2}$', linewidth='4')
if nchoice>2:
    plt.plot(t3,abs(ene3/E03),'--',label=r'CG2/$\frac{\Delta t}{4}$', linewidth='4')
if nchoice>3:
    plt.plot(t4,abs(ene4/E04),':',label=r'CG4/$\Delta t$', linewidth='4')


plt.xlabel(' $t (s)$ ',size=16)
plt.legend(loc='lower right',fontsize="14")
plt.ylabel( r'$\left|(E(t)-E(0))/E(0)\right|$ ',size=16)
plt.grid()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('energy_sp3.png')
plt.show(block=True)
plt.gcf().clear()

print("Finished program!")
