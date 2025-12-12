import subprocess
import pandas as pd
import pickle
import time
import os

##################################################################
#                        Setting parameters                      #
##################################################################
FILE='600N23' # the number of training samples (600) and the number of nodal points-1 (24-1) 
Equation='NS2d' #a governing equation
kind='force2d'  # a type of input data
eps=0.1         # viscosity
filename = f'./training/{Equation}{eps}/{FILE}/order1' #a path to save result data
data_path=f"data/{Equation}{eps}/{kind}"               # a path to save input data
forcing='sigma5' # setting standard deviation (sigam=5) to randomly generate input samples                    
final_step=3    #the final time step 
   
time0 = time.time()


##################################################################
#           Generating training and testing data                 #
##################################################################

if os.path.isdir(data_path) == False: 
   os.makedirs(data_path)
   try:
       subprocess.run(f'python ns_solver.py --case train --Nsamples 600 --Ntimes {final_step} --Equation {Equation} --forcing {forcing}', shell=True)
       subprocess.run(f'python ns_solver.py --case test --Nsamples 100 --Ntimes {final_step} --Equation {Equation} --forcing {forcing}', shell=True)
       print("Script executed successfully.")
       
   except subprocess.CalledProcessError:
       print('error')
    

##################################################################
#           Training SpecOnet for solutions at t=0.01            #
##################################################################   
try:
    subprocess.run(f'python training2alp.py --blocks 0 --file 600N23 --ks 9 --filters 10'\
                    f' --epochs 50 --dt 0.01 --forcing {forcing} --ndt 1 --eps 0.1 --kind {kind}', shell=True)
    
    df1 = pd.read_csv(filename+f"/call1_alp.csv")
    
    PATH=df1['path'][0]
    subprocess.run(f'python training2alp.py --blocks 0 --file 600N23 --ks 9 --filters 10'\
                    f' --epochs 20 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --kind {kind} --pretrained true', shell=True)
    
    df1 = pd.read_csv(filename+f"/call1_alp.csv")
    PATH=df1['path'][0]
    subprocess.run(f'python training2pressure.py --blocks 0 --file 600N23 --kind {kind}'\
                f' --epochs 10 --ks 9 --filters 10 --dt 0.01 --forcing {forcing} --ndt 1 --eps 0.1 --path {PATH} --order 1', shell=True)
    
    df2 = pd.read_csv(filename+f"/call1_pp.csv")    
    PATH2=df2['path'][0]
    subprocess.run(f'python training2pressure2.py --blocks 0 --file 600N23 --kind {kind}'\
                f' --epochs 10 --ks 9 --filters 10 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --path2 {PATH2} --order 1', shell=True)
    
    
    print("Script executed successfully.")
    
except subprocess.CalledProcessError:
    print('error')
    
df1 = pd.read_csv(filename+f"/call1_alp.csv")
PATH_alp=df1['path'][0]

##################################################################
#   Training SpecOnet for solutions at 0.01<t<=0.01*final_step   #
##################################################################   
for ii in range(2,final_step+1):
    
    df1 = pd.read_csv(filename+f"/call{ii-1}_pp.csv")
    PATH=df1['path'][0]
    
    
    ORDER=ii
    
    if ORDER%10==1:
        
        try:
            subprocess.run(f'python training3alp.py --blocks 0 --file 600N23 --ks 9 --filters 10'\
                            f' --epochs 50 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --order {ORDER} --kind {kind}', shell=True)
            print("Script executed successfully.")
            df1 = pd.read_csv(filename+f"/call{ii}_alp.csv")
            PATH_alp=df1['path'][0]
        except subprocess.CalledProcessError:
                break
    try:
            
            subprocess.run(f'python training3alp.py --blocks 0 --file 600N23 --ks 9 --filters 10'\
                            f' --epochs 20 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --path_alp {PATH_alp} --order {ORDER} --kind {kind} --pretrained true', shell=True)
            
            df1 = pd.read_csv(filename+f"/call{ii}_alp.csv")   
            PATH=df1['path'][0]
            ORDER=df1['order'][0]
            subprocess.run(f'python training2pressure.py --blocks 0 --file 600N23 --kind {kind}'\
                    f' --epochs 10 --ks 9 --filters 10 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --order {ORDER}', shell=True)
            
            df2 = pd.read_csv(filename+f"/call{ii}_pp.csv")    
            PATH2=df2['path'][0]
            subprocess.run(f'python training2pressure2.py --blocks 0 --file 600N23 --kind {kind}'\
                    f' --epochs 10 --ks 9 --filters 10 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --path2 {PATH2} --order {ORDER}', shell=True)
        
            print("Script executed successfully.")
    except subprocess.CalledProcessError:
            break
    
    
   
    
    
   
print(time.time()-time0)



