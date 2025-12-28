import subprocess
import pandas as pd
import pickle
import time
import os

##################################################################
#                        Setting parameters                      #
##################################################################
FILE='700N19' # the number of training samples (600) and the number of nodal points-1 (24-1) 
FILE_test='100N19'
Equation='NS3d' #a governing equation
kind='force3d'  # a type of input data
eps=1.0         # viscosity
forcing='sigma5' # setting standard deviation (sigam=5) to randomly generate input samples                    
filename = f'./training/{Equation}{eps}/{FILE}/order1' #a path to save result data
data_path=f"data/{Equation}{eps}/{forcing}"               # a path to save input data
final_step=100   #the final time step 
   
time0 = time.time()


##################################################################
#           Generating training and testing data                 #
##################################################################

if os.path.isdir(data_path) == False: 
   os.makedirs(data_path)
   try:
       subprocess.run(f'python ns3d_paper.py --case train --Nsamples {FILE} --Ntimes {final_step} --Equation {Equation} --forcing {forcing} --epsilon {eps}', shell=True)
       subprocess.run(f'python ns3d_paper.py --case test --Nsamples {FILE_test} --Ntimes {final_step} --Equation {Equation} --forcing {forcing} --epsilon {eps}', shell=True)
       print("Script executed successfully.")
       
   except subprocess.CalledProcessError:
       print('error')
    

##################################################################
#           Training SpecOnet for solutions at t=0.01            #
##################################################################   
try:
    subprocess.run(f'python training2alp.py --equation NS3d --blocks 0 --file {FILE} --ks 9 --filters 3'\
                    f' --epochs 30000 --dt 0.01 --forcing {forcing} --kind {kind} --ndt 1 --eps {eps}', shell=True)
    
    df1 = pd.read_csv(filename+f"/call1_alp.csv")
    
    PATH=df1['path'][0]
    subprocess.run(f'python training2alp.py --blocks 0 --file {FILE} --ks 9 --filters 3'\
                    f' --epochs 10000 --dt 0.01 --forcing {forcing}  --ndt 1 --eps {eps} --path {PATH} --kind {kind} --pretrained true', shell=True)
    
    df1 = pd.read_csv(filename+f"/call1_alp.csv")
    PATH=df1['path'][0]
    subprocess.run(f'python training2pressure.py --equation NS3d --blocks 0 --file {FILE} --ks 9 --filters 3 --dt 0.01'\
                f' --epochs 15000 --forcing {forcing}  --ndt 1 --eps {eps} --order 1 --kind {kind} --path {PATH}', shell=True)
    
    df2 = pd.read_csv(filename+f"/call1_pp.csv")    
    PATH2=df2['path'][0]
    subprocess.run(f'python training2pressure2.py --equation NS3d --blocks 0 --file {FILE} --kind {kind}'\
                f' --epochs 5000 --ks 9 --filters 3 --dt 0.01 --forcing {forcing}  --ndt 1 --eps {eps} --path {PATH} --path2 {PATH2} --order 1', shell=True)
    
    
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
            subprocess.run(f'python training3alp.py --equation NS3d --blocks 0 --file {FILE} --ks 9 --filters 3 --dt 0.01'\
                            f' --epochs 30000 --forcing {forcing} --ndt 1 --eps {eps} --order {ORDER} --path {PATH} --path_alp {PATH_alp} --kind {kind}', shell=True)
            print("Script executed successfully.")
            df1 = pd.read_csv(filename+f"/call{ii}_alp.csv")
            PATH_alp=df1['path'][0]
        except subprocess.CalledProcessError:
                break
    try:
            
            subprocess.run(f'python training3alp.py --equation NS3d --blocks 0 --file {FILE} --ks 9 --filters 3 --dt 0.01'\
                            f' --epochs 10000 --forcing {forcing} --ndt 1 --eps {eps} --order {ORDER} --path {PATH} --path_alp {PATH_alp} --kind {kind} --pretrained true', shell=True)
            print("Script executed successfully.")
            
            df1 = pd.read_csv(filename+f"/call{ii}_alp.csv")   
            PATH=df1['path'][0]
            ORDER=df1['order'][0]
            subprocess.run(f'python training2pressure.py --equation NS3d --blocks 0 --file {FILE} --ks 9 --filters 3 --dt 0.01  --forcing {forcing}'\
                    f' --epochs 15000 --ndt 1 --eps {eps} --order {ORDER} --kind {kind} --path {PATH}', shell=True)
            
            df2 = pd.read_csv(filename+f"/call{ii}_pp.csv")    
            PATH2=df2['path'][0]
            subprocess.run(f'python training2pressure2.py --equation NS3d --blocks 0 --file {FILE} --kind {kind}'\
                    f' --epochs 5000 --ks 9 --filters 3 --dt 0.01 --forcing {forcing}  --ndt 1 --eps {eps} --path {PATH} --path2 {PATH2} --order {ORDER}', shell=True)
        
            print("Script executed successfully.")
    except subprocess.CalledProcessError:
            break
    
    
   
    
    
   
print(time.time()-time0)



