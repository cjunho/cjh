import subprocess
import pandas as pd
import pickle
import time
import os



FILE='600N23'
Equation='NS2d'
kind='force'
eps=0.1
filename = f'./training/{Equation}{eps}/{FILE}/order1'



data_path=f"data/{Equation}{eps}/{kind}"
if os.path.isdir(data_path) == False: os.makedirs(data_path)
   
time0 = time.time()

forcing='sigma5'

try:
    subprocess.run(f'python ns_solver.py --case train --Nsamples 600 --Ntimes 100 --Equation {Equation}', shell=True)
    subprocess.run(f'python ns_solver.py --case test --Nsamples 100 --Ntimes 100 --Equation {Equation}', shell=True)
    print("Script executed successfully.")
    
except subprocess.CalledProcessError:
    print('error')
    

    
try:
    subprocess.run(f'python training2alp.py --equation {Equation} --model Net3D --loss MSE --blocks 0 --file 600N23 --ks 9 --filters 10'\
                    f' --epochs 50000 --nbfuncs 30 --U 9 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --kind force', shell=True)
    
    df1 = pd.read_csv(filename+f"/call1_alp.csv")
    
    PATH=df1['path'][0]
    subprocess.run(f'python training2alp.py --equation {Equation} --model Net3D --loss MSE --blocks 0 --file 600N23 --ks 9 --filters 10'\
                    f' --epochs 20000 --nbfuncs 30 --U 9 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --kind force --pretrained true', shell=True)
    
    df1 = pd.read_csv(filename+f"/call1_alp.csv")
    PATH=df1['path'][0]
    subprocess.run(f'python training2pressure.py --equation {Equation} --model Net3Dpressure --loss MSE --blocks 0 --file 600N23 --kind force'\
                f' --epochs 10000 --ks 9 --filters 10 --nbfuncs 30 --U 9 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --order 1', shell=True)
    
    df2 = pd.read_csv(filename+f"/call1_pp.csv")    
    PATH2=df2['path'][0]
    subprocess.run(f'python training2pressure2.py --equation {Equation} --model Net3Dpressure --loss MSE --blocks 0 --file 600N23 --kind force'\
                f' --epochs 10000 --ks 9 --filters 10 --nbfuncs 30 --U 9 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --path2 {PATH2} --order 1', shell=True)
    
    
    print("Script executed successfully.")
    
except subprocess.CalledProcessError:
    print('error')
    




for ii in range(2,100+1):
    
    df1 = pd.read_csv(filename+f"/call{ii-1}_pp.csv")
    PATH=df1['path'][0]
    
    
    ORDER=ii
    
    if ORDER%10==1:
        
        try:
            subprocess.run(f'python training3alp.py --equation {Equation} --model Net3D --loss MSE --blocks 0 --file 600N23 --ks 9 --filters 10'\
                            f' --epochs 50000 --nbfuncs 30 --U 9 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --order {ORDER} --kind force', shell=True)
            print("Script executed successfully.")
            df1 = pd.read_csv(filename+f"/call{ii}_alp.csv")
            PATH_alp=df1['path'][0]
        except subprocess.CalledProcessError:
                break
    try:
            
            subprocess.run(f'python training3alp.py --equation {Equation} --model Net3D --loss MSE --blocks 0 --file 600N23 --ks 9 --filters 10'\
                            f' --epochs 20000 --nbfuncs 30 --U 9 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --path_alp {PATH_alp} --order {ORDER} --kind force --pretrained true', shell=True)
            
            df1 = pd.read_csv(filename+f"/call{ii}_alp.csv")   
            PATH=df1['path'][0]
            ORDER=df1['order'][0]
            subprocess.run(f'python training2pressure.py --equation {Equation} --model Net3Dpressure --loss MSE --blocks 0 --file 600N23 --kind force'\
                    f' --epochs 10000 --ks 9 --filters 10 --nbfuncs 30 --U 9 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --order {ORDER}', shell=True)
            
            df2 = pd.read_csv(filename+f"/call{ii}_pp.csv")    
            PATH2=df2['path'][0]
            subprocess.run(f'python training2pressure2.py --equation {Equation} --model Net3Dpressure --loss MSE --blocks 0 --file 600N23 --kind force'\
                    f' --epochs 10000 --ks 9 --filters 10 --nbfuncs 30 --U 9 --dt 0.01 --forcing {forcing}  --ndt 1 --eps 0.1 --path {PATH} --path2 {PATH2} --order {ORDER}', shell=True)
        
            print("Script executed successfully.")
    except subprocess.CalledProcessError:
            break
    
    
   
    
    
   
print(time.time()-time0)

