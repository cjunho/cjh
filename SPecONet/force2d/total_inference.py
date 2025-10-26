import subprocess
import pandas as pd
import pickle
import time


fdata="sigma5"
# uniform3, uniform3exf4, uniform3sigma6, uniform3sigma7

for ii in range(1,4+1):
    try:
        subprocess.run(f'python inference3error_rec.py --equation NS2d --model Net3D0 --loss MSE --blocks 0 --epochs 20 --ks 9 --filters 10 --nbfuncs 30 --U 9 --pre_epochs 5000 --dt 0.01'\
                        f' --forcing {fdata} --ndt 1 --eps 0.1 --kind force --A 0 --U 5 --order {ii} --start {ii} --file 100N23', shell=True)
      
        print("Script executed successfully.")
    except subprocess.CalledProcessError:
            break
    
                
# print(time.time()-time0)
