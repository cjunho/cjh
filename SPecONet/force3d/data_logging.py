#data_logging.py

import pandas as pd
import pickle
import subprocess
import numpy as np
import os


def record_path(path):
    entry = str(path) + '\n'
    with open("paths.txt", 'a') as f:
        f.write(entry)


def log_loss(losses,loss_train):     
    
    
    losses['loss_train'].append(loss_train.item())
    return losses


def log_gparams(gparams):
    cwd = os.getcwd()
    # print(gparams['path'])
    os.chdir(gparams['path'])
    with open('parameters.txt', 'w') as f:
        for k, v in gparams.items():
            if k == 'losses':
                df = pd.DataFrame(gparams['losses'])
                df.to_csv('losses.csv')
            else:
                entry = f"{k}:{v}\n"
                f.write(entry)
    os.chdir(cwd)


def log_path(path):
    with open("paths.txt", "a") as f:
        f.write(str(path) + '\n')
        f.close()
