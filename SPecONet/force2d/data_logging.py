#data_logging.py

import pandas as pd
import pickle
import subprocess
import numpy as np
from evaluate import *
import os, json


def record_path(path):
    entry = str(path) + '\n'
    with open("paths.txt", 'a') as f:
        f.write(entry)


def log_loss(nn,losses, loss_a, loss_u, loss_f, loss_wf,  loss_train, loss_validate, dataset, avg_l2_u): #loss_wf1, loss_wf2, loss_wf3, 
   
    if type(loss_u) == int:
        losses['loss_u'].append(loss_u/dataset)
    else:
        losses['loss_u'].append(loss_u/dataset)
    
    losses['loss_train'].append(loss_train.item()/dataset)
    losses['loss_validate'].append(loss_validate/1000)
    losses['avg_l2_u'].append((avg_l2_u)/1000)
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
