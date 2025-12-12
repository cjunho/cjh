"find ap at first time step by using training2alp"
# python training2pressure2.py --blocks 0 --file 600N23 --epochs 60 --ks 9 --filters 10 --dt 0.01 --forcing sigma5  --ndt 1 --eps 0.1 --order 1 --kind force2d --path 50_20251212T063320 --path2 60_20251212T064446
import torch
import time
import datetime
import os
import argparse
import gc
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from net.data_loader import *
from net.network import *
from sem.sem import *
from reconstruct import *
from data_logging import *
from funsjax import matA

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
# ARGS
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--blocks", type=int, default=0)
parser.add_argument("--file", type=str, default='10000N15', help='Example: --file 2000N31') # 2^5-1, 2^6-1
parser.add_argument("--forcing", type=str, default='normal')
parser.add_argument("--epochs", type=int, default=80000)
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--filters", type=int, default=32)
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--ndt", type=int, default=5)
parser.add_argument("--order", type=int, default=1)
parser.add_argument("--eps", type=float, default=1)
parser.add_argument("--path", type=str)
parser.add_argument("--path2", type=str)
parser.add_argument("--kind", type=str, default='trainN10')


args = parser.parse_args()
gparams = args.__dict__

ndt=args.ndt
PATH0=args.path
PATH2=args.path2
D_in = 1
kind=args.kind
ORDER=args.order
EQUATION = 'NS2d'
EPSILON = args.eps
MODEL = Net3Dpressure

#GLOBALS
gparams['epsilon'] = EPSILON
FILE = gparams['file']
DATASET = int(FILE.split('N')[0])
SHAPE = int(FILE.split('N')[1]) + 1
BLOCKS = int(gparams['blocks'])
EPOCHS = int(gparams['epochs'])
dt = gparams['dt']
FILTERS = int(gparams['filters'])
KERNEL_SIZE = int(gparams['ks'])
# PADDING = (KERNEL_SIZE - 1)//2
PADDING = int(3)
cur_time = str(datetime.datetime.now()).replace(' ', 'T')
cur_time = cur_time.replace(':','').split('.')[0].replace('-','')
FOLDER = f'Net3Dpressure_{args.forcing}_epochs{EPOCHS}_{cur_time}'
FOLDER0 = f'Net3D_{args.forcing}_epochs{PATH0}'
FOLDER2 = f'Net3Dpressure_{args.forcing}_epochs{PATH2}'
PATH = os.path.join('training', f"{EQUATION}{EPSILON}", FILE,f"order{ORDER}" ,FOLDER)
PATH_prev1=os.path.join('training', f"{EQUATION}{EPSILON}", FILE,f"order{ORDER}" ,FOLDER0)
PATH_prev2=os.path.join('training', f"{EQUATION}{EPSILON}", FILE,f"order{ORDER}" ,FOLDER2)
BATCH_SIZE, Filters, D_out = int(DATASET), FILTERS, SHAPE
NN=SHAPE-1



# CREATE PATHING
if os.path.isdir(PATH) == False: os.makedirs(PATH)
   
# CREATE BASIS VECTORS
xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx, D,aa1,bb1 = basis_vectors(D_out,EPSILON ,equation=EQUATION)

shuffle1=False
NORM = False
gparams['norm'] = False
transform_f = None

# LOAD DATASET
lg_dataset = get_data(gparams, kind, transform_f=transform_f)
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=shuffle1)

# INITIALIZE a model
model= MODEL(1,D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
#Load the trained model
model.load_state_dict(torch.load(f'{PATH_prev2}/model.pt'), strict=False)
model.train()

# Check if CUDA is available and then use it.
device = get_device()
gparams['device'] = device

# SEND TO GPU (or CPU)
model.to(device).double()

# Count the number of the network's parameters
param_size = 0
r0=1
for name,param in model.named_parameters(): 
    print(name,r0,param.shape)
    param_size += param.nelement() * param.element_size()
    r0+=1
    if name !='fcH.weight':
        param.requires_grad = False

buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**3
print(param_size,buffer_size)
print('model size: {:.3f}GiB'.format(size_all_mb))
print(torch.cuda.memory_allocated()/1024**3)


#INIT OPTIMIZER
optimizer = init_optim(model)
BEST_LOSS = float('inf')
losses = {'loss_train':[]}
gparams['path'] = PATH
log_gparams(gparams)

#Generate matrices to compute the weak formulation
_,oden_data0,_,En,_,_,_,_,_,_,_,_,Mm,Mmx,phisets,phixsets,_,_,_=matA(NN,dt,EPSILON)

oden_data=np.zeros((NN-1,NN-1,NN-1))
pre_condn=np.zeros((NN-1,NN-1,NN-1))
ipre_condn=np.zeros((NN-1,NN-1,NN-1))
for jj in range(NN-1):
        # ode1=(eie[jj]*3*.5/dt+1)*eie[0]*M+eie[jj]*M+eie[jj]*eie[0]*np.eye(N-1)
       
        ode1=oden_data0[jj]
        pre_condn[jj,]=np.diag(1/np.diag(ode1)**.5)
        ipre_condn[jj,]=np.diag(np.diag(ode1)**.5)
        oden_data[jj,]=(pre_condn[jj,]@ode1)@pre_condn[jj,]

#Convert numpy files into torch files
En=torch.from_numpy(En).to(device).double()
Mm=torch.from_numpy(Mm).to(device).double()
Mmx=torch.from_numpy(Mmx).to(device).double()
oden_data=torch.from_numpy(oden_data).to(device).double()
pre_condn=torch.from_numpy(pre_condn).to(device).double()
ipre_condn=torch.from_numpy(ipre_condn).to(device).double()
phisets=torch.from_numpy(phisets).to(device).double()
phixsets=torch.from_numpy(phixsets).to(device).double()


def closure(dt,fdata0,alp):
 
    model.train()
    if torch.is_grad_enabled():
        optimizer.zero_grad()
    
    # Mapping outputs
    a_pred = model(fdata0)

    # Compute the residual of the weak formulation
    phial00,Pexfx = weak_pressure(alp[:,0,:ndt,],alp[:,1,:ndt,], a_pred,Mmx,Mm,dt, oden_data,pre_condn,En )
    
    #Loss of the residual in l2 norm
    loss = 10**7*(torch.sum((phial00-Pexfx)**2))#+torch.sum((abs(a_pred-aex))**2))
    
    if loss.requires_grad:
        loss.backward()
    
    #Inference, phi  
    phi0=En@torch.sum(pre_condn@(a_pred.reshape((BATCH_SIZE,1,NN-1,NN-1,1))),4)
    
    
    return  loss, phi0

#


torch.autograd.set_detect_anomaly(True)
################################################
time0 = time.time()

#Compute the input data, laplace*u
alp1=torch.load(PATH_prev1+'/data.pt').detach().double().to(device)
ux=reconstructx(alp1[:,0], phisets,phixsets)
vy=reconstructx(alp1[:,1],phixsets, phisets)

fdata=ux+vy


print(fdata.shape)






for epoch in tqdm(range(1, EPOCHS+1)):
        
        loss,a_pred = closure(dt,fdata,alp1)
        optimizer.step(loss.item)
        
        
        
        loss_train = np.round(float(loss.item()), 12)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        #SAVE train loss
        if epoch % int(2) == 0:
            
            losses = log_loss(losses,loss_train)
        

torch.save(model.state_dict(), PATH + '/model.pt')

print('Final loss:',loss_train)


#Save inference data
if ORDER>1:
    os.replace(PATH_prev2+'/cu0.pt',PATH +"/cu0.pt")
    os.replace(PATH_prev2+'/cv0.pt',PATH +"/cv0.pt")
   
    os.replace(PATH_prev2+'/cFx0.pt',PATH +"/cFx0.pt")
    os.replace(PATH_prev2+'/cFy0.pt',PATH +"/cFy0.pt")
   
    
    
    os.replace(PATH_prev2+'/cuu0.pt',PATH +"/cuu0.pt")
    os.replace(PATH_prev2+'/cvv0.pt',PATH +"/cvv0.pt")
   







torch.save(alp1, PATH +'/alpha.pt')
torch.save(a_pred, PATH +'/alphi.pt')

df = pd.DataFrame(losses)
df.to_csv(PATH + '/losses.csv')
del df

time1 = time.time()
dt1 = time1 - time0
AVG_ITER = np.round(dt/EPOCHS, 6)
NPARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)

gparams['dt'] = dt
gparams['avgIter'] = AVG_ITER
gparams['nParams'] = NPARAMS
gparams['batchSize'] = BATCH_SIZE
# gparams['bestLoss'] = BEST_LOSS
gparams['losses'] = losses


log_path(PATH)

log_gparams(gparams)

import pandas as pd
newcall={'blocks':[],'file':[],'ks':[],'dt':[],'forcing':[],'ndt':[],'eps':[],'path':[],'order':[]}

newcall['blocks'].append(BLOCKS)
newcall['file'].append(FILE)
newcall['ks'].append(KERNEL_SIZE)

newcall['dt'].append(dt)
newcall['forcing'].append(args.forcing)
newcall['ndt'].append(ndt)
newcall['eps'].append(EPSILON)
newcall['path'].append(f'{EPOCHS}_{cur_time}')
newcall['order'].append(ORDER)

df = pd.DataFrame(newcall)
PATH_call=os.path.join('training', f"{EQUATION}{EPSILON}", FILE, "order1")
df.to_csv(PATH_call + f'/call{ORDER}_pp.csv')

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()


# Evaluate the error of inference upon test samples

if os.path.isdir(f'training/{EQUATION}{EPSILON}/pp') == False: os.makedirs(f'training/{EQUATION}{EPSILON}/pp')
try:
    subprocess.run(f'python inference3error_rec.py --blocks 0 --ks 9 --filters 10 --dt 0.01'\
                    f' --forcing {args.forcing} --ndt 1 --eps 0.1 --kind {kind} --order {ORDER} --start {ORDER} --file 100N23', shell=True)
    
    print("Script executed successfully.")
except subprocess.CalledProcessError:
    print('error')