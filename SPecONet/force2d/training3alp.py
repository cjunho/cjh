"find au, av, aw at second time step by using training2alp"
# python training3alp.py --blocks 0 --file 600N23 --ks 9 --filters 10 --epochs 50 --dt 0.01 --forcing sigma5  --ndt 1 --eps 0.1 --path 60_20251213T062112 --order 2 --kind force2d
import random
import torch
import time
import datetime
import os
import LG_1d
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
from plotting import *
from reconstruct import *
from data_logging import *
from evaluate import *
from pprint import pprint
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
parser.add_argument("--eps", type=float, default=1)
parser.add_argument("--path", type=str)
parser.add_argument("--path_alp", type=str, default=None)
parser.add_argument("--order", type=int, default=1)
parser.add_argument("--kind", type=str, default='trainN10')


args = parser.parse_args()
gparams = args.__dict__


ndt=args.ndt
kind=args.kind
D_in = 2*ndt
ORDER=args.order
EQUATION = 'NS2d'
EPSILON = args.eps
PATH0 = args.path
PATH_alp=args.path_alp
MODEL = Net3D

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
PADDING = int(3)
cur_time = str(datetime.datetime.now()).replace(' ', 'T')
cur_time = cur_time.replace(':','').split('.')[0].replace('-','')
FOLDER = f'Net3D_{args.forcing}_epochs{EPOCHS}_{cur_time}'
FOLDER0 = f'Net3Dpressure_{args.forcing}_epochs{PATH0}'
PATH = os.path.join('training', f"{EQUATION}{EPSILON}", FILE,f"order{ORDER}" ,FOLDER)
PATH_prev=os.path.join('training', f"{EQUATION}{EPSILON}", FILE,f"order{ORDER-1}", FOLDER0)
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
model= MODEL(ndt,2*1, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)

# LOAD the trained model
if args.pretrained is not None:
    ORDER_alp=10*((ORDER-1)//10)+1
    print('ORDER_alp',ORDER_alp)  
    args.pretrained = 'N' + args.file.split('N')[-1] + '_' + f'{EQUATION}' + '_' + args.forcing
    model.load_state_dict(torch.load(f'training/{EQUATION}{EPSILON}/{BATCH_SIZE}N23/order{ORDER_alp}/Net3D_{args.forcing}_epochs{PATH_alp}/model.pt'), strict=False)
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
    if args.pretrained is not None:
        if name !='fcH.weight':
            param.requires_grad = False
    
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**3
print(param_size,buffer_size)
print('model size: {:.3f}GiB'.format(size_all_mb))
print(torch.cuda.memory_allocated()/1024**3)

#KAIMING HE INIT
if args.pretrained is None:
    model.apply(weights_init)


#INIT OPTIMIZER
optimizer = init_optim(model)
BEST_LOSS = float('inf')
losses = {'loss_train':[]}
gparams['path'] = PATH
log_gparams(gparams)

#Generate matrices to compute the weak formulation
ode_data,_, Ed,_,_,_,Mnd,Mxdd,Mxnd,Mdxd,Md,iMd,Mm,Mmx,phisets,_,_,sd_diag,lep=matA(NN,dt,EPSILON)
iode_data=np.zeros((NN-1,NN-1,NN-1))
ode_eye=np.zeros((NN-1,NN-1,NN-1))

for jj in range(NN-1):       
            ode_data0=ode_data[jj]  
            iode_data[jj,]=np.diag(1/np.diag(ode_data0)**.5)
            ode_eye[jj,]=(iode_data[jj,]@ode_data0)@iode_data[jj,]


#Convert numpy files into torch files
Mnd=torch.from_numpy(Mnd).to(device).double()
Mdxd=torch.from_numpy(Mdxd).to(device).double()
Mxdd=torch.from_numpy(Mxdd).to(device).double()
Md=torch.from_numpy(Md).to(device).double()
iMd=torch.from_numpy(iMd).to(device).double()
phisets=torch.from_numpy(phisets).to(device).double()
lep=torch.from_numpy(lep).to(device).double()
Mxnd=torch.from_numpy(Mxnd).to(device).double()
Ed=torch.from_numpy(Ed).to(device).double()
Mm=torch.from_numpy(Mm).to(device).double()
Mmx=torch.from_numpy(Mmx).to(device).double()
iode_data=torch.from_numpy(iode_data).to(device).double()
ode_eye=torch.from_numpy(ode_eye).to(device).double()
sd_diag=torch.from_numpy(sd_diag).to(device).double()


def closure(fdata0,cf,cu1,cv1,cFx01,cFy01):   
    model.train()
    
    if torch.is_grad_enabled():
        optimizer.zero_grad()
        
    # Mapping outputs
    a_pred = model(fdata0)
    
    cFx=cf[:,0:1]
    cFy=cf[:,1:2]
    alx=a_pred[:,0]
    aly=a_pred[:,1]
   
    # Compute the residual of the weak formulation   
    al_unext,al_vnext,exfx,exfy = weak_form1(cu0,cv0,cu1,cv1,cuu0,cvv0,cuu1,cvv1,cFx01,cFy01,cFx,cFy,dt,ode_eye,iode_data, a_pred,Ed )
    #Loss of the residual in l2 norm
    loss=10**7*((torch.sum((al_unext-exfx)**2))+(torch.sum((al_vnext-exfy)**2)))
    
   
    if loss.requires_grad:
        loss.backward()
    #Inference, alpha   
    alx0=Ed@torch.sum(iode_data@(alx.reshape((BATCH_SIZE,1,NN-1,NN-1,1))),4)
    aly0=Ed@torch.sum(iode_data@(aly.reshape((BATCH_SIZE,1,NN-1,NN-1,1))),4)
    
    ald1=torch.stack((alx0,aly0),dim=1)

    return  loss, ald1

#


torch.autograd.set_detect_anomaly(True)
################################################
time0 = time.time()

for batch_idx, sample_batch in enumerate(trainloader):
        fdata = sample_batch['f'][:BATCH_SIZE,:,ORDER-1].double().to(device)
        cf1 = sample_batch['cf'][:BATCH_SIZE,:,ORDER-1].double().to(device)[:,:2*ndt,]

print(fdata.shape)
print(cf1.shape)

# Load inference data for the previous time step
alp1=torch.load(PATH_prev+'/alpha.pt').detach().to(device).double()
alphi1=torch.load(PATH_prev+'/alphi.pt').detach().to(device).double()

# Compute a source function consisting of the inference data for the previous time step  
if ORDER==2:
    cu0=0
    cv0=0
    cuu0=0
    cvv0=0
    cFx0=0
    cFy0=0
    cu11,cv11,cFx011,cFy011,cuu1,cvv1=weak_combine(EPSILON,alp1[:,0,0:ndt,],alp1[:,1,0:ndt,]\
                                                        ,cFx0,cFy0,dt, NN,alphi1[:BATCH_SIZE,0:ndt,],Mxnd,Mnd,Md,Mxdd,Mdxd,Mm,Mmx,iMd,phisets,lep,sd_diag )
   
else:
    cu0=torch.load(PATH_prev+'/cu0.pt').detach().to(device).double()
    cv0=torch.load(PATH_prev+'/cv0.pt').detach().to(device).double()
   
   
    cuu0=torch.load(PATH_prev+'/cuu0.pt').detach().to(device).double()
    cvv0=torch.load(PATH_prev+'/cvv0.pt').detach().to(device).double()
   
    cFx0=torch.load(PATH_prev+'/cFx0.pt').detach().to(device).double()
    cFy0=torch.load(PATH_prev+'/cFy0.pt').detach().to(device).double()
    
    cu11,cv11,cFx011,cFy011,cuu1,cvv1=weak_combine(EPSILON,alp1[:,0,0:ndt,],alp1[:,1,0:ndt,]\
                                                        ,cFx0,cFy0,dt, NN,alphi1[:BATCH_SIZE,0:ndt,],Mxnd,Mnd,Md,Mxdd,Mdxd,Mm,Mmx,iMd,phisets,lep,sd_diag )
    

torch.save(cu11, PATH + '/cu0.pt')
torch.save(cv11, PATH + '/cv0.pt')
torch.save(cFx011, PATH + '/cFx0.pt')
torch.save(cFy011, PATH + '/cFy0.pt')
torch.save(cuu1, PATH + '/cuu0.pt')
torch.save(cvv1, PATH + '/cvv0.pt')
del alp1,alphi1


for epoch in tqdm(range(1, EPOCHS+1)):
        
        loss,a_pred = closure(fdata,cf1,cu11,cv11,cFx011,cFy011)
       
        optimizer.step(loss.item) 
        
        loss_train = np.round(float(loss.item()), 12)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        #SAVE train loss
        if epoch % int(2) == 0:
            
            losses = log_loss(losses, loss_train)
        

torch.save(model.state_dict(), PATH + '/model.pt')


torch.save(a_pred, PATH + '/data.pt')

print('Final loss:',loss_train)


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
df.to_csv(PATH_call + f'/call{ORDER}_alp.csv')

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()


