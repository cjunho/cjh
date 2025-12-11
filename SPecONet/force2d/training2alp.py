"find au, av, aw at first time step"
# python training2alp.py --blocks 0 --file 600N23 --ks 9 --filters 10 --epochs 50 --dt 0.01 --forcing sigma5  --ndt 1 --eps 0.1 --kind train
import random
import torch
import time
import datetime
import subprocess
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
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
# ARGS
# python training.py --equation Burgers --model NetC --blocks 4 --file 10000N63 --forcing uniform --epochs 50000
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
parser.add_argument("--kind", type=str, default='trainN10')
parser.add_argument("--path", type=str, default=None)


args = parser.parse_args()
gparams = args.__dict__


ndt=args.ndt
kind=args.kind
D_in = 2*ndt

EQUATION = 'NS2d'


EPSILON = args.eps
PATH_alp=args.path


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
# PADDING = (KERNEL_SIZE - 1)//2
PADDING = int(3)
cur_time = str(datetime.datetime.now()).replace(' ', 'T')
cur_time = cur_time.replace(':','').split('.')[0].replace('-','')
FOLDER = f'{MODEL}_{args.forcing}_epochs{EPOCHS}_{cur_time}'

PATH = os.path.join('training', f"{EQUATION}{EPSILON}", FILE,'order1', FOLDER)




BATCH_SIZE, Filters, D_out = int(DATASET), FILTERS, SHAPE
# LOSS SCALE FACTORS


NN=SHAPE-1



# CREATE PATHING
if os.path.isdir(PATH) == False: os.makedirs(PATH)

        
# CREATE BASIS VECTORS
xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx, D,aa1,bb1 = basis_vectors(D_out,EPSILON ,equation=EQUATION)

# if BATCH_SIZE+1<DATASET:
#     shuffle1=True
# else: 
shuffle1=False


NORM = False
gparams['norm'] = False
transform_f = None

# LOAD DATASET
lg_dataset = get_data(gparams, kind, transform_f=transform_f)
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=shuffle1)

# lg_dataset = get_data(gparams, kind='validate', transform_f=transform_f)
# validateloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=True)

# INITIALIZE a model
model = MODEL(ndt,D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)

# LOAD the trained model
if args.pretrained is not None:
    args.pretrained = 'N' + args.file.split('N')[-1] + '_' + args.equation + '_' + args.forcing
    model.load_state_dict(torch.load(f'training/{EQUATION}{EPSILON}/{BATCH_SIZE}N23/order1/Net3D_{args.forcing}_epochs{PATH_alp}/model.pt'), strict=False)
    model.train()

# Check if CUDA is available and then use it.
device = get_device()
gparams['device'] = device

# SEND TO GPU (or CPU)
model.to(device).double()


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


#KAIMING HE INIT

model.apply(weights_init)


#INIT OPTIMIZER
optimizer = init_optim(model)

# Construct our loss function and an Optimizer.


BEST_LOSS = float('inf')
losses = {'loss_u':[],
          'loss_train':[],
          'loss_validate':[],
          'avg_l2_u': []}
gparams['path'] = PATH
log_gparams(gparams)



dt=0.01




# Md,sd_diag,Ed,eid=basic_mat(b,NN,'dirichlet')

# Mn,sn_diag,En,ein=basic_mat(bn,NN,'neumann')


# Mm=np.zeros((NN-1,NN-1))
# Mmx=np.zeros((NN-1,NN-1))

# iMd=Ed@np.diag(1/eid)@Ed.T
# iMn=En@np.diag(1/ein)@En.T





# for ii in range(NN-1):
#     phi=(lepolys[ii]- lepolys[ii+2])/(sd_diag[ii])**.5
#     phix=(lepolysx[ii].T-lepolysx[ii+2].T)/(sd_diag[ii])**.5
#     for jj in range(NN-1):
#         psi=(lepolys[jj]+ bn[jj]*lepolys[jj+2])/(sn_diag[jj])**.5
#         Mm[jj,ii]=np.sum(psi*phi/(lepolys[NN])**2)*(2/(NN*(NN+1)))
#         Mmx[jj,ii]=np.sum(psi*phix/(lepolys[NN])**2)*(2/(NN*(NN+1)))

# Mm[abs(Mm)<10**-8]=0
# Mmx[abs(Mmx)<10**-8]=0


# Mxnd=np.zeros((NN-1,NN-1))
# Mdxd=np.zeros((NN-1,NN-1))
# Mxdd=np.zeros((NN-1,NN-1))
# Mnd=np.zeros((NN-1,NN-1))


# mnd1=np.zeros((NN-1,))
# mnd2=np.zeros((NN-1,))
# mnd3=np.zeros((NN-1,))

# mxdd=np.zeros((NN-1,))

# for ii in range(NN-1):
#     mnd2[ii]=2*(1/(2*ii+1)+b[ii]*bn[ii]/(2*ii+5))/(sd_diag[ii]*sn_diag[ii])**.5
#     mnd1[ii]=(b[ii])*2/(2*ii+5)/(sd_diag[ii]*sn_diag[ii+2])**.5
#     mnd3[ii]=(bn[ii])*2/(2*ii+5)/(sd_diag[2+ii]*sn_diag[ii])**.5
#     if ii< NN-2:
#         diri = (lepolys[ii]-lepolys[ii+2])/(sd_diag[ii])**.5
#         dirix = (lepolysx[ii+1].T-lepolysx[ii+3].T)/(sd_diag[ii+1])**.5
#         qwe=diri*dirix/lepolys[NN]**2
#         mxdd[ii]=np.sum(qwe)*(2/(NN*(NN+1)))
  
# Mnd=  mnd2*np.eye(NN-1)+np.diag(mnd1[0:NN-3],2)+np.diag(mnd3[0:NN-3],-2)
# Mdxd=np.diag(mxdd[:NN-2],1)-np.diag(mxdd[:NN-2],-1)
# Mxdd=Mdxd.T

# for ii in range(NN-1):
   
    
#     neunx = (lepolysx[ii].T+ bn[ii]*lepolysx[ii+2].T)/(sn_diag[ii])**.5
   
#     for jj in range(NN-1):
#         diri1=(lepolys[jj]-lepolys[jj+2])/(sd_diag[jj])**.5
#         dirix1 = (lepolysx[jj].T  -lepolysx[jj+2].T)/(sd_diag[jj])**.5
        
       
#         phi1=neunx*diri1/lepolys[NN]**2
 
#         Mxnd[jj,ii]=np.sum(phi1)*(2/(NN*(NN+1)))







ode_data,_, Ed,_,_,_,_,_,_,_,_,_,_,_,_,_=matA(NN,dt,EPSILON)




# ode_data=np.zeros((NN-1,NN-1,NN-1))
iode_data=np.zeros((NN-1,NN-1,NN-1))
ode_eye=np.zeros((NN-1,NN-1,NN-1))

for jj in range(NN-1):       
            ode_data0=ode_data[jj]
            ode_data[jj,]=np.diag(np.diag(ode_data0)**.5)
            
            iode_data[jj,]=np.diag(1/np.diag(ode_data0)**.5)
            ode_eye[jj,]=(iode_data[jj,]@ode_data0)@iode_data[jj,]






# Mxnd[abs(Mxnd)<10**-8]=0  #diri*neumann





# al_upre=torch.zeros((BATCH_SIZE,SHAPE-2,SHAPE-2,SHAPE-2)).to(device).double()
# al_vpre=torch.zeros((BATCH_SIZE,SHAPE-2,SHAPE-2,SHAPE-2)).to(device).double()
# al_wpre=torch.zeros((BATCH_SIZE,SHAPE-2,SHAPE-2,SHAPE-2)).to(device).double()
# Mnd=torch.from_numpy(Mnd).to(device).double()
# Mdxd=torch.from_numpy(Mdxd).to(device).double()
# Mxdd=torch.from_numpy(Mxdd).to(device).double()
# Md=torch.from_numpy(Md).to(device).double()
# Mxnd=torch.from_numpy(Mxnd).to(device).double()
Ed=torch.from_numpy(Ed).to(device).double()
# En=torch.from_numpy(En).to(device).double()
# Mm=torch.from_numpy(Mm).to(device).double()
# Mmx=torch.from_numpy(Mmx).to(device).double()



ode_data=torch.from_numpy(ode_data).to(device).double()

iode_data=torch.from_numpy(iode_data).to(device).double()


ode_eye=torch.from_numpy(ode_eye).to(device).double()


def closure(ald,fdata0,cf0):
 
    
    model.train()
    
    if torch.is_grad_enabled():
        optimizer.zero_grad()
    
    a_pred = model(fdata0)
    
    alx=a_pred[:,0]
    aly=a_pred[:,1]
   
    loss_u=torch.zeros(1)
    cfx0=cf0[:,0]
    cfy0=cf0[:,1]
   
    al_unext,al_vnext,exfx,exfy = weak_form0(alx,aly,cfx0,cfy0, NN,ode_eye,iode_data, Ed )
    
    
    
    alx0=Ed@torch.sum(iode_data@(alx.reshape((BATCH_SIZE,1,NN-1,NN-1,1))),4)
    aly0=Ed@torch.sum(iode_data@(aly.reshape((BATCH_SIZE,1,NN-1,NN-1,1))),4)
    loss=10**9*((torch.sum((al_unext-exfx)**2))+(torch.sum((al_vnext-exfy)**2)))
       
    

    loss_u1 = torch.max(abs(alx0-ald[:,0]))+torch.max(abs(aly0-ald[:,1]))
   
    if loss.requires_grad:
        loss.backward()
    
    ald1=torch.stack((alx0,aly0),dim=1)

    return  loss_u,loss_u1, loss, ald1
#

f_pred=0.0
torch.autograd.set_detect_anomaly(True)
################################################
time0 = time.time()
test1=int(1)
loss_a,  loss_f,    loss_validate, avg_l2_u=0,0,0,0

loss_u_test, loss_wf_test=0,0



print(torch.cuda.memory_allocated()/1024**3)

for batch_idx, sample_batch in enumerate(trainloader):
    all0 = sample_batch['data_u'][:BATCH_SIZE,:2,:1].double().to(device)
       
    fdata = sample_batch['f'][:BATCH_SIZE,:,0].double().to(device)
    cf00 = sample_batch['cf0'].double().to(device)
       

print(all0.shape)
print(fdata.shape)
print(cf00.shape)


loss_wf1=0



for epoch in tqdm(range(1, EPOCHS+1)):
        
        loss_u,loss_u1,  loss,a_pred = closure(all0,fdata,cf00)
        optimizer.step(loss.item)
        
        
       
        
        loss_u11 = np.round(float(loss_u1.item()), 12)        
        loss_train = np.round(float(loss.item()), 12)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        #SAVE train data
        if epoch % int(2) == 0:
            
            losses = log_loss(losses, loss_a,loss_u11,loss_f, loss_wf1, loss_train,  loss_wf_test,BATCH_SIZE, loss_u_test)
        

torch.save(model.state_dict(), PATH + '/model.pt')
torch.save(a_pred, PATH + '/data.pt')


print(loss_train)


print(torch.max(abs(a_pred[:,0]-all0[:,0,:ndt,])),torch.max(abs(a_pred[:,1]-all0[:,1,:ndt,])))



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

# loss_plot(gparams)
#values = model_stats(PATH, kind='validate', gparams=gparams)
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
newcall['order'].append(1)

df = pd.DataFrame(newcall)
PATH_call=os.path.join('training', f"{EQUATION}{EPSILON}", FILE, "order1")
df.to_csv(PATH_call + f'/call1_alp.csv')


# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()

