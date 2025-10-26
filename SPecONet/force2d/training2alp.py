"find au, av, aw at first time step"
# python training2alp00.py --equation ConvDiff2D --model Net3D --loss MSE --blocks 0 --file 800N23 --epochs 50000 --ks 9 --filters 10 --nbfuncs 30 --U 9 --pre_epochs 5000 --dt 0.01 --forcing num444am2  --kind cosN30 --ndt 1 --eps 0.1

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
from funs import *

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
# ARGS
# python training.py --equation Burgers --model NetC --blocks 4 --file 10000N63 --forcing uniform --epochs 50000
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--equation", type=str, default='ConvDiff2D', choices=['NS2d','Standard','Standard1', 'Burgers', 'Helmholtz', 'Standard2D', 'ConvDiff2D']) #, 'BurgersT' 
parser.add_argument("--pre_test", type=str, default='pre_Standard1', choices=['pre_Standard','pre_Standard1', 'pre_Burgers', 'pre_Helmholtz', 'pre_Standard2D', 'pre_ConvDiff2D'])
parser.add_argument("--model", type=str, default='Net3D', choices=['ResNet', 'NetA', 'NetB', 'NetC', 'NetD', 'Net2D', 'Net3D']) 
parser.add_argument("--blocks", type=int, default=0)
parser.add_argument("--loss", type=str, default='MSE', choices=['MAE', 'MSE', 'RMSE', 'RelMSE'])
parser.add_argument("--file", type=str, default='10000N15', help='Example: --file 2000N31') # 2^5-1, 2^6-1
parser.add_argument("--forcing", type=str, default='normal')
parser.add_argument("--epochs", type=int, default=80000)
parser.add_argument("--pre_epochs", type=int, default=5000)
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--filters", type=int, default=32)
parser.add_argument("--nbfuncs", type=int, default=1)
parser.add_argument("--A", type=float, default=0)
parser.add_argument("--F", type=float, default=0)
parser.add_argument("--U", type=float, default=1)
parser.add_argument("--WF", type=float, default=1) # 1 = include weaf form
parser.add_argument("--sd", type=float, default=1)
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--ndt", type=int, default=5)
parser.add_argument("--eps", type=float, default=1)
parser.add_argument("--kind", type=str, default='trainN10')
parser.add_argument("--path", type=str, default=None)


args = parser.parse_args()
gparams = args.__dict__
#pprint(gparams)

ndt=args.ndt
kind=args.kind
D_in = 2*ndt

EQUATION = args.equation
pre_test=args.pre_test

EPSILON = args.eps
PATH_alp=args.path

models = {
          'ResNet': ResNet,
          'NetA': NetA,
          'NetB': NetB,
          'NetC': NetC,
          'NetD': NetD,
          'Net3D': Net3D,
          'Net2D': Net2D
          }
MODEL = models[args.model]

#GLOBALS
gparams['epsilon'] = EPSILON
FILE = gparams['file']
DATASET = int(FILE.split('N')[0])
SHAPE = int(FILE.split('N')[1]) + 1

BLOCKS = int(gparams['blocks'])
EPOCHS = int(gparams['epochs'])
pre_EPOCHS = int(gparams['pre_epochs'])
NBFUNCS = int(gparams['nbfuncs'])
dt = gparams['dt']
FILTERS = int(gparams['filters'])
KERNEL_SIZE = int(gparams['ks'])
# PADDING = (KERNEL_SIZE - 1)//2
PADDING = int(3)
cur_time = str(datetime.datetime.now()).replace(' ', 'T')
cur_time = cur_time.replace(':','').split('.')[0].replace('-','')
FOLDER = f'{gparams["model"]}_{args.forcing}_epochs{EPOCHS}_{cur_time}'
pre_PATH = os.path.join('training', f"{pre_test}", FILE, FOLDER)
PATH = os.path.join('training', f"{EQUATION}{EPSILON}", FILE,'order1', FOLDER)
gparams['path'] = pre_PATH



BATCH_SIZE, Filters, D_out = int(DATASET), FILTERS, SHAPE
# LOSS SCALE FACTORS
A, U, F, WF = int(gparams['A']), 10**(gparams['U']), int(gparams['F']), (gparams['WF'])

NN=SHAPE-1



# CREATE PATHING
if os.path.isdir(PATH) == False: os.makedirs(PATH)
elif os.path.isdir(PATH) == True:
    if args.pretrained is None:
        print("\n\nPATH ALREADY EXISTS!\n\nEXITING\n\n")
        exit()
    else:
        print("\n\nPATH ALREADY EXISTS!\n\nLOADING MODEL\n\n")
        
# CREATE BASIS VECTORS
xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx, D,aa1,bb1 = basis_vectors(D_out,EPSILON ,equation=EQUATION)

# if BATCH_SIZE+1<DATASET:
#     shuffle1=True
# else: 
shuffle1=False

if args.model != 'ResNet' and args.model != 'Net3D'and args.model != 'Net3Dpressure':
    # NORMALIZE DATASET    
    NORM = True
    gparams['norm'] = True
    lg_dataset = get_data(gparams, kind)
    trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=shuffle1)
    gparams, transform_f = normalize(gparams, trainloader)
else:
    NORM = False
    gparams['norm'] = False
    transform_f = None

# LOAD DATASET
lg_dataset = get_data(gparams, kind, transform_f=transform_f)
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=shuffle1)

# lg_dataset = get_data(gparams, kind='validate', transform_f=transform_f)
# validateloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=True)

# INITIALIZE a model
if EQUATION=="Standardb":
    model = MODEL(D_in, Filters, D_out - 1, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
else:
    model = MODEL(10**A,ndt,D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)

# LOAD the trained model
if args.pretrained is not None:
    args.pretrained = 'N' + args.file.split('N')[-1] + '_' + args.equation + '_' + args.forcing
    # model.load_state_dict(torch.load(r'training/ConvDiff2D0.1/10N23/order1/Net3D_num333am2sigma5_epochs150000_20251025T192136/model.pt'), strict=False)
    model.load_state_dict(torch.load(f'training/{EQUATION}{EPSILON}/{BATCH_SIZE}N23/order1/Net3D_{args.forcing}_epochs{PATH_alp}/model.pt'), strict=False)
    model.train()

# Check if CUDA is available and then use it.
device = get_device()
gparams['device'] = device

# SEND TO GPU (or CPU)
model.to(device).double()
# www1=np.load('neww200.npy')
# www=torch.from_numpy(www1).contiguous().to(device).double()

# with torch.no_grad():
#     model.fcH.weight = torch.nn.parameter.Parameter(www)


param_size = 0
r0=1


for name,param in model.named_parameters():
    # print(name,r0,param.nelement())
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
if args.pretrained is None and args.model != 'NetB':
    model.apply(weights_init)
elif args.model == 'NetB':
    model.apply(weights_xavier)

#INIT OPTIMIZER
optimizer = init_optim(model)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.5)
# Construct our loss function and an Optimizer.
LOSS_TYPE = args.loss
if args.loss == 'MAE':
    criterion_a, criterion_u, criterion_wf = torch.nn.L1Loss(), torch.nn.L1Loss(), torch.nn.L1Loss()
elif args.loss == 'MSE':
    criterion_a, criterion_u, criterion_wf = torch.nn.MSELoss(reduction="sum"), L2, torch.nn.MSELoss(reduction="sum")
elif args.loss == 'RMSE':
    criterion_a, criterion_u, criterion_wf = RMSELoss(), RMSELoss(), RMSELoss()
elif args.loss == 'RelMSE':
    criterion_a, criterion_u, criterion_wf = RelMSELoss(batch=BATCH_SIZE), RelMSELoss(batch=BATCH_SIZE), RelMSELoss(batch=BATCH_SIZE)
criterion_f = torch.nn.L1Loss()

criterion = {
             'a': criterion_a,
             'f': criterion_f,
             'u': criterion_u,
             'wf': criterion_wf,
            }
BEST_LOSS = float('inf')
losses = {'loss_u':[],
          'loss_train':[],
          'loss_validate':[],
          'avg_l2_u': []}
gparams['path'] = PATH
log_gparams(gparams)

# for parameter in model.parameters():
#     print(parameter)
# print(sum(p.numel() for p in model.parameters()))

dt=0.01


Y,X,Z=np.meshgrid(xx,xx,xx)

Md,sd_diag,Ed,eid=basic_mat(b,NN,'dirichlet')

Mn,sn_diag,En,ein=basic_mat(bn,NN,'neumann')


Mm=np.zeros((NN-1,NN-1))
Mmx=np.zeros((NN-1,NN-1))

iMd=Ed@np.diag(1/eid)@Ed.T
iMn=En@np.diag(1/ein)@En.T





for ii in range(NN-1):
    phi=(lepolys[ii]- lepolys[ii+2])/(sd_diag[ii])**.5
    phix=(lepolysx[ii].T-lepolysx[ii+2].T)/(sd_diag[ii])**.5
    for jj in range(NN-1):
        psi=(lepolys[jj]+ bn[jj]*lepolys[jj+2])/(sn_diag[jj])**.5
        Mm[jj,ii]=np.sum(psi*phi/(lepolys[NN])**2)*(2/(NN*(NN+1)))
        Mmx[jj,ii]=np.sum(psi*phix/(lepolys[NN])**2)*(2/(NN*(NN+1)))

Mm[abs(Mm)<10**-8]=0
Mmx[abs(Mmx)<10**-8]=0


Mxnd=np.zeros((NN-1,NN-1))
Mdxd=np.zeros((NN-1,NN-1))
Mxdd=np.zeros((NN-1,NN-1))
Mnd=np.zeros((NN-1,NN-1))


mnd1=np.zeros((NN-1,))
mnd2=np.zeros((NN-1,))
mnd3=np.zeros((NN-1,))

mxdd=np.zeros((NN-1,))

for ii in range(NN-1):
    mnd2[ii]=2*(1/(2*ii+1)+b[ii]*bn[ii]/(2*ii+5))/(sd_diag[ii]*sn_diag[ii])**.5
    mnd1[ii]=(b[ii])*2/(2*ii+5)/(sd_diag[ii]*sn_diag[ii+2])**.5
    mnd3[ii]=(bn[ii])*2/(2*ii+5)/(sd_diag[2+ii]*sn_diag[ii])**.5
    if ii< NN-2:
        diri = (lepolys[ii]-lepolys[ii+2])/(sd_diag[ii])**.5
        dirix = (lepolysx[ii+1].T-lepolysx[ii+3].T)/(sd_diag[ii+1])**.5
        qwe=diri*dirix/lepolys[NN]**2
        mxdd[ii]=np.sum(qwe)*(2/(NN*(NN+1)))
    # mxnxd[ii]=-bn[ii]*(4*ii+6)/(sd_diag[ii]*sn_diag[ii])**.5
# mxnxd[0]=1/sd_diag[0]**.5

Mnd=  mnd2*np.eye(NN-1)+np.diag(mnd1[0:NN-3],2)+np.diag(mnd3[0:NN-3],-2)
Mdxd=np.diag(mxdd[:NN-2],1)-np.diag(mxdd[:NN-2],-1)
Mxdd=Mdxd.T

for ii in range(NN-1):
    # dirix = (lepolysx[ii].T-lepolysx[ii+2].T)/(sd_diag[ii])**.5
    
    neunx = (lepolysx[ii].T+ bn[ii]*lepolysx[ii+2].T)/(sn_diag[ii])**.5
    # neun = (lepolys[ii]+ bn[ii]*lepolys[ii+2])/(sn_diag[ii])**.5
    for jj in range(NN-1):
        diri1=(lepolys[jj]-lepolys[jj+2])/(sd_diag[jj])**.5
        dirix1 = (lepolysx[jj].T  -lepolysx[jj+2].T)/(sd_diag[jj])**.5
        
        # psi_l_M = (lepolysx[jj].T  -lepolysx[jj+2].T)/(sd_diag[jj])**.5
        phi1=neunx*diri1/lepolys[NN]**2
 
        Mxnd[jj,ii]=np.sum(phi1)*(2/(NN*(NN+1)))


ode_data=np.zeros((NN-1,NN-1,NN-1))

iode_data=np.zeros((NN-1,NN-1,NN-1))

# ode_cond=np.zeros((NN-1,NN-1,NN-1))
# iode_cond=np.zeros((NN-1,NN-1,NN-1))



pre_cond=np.zeros((NN-1,NN-1,NN-1))
# ipre_cond=np.zeros((NN-1,NN-1,NN-1))
ode_eye=np.zeros((NN-1,NN-1,NN-1))
for jj in range(NN-1):
       
            ode_data0=(1.5*eid[jj]/dt+EPSILON)*Md+EPSILON*eid[jj]*np.eye(NN-1)
            ode_data[jj,]=np.diag(np.diag(ode_data0)**.5)
            # iode_data[jj,]=np.linalg.solve(ode_data0,np.eye(NN-1))
            iode_data[jj,]=np.diag(1/np.diag(ode_data0)**.5)
            ode_eye[jj,]=(iode_data[jj,]@ode_data0)@iode_data[jj,]
            # ode_data[jj,]=(eid[jj]+eps)*Md+eps*eid[jj]*np.eye(NN-1)
            # ode_data[jj,]=ode1


# cond=np.sum(np.sum(ode_data**2,-1),-1)*np.sum(np.sum(iode_data**2,-1),-1)


Mxnd[abs(Mxnd)<10**-8]=0  #diri*neumann


t=0




al_upre=torch.zeros((BATCH_SIZE,SHAPE-2,SHAPE-2,SHAPE-2)).to(device).double()
al_vpre=torch.zeros((BATCH_SIZE,SHAPE-2,SHAPE-2,SHAPE-2)).to(device).double()
al_wpre=torch.zeros((BATCH_SIZE,SHAPE-2,SHAPE-2,SHAPE-2)).to(device).double()
Mnd=torch.from_numpy(Mnd).to(device).double()
Mdxd=torch.from_numpy(Mdxd).to(device).double()
Mxdd=torch.from_numpy(Mxdd).to(device).double()
Md=torch.from_numpy(Md).to(device).double()
Mxnd=torch.from_numpy(Mxnd).to(device).double()
Ed=torch.from_numpy(Ed).to(device).double()
En=torch.from_numpy(En).to(device).double()
Mm=torch.from_numpy(Mm).to(device).double()
Mmx=torch.from_numpy(Mmx).to(device).double()



ode_data=torch.from_numpy(ode_data).to(device).double()

iode_data=torch.from_numpy(iode_data).to(device).double()


ode_eye=torch.from_numpy(ode_eye).to(device).double()

# cond=torch.from_numpy(cond.reshape(1,1,NN-1,NN-1)).to(device).double()

# ode_cond=torch.from_numpy(ode_cond).to(device).double()

# icc=torch.sum(torch.sum(iode_cond,2),2)

# cc=torch.sum(torch.sum(ode_cond,2),2)

# print(icc,cc)

def closure(ald,fdata0,cf0):
 
    # print('111',torch.cuda.memory_allocated()/1024**3)
    model.train()
    # print('222',torch.cuda.memory_allocated()/1024**3)
    if torch.is_grad_enabled():
        optimizer.zero_grad()
    # print('333',torch.cuda.memory_allocated()/1024**3)
    # f0=torch.reshape(fdata,(1,1,NN-1,NN-1,NN-1) ).to(device).double()
    a_pred = model(fdata0)
    # print('444',torch.cuda.memory_allocated()/1024**3)
    # print(a_pred.shape)
    # input('dddd')
    # print(a_pred.shape)
    "check weak form"
    # a_pred=torch.sum(ode_data@((Ed.T@ald).reshape((BATCH_SIZE,2,1,NN-1,NN-1,1))),5)
    # print(a_pred.shape)
    
    alx=a_pred[:,0]
    aly=a_pred[:,1]
    # print(alx.shape)
    # input('sdf')
    # u_pred = torch.sum(phiset*a_pred0,dim=1)
    loss_u=torch.zeros(1)
    cfx0=cf0[:,0]
    cfy0=cf0[:,1]
   
    
    # cFx=cf[:,0:ndt]
    # cFy=cf[:,ndt:2*ndt]
    # cFz=cf[:,2*ndt:3*ndt]
    
    # al_unext,al_vnext,al_wnext,exfx,exfy,exfz,phial,Pexfx = weak_form0(EPSILON,al_upre,al_vpre,al_wpre,cfx0,cfy0,cfz0,cFx,cFy,cFz,dt, NN,ode_data,oden_data,pre_cond,pre_condn, a_pred,Mxnd,Mnd,Md,Mxdd,Mdxd,Ed,En,Mm,Mmx )
    al_unext,al_vnext,exfx,exfy = weak_form0(alx,aly,cfx0,cfy0, NN,ode_eye,iode_data, Ed )
    
    
    
    alx0=Ed@torch.sum(iode_data@(alx.reshape((BATCH_SIZE,1,NN-1,NN-1,1))),4)
    aly0=Ed@torch.sum(iode_data@(aly.reshape((BATCH_SIZE,1,NN-1,NN-1,1))),4)
    #loss = U*(torch.sum((al_unext-exfx)**2))+U*torch.sum((alx-ald0)**2)
    # loss = U*torch.sum((alx-ald0)**2)
    loss=10**9*((torch.sum((al_unext-exfx)**2))+(torch.sum((al_vnext-exfy)**2)))
       
    


            #           +(BATCH_SIZE/inx)*torch.sum((a_pred1[:3*inx:3,:ndt,]-ald[:4*inx:4,0:ndt,])**2+(a_pred1[1:3*inx:3,:ndt,]-ald[1:4*inx:4,0:ndt,])**2\
            # +(a_pred1[2:3*inx:3,:ndt,]-ald[2:4*inx:4,0:ndt,])**2))
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
        # all0 = sample_batch['data_u'][:BATCH_SIZE,:2,:1].double().to(device).reshape((BATCH_SIZE,2,ndt,SHAPE-2,SHAPE-2))
        all0 = sample_batch['data_u'][:BATCH_SIZE,:2,:1].double().to(device)
       
        fdata = sample_batch['f'][:BATCH_SIZE,:,0].double().to(device)
        cf00 = sample_batch['cf0'].double().to(device)
        # cf1 = sample_batch['cf'][:BATCH_SIZE,:,0].double().to(device)[:,:3*ndt,]

print(all0.shape)
print(fdata.shape)
print(cf00.shape)
# print(cf1.shape)

# fdata00=fdata.detach().cpu().numpy()
# with open('fxtorch.npy', 'wb') as data_ex:
#     np.save(data_ex, fdata00[:,0])
# input('dddff')

loss_wf1=0

# print(torch.cuda.memory_allocated()/1024**3)


for epoch in tqdm(range(1, EPOCHS+1)):
        
        loss_u,loss_u1,  loss,a_pred = closure(all0,fdata,cf00)
        # print(torch.cuda.memory_summary())
        # print(torch.cuda.memory_allocated()/1024**3)
        optimizer.step(loss.item)
        
        
       # optimizer.step()
        # print(torch.cuda.memory_allocated()/1024**3)
        # input('ggg')
        
        loss_u11 = np.round(float(loss_u1.item()), 12)        
        loss_train = np.round(float(loss.item()), 12)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        #SAVE train data
        if epoch % int(2) == 0:
            
            losses = log_loss(NBFUNCS,losses, loss_a,loss_u11,loss_f, loss_wf1, loss_train,  loss_wf_test,BATCH_SIZE, loss_u_test)
        #if lu1<5*10**-9 and lv1<5*10**-9 and lw1<5*10**-9:
         #   break
        
        #scheduler.step()

# u_test_save.append(np.reshape(dd[:,0][700][:,1],(D_out,))) 
# u_test_save.append(np.reshape(xx,(D_out,)))            
# with open(PATH+"/u_test.pkl", "wb") as fp:   #Pickling
#     pickle.dump(u_test_save, fp)

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
gparams['lossType'] = LOSS_TYPE

log_path(PATH)

# loss_plot(gparams)
#values = model_stats(PATH, kind='validate', gparams=gparams)
log_gparams(gparams)

import pandas as pd
newcall={'blocks':[],'file':[],'ks':[],'nbfuncs':[],'dt':[],'forcing':[],'ndt':[],'eps':[],'path':[],'order':[]}

newcall['blocks'].append(BLOCKS)
newcall['file'].append(FILE)
newcall['ks'].append(KERNEL_SIZE)
newcall['nbfuncs'].append(NBFUNCS)
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

