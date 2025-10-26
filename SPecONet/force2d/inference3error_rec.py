"inference time check"
# python inference3error_rec00.py --equation ConvDiff2D --model Net3D0 --loss MSE --blocks 0 --epochs 5000 --ks 9 --filters 10 --nbfuncs 10 --U 9 --pre_epochs 5000 --dt 0.01  --ndt 1 --eps 0.1 --kind cosN30 --file 100N23 --forcing num444am2sigma5 --order 20 --start 20
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
parser.add_argument("--equation", type=str, default='ConvDiff2D', choices=['NS2d','Standard', 'test3d','Standard1', 'Burgers', 'test3d', 'Helmholtz', 'Standard2D', 'ConvDiff2D']) #, 'BurgersT' 
parser.add_argument("--pre_test", type=str, default='pre_Standard1', choices=['pre_Standard','pre_Standard1', 'pre_Burgers', 'pre_Helmholtz', 'pre_Standard2D', 'pre_ConvDiff2D'])
parser.add_argument("--model", type=str, default='Net3D', choices=['ResNet', 'NetA', 'NetB', 'NetC', 'NetD', 'Net2D', 'Net3D', 'Net3D0']) 
parser.add_argument("--blocks", type=int, default=0)
parser.add_argument("--loss", type=str, default='MSE', choices=['MAE', 'MSE', 'RMSE', 'RelMSE'])
parser.add_argument("--file", type=str, default='10000N15', help='Example: --file 2000N31') # 2^5-1, 2^6-1
parser.add_argument("--forcing", type=str, default='normal')
parser.add_argument("--kind", type=str, default='trainN10')
parser.add_argument("--epochs", type=int, default=80000)
parser.add_argument("--pre_epochs", type=int, default=5000)
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--filters", type=int, default=32)
parser.add_argument("--nbfuncs", type=int, default=1)
parser.add_argument("--A", type=float, default=0)
parser.add_argument("--F", type=float, default=0)
parser.add_argument("--U", type=float, default=1)
parser.add_argument("--WF", type=int, default=1) # 1 = include weaf form
parser.add_argument("--sd", type=float, default=1)
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--ndt", type=int, default=5)
parser.add_argument("--eps", type=float, default=1)
parser.add_argument("--order", type=int, default=1)
parser.add_argument("--start", type=int, default=1)
parser.add_argument("--path", type=str)

args = parser.parse_args()
gparams = args.__dict__
#pprint(gparams)

ndt=args.ndt

ORDER=args.order
start1=args.start

D_in = 2*ndt

EQUATION = args.equation
pre_test=args.pre_test

EPSILON = args.eps
models = {
          'ResNet': ResNet,
          'NetA': NetA,
          'NetB': NetB,
          'NetC': NetC,
          'NetD': NetD,
          'Net3D': Net3D,
          'Net3D0': Net3D0,
          'Net2D': Net2D,
          'Net3Dpressure0':Net3Dpressure0
          }
MODEL = models[args.model]

MODEL2 = models['Net3Dpressure0']
kind=args.kind

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
PATH0 = args.path
FOLDER0 = f'Net3Dpressure_{args.forcing}_epochs{PATH0}'

PATH = os.path.join('training', f"{EQUATION}{EPSILON}", FILE,f"order{ORDER}" ,FOLDER)

PATH_prev=os.path.join('training', f"{EQUATION}{EPSILON}", FILE,f"order{ORDER-1}", FOLDER0)

gparams['PATH_prev']=PATH_prev

BATCH_SIZE, Filters, D_out = int(DATASET), FILTERS, SHAPE
# LOSS SCALE FACTORS
A, U, F, WF = int(gparams['A']), (gparams['U']), int(gparams['F']), int(gparams['WF'])

NN=SHAPE-1



# CREATE BASIS VECTORS
xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx, D,aa1,bb1 = basis_vectors(D_out,EPSILON ,equation=EQUATION)

# if BATCH_SIZE+1<DATASET:
#     shuffle1=True
# else: 
shuffle1=False
lg_dataset = get_data(gparams, kind, transform_f=None)
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=shuffle1)


NORM = False
gparams['norm'] = False
transform_f = None

# LOAD DATASET

# INITIALIZE a model

# lin_weight=torch.zeros((ORDER+1-start1,17496, 17496)).to(device).double()

# lin_weight2=torch.zeros((ORDER+1-start1, 17496, 5832)).to(device).double()

lin_weight=torch.zeros((ORDER+1-start1,4840, 968)).to(device).double()

lin_weight2=torch.zeros((ORDER+1-start1, 4840, 484)).to(device).double()

for ii in range(start1,ORDER+1):
    if ii>start1:        del model, param#,model2
    model = MODEL(1,ndt,D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
    model2 = MODEL2(1,1,1, 10, D_out - 2,  kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
    
 
    device = get_device()
    gparams['device'] = device
    model.to(device).double()
    model2.to(device).double()

    print({ii})
    if ii>10:
        ii=10
    # else: ii=ii%10    
    df1 = pd.read_csv(f'./training/{EQUATION}{EPSILON}/600N23/order1'+f"/call{ii}_alp.csv")
    df2 = pd.read_csv(f'./training/{EQUATION}{EPSILON}/600N23/order1'+f"/call{ii}_pp.csv")
    PATH=f'Net3D_{args.forcing}_epochs'+df1['path'][0]
    PATH2=f'Net3Dpressure_{args.forcing}_epochs'+df2['path'][0]
    # PATH2='Net3Dpressure_num444sigma5_epochs15000_20250226T042242'
    model.load_state_dict(torch.load(f'training/{EQUATION}{EPSILON}/600N23/order{ii}/'+PATH+'/model.pt'), strict=False)
    model2.load_state_dict(torch.load(f'training/{EQUATION}{EPSILON}/600N23/order{ii}/'+PATH2+'/model.pt'), strict=False)
    # model2.load_state_dict(torch.load(f'training/ConvDiff2D1.0/1500N19/order{ii}/Net3Dpressure_num444sigma5_epochs10000_20250213T170404/model.pt'), strict=False)



    param_size = 0
    r0=1
    # net.get_parameter('layer1.0.weight')

    for name,param in model.named_parameters():
       
        if name =='fcH.weight':
           lin_weight[0]=param.T

    for name,param in model2.named_parameters():
        
        if name =='fcH.weight':
           lin_weight2[0]=param.T
# Check if CUDA is available and then use it.


Mxnd=np.zeros((NN-1,NN-1))
phisets=np.zeros((N+1,N-1))
phinsets=np.zeros((N+1,N-1))
phixsets=np.zeros((N+1,N-1))
Md,sd_diag,Ed,eid=basic_mat(b,NN,'dirichlet')
Mn,sn_diag,En,ein=basic_mat(bn,NN,'neumann')
# for ii in range(NN-1):
#     
iMd=Ed@np.diag(1/eid)@Ed.T

mnd1=np.zeros((NN-1,))
mnd2=np.zeros((NN-1,))
mnd3=np.zeros((NN-1,))


for ii in range(NN-1):
    phi=(lepolys[ii]- lepolys[ii+2])/(sd_diag[ii])**.5
    phin=(lepolys[ii]+bn[ii]*lepolys[ii+2])/(sn_diag[ii])**.5
    phix=(lepolysx[ii].T-lepolysx[ii+2].T)/(sd_diag[ii])**.5
    phinsets[:,ii]=phin[:,0]
    phisets[:,ii]=phi[:,0]
    phixsets[:,ii]=phix[:,0]
    neunx = (lepolysx[ii].T+ bn[ii]*lepolysx[ii+2].T)/(sn_diag[ii])**.5
    dirix = (lepolysx[ii].T-lepolysx[ii+2].T)/(sd_diag[ii])**.5
    mnd2[ii]=2*(1/(2*ii+1)+b[ii]*bn[ii]/(2*ii+5))/(sd_diag[ii]*sn_diag[ii])**.5
    mnd1[ii]=(b[ii])*2/(2*ii+5)/(sd_diag[ii]*sn_diag[ii+2])**.5
    mnd3[ii]=(bn[ii])*2/(2*ii+5)/(sd_diag[2+ii]*sn_diag[ii])**.5
    for jj in range(NN-1):
          diri1=(lepolys[jj]-lepolys[jj+2])/(sd_diag[jj])**.5
          phi1=neunx*diri1/lepolys[NN]**2
          Mxnd[jj,ii]=np.sum(phi1)*(2/(NN*(NN+1)))

Mnd=  mnd2*np.eye(NN-1)+np.diag(mnd1[0:NN-3],2)+np.diag(mnd3[0:NN-3],-2)

# ode_eye=np.zeros((NN-1,NN-1,NN-1))
# ode_data=np.zeros((NN-1,NN-1,NN-1))
iode_data=np.zeros((NN-1,NN-1,NN-1))
ioden_data=np.zeros((NN-1,NN-1,NN-1))
for jj in range(NN-1):
       
            ode_data0=(1.5*eid[jj]/dt+EPSILON)*Md+EPSILON*eid[jj]*np.eye(NN-1)
            oden_data0=Mn+ein[jj]*np.eye(NN-1)
            # iode_data[jj,]=np.linalg.solve(ode_data0,np.eye(NN-1))
            iode_data[jj,]=np.diag(1/np.diag(ode_data0)**.5)
           
            ioden_data[jj,]=np.diag(1/np.diag(oden_data0)**.5)

phisets=torch.from_numpy(phisets).to(device).double()
phinsets=torch.from_numpy(phinsets).to(device).double()
phixsets=torch.from_numpy(phixsets).to(device).double()
Ed=torch.from_numpy(Ed).to(device).double()
En=torch.from_numpy(En).to(device).double()
iode_data=torch.from_numpy(iode_data).to(device).double()
ioden_data=torch.from_numpy(ioden_data).to(device).double()
Mnd=torch.from_numpy(Mnd).to(device).double()
Mxnd=torch.from_numpy(Mxnd).to(device).double()
Md=torch.from_numpy(Md).to(device).double()
iMd=torch.from_numpy(iMd).to(device).double()
D=torch.from_numpy(D).to(device).double()
t00 = time.time()
for batch_idx, sample_batch in enumerate(trainloader):
        
        # aa= sample_batch['data_u'][:BATCH_SIZE,3:4,start1-1:ORDER].double().to(device)
        all0 = sample_batch['data_u'][:BATCH_SIZE,:3,start1-1:ORDER].double().to(device)
        udata00 = sample_batch['uex'].double().to(device)[:BATCH_SIZE,:2,start1-1:ORDER]
        fdata000 = sample_batch['f'][:BATCH_SIZE,:,0:1].double().to(device)
        # fdata000 = sample_batch['f'][:BATCH_SIZE,:,0:1].double().to(device)


Y,X=np.meshgrid(xx,xx)


       
fdata0=torch.permute(fdata000,(2,0,1,3,4)).reshape(BATCH_SIZE*(ORDER+1-start1),2,SHAPE,SHAPE)



a_pred0 = model(fdata0).reshape((ORDER+1-start1),BATCH_SIZE,-1)
a_pred=(a_pred0@lin_weight).reshape((ORDER+1-start1),BATCH_SIZE,2,SHAPE-2,SHAPE-2,1)

# alp=torch.permute(a_pred,(1,2,0,3,4))
# alp111=torch.sum(pre_cond@(alp.reshape((BATCH_SIZE,2,(ORDER+1-start1),NN-1,NN-1,1))),5)
# alp11=Ed@alp111
# alp1=torch.transpose(Ed@torch.transpose(alp11,2,3),2,3)

alp1=Ed@torch.sum(iode_data@a_pred,5)



ux=reconstructx(alp1[0,:,0:1], phisets,phixsets)
vx=reconstructx(alp1[0,:,1:2],phixsets, phisets)

uhat=(ux+vx).reshape(BATCH_SIZE*(ORDER+1-start1),1,SHAPE,SHAPE)

a_phi0 = model2(uhat)
# a_phi0 = model2(uhat)

a_phi=a_phi0.reshape(BATCH_SIZE,1,-1)@lin_weight2
a_phi=a_phi.reshape(BATCH_SIZE,1,SHAPE-2,SHAPE-2,1)
a_pred1=En@torch.sum(ioden_data@(a_phi),4)
# a_pred1=(torch.transpose(En@torch.transpose(alphi2,2,3),2,3))



t00 = time.time()


ubar,vbar=sol(alp1[0,:,0:1],alp1[0,:,1:2],dt,a_pred1[:,0:1],Mxnd,Mnd,Md,iMd,phisets )


print('inference time',time.time() - t00)
ubar=ubar.detach().cpu().numpy()
vbar=vbar.detach().cpu().numpy()

if ORDER==1:
    pp0ex=0
    pp0=0
elif ORDER>1.1:
    pp0ex=torch.load(f"training/{EQUATION}{EPSILON}/pp/ppex{ORDER-1}.pt")
    pp0=torch.load(f"training/{EQUATION}{EPSILON}/pp/pp{ORDER-1}.pt")


pexx,pexy,pex=psol(all0[:,0],all0[:,1],all0[:,2],pp0ex,phisets,phixsets,phinsets,EPSILON,D )
pbarx,pbary,pbar=psol(alp1[0,:,0:1],alp1[0,:,1:2],a_pred1[:,0:1],pp0,phisets,phixsets,phinsets,EPSILON,D )


torch.save(pex, f"training/{EQUATION}{EPSILON}/pp/ppex{ORDER}.pt")
torch.save(pbar, f"training/{EQUATION}{EPSILON}/pp/pp{ORDER}.pt")


uex=udata00[:,0]
vex=udata00[:,1]


pbarx=pbarx.detach().cpu().numpy()
pbary=pbary.detach().cpu().numpy()


uex=uex.detach().cpu().numpy()
vex=vex.detach().cpu().numpy()


pexx=pexx.detach().cpu().numpy()
pexy=pexy.detach().cpu().numpy()





lepp=(lepolys[N]*lepolys[N].T).reshape(1,1,SHAPE,SHAPE)

def intt(f,le):
    jj=SHAPE
    f1=((f/le)**2).reshape(BATCH_SIZE,-1)
    iit=(2/((jj-1)*jj))**2*np.sum(f1,-1)
    return iit

# fdata0=fdata0.detach().cpu().numpy()
# with open(f'sigma{WF}/force{BATCH_SIZE}_{ORDER}sigma{WF}.npy', 'wb') as data_ex:
#             np.save(data_ex, fdata0[79-5:79+5])
ddata='sigma5all'
if os.path.isdir(f'training/{EQUATION}{EPSILON}/uex{ddata}') == False: os.makedirs(f'training/{EQUATION}{EPSILON}/uex{ddata}')
if os.path.isdir(f'training/{EQUATION}{EPSILON}/ubar{ddata}') == False: os.makedirs(f'training/{EQUATION}{EPSILON}/ubar{ddata}')


if ORDER%20==0:
    with open(f'training/{EQUATION}{EPSILON}/ubar{ddata}/ubar{ORDER}.npy', 'wb') as data_ex:
                np.save(data_ex, ubar)

    with open(f'training/{EQUATION}{EPSILON}/ubar{ddata}/vbar{ORDER}.npy', 'wb') as data_ex:
                np.save(data_ex, vbar)

                
    with open(f'training/{EQUATION}{EPSILON}/uex{ddata}/usol{ORDER}.npy', 'wb') as data_ex:
                np.save(data_ex, uex)

    with open(f'training/{EQUATION}{EPSILON}/uex{ddata}/vsol{ORDER}.npy', 'wb') as data_ex:
                np.save(data_ex, vex)


ul21=intt(ubar-uex,lepp)
vl21=intt(vbar-vex,lepp)


ul22=intt(uex,lepp)
vl22=intt(vex,lepp)


pxl21=intt(pbarx-pexx,lepp)
pyl21=intt(pbary-pexy,lepp)


pxl22=intt(pexx,lepp)
pyl22=intt(pexy,lepp)




ul2=(ul21/ul22)**.5
vl2=(vl21/vl22)**.5


pl2=((pxl21+pyl21)/(pxl22+pyl22))**.5
# pl2=ul2
# pexx=uex

jj=BATCH_SIZE
print('ML2u',np.max(ul2[:jj]),np.max(vl2[:jj]),np.max(pl2[:jj]))
print('AL2u',np.mean(ul2[:jj]),np.mean(vl2[:jj]),np.mean(pl2[:jj]))
print('stdL2u',np.std(ul2),np.std(vl2),np.std(pl2))
# print('rel0',ul2[0],vl2[0],wl2[0])
# print('ML2u',ul2[0],ul2[1],ul2[2])

print('max u',np.max(abs(uex)),np.max(abs(vex)),np.max(abs(pexx)))

with open(f'training/{EQUATION}{EPSILON}/uex{ddata}/uex{ORDER}.npy', 'wb') as data_ex:
            np.save(data_ex, ul22)
            
with open(f'training/{EQUATION}{EPSILON}/uex{ddata}/vex{ORDER}.npy', 'wb') as data_ex:
            np.save(data_ex, vl22)



with open(f'training/{EQUATION}{EPSILON}/uex{ddata}/pex{ORDER}.npy', 'wb') as data_ex:
            np.save(data_ex, (pxl22+pyl22))

with open(f'training/{EQUATION}{EPSILON}/ubar{ddata}/lu{ORDER}.npy', 'wb') as data_ex:
            np.save(data_ex, ul21)
with open(f'training/{EQUATION}{EPSILON}/ubar{ddata}/lv{ORDER}.npy', 'wb') as data_ex:
            np.save(data_ex, vl21)
            

with open(f'training/{EQUATION}{EPSILON}/ubar{ddata}/lp{ORDER}.npy', 'wb') as data_ex:
            np.save(data_ex, (pxl21+pyl21))    



import pandas as pd
# data=pd.read_csv('call_alp.csv')
newcall = {'order':[],'uL2':[],'vL2':[],'pL2':[]}
# newcall=pd.read_csv('call_alp.csv')  


newcall['order'].append(ORDER)
newcall['uL2'].append(np.mean(ul2))
newcall['vL2'].append(np.mean(vl2))

newcall['pL2'].append(np.mean(pl2))

df = pd.DataFrame(newcall)

if ORDER==1:
    df.to_csv( f'2dforce{BATCH_SIZE}{ddata}.csv', index=False)
    
elif ORDER>1.1:
    df.to_csv( f'2dforce{BATCH_SIZE}{ddata}.csv', mode='a', index=False, header=False)
