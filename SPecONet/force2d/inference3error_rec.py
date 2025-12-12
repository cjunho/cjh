"inference time check"
# python inference3error_rec.py --blocks 0 --ks 9 --filters 10 --dt 0.01  --ndt 1 --eps 0.1 --kind force2d --file 100N23 --forcing sigma5 --order 1 --start 1
import random
import torch
import time
import datetime
import subprocess
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
parser.add_argument("--equation", type=str, default='ConvDiff2D', choices=['NS2d','Standard', 'test3d','Standard1', 'Burgers', 'test3d', 'Helmholtz', 'Standard2D', 'ConvDiff2D']) #, 'BurgersT' 
parser.add_argument("--blocks", type=int, default=0)
parser.add_argument("--file", type=str, default='10000N15', help='Example: --file 2000N31') # 2^5-1, 2^6-1
parser.add_argument("--forcing", type=str, default='normal')
parser.add_argument("--kind", type=str, default='trainN10')
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--filters", type=int, default=32)
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--ndt", type=int, default=5)
parser.add_argument("--eps", type=float, default=1)
parser.add_argument("--order", type=int, default=1)
parser.add_argument("--start", type=int, default=1)


args = parser.parse_args()
gparams = args.__dict__
ndt=args.ndt
ORDER=args.order
start1=args.start
D_in = 2*ndt
EQUATION = 'NS2d'
EPSILON = args.eps
MODEL = Net3D0
MODEL2 = Net3Dpressure0

kind=args.kind

#GLOBALS
gparams['epsilon'] = EPSILON
FILE = gparams['file']
DATASET = int(FILE.split('N')[0])
SHAPE = int(FILE.split('N')[1]) + 1
BLOCKS = int(gparams['blocks'])
dt = gparams['dt']
FILTERS = int(gparams['filters'])
KERNEL_SIZE = int(gparams['ks'])
PADDING = int(3)
BATCH_SIZE, Filters, D_out = int(DATASET), FILTERS, SHAPE
NN=SHAPE-1



# CREATE BASIS VECTORS
xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx, D,aa1,bb1 = basis_vectors(D_out,EPSILON ,equation=EQUATION)

shuffle1=False
NORM = False
gparams['norm'] = False
transform_f = None


# LOAD DATASET
lg_dataset = get_data(gparams, kind, transform_f=None)
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=shuffle1)


# LOAD the trained model
lin_weight=torch.zeros((ORDER+1-start1,4840, 968)).to(device).double()
lin_weight2=torch.zeros((ORDER+1-start1, 4840, 484)).to(device).double()

for ii in range(start1,ORDER+1):
    if ii>start1:        del model, param#,model2
    model = MODEL(ndt,D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
    model2 = MODEL2(1,1, 10, D_out - 2,  kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
    
 
    device = get_device()
    gparams['device'] = device
    model.to(device).double()
    model2.to(device).double()

    print({ii})
   
    df1 = pd.read_csv(f'./training/{EQUATION}{EPSILON}/600N23/order1'+f"/call{ii}_alp.csv")
    df2 = pd.read_csv(f'./training/{EQUATION}{EPSILON}/600N23/order1'+f"/call{ii}_pp.csv")
    PATH=f'Net3D_{args.forcing}_epochs'+df1['path'][0]
    PATH2=f'Net3Dpressure_{args.forcing}_epochs'+df2['path'][0]
    model.load_state_dict(torch.load(f'training/{EQUATION}{EPSILON}/600N23/order{ii}/'+PATH+'/model.pt'), strict=False)
    model2.load_state_dict(torch.load(f'training/{EQUATION}{EPSILON}/600N23/order{ii}/'+PATH2+'/model.pt'), strict=False)
   

  

    for name,param in model.named_parameters():
       
        if name =='fcH.weight':
           lin_weight[0]=param.T

    for name,param in model2.named_parameters():
        
        if name =='fcH.weight':
           lin_weight2[0]=param.T

#Generate matrices to compute the weak formulation
ode_data,oden_data, Ed,En,_,_,Mnd,Mxdd,Mxnd,Mdxd,Md,iMd,Mm,Mmx,phisets,phixsets,phinsets,sd_diag,lepp=matA(NN,dt,EPSILON)
iode_data=np.zeros((NN-1,NN-1,NN-1))
ioden_data=np.zeros((NN-1,NN-1,NN-1))
for jj in range(NN-1):
        iode_data[jj,]=np.diag(1/np.diag(ode_data[jj])**.5)
           
        ioden_data[jj,]=np.diag(1/np.diag(oden_data[jj])**.5)

#Convert numpy files into torch files
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

#Load input data and reference solutions
for batch_idx, sample_batch in enumerate(trainloader):
        
        all0 = sample_batch['data_u'][:BATCH_SIZE,:3,start1-1:ORDER].double().to(device)
        udata00 = sample_batch['uex'].double().to(device)[:BATCH_SIZE,:2,start1-1:ORDER]
        fdata000 = sample_batch['f'][:BATCH_SIZE,:,start1-1:ORDER].double().to(device)
      

#input data for u      
fdata0=torch.permute(fdata000,(2,0,1,3,4)).reshape(BATCH_SIZE*(ORDER+1-start1),2,SHAPE,SHAPE)

#Mapping outputs
a_pred0 = model(fdata0).reshape((ORDER+1-start1),BATCH_SIZE,-1)
a_pred=(a_pred0@lin_weight).reshape((ORDER+1-start1),BATCH_SIZE,2,SHAPE-2,SHAPE-2,1)

#Inference, alpha    
alp1=Ed@torch.sum(iode_data@a_pred,5)


#input data for Phi 
ux=reconstructx(alp1[0,:,0:1], phisets,phixsets)
vy=reconstructx(alp1[0,:,1:2],phixsets, phisets)
uhat=(ux+vy).reshape(BATCH_SIZE*(ORDER+1-start1),1,SHAPE,SHAPE)

#Mapping outputs
a_phi0 = model2(uhat)
a_phi=a_phi0.reshape(BATCH_SIZE,1,-1)@lin_weight2
a_phi=a_phi.reshape(BATCH_SIZE,1,SHAPE-2,SHAPE-2,1)

#Inference, phi  
a_pred1=En@torch.sum(ioden_data@(a_phi),4)



t00 = time.time()

#Construct inferences
ubar,vbar=sol(alp1[0,:,0:1],alp1[0,:,1:2],dt,a_pred1[:,0:1],Mxnd,Mnd,Md,iMd,phisets )


print('inference time',time.time() - t00)


ubar=ubar.detach().cpu().numpy()
vbar=vbar.detach().cpu().numpy()

#Load the previous p
if ORDER==1:
    pp0ex=0
    pp0=0
elif ORDER>1.1:
    pp0ex=torch.load(f"training/{EQUATION}{EPSILON}/pp/ppex{ORDER-1}.pt")
    pp0=torch.load(f"training/{EQUATION}{EPSILON}/pp/pp{ORDER-1}.pt")

#Compute p
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






#Relative l2 norm
def intt(f,le):
    jj=SHAPE
    f1=((f/le)**2).reshape(BATCH_SIZE,-1)
    iit=(2/((jj-1)*jj))**2*np.sum(f1,-1)
    return iit


ddata='sigma5'
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



print('ML2u',np.max(ul2),np.max(vl2),np.max(pl2))
print('AL2u',np.mean(ul2),np.mean(vl2),np.mean(pl2))
print('stdL2u',np.std(ul2),np.std(vl2),np.std(pl2))

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

newcall = {'order':[],'uL2':[],'vL2':[],'pL2':[]}


newcall['order'].append(ORDER)
newcall['uL2'].append(np.mean(ul2))
newcall['vL2'].append(np.mean(vl2))

newcall['pL2'].append(np.mean(pl2))

df = pd.DataFrame(newcall)

if ORDER==1:
    df.to_csv( f'2dforce{BATCH_SIZE}{ddata}.csv', index=False)
    
elif ORDER>1.1:
    df.to_csv( f'2dforce{BATCH_SIZE}{ddata}.csv', mode='a', index=False, header=False)
