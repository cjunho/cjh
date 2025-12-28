"inference time to check simple L2"
# python inference3.py --equation NS3d  --blocks 0 --ks 9 --filters 3 --dt 0.01 --forcing num444sigma5 --ndt 1 --eps 1.0 --kind force3d --file 5N19 --order 1 --start 1
import random
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
from funsjax import matA,basic_mat

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
# ARGS
# python training.py --equation Burgers --model NetC --blocks 4 --file 10000N63 --forcing uniform --epochs 50000
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--equation", type=str, default='NS3d', choices=['Standard','Standard1', 'Burgers', 'test3d', 'Helmholtz', 'Standard2D', 'NS3d']) #, 'BurgersT' 
parser.add_argument("--blocks", type=int, default=0)
parser.add_argument("--file", type=str, default='10000N15', help='Example: --file 2000N31') # 2^5-1, 2^6-1
parser.add_argument("--train", type=str, default='10000N15', help='Example: --file 2000N31') # 2^5-1, 2^6-1
parser.add_argument("--forcing", type=str, default='normal')
parser.add_argument("--kind", type=str, default='trainN10')
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--filters", type=int, default=32)
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

D_in = 3*ndt

EQUATION = args.equation


EPSILON = args.eps

MODEL = Net3D0

MODEL2 = Net3Dpressure0
kind=args.kind
TRAIN=args.train

#GLOBALS
gparams['epsilon'] = EPSILON
FILE = gparams['file']
DATASET = int(FILE.split('N')[0])
SHAPE = int(FILE.split('N')[1]) + 1
BLOCKS = int(gparams['blocks'])



dt = gparams['dt']
FILTERS = int(gparams['filters'])
KERNEL_SIZE = int(gparams['ks'])
# PADDING = (KERNEL_SIZE - 1)//2
PADDING = int(3)


BATCH_SIZE, Filters, D_out = int(DATASET), FILTERS, SHAPE
# LOSS SCALE FACTORS

NN=SHAPE-1



# CREATE BASIS VECTORS
xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx, D,aa1,bb1 = basis_vectors(D_out,EPSILON ,equation='NS3d')

# if BATCH_SIZE+1<DATASET:
#     shuffle1=True
# else: 
shuffle1=False


NORM = False
gparams['norm'] = False
transform_f = None

# LOAD DATASET

# INITIALIZE a model
lg_dataset = get_data(gparams, kind, transform_f=transform_f)
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=shuffle1)
# lg_dataset = get_data(gparams, kind='validate', transform_f=transform_f)
# validateloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=True)

lin_weight=torch.zeros((ORDER+1-start1,17496, 17496)).to(device).double()

lin_weight2=torch.zeros((ORDER+1-start1, 17496, 5832)).to(device).double()

for ii in range(start1,ORDER+1):
    if ii>start1:
        del model, param#,model2
    model = MODEL(ndt,D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
    model2 = MODEL2(1,1, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
    device = get_device()
    gparams['device'] = device
    model.to(device).double()
    model2.to(device).double()

    print({ii})
    df1 = pd.read_csv(f'./training/{EQUATION}{EPSILON}/{TRAIN}/order1'+f"/call{ii}_alp.csv")
    df2 = pd.read_csv(f'./training/{EQUATION}{EPSILON}/{TRAIN}/order1'+f"/call{ii}_pp.csv")
    PATH=f'Net3D_{args.forcing}_epochs'+df1['path'][0]
    PATH2=f'Net3Dpressure_{args.forcing}_epochs'+df2['path'][0]
    # PATH2='Net3Dpressure_num444sigma5_epochs15000_20250226T042242'
    model.load_state_dict(torch.load(f'training/{EQUATION}{EPSILON}/{TRAIN}/order{ii}/'+PATH+'/model.pt'), strict=False)
    model2.load_state_dict(torch.load(f'training/{EQUATION}{EPSILON}/{TRAIN}/order{ii}/'+PATH2+'/model.pt'), strict=False)
    # model2.load_state_dict(torch.load(f'training/NS3d1.0/1500N19/order{ii}/Net3Dpressure_num444sigma5_epochs10000_20250213T170404/model.pt'), strict=False)



    param_size = 0
    r0=1
    # net.get_parameter('layer1.0.weight')

    for name,param in model.named_parameters():
        
        if name =='fcH.weight':
           lin_weight[ii-start1]=param.T

    for name,param in model2.named_parameters():
        
        if name =='fcH.weight':
           lin_weight2[ii-start1]=param.T
# Check if CUDA is available and then use it.

ode_data,oden_data,Ed,En,_,_,_,Mnd,Mxdd,Mxnd,Mdxd,Md,iMd,Mm,Mmx,phisets,phixsets,phinsets,sd_diag,lep=matA(NN,dt,EPSILON)


pre_cond=np.zeros((NN-1,NN-1,NN-1,NN-1))
pre_condn=np.zeros((NN-1,NN-1,NN-1,NN-1))


for jj in range(NN-1):
        for ii in range(NN-1):
            ode1=ode_data[jj,ii]
            pre_cond[jj,ii,]=np.diag(1/np.diag(ode1)**.5)
            ode2=oden_data[jj,ii]
            pre_condn[jj,ii]=np.diag(1/np.diag(ode2)**.5)


            


pre_cond=torch.from_numpy(pre_cond).to(device).double()
pre_condn=torch.from_numpy(pre_condn).to(device).double()
Ed=torch.from_numpy(Ed).to(device).double()
En=torch.from_numpy(En).to(device).double()
phisets=torch.from_numpy(phisets).to(device).double()
phinsets=torch.from_numpy(phinsets).to(device).double()
phixsets=torch.from_numpy(phixsets).to(device).double()
lep=torch.from_numpy(lep).to(device).double()
Mxnd=torch.from_numpy(Mxnd).to(device).double()
iMd=torch.from_numpy(iMd).to(device).double()
Md=torch.from_numpy(Md).to(device).double()
Mnd=torch.from_numpy(Mnd).to(device).double()
Mm=torch.from_numpy(Mm).to(device).double()
Mmx=torch.from_numpy(Mmx).to(device).double()
oden_data=torch.from_numpy(oden_data).to(device).double()
D=torch.from_numpy(D).to(device).double()
# torch.manual_seed(0)

# fdata=torch.rand(DATASET,3*ndt,SHAPE,SHAPE,SHAPE).double().to(device)

# aa=torch.rand(4*DATASET,1,1,1,1).double().to(device)
t00 = time.time()
for batch_idx, sample_batch in enumerate(trainloader):
        
        # aa= sample_batch['data_u'][:BATCH_SIZE,3:4,start1-1:ORDER].double().to(device)
        all0 = sample_batch['data_u'][:BATCH_SIZE,:,start1-1:ORDER].double().to(device)
        # alp2 = sample_batch['data_u'][:BATCH_SIZE,:,ORDER-1:ORDER].double().to(device).reshape((BATCH_SIZE,4,ndt,SHAPE-2,SHAPE-2,SHAPE-2))
        udata00 = sample_batch['uex'].double().to(device)[:BATCH_SIZE,:,start1-1:ORDER]
        fdata00 = sample_batch['f'][:BATCH_SIZE,:,start1-1:ORDER].double().to(device)
        
fdata0=torch.permute(fdata00,(2,0,1,3,4,5)).reshape(BATCH_SIZE*(ORDER+1-start1),3,SHAPE,SHAPE,SHAPE)

a_pred0 = model(fdata0).reshape((ORDER+1-start1),BATCH_SIZE,-1)
a_pred=(a_pred0@lin_weight).reshape(ORDER+1-start1,BATCH_SIZE,3,SHAPE-2,SHAPE-2,SHAPE-2)

alp=torch.permute(a_pred,(1,2,0,3,4,5))
alp111=torch.sum(pre_cond@(alp.reshape((BATCH_SIZE,3,ORDER+1-start1,NN-1,NN-1,NN-1,1))),6)
alp11=Ed@alp111
alp1=torch.transpose(Ed@torch.transpose(alp11,3,4),3,4)

# err=abs(alp2[:,:3]-alp1)
# print(torch.max(err))

cont=1
# alp11=(alp1[:,:3])/cont
ux=reconstruct2(alp1[:,0], phisets,phixsets,'x')
vx=reconstruct2(alp1[:,1], phisets,phixsets,'y')
wx=reconstruct2(alp1[:,2], phisets,phixsets,'z')
uhat=ux+vx+wx

a_phi0 = model2(uhat).reshape((ORDER+1-start1),BATCH_SIZE,-1)

a_phi=(a_phi0@lin_weight2).reshape(ORDER+1-start1,BATCH_SIZE,1,SHAPE-2,SHAPE-2,SHAPE-2)

a_phi1=torch.permute(a_phi,(1,2,0,3,4,5))
a_phi1[:,0,0,-1,-1,0]=0
a_pred111=torch.sum(pre_condn@(a_phi1.reshape((BATCH_SIZE,1,ORDER+1-start1,NN-1,NN-1,NN-1,1))),6)

a_pred11=En@(a_pred111)
a_pred1=cont*(torch.transpose(En@torch.transpose(a_pred11,3,4),3,4))

t00 = time.time()


# a_phi0 = model2(fdata00).reshape((ORDER+1-start1),BATCH_SIZE,-1)

# print('input size',a_pred0.shape)
# print('input size',a_phi0.shape)







# all0=all0.detach().cpu().numpy()

# with open('data100.npy', 'wb') as data_ex:
#             np.save(data_ex, all0)
# input('dsfds')

# alp=torch.permute(a_pred,(1,2,0,3,4,5))
# alp111=torch.sum(pre_cond@(alp.reshape((BATCH_SIZE,3,ORDER+1-start1,NN-1,NN-1,NN-1,1))),6)
# alp11=Ed@alp111
# alp1=torch.transpose(Ed@torch.transpose(alp11,3,4),3,4)


# print(a_pred111.shape)
# input('dafad')








# # a_pred111=phi_combine(alp1[:,0],alp1[:,1],alp1[:,2],dt, N,oden_data,En,Mm,Mmx,BATCH_SIZE,ORDER+1-start1)



# err=abs(a_pred1-all0[:,3])
# print(torch.max(err))
# input('sdfa')
""""""
# a_pred1=all0[:,3:]
""""""


ubar,vbar,wbar=sol(alp1[:,0],alp1[:,1],alp1[:,2],dt,a_pred1[:,0],Mxnd,Mnd,Md,iMd,phisets )


print('inference time',time.time() - t00)

if ORDER==1:
    pp0ex=0
    pp0=0
elif ORDER>1.1:
    pp0ex=torch.load(f"training/NS3d1.0/pp/ppex{ORDER-1}.pt")
    pp0=torch.load(f"training/NS3d1.0/pp/pp{ORDER-1}.pt")
pexx,pexy,pexz,pex=psol(all0[:,0],all0[:,1],all0[:,2],all0[:,3],pp0ex,phisets,phixsets,phinsets,EPSILON,D )
pbarx,pbary,pbarz,pbar=psol(alp1[:,0],alp1[:,1],alp1[:,2],a_pred1[:,0],pp0,phisets,phixsets,phinsets,EPSILON,D )

# print(pexx.shape)

# ppex=torch.cat(torch.cat(pexx,pexy,dim=1),pexz,dim=1)
# input('sdfdsf')

torch.save(pex, f"training/NS3d1.0/pp/ppex{ORDER}.pt")
torch.save(pbar, f"training/NS3d1.0/pp/pp{ORDER}.pt")

uex=udata00[:,0]
vex=udata00[:,1]
wex=udata00[:,2]



# with open('alp01.npy', 'wb') as data_ex:
#             np.save(data_ex, alp01)

# with open('alp02.npy', 'wb') as data_ex:
#             np.save(data_ex, alp02)

""""""
# ubar=reconstruct(alp1[:,0,],phisets)
# vbar=reconstruct(alp1[:,1,],phisets)
# wbar=reconstruct(alp1[:,2,],phisets)

# uex=reconstruct(all0[:,0,],phisets)
# vex=reconstruct(all0[:,1,],phisets)
# wex=reconstruct(all0[:,2,],phisets)
""""""
print(ubar.shape)

print(uex.shape)

ubar=ubar.detach().cpu().numpy()
vbar=vbar.detach().cpu().numpy()
wbar=wbar.detach().cpu().numpy()

pbarx=pbarx.detach().cpu().numpy()
pbary=pbary.detach().cpu().numpy()
pbarz=pbarz.detach().cpu().numpy()

uex=uex.detach().cpu().numpy()
vex=vex.detach().cpu().numpy()
wex=wex.detach().cpu().numpy()

pexx=pexx.detach().cpu().numpy()
pexy=pexy.detach().cpu().numpy()
pexz=pexz.detach().cpu().numpy()

# with open('ubar.npy', 'wb') as data_ex:
#             np.save(data_ex, ubar[:10,0])

# with open('uex.npy', 'wb') as data_ex:
#             np.save(data_ex, uex[:10,0])

# input('dsffd')

lep0=lep.detach().cpu().numpy()
lep=(lep0*lep0.T).reshape(1,1,SHAPE,SHAPE,1)*(lep0.reshape(1,1,1,1,SHAPE))


def intt(f,le):
    jj=SHAPE
    f1=((f/le)**2).reshape(BATCH_SIZE,-1)
    iit=(2/((jj-1)*jj))**3*np.sum(f1,-1)
    return iit



# ul21=np.sum((ubar-uex).reshape(BATCH_SIZE,-1)**2,-1)
# vl21=np.sum((vbar-vex).reshape(BATCH_SIZE,-1)**2,-1)
# wl21=np.sum((wbar-wex).reshape(BATCH_SIZE,-1)**2,-1)

# ul22=np.sum(uex.reshape(BATCH_SIZE,-1)**2,-1)
# vl22=np.sum(vex.reshape(BATCH_SIZE,-1)**2,-1)
# wl22=np.sum(wex.reshape(BATCH_SIZE,-1)**2,-1)

ul21=intt(ubar-uex,lep)
vl21=intt(vbar-vex,lep)
wl21=intt(wbar-wex,lep)

ul22=intt(uex,lep)
vl22=intt(vex,lep)
wl22=intt(wex,lep)

pxl21=intt(pbarx-pexx,lep)
pyl21=intt(pbary-pexy,lep)
pzl21=intt(pbarz-pexz,lep)

pxl22=intt(pexx,lep)
pyl22=intt(pexy,lep)
pzl22=intt(pexz,lep)


      


ul2=(ul21/ul22)**.5
vl2=(vl21/vl22)**.5
wl2=(wl21/wl22)**.5
pl2=((pxl21+pyl21+pzl21)/(pxl22+pyl22+pzl22))**.5



# kkk=abs(ul2-np.mean(ul2))
# kkk1=np.where(np.min(kkk)==kkk)[0][0]
# kkk1=np.where(np.min(kkk)==kkk)
# kkk1=int(79)
kkk1=range(3)
print(kkk1,ul2[kkk1],vl2[kkk1],wl2[kkk1],pl2[kkk1])
jj=BATCH_SIZE
print('ML2u',np.max(ul2[:jj]),np.max(vl2[:jj]),np.max(wl2[:jj]),np.max(pl2))
print('AL2u',np.mean(ul2[:jj]),np.mean(vl2[:jj]),np.mean(wl2[:jj]),np.mean(pl2))
print('stdL2u',np.std(ul2),np.std(vl2),np.std(wl2),np.std(pl2))
# print('rel0',ul2[0],vl2[0],wl2[0])
# print('ML2u',ul2[0],ul2[1],ul2[2])

print('max u',np.max(abs(uex)),np.max(abs(vex)),np.max(abs(wex)))




# fdata=a_pred1.detach().cpu().numpy()

# with open(f'training/NS3d1.0/ubar/ubar{ORDER}.npy', 'wb') as data_ex:
#             np.save(data_ex, ubar[kkk1])
# with open(f'training/NS3d1.0/ubar/vbar{ORDER}.npy', 'wb') as data_ex:
#             np.save(data_ex, vbar[kkk1])
# with open(f'training/NS3d1.0/ubar/wbar{ORDER}.npy', 'wb') as data_ex:
#             np.save(data_ex, wbar[kkk1])

# with open(f'training/NS3d1.0/uex/uex{ORDER}.npy', 'wb') as data_ex:
#             np.save(data_ex, uex[kkk1])
# with open(f'training/NS3d1.0/uex/vex{ORDER}.npy', 'wb') as data_ex:
#             np.save(data_ex, vex[kkk1])
# with open(f'training/NS3d1.0/uex/wex{ORDER}.npy', 'wb') as data_ex:
#             np.save(data_ex, wex[kkk1])

import pandas as pd
# data=pd.read_csv('call_alp.csv')
newcall = {'order':[],'uL2':[],'vL2':[],'wL2':[],'pL2':[]}
# newcall=pd.read_csv('call_alp.csv')  


newcall['order'].append(ORDER)
newcall['uL2'].append(np.mean(ul2))
newcall['vL2'].append(np.mean(vl2))
newcall['wL2'].append(np.mean(wl2))
newcall['pL2'].append(np.mean(pl2))

df = pd.DataFrame(newcall)

if ORDER==1:
    df.to_csv( '3dforce_l2.csv', index=False)
elif ORDER>1.1:
    df.to_csv( '3dforce_l2.csv', mode='a', index=False, header=False)