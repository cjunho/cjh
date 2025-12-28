"find au, av, aw at second time step by using training2alp"
# python training3alp00.py --equation NS3d --blocks 0 --file 5N19 --epochs 20 --ks 9 --filters 3 --dt 0.01 --forcing num444sigma5 --ndt 1 --eps 1.0 --order 2 --path 10_20251226T175407 --kind cosN30 --path_alp
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
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--equation", type=str, default='NS3d', choices=['Standard','Standard1', 'Burgers', 'Helmholtz', 'Standard2D', 'NS3d']) #, 'BurgersT' 
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

D_in = 3*ndt
ORDER=args.order
EQUATION = args.equation

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



model= MODEL(ndt,D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
# LOAD the trained model
if args.pretrained is not None:
    ORDER_alp=10*((ORDER-1)//10)+1
    print('ORDER_alp',ORDER_alp)  
    args.pretrained = 'N' + args.file.split('N')[-1] + '_' + f'{EQUATION}' + '_' + args.forcing
    model.load_state_dict(torch.load(f'training/{EQUATION}{EPSILON}/{FILE}/order{ORDER_alp}/Net3D_{args.forcing}_epochs{PATH_alp}/model.pt'), strict=False)
    model.train()

# Check if CUDA is available and then use it.
device = get_device()
gparams['device'] = device

# SEND TO GPU (or CPU)
# model0.to(device).double()
# modelp.to(device).double()
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
# input('gggg')

#KAIMING HE INIT
if args.pretrained is None:
    model.apply(weights_init)

#INIT OPTIMIZER
optimizer = init_optim(model)

BEST_LOSS = float('inf')
losses = {'loss_train':[]}
gparams['path'] = PATH
log_gparams(gparams)

# for parameter in model.parameters():
#     print(parameter)
# print(sum(p.numel() for p in model.parameters()))

dt=0.01
ode_data0,oden_data0, Ed,_,_,_,_,Mnd,Mxdd,Mxnd,Mdxd,Md,iMd,Mm,Mmx,phisets,_,_,sd_diag,lep=matA(NN,dt,EPSILON)




ode_eye=np.zeros((NN-1,NN-1,NN-1,NN-1))
pre_cond=np.zeros((NN-1,NN-1,NN-1,NN-1))
ipre_cond=np.zeros((NN-1,NN-1,NN-1,NN-1))


for jj in range(NN-1):
        for ii in range(NN-1):
            ode1=ode_data0[jj,ii]
            pre_cond[jj,ii,]=np.diag(1/np.diag(ode1)**.5)
            ipre_cond[jj,ii,]=np.diag(np.diag(ode1)**.5)
            ode_eye[jj,ii,]=(pre_cond[jj,ii,]@ode1)@pre_cond[jj,ii,]
            
oden_eye=np.zeros((NN-1,NN-1,NN-1,NN-1))

for jj in range(NN-1):
        # ode1=(eie[jj]*3*.5/dt+1)*eie[0]*M+eie[jj]*M+eie[jj]*eie[0]*np.eye(N-1)
        for ii in range(NN-1):
            ode1=oden_data0[jj,ii]
            pre_condn=np.diag(1/np.diag(ode1)**.5)
            oden_eye[jj,ii,]=(pre_condn@ode1)@pre_condn





Mnd=torch.from_numpy(Mnd).to(device).double()
Mdxd=torch.from_numpy(Mdxd).to(device).double()
Mxdd=torch.from_numpy(Mxdd).to(device).double()
Md=torch.from_numpy(Md).to(device).double()
iMd=torch.from_numpy(iMd).to(device).double()
phisets=torch.from_numpy(phisets).to(device).double()
lep=torch.from_numpy(lep).to(device).double()

Mxnd=torch.from_numpy(Mxnd).to(device).double()
Ed=torch.from_numpy(Ed).to(device).double()
# En=torch.from_numpy(En).to(device).double()
Mm=torch.from_numpy(Mm).to(device).double()
Mmx=torch.from_numpy(Mmx).to(device).double()

oden_eye=torch.from_numpy(oden_eye).to(device).double()

ode_eye=torch.from_numpy(ode_eye).to(device).double()



pre_cond=torch.from_numpy(pre_cond).to(device).double()



sd_diag=torch.from_numpy(sd_diag).to(device).double()

# cu0,cv0,cw0=0,0,0

def closure(fdata1,cf,cu1,cv1,cw1,cFx01,cFy01,cFz01):
 
    # print('111',torch.cuda.memory_allocated()/1024**3)
    model.train()
    # print('222',torch.cuda.memory_allocated()/1024**3)
    if torch.is_grad_enabled():
        optimizer.zero_grad()
    # print('333',torch.cuda.memory_allocated()/1024**3)
    # f0=torch.reshape(fdata,(1,1,NN-1,NN-1,NN-1) ).to(device).double()
    a_pred = model(fdata1)

    
    #a_pred=aex
    loss_u=torch.zeros(1)
    
    
    cFx=cf[:,0:ndt]
    cFy=cf[:,ndt:2*ndt]
    cFz=cf[:,2*ndt:3*ndt]
    
    # al_unext,al_vnext,al_wnext,exfx,exfy,exfz,phial,Pexfx = weak_form0(EPSILON,al_upre,al_vpre,al_wpre,cfx0,cfy0,cfz0,cFx,cFy,cFz,dt, NN,ode_eye,oden_eye,pre_cond,pre_condn, a_pred,Mxnd,Mnd,Md,Mxdd,Mdxd,Ed,En,Mm,Mmx )
    al_unext,al_vnext,al_wnext,exfx,exfy,exfz = weak_form1(EPSILON,cu0,cv0,cw0,cu1,cv1,cw1,cuu0,cvv0,cww0,cuu1,cvv1,cww1,cFx01,cFy01,cFz01,cFx,cFy,cFz,dt, NN,ode_eye,pre_cond, a_pred,Mxnd,Mnd,Md,Mxdd,Mdxd,Ed,Mm,Mmx )
   
    lossUx=torch.sum((al_unext-exfx)**2)
    
    lossUy=torch.sum((al_vnext-exfy)**2)
    
    lossUz=torch.sum((al_wnext-exfz)**2)

    loss=10**8*(lossUx+lossUy+lossUz)
    
    # LOSS with weak phialormulations
   
    a_pred111=torch.sum(pre_cond@(a_pred.reshape((BATCH_SIZE,3,ndt,NN-1,NN-1,NN-1,1))),6)
    a_pred11=Ed@a_pred111
    a_pred1=torch.transpose(Ed@torch.transpose(a_pred11,3,4),3,4)
    
    
    #
    # print('555',torch.cuda.memory_allocated()/1024**3)
    if loss.requires_grad:
        loss.backward()
    # print(f[0,0,:,4])
    # print(a_pred[0,0,:,4])
    # print(u_pred[0,0,:,4])
    # input('dfgsfd')
    # input('dsfgdsfg')
    # print('666',torch.cuda.memory_allocated()/1024**3)
    return  loss, a_pred1

#

f_pred=0.0
torch.autograd.set_detect_anomaly(True)
################################################
time0 = time.time()


print(torch.cuda.memory_allocated()/1024**3)

for batch_idx, sample_batch in enumerate(trainloader):
        # all000 = sample_batch['data_u'][:BATCH_SIZE,:,ORDER-3:ORDER-2].double().to(device).reshape((4*BATCH_SIZE,ndt,SHAPE-2,SHAPE-2,SHAPE-2))
        # all00 = sample_batch['data_u'][:BATCH_SIZE,:,ORDER-2:ORDER-1].double().to(device).reshape((4*BATCH_SIZE,ndt,SHAPE-2,SHAPE-2,SHAPE-2))
        # all0 = sample_batch['data_u'][:BATCH_SIZE,:,ORDER-1:ORDER].double().to(device).reshape((BATCH_SIZE,4,ndt,SHAPE-2,SHAPE-2,SHAPE-2))
        # fdata0 = sample_batch['f'][:BATCH_SIZE,0::3].double().to(device)
        fdata = sample_batch['f'][:BATCH_SIZE,:,ORDER-1].double().to(device)
        # cf00 = sample_batch['cf0'][:200,:,1:2].double().to(device)   
        cf1 = sample_batch['cf'][:BATCH_SIZE,:,ORDER-1].double().to(device)[:,:3*ndt,]
# print(all0.shape)
print(fdata.shape)
# print(cf00.shape)
print(cf1.shape)

# print(all0[1,0,0,0,:10])
# print(fdata[0,1,0,0,:10])
# print(cf1[0,1,0,0,:10])
# input('kkkk')

# alp=model0(fdata0)
# alphi=modelp(fdata0)

# del model0, modelp,fdata0

# r0=0
# for name,param in model0.named_parameters():
#     # print(name,r0,param.nelement())
#     print(name,r0,param.shape)
#     param_size += param.nelement() * param.element_size()
#     r0+=1
   
    

# size_all_mb = (param_size ) / 1024**3
# print(param_size,buffer_size)

# print('model size: {:.3f}GiB'.format(size_all_mb))

# alp111=torch.sum(pre_cond@(alp.reshape((3*200,1,NN-1,NN-1,NN-1,1))),5)
# alp11=Ed@alp111
# alp1=torch.transpose(Ed@torch.transpose(alp11,2,3),2,3)

# alphi111=torch.sum(pre_condn@(alphi.reshape((200,1,NN-1,NN-1,NN-1,1))),5)
# alphi11=En@alphi111
# alphi1=torch.transpose(En@torch.transpose(alphi11,2,3),2,3)

alp1=torch.load(PATH_prev+'/alpha.pt').detach().to(device).double()

alphi1=torch.load(PATH_prev+'/alphi.pt').detach().to(device).double()

# alp1=all00[0:4*BATCH_SIZE:4,0:ndt,]
# alphi1=all00[3:4*BATCH_SIZE:4,0:ndt,]

# cu11,cv11,cw11,cFx011,cFy011,cFz011,_,_,_=weak_combine(EPSILON,alp1[:3*BATCH_SIZE:3,:ndt,],alp1[1:3*BATCH_SIZE:3,:ndt,],alp1[2:3*BATCH_SIZE:3,:ndt,]\
#                                                        ,dt, NN,alphi1,Mxnd,Mnd,Md,Mxdd,Mdxd,Mm,Mmx,iMd,phisets,lep,sd_diag )









if ORDER==2:
    cu0=0
    cv0=0
    cw0=0
    cuu0=0
    cvv0=0
    cww0=0
    
    cFx0=0
    cFy0=0
    cFz0=0
    # cu11,cv11,cw11,cFx011,cFy011,cFz011,_,_,_=weak_combine(EPSILON,alp1[0:3*BATCH_SIZE:3,0:ndt,],2*alp1[1:3*BATCH_SIZE:3,0:ndt,],alp1[2:3*BATCH_SIZE:3,0:ndt,]\
    #                                                     ,cu0,cv0,cw0,dt, NN,alphi1,Mxnd,Mnd,Md,Mxdd,Mdxd,Mm,Mmx,iMd,phisets,lep,sd_diag )
    
    cu11,cv11,cw11,cFx011,cFy011,cFz011,cuu1,cvv1,cww1=weak_combine(EPSILON,alp1[:,0,0:ndt,],alp1[:,1,0:ndt,],alp1[:,2,0:ndt,]\
                                                        ,cFx0,cFy0,cFz0,dt, NN,alphi1[:BATCH_SIZE,0:ndt,],Mxnd,Mnd,Md,Mxdd,Mdxd,Mm,Mmx,iMd,phisets,lep,sd_diag )
    
else:
    cu0=torch.load(PATH_prev+'/cu0.pt').detach().to(device).double()
    cv0=torch.load(PATH_prev+'/cv0.pt').detach().to(device).double()
    cw0=torch.load(PATH_prev+'/cw0.pt').detach().to(device).double()
    # cuu0=0
    # cvv0=0
    # cww0=0
    cuu0=torch.load(PATH_prev+'/cuu0.pt').detach().to(device).double()
    cvv0=torch.load(PATH_prev+'/cvv0.pt').detach().to(device).double()
    cww0=torch.load(PATH_prev+'/cww0.pt').detach().to(device).double()
    
    cFx0=torch.load(PATH_prev+'/cFx0.pt').detach().to(device).double()
    cFy0=torch.load(PATH_prev+'/cFy0.pt').detach().to(device).double()
    cFz0=torch.load(PATH_prev+'/cFz0.pt').detach().to(device).double()
    cu11,cv11,cw11,cFx011,cFy011,cFz011,cuu1,cvv1,cww1=weak_combine(EPSILON,alp1[:,0,0:ndt,],alp1[:,1,0:ndt,],alp1[:,2,0:ndt,]\
                                                        ,cFx0,cFy0,cFz0,dt, NN,alphi1[:BATCH_SIZE,0:ndt,],Mxnd,Mnd,Md,Mxdd,Mdxd,Mm,Mmx,iMd,phisets,lep,sd_diag )
    
# cu0,cv0,cw0,cFx001,cFy001,cFz001,_,_,_=weak_combine(EPSILON,all000[0:4*BATCH_SIZE:4,0:ndt,],all000[1:4*BATCH_SIZE:4,0:ndt,],all000[2:4*BATCH_SIZE:4,0:ndt,]\
#                                                        ,0,0,0,dt, NN,all000[3:4*BATCH_SIZE:4,0:ndt,],Mxnd,Mnd,Md,Mxdd,Mdxd,Mm,Mmx,iMd,phisets,lep,sd_diag )
# del all000
# cu11,cv11,cw11,cFx011,cFy011,cFz011,_,_,_=weak_combine(EPSILON,all00[0:4*BATCH_SIZE:4,0:ndt,],all00[1:4*BATCH_SIZE:4,0:ndt,],all00[2:4*BATCH_SIZE:4,0:ndt,]\
                                                       # ,cFx001,cFy001,cFz001,dt, NN,all00[3:4*BATCH_SIZE:4,0:ndt,],Mxnd,Mnd,Md,Mxdd,Mdxd,Mm,Mmx,iMd,phisets,lep,sd_diag )

torch.save(cu11, PATH + '/cu0.pt')
torch.save(cv11, PATH + '/cv0.pt')
torch.save(cw11, PATH + '/cw0.pt')

torch.save(cFx011, PATH + '/cFx0.pt')
torch.save(cFy011, PATH + '/cFy0.pt')
torch.save(cFz011, PATH + '/cFz0.pt')


torch.save(cuu1, PATH + '/cuu0.pt')
torch.save(cvv1, PATH + '/cvv0.pt')
torch.save(cww1, PATH + '/cww0.pt')
# cw11=cu11.detach().cpu().numpy()



# err_alu=abs(alp1[0:3*BATCH_SIZE:3,:ndt,]-all00[0:4*BATCH_SIZE:4,0:ndt,])+abs(alp1[1:3*BATCH_SIZE:3,:ndt,]-all00[1:4*BATCH_SIZE:4,0:ndt,])+abs(alp1[2:3*BATCH_SIZE:3,:ndt,]-all00[2:4*BATCH_SIZE:4,0:ndt,])

# err_p=abs(alphi1[0:BATCH_SIZE,:ndt,]-all00[3:4*BATCH_SIZE:4,0:ndt,])

# print(torch.max(err_alu))

# print(torch.max(err_p))

del alp1,alphi1

loss_wf1=0

# print(torch.cuda.memory_allocated()/1024**3)

rr=0

for epoch in tqdm(range(1, EPOCHS+1)):
        
        loss,a_pred = closure(fdata,cf1,cu11,cv11,cw11,cFx011,cFy011,cFz011)
        # print(torch.cuda.memory_summary())
        # print(torch.cuda.memory_allocated()/1024**3)
        optimizer.step(loss.item)
        
        
       # optimizer.step()
        # print(torch.cuda.memory_allocated()/1024**3)
        # input('ggg')
        
         
        loss_train = np.round(float(loss.item()), 12)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        #SAVE train data
        if epoch % int(2) == 0:
            
            losses = log_loss(losses, loss_train)
        #SAVE test data
        # if epoch % int(100)==0:
        #         loss_u,loss_u1,  loss,u_pred = closure(dt,aa1,bb1,  test_f, test_u,xx)
        #         u_save1=np.reshape(u_pred.detach().cpu().numpy(),(D_out,))
        #         u_test_save.append(u_save1)
        #scheduler.step()
        # if loss<30:
        #     break
# u_test_save.append(np.reshape(dd[:,0][700][:,1],(D_out,))) 
# u_test_save.append(np.reshape(xx,(D_out,)))            
# with open(PATH+"/u_test.pkl", "wb") as fp:   #Pickling
#     pickle.dump(u_test_save, fp)

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
newcall['order'].append(ORDER)

df = pd.DataFrame(newcall)
PATH_call=os.path.join('training', f"{EQUATION}{EPSILON}", FILE, "order1")
df.to_csv(PATH_call + f'/call{ORDER}_alp.csv')

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()



