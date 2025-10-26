"find ap at first time step by using training2alp"
# python training2pressure00.py --equation ConvDiff2D --model Net3Dpressure --loss MSE --blocks 0 --file 1000N23 --epochs 10000 --ks 9 --filters 10 --nbfuncs 30 --U 9 --pre_epochs 5000 --dt 0.01 --forcing num444am4  --ndt 1 --eps 0.1 --order 15 --kind cosN30 --path 20000_20250731T003516
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
parser.add_argument("--model", type=str, default='Net3D', choices=['ResNet', 'NetA', 'NetB', 'NetC', 'NetD', 'Net2D', 'Net3D', 'Net3Dpressure']) 
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
parser.add_argument("--order", type=int, default=1)
parser.add_argument("--eps", type=float, default=1)
parser.add_argument("--path", type=str)
parser.add_argument("--kind", type=str, default='trainN10')

args = parser.parse_args()
gparams = args.__dict__
#pprint(gparams)

ndt=args.ndt
PATH0=args.path

D_in = 1
kind=args.kind
ORDER=args.order

EQUATION = args.equation
pre_test=args.pre_test

EPSILON = args.eps
# models0 = {'Net3D': Net3D}


models = {
          'ResNet': ResNet,
          'NetA': NetA,
          'NetB': NetB,
          'NetC': NetC,
          'NetD': NetD,
          'Net3D': Net3D,
          'Net2D': Net2D,
          'Net3Dpressure':Net3Dpressure
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

FOLDER0 = f'Net3D_{args.forcing}_epochs{PATH0}'

PATH = os.path.join('training', f"{EQUATION}{EPSILON}", FILE,f"order{ORDER}" ,FOLDER)

PATH_prev=os.path.join('training', f"{EQUATION}{EPSILON}", FILE,f"order{ORDER}" ,FOLDER0)



BATCH_SIZE, Filters, D_out = int(DATASET), FILTERS, SHAPE
# LOSS SCALE FACTORS
A, U, num2, WF = int(gparams['A']), 10**(gparams['U']),gparams['F'], (gparams['WF'])

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

# model0 = MODEL0(1,D_in, 3*Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
# model0.load_state_dict(torch.load(r'training/ConvDiff2D1.0/200N15/Net3D_uniform_epochs10000_20240811T224627fil21/model.pt'), strict=False)

# for name,param in model0.named_parameters():
#     param.requires_grad = False

model= MODEL(10**(A),1,D_in, Filters, D_out - 2, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
# LOAD the trained model
if args.pretrained is not None:
    args.pretrained = 'N' + args.file.split('N')[-1] + '_' + args.equation + '_' + args.forcing
    model.load_state_dict(torch.load(r'training/ConvDiff2D0.1/600N23/order1/Net3Dpressure_num444am2_epochs50000_20250805T200851/model.pt'), strict=False)
    # model.load_state_dict(torch.load(r'training/ConvDiff2D1.0/70N19/order1/Net3Dpressure_num222sigma5_epochs10000_20241203T022537/model.pt'), strict=False)
    model.train()

# Check if CUDA is available and then use it.
device = get_device()
gparams['device'] = device

# SEND TO GPU (or CPU)
# model0.to(device).double()
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
    
    # if args.pretrained is not None:
    #   if name !='fcH.weight':
    #      param.requires_grad = False

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


iMn=En@np.diag(1/ein)@En.T


phisets=np.zeros((N+1,N-1))
phixsets=np.zeros((N+1,N-1))


for ii in range(NN-1):
    phi=(lepolys[ii]- lepolys[ii+2])/(sd_diag[ii])**.5
    phix=(lepolysx[ii].T-lepolysx[ii+2].T)/(sd_diag[ii])**.5
    phisets[:,ii]=phi[:,0]
    phixsets[:,ii]=phix[:,0]
    for jj in range(NN-1):
        psi=(lepolys[jj]+ bn[jj]*lepolys[jj+2])/(sn_diag[jj])**.5
        Mm[jj,ii]=np.sum(psi*phi/(lepolys[NN])**2)*(2/(NN*(NN+1)))
        Mmx[jj,ii]=np.sum(psi*phix/(lepolys[NN])**2)*(2/(NN*(NN+1)))

Mm[abs(Mm)<10**-8]=0
Mmx[abs(Mmx)<10**-8]=0





oden_data=np.zeros((NN-1,NN-1,NN-1))
pre_condn=np.zeros((NN-1,NN-1,NN-1))
ipre_condn=np.zeros((NN-1,NN-1,NN-1))
for jj in range(NN-1):
        # ode1=(eie[jj]*3*.5/dt+1)*eie[0]*M+eie[jj]*M+eie[jj]*eie[0]*np.eye(N-1)
       
        ode1=Mn+ein[jj]*np.eye(NN-1)
        pre_condn[jj,]=np.diag(1/np.diag(ode1)**.5)
        ipre_condn[jj,]=np.diag(np.diag(ode1)**.5)
        oden_data[jj,]=(pre_condn[jj,]@ode1)@pre_condn[jj,]




t=0




al_upre=torch.zeros((BATCH_SIZE,SHAPE-2,SHAPE-2)).to(device).double()
al_vpre=torch.zeros((BATCH_SIZE,SHAPE-2,SHAPE-2)).to(device).double()

En=torch.from_numpy(En).to(device).double()
Mm=torch.from_numpy(Mm).to(device).double()
Mmx=torch.from_numpy(Mmx).to(device).double()

oden_data=torch.from_numpy(oden_data).to(device).double()



pre_condn=torch.from_numpy(pre_condn).to(device).double()
ipre_condn=torch.from_numpy(ipre_condn).to(device).double()



phisets=torch.from_numpy(phisets).to(device).double()
phixsets=torch.from_numpy(phixsets).to(device).double()

def closure(dt,ald,fdata0,alp):
 
    # print('111',torch.cuda.memory_allocated()/1024**3)
    model.train()
    # print('222',torch.cuda.memory_allocated()/1024**3)
    if torch.is_grad_enabled():
        optimizer.zero_grad()
    # print('333',torch.cuda.memory_allocated()/1024**3)
    # f0=torch.reshape(fdata,(1,1,NN-1,NN-1,NN-1) ).to(device).double()
    
    a_pred = model(fdata0)

    "check weak form"
    # a_pred=torch.sum(ipre_condn@((En.T@ald[:,2]).reshape((BATCH_SIZE,1,NN-1,NN-1,1))),4)
   
    loss_u=torch.zeros(1)
    
    phial00,Pexfx = weak_pressure(alp[:,0,:ndt,],alp[:,1,:ndt,], a_pred,Mmx,Mm,dt, oden_data,pre_condn,En )

    # phial00,Pexfx = weak_pressure(ald[:,0],ald[:,1], a_pred,Mmx,Mm,dt, oden_data,pre_condn,En )
    
    phi0=En@torch.sum(pre_condn@(a_pred.reshape((BATCH_SIZE,1,NN-1,NN-1,1))),4)
    #phi0=p_pred
    #loss_u1 = torch.max(abs(a_pred1[:BATCH_SIZE,:ndt,]-ald[:,3,:ndt,]))
    
    loss = 10**7*(torch.sum((phial00-Pexfx)**2))#+torch.sum((abs(a_pred-aex))**2))
    #loss = U*(torch.sum((abs(a_pred-aex))**2))
   
    # loss_u1= torch.max(abs(num2*(a_pred1+malphi0)-ald[:,:ndt,]))
    # loss_u1 = torch.max(abs(phi0-aphiex))
    loss_u1=torch.zeros((1))
    # print('555',torch.cuda.memory_allocated()/1024**3)
    if loss.requires_grad:
        loss.backward()
    # aa=a_pred1[:BATCH_SIZE,:ndt,].detach().cpu().numpy()
    # 
    # print('ff',fff.shape)
   
    # with open('rhd.npy', 'wb') as data_ex:
    #     np.save(data_ex, fff)
    
    return  loss_u,loss_u1, loss, phi0

#

f_pred=0.0
torch.autograd.set_detect_anomaly(True)
################################################
time0 = time.time()
test1=int(1)
loss_a,  loss_f,    loss_validate, avg_l2_u=0,0,0,0

loss_u_test, loss_wf_test=0,0

ini=1-1

print(torch.cuda.memory_allocated()/1024**3)

# for batch_idx, sample_batch in enumerate(trainloader):
#         all0 = sample_batch['data_u'][:BATCH_SIZE,:,ORDER-1:ORDER-0].double().to(device).reshape((BATCH_SIZE,3,ndt,SHAPE-2,SHAPE-2))
        # alp11 = sample_batch['data_u'][:BATCH_SIZE,:3,ORDER-1-ini:ORDER-ini].double().to(device).reshape((BATCH_SIZE,3,ndt,SHAPE-2,SHAPE-2,SHAPE-2))
        # all0 = sample_batch['data_u'][:BATCH_SIZE,3,ORDER-1-ini:ORDER-ini].double().to(device).reshape((BATCH_SIZE,ndt,SHAPE-2,SHAPE-2,SHAPE-2))
        # fdata00 = sample_batch['f'][:BATCH_SIZE,0:3,ORDER-1-ini].double().to(device)
        # cf00 = sample_batch['cf0'].double().to(device)
        # cf1 = sample_batch['cf'].double().to(device)[:,:3*1,]
# alp=model0(fdata)

# del model0
# fdata=(torch.mean(fdata0,dim=1)).reshape((BATCH_SIZE,ndt,SHAPE,SHAPE,SHAPE))

# fdata=torch.zeros((BATCH_SIZE,num2,3,SHAPE,SHAPE,SHAPE)).double().to(device)
alp1=torch.load(PATH_prev+'/data.pt').detach().double().to(device)

# malp1=1*torch.mean(alp11,dim=0)
# alp1=(alp11)/num2
ux=reconstructx(alp1[:,0], phisets,phixsets)
vx=reconstructx(alp1[:,1],phixsets, phisets)

fdata=ux+vx

# malphi=mphi(malp1,Mm,Mmx,En,oden_data,dt)
# malphi=malphi.reshape(1,1,SHAPE-2,SHAPE-2,SHAPE-2)
# malphi=0

# fdata=fdata.reshape((BATCH_SIZE*num2,3,SHAPE,SHAPE,SHAPE))
# fdata[:,0]=reconstruct(alp1[:,0],phisets)
# fdata[:,1]=reconstruct(alp1[:,1],phisets)
# fdata[:,2]=reconstruct(alp1[:,2],phisets)

# 
# alp1=alp11.detach().double().to(device)
# print(all0.shape)
print(fdata.shape)

loss_wf1=0

all0=0


for epoch in tqdm(range(1, EPOCHS+1)):
        
        loss_u,loss_u1,  loss,a_pred = closure(dt,all0,fdata,alp1)
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
        #SAVE test data
        # if epoch % int(100)==0:
        #         loss_u,loss_u1,  loss,u_pred = closure(dt,aa1,bb1,  test_f, test_u,xx)
        #         u_save1=np.reshape(u_pred.detach().cpu().numpy(),(D_out,))
        #         u_test_save.append(u_save1)
        #scheduler.step()
        # if loss_train<10**-3:
        #     break
# u_test_save.append(np.reshape(dd[:,0][700][:,1],(D_out,))) 
# u_test_save.append(np.reshape(xx,(D_out,)))            
# with open(PATH+"/u_test.pkl", "wb") as fp:   #Pickling
#     pickle.dump(u_test_save, fp)

torch.save(model.state_dict(), PATH + '/model.pt')

print(loss_train)
# print(torch.max(abs(a_pred-all0[:,:ndt,])))



if ORDER>1:
    os.replace(PATH_prev+'/cu0.pt',PATH +"/cu0.pt")
    os.replace(PATH_prev+'/cv0.pt',PATH +"/cv0.pt")
   
    os.replace(PATH_prev+'/cFx0.pt',PATH +"/cFx0.pt")
    os.replace(PATH_prev+'/cFy0.pt',PATH +"/cFy0.pt")
   
    
    
    os.replace(PATH_prev+'/cuu0.pt',PATH +"/cuu0.pt")
    os.replace(PATH_prev+'/cvv0.pt',PATH +"/cvv0.pt")
   







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
newcall['order'].append(ORDER)

df = pd.DataFrame(newcall)
PATH_call=os.path.join('training', f"{EQUATION}{EPSILON}", FILE, "order1")
df.to_csv(PATH_call + f'/call{ORDER}_pp.csv')

# EVERYONE APRECIATES A CLEAN WORKSPACE
gc.collect()
torch.cuda.empty_cache()


# if ORDER >1 and ORDER <10:
#     os.kill(os.getpid(), signal.SIGTERM)

#     subprocess.run(f'python training3alp.py --equation ConvDiff2D --model Net3D --loss MSE --blocks {BLOCKS} --file {FILE} --epochs 2000 --ks {KERNEL_SIZE} --filters 21'\
#                     f' --nbfuncs {NBFUNCS} --U 1 --dt {dt} --forcing {args.forcing}  --ndt {ndt} --eps {EPSILON} --path {EPOCHS}_{cur_time} --order {ORDER+1}', shell=True)
    
        # os.system(f'python training3alp.py --equation ConvDiff2D --model Net3D --loss MSE --blocks {BLOCKS} --file {FILE} --epochs 20000 --ks {KERNEL_SIZE} --filters 21'\
    #                f' --nbfuncs {NBFUNCS} --U 1 --dt {dt} --forcing {args.forcing}  --ndt {ndt} --eps {EPSILON} --path {EPOCHS}_{cur_time} --order {ORDER+1}')
