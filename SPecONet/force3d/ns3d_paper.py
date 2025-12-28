"3D NS with dirichlet condition for my paper"

import numpy as np
from numpy.linalg import inv
import pickle
import os
from sem import sem as sem
from tqdm import tqdm
import argparse
from funsjax import *
import time

parser = argparse.ArgumentParser("SEM")

parser.add_argument("--case", type=str, default='train', choices=['train', 'test']) 
parser.add_argument("--Nsamples", type=str)
parser.add_argument("--Ntimes", type=int)
parser.add_argument("--Equation", type=str, default='NS3d', choices=['NS3d'])
parser.add_argument("--forcing", type=str, default='sigma5')
parser.add_argument("--epsilon", type=float)

args = parser.parse_args()
case=args.case
equation=args.Equation

Nsamples=args.Nsamples

Jnd=int(Nsamples.split('N')[0])
Ind=args.Ntimes
forcing=args.forcing


N=int(Nsamples.split('N')[1])
dt=0.01
eps=args.epsilon



ode_data, oden_data, Ed,En,X,Y,Z,Mnd,Mxdd,Mxnd,Mdxd,Md,iMd,Mm,Mmx,phisets,_,_,_,lep=matA(N,dt,eps)



t=0






T=Ind*dt



uun1=0
vvn1=0
wwn1=0


u1=0
v1=0
w1=0

cpp0=np.zeros((1,3))

cuux0,cvvy0,cwwz0=0,0,0


data = []

if case=='train':
    np.random.seed(0)
  
elif case=='test':
    np.random.seed(1) 


    
al_upre=0
al_vpre=0
al_wpre=0
cFx01=0
cFy01=0
cFz01=0
cu_data=np.zeros((Jnd,Ind,N-1,N-1,N-1))
cv_data=np.zeros((Jnd,Ind,N-1,N-1,N-1))
cw_data=np.zeros((Jnd,Ind,N-1,N-1,N-1))
cp_data=np.zeros((Jnd,Ind,N-1,N-1,N-1))


u_data=np.zeros((Jnd,Ind,N+1,N+1,N+1))
v_data=np.zeros((Jnd,Ind,N+1,N+1,N+1))
w_data=np.zeros((Jnd,Ind,N+1,N+1,N+1))
p_data=np.zeros((1,Ind,N+1,N+1,N+1))

cfdata=np.zeros((Jnd,3,(Ind),N-1,N-1,N-1))
cfdata0=np.zeros((Jnd,3,N-1,N-1,N-1))

fxdata=np.zeros((Jnd,Ind,N+1,N+1,N+1))

fydata=np.zeros((Jnd,Ind,N+1,N+1,N+1))

fzdata=np.zeros((Jnd,Ind,N+1,N+1,N+1))

tt=dt*(np.arange(1,Ind+1).reshape(Ind,1,1,1))

num, sigma =4,int(forcing.split('a')[1])



qq2=np.random.normal(0,sigma,1500*3*2*num**3)  # if Jnd=700, 800 qq2.
qq=10*(N+1)**2*qq2.reshape(1500,2,3,num,num,num)

qq1=qq[:,0]+1j*qq[:,1]

filename=f'./data/{equation}{eps}/{forcing}'



X1=np.pi*(X+1)
Y1=np.pi*(Y+1)
Z1=np.pi*(Z+1)


for jnd in range(Jnd):    
    
    
    fx=exf2(tt,qq1[jnd,0],X1,Y1,Z1,dt)   
    fy=exf2(tt,qq1[jnd,1],X1,Y1,Z1,dt)    
    fz=exf2(tt,qq1[jnd,2],X1,Y1,Z1,dt)
    
    fxdata[jnd,]=fx
    fydata[jnd,]=fy
    fzdata[jnd,]=fz
    

aa=fxdata[:,-1]**2+fydata[:,-1]**2+fzdata[:,-1]**2
aa=aa.reshape(100,-1)
aa1=np.max(aa,1)

qq=np.where(aa1==np.max(aa1))




x_rhs=fxdata[:,0,]+0.5*(4*u1-uun1)/dt
y_rhs=fydata[:,0,]+0.5*(4*v1-vvn1)/dt
z_rhs=fzdata[:,0,]+0.5*(4*w1-wwn1)/dt


cFx0=conv(x_rhs,phisets,lep)
cFy0=conv(y_rhs,phisets,lep)
cFz0=conv(z_rhs,phisets,lep)


cFx=cFx0-cpp0[:,0]
cFy=cFy0-cpp0[:,1]
cFz=cFz0-cpp0[:,2]

cfdata0[:,0]=cFx
cfdata0[:,1]=cFy
cfdata0[:,2]=cFz

cfdata[:,0,0]=cFx0
cfdata[:,1,0]=cFy0
cfdata[:,2,0]=cFz0

t00 = time.time()
for ind in range(1,Ind+1):
# ind=1
    
    exfx=np.zeros((Jnd,N-1,N-1,N-1))
    exfy=np.zeros((Jnd,N-1,N-1,N-1))
    exfz=np.zeros((Jnd,N-1,N-1,N-1))
   
    
    for jj in range(N-1):
        exfx[:,jj,]=Ed.T@np.sum(np.reshape(Ed[:,jj],(1,N-1,1,1))*cFx,axis=1)
        exfy[:,jj,]=Ed.T@np.sum(np.reshape(Ed[:,jj],(1,N-1,1,1))*cFy,axis=1)
        exfz[:,jj,]=Ed.T@np.sum(np.reshape(Ed[:,jj],(1,N-1,1,1))*cFz,axis=1)
   
    alx1=np.linalg.solve(ode_data, np.transpose(exfx,(1,2,3,0)))
    aly1=np.linalg.solve(ode_data, np.transpose(exfy,(1,2,3,0)))
    alz1=np.linalg.solve(ode_data, np.transpose(exfz,(1,2,3,0)))
    
   
    
    alx2=Ed@np.transpose(alx1,(3,0,1,2))
    aly2=Ed@np.transpose(aly1,(3,0,1,2))
    alz2=Ed@np.transpose(alz1,(3,0,1,2))
    
    
    alx=np.swapaxes(Ed@np.swapaxes(alx2,1,2),1,2)
    aly=np.swapaxes(Ed@np.swapaxes(aly2,1,2),1,2)
    alz=np.swapaxes(Ed@np.swapaxes(alz2,1,2),1,2)
   
    
       
    cFnx3=Mm@alx  #second
    
    cFnx2=np.transpose(Mm@np.transpose(cFnx3,(0,1,3,2)),(0,1,3,2)) #third
    
    cFnx1=np.transpose(Mmx@np.transpose(cFnx2,(0,2,1,3)),(0,2,1,3))
    
    
    
    cFny3=np.transpose(Mm@np.transpose(aly,(0,1,3,2)),(0,1,3,2))  #third
        
    cFny2=Mmx@cFny3 #second
    
    cFny1=np.transpose(Mm@np.transpose(cFny2,(0,2,1,3)),(0,2,1,3)) #first
   
    cFnz3=Mm@alz  #third
    
    cFnz2=np.transpose(Mmx@np.transpose(cFnz3,(0,1,3,2)),(0,1,3,2)) #second
    
    cFnz1=np.transpose(Mm@np.transpose(cFnz2,(0,2,1,3)),(0,2,1,3)) #first
    
    cFn=1.5*((cFnx1)+(cFny1)+(cFnz1))/dt
    
    Pf=np.zeros(cFn.shape)
    Pexfx=np.zeros((Jnd,N-1,N-1,N-1))
    
    for jj in range(N-1):
       
        Pf[:,jj,]=np.sum(np.reshape(En[:,jj],(1,N-1,1,1))*cFn,axis=1)
        
        Pexfx[:,jj,]=En.T@Pf[:,jj,]
    
    
    phial1=np.linalg.solve(oden_data, np.transpose(Pexfx,(1,2,3,0)))
    
  
    phial2=En@np.transpose(phial1,(3,0,1,2))
    
    phial=-np.swapaxes(En@np.swapaxes(phial2,1,2),1,2)
    
    
    if ind<Ind:
        
        cFx0=conv(fxdata[:,ind,],phisets,lep)
        cFy0=conv(fydata[:,ind,],phisets,lep)
        cFz0=conv(fzdata[:,ind,],phisets,lep)
        cfdata[:,0,ind]=cFx0
        cfdata[:,1,ind]=cFy0
        cfdata[:,2,ind]=cFz0
        
    phiall=phial.copy()
    
    ""
    cu_data[:,ind-1,]=alx    
    cv_data[:,ind-1,]=aly
    cw_data[:,ind-1,]=alz
    cp_data[:,ind-1,]=phial
       
    phiall[:,0,:,:]=0   
    Px3=np.transpose(Mxnd@np.transpose(phiall,(0,2,1,3)),(0,2,1,3))
    Px2=Mnd@Px3 #second
    px1=np.transpose(Mnd@np.transpose(Px2,(0,1,3,2)),(0,1,3,2)) #first
    
   
    uu3=np.transpose(Md@np.transpose(alx,(0,1,3,2)),(0,1,3,2))
    uu1=Md@uu3
    
       
    uu3=np.transpose(Md@np.transpose(aly,(0,1,3,2)),(0,1,3,2))
    uu2=np.transpose(Mdxd@np.transpose(uu3,(0,2,1,3)),(0,2,1,3))
    vv1=Mxdd@uu2
    
    uu3=np.transpose(Mxdd@np.transpose(alz,(0,1,3,2)),(0,1,3,2))
    uu2=np.transpose(Mdxd@np.transpose(uu3,(0,2,1,3)),(0,2,1,3))
    ww1=Md@uu2
    
    
    ""
   
    phiall=phial.copy()
    
    phiall[:,:,0,:]=0
   
    Py3=np.transpose(Mnd@np.transpose(phiall,(0,2,1,3)),(0,2,1,3))
    Py2=Mxnd@Py3 #second
    py1=np.transpose(Mnd@np.transpose(Py2,(0,1,3,2)),(0,1,3,2))
    
   
    uu3=np.transpose(Md@np.transpose(alx,(0,1,3,2)),(0,1,3,2))
    uu2=Mdxd@uu3
    uuu1=np.transpose(Mxdd@np.transpose(uu2,(0,2,1,3)),(0,2,1,3))
    
    uu3=np.transpose(Md@np.transpose(aly,(0,1,3,2)),(0,1,3,2))
    vvv1=np.transpose(Md@np.transpose(uu3,(0,2,1,3)),(0,2,1,3))
   
    
    uu3=np.transpose(Mxdd@np.transpose(alz,(0,1,3,2)),(0,1,3,2))
    uu2=Mdxd@uu3
    www1=np.transpose(Md@np.transpose(uu2,(0,2,1,3)),(0,2,1,3))
    
    
    phiall=phial.copy()   
    phiall[:,:,:,0]=0
    Pz3=np.transpose(Mnd@np.transpose(phiall,(0,2,1,3)),(0,2,1,3))
    Pz2=Mnd@Pz3 #second
    pz1=np.transpose(Mxnd@np.transpose(Pz2,(0,1,3,2)),(0,1,3,2))
    
    uu3=np.transpose(Mdxd@np.transpose(alx,(0,1,3,2)),(0,1,3,2))
    uu2=Md@uu3
    uuuu1=np.transpose(Mxdd@np.transpose(uu2,(0,2,1,3)),(0,2,1,3))
    
    uu3=np.transpose(Mdxd@np.transpose(aly,(0,1,3,2)),(0,1,3,2))
    uu2=Mxdd@uu3
    vvvv1=np.transpose(Md@np.transpose(uu2,(0,2,1,3)),(0,2,1,3))
   
    
    
    uu2=Md@alz
    wwww1=np.transpose(Md@np.transpose(uu2,(0,2,1,3)),(0,2,1,3))
    
    
    
    cFx01=px1+cFx01+eps*(uu1+vv1+ww1)
    cFy01=py1+cFy01+eps*(uuu1+vvv1+www1)
    cFz01=pz1+cFz01+eps*(uuuu1+vvvv1+wwww1)
    

    
    
   
    al_unext1=Md@alx
    al_unext2=np.transpose(Md@np.transpose(al_unext1,(0,2,1,3)),(0,2,1,3))
    al_unext3=np.transpose(Md@np.transpose(al_unext2,(0,1,3,2)),(0,1,3,2))
    
   
    phiall=phial.copy()
    
   
    phiall[:,0,:,:]=0   
    phixnext1=np.transpose(Mxnd@np.transpose(phiall,(0,2,1,3)),(0,2,1,3))
    phixnext2=Mnd@phixnext1
    phixnext3=np.transpose(Mnd@np.transpose(phixnext2,(0,1,3,2)),(0,1,3,2))
    
    al_vnext1=Md@aly
    al_vnext2=np.transpose(Md@np.transpose(al_vnext1,(0,2,1,3)),(0,2,1,3))
    al_vnext3=np.transpose(Md@np.transpose(al_vnext2,(0,1,3,2)),(0,1,3,2))
    
   
    phiall=phial.copy()
    phiall[:,:,0,:]=0
        
    phiynext1=np.transpose(Mnd@np.transpose(phiall,(0,2,1,3)),(0,2,1,3))
    phiynext2=Mxnd@phiynext1
    phiynext3=np.transpose(Mnd@np.transpose(phiynext2,(0,1,3,2)),(0,1,3,2))
    
    al_wnext1=Md@alz
    al_wnext2=np.transpose(Md@np.transpose(al_wnext1,(0,2,1,3)),(0,2,1,3))
    al_wnext3=np.transpose(Md@np.transpose(al_wnext2,(0,1,3,2)),(0,1,3,2))
    
   
    phiall=phial.copy()
    phiall[:,:,:,0]=0
    phiznext1=np.transpose(Mnd@np.transpose(phiall,(0,2,1,3)),(0,2,1,3))
    phiznext2=Mnd@phiznext1
    phiznext3=np.transpose(Mxnd@np.transpose(phiznext2,(0,1,3,2)),(0,1,3,2))
   
 
    al_unext=al_unext3-2*dt*phixnext3/3
    al_vnext=al_vnext3-2*dt*phiynext3/3
    al_wnext=al_wnext3-2*dt*phiznext3/3

 
    
    alu13=iMd@al_unext
    alu12=np.transpose(iMd@np.transpose(alu13,(0,2,1,3)),(0,2,1,3))
    auu=np.transpose(iMd@np.transpose(alu12,(0,1,3,2)),(0,1,3,2)) #first
    
    alu13=iMd@al_vnext
    alu12=np.transpose(iMd@np.transpose(alu13,(0,2,1,3)),(0,2,1,3))
    avv=np.transpose(iMd@np.transpose(alu12,(0,1,3,2)),(0,1,3,2)) #first
    
    alu13=iMd@al_wnext
    alu12=np.transpose(iMd@np.transpose(alu13,(0,2,1,3)),(0,2,1,3))
    aww=np.transpose(iMd@np.transpose(alu12,(0,1,3,2)),(0,1,3,2)) #first


   


    u1=phiset(auu,phisets)
    v1=phiset(avv,phisets)
    w1=phiset(aww,phisets)
    
      
    u_data[:,ind-1,]=u1
    v_data[:,ind-1,]=v1
    w_data[:,ind-1,]=w1


    cuux1,cvvy1,cwwz1=nonlinear(u1,v1,w1,phisets,lep,iMd,Mdxd)
  
    cFx=-cFx01+0.5*(4*al_unext-al_upre)/dt+cFx0-(2*cuux1-cuux0)
    cFy=-cFy01+0.5*(4*al_vnext-al_vpre)/dt+cFy0-(2*cvvy1-cvvy0)
    cFz=-cFz01+0.5*(4*al_wnext-al_wpre)/dt+cFz0-(2*cwwz1-cwwz0)
    
    cuux0=cuux1
    cvvy0=cvvy1
    cwwz0=cwwz1
    al_upre=al_unext
    al_vpre=al_vnext
    al_wpre=al_wnext  
    
   
print('compuational time',time.time() - t00)
print(u_data.shape)


data_uu=np.zeros((Jnd,3,Ind,N+1,N+1,N+1))
data_alp=np.zeros((Jnd,4,Ind,N-1,N-1,N-1))
fdata=np.zeros((Jnd,3,(Ind),N+1,N+1,N+1))



data_alp[:,0]=cu_data[:,0:0+Ind,]
data_alp[:,1]=cv_data[:,0:0+Ind,]
data_alp[:,2]=cw_data[:,0:0+Ind,]
data_alp[:,3]=cp_data[:,0:0+Ind,]

data_uu[:,0]=u_data[:,0:0+Ind,]
data_uu[:,1]=v_data[:,0:0+Ind,]
data_uu[:,2]=w_data[:,0:0+Ind,]

fdata[:,0,]=fxdata[:,0:0+Ind,]
fdata[:,1,]=fydata[:,0:0+Ind,]
fdata[:,2,]=fzdata[:,0:0+Ind,]



print(np.max(abs(fxdata)))
print(np.max(abs(fydata)))
print(np.max(abs(fzdata)))



for jnd in range(Jnd):
    data.append([data_alp[jnd,], fdata[jnd,],cfdata0[jnd],cfdata[jnd,],data_uu[jnd,:]])
    

data = np.array(data, dtype=object)

print('max',np.max(abs(u1)))
print(np.max(abs(v1)))
print(np.max(abs(w1)))

with open(filename+f'/{Jnd}N{N}sigma{sigma}.pkl', 'wb') as f:
        
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


