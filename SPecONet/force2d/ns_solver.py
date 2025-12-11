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
parser.add_argument("--Nsamples", type=int)
parser.add_argument("--Ntimes", type=int)
parser.add_argument("--Equation", type=str, default='NS2d', choices=['NS2d'])
args = parser.parse_args()
case=args.case
equation=args.Equation

Jnd=args.Nsamples
Ind=args.Ntimes

N=int(24-1)
dt=0.01
eps=0.1

T=Ind*dt



ode_data, oden_data, Ed,En,X,Y,Mnd,Mxdd,Mxnd,Mdxd,Md,iMd,Mm,Mmx,phisets,lep=matA(N,dt,eps)


uun1=0
vvn1=0


u1=0
v1=0


cpp0=np.zeros((1,2))

cuux0,cvvy0=0,0


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
cu_data=np.zeros((Jnd,Ind,N-1,N-1))
cv_data=np.zeros((Jnd,Ind,N-1,N-1))

cp_data=np.zeros((Jnd,Ind,N-1,N-1))


u_data=np.zeros((Jnd,Ind,N+1,N+1))
v_data=np.zeros((Jnd,Ind,N+1,N+1))


p_data=np.zeros((1,Ind,N+1,N+1,N+1))

cfdata=np.zeros((Jnd,2,(Ind),N-1,N-1))
cfdata0=np.zeros((Jnd,2,N-1,N-1))

fxdata=np.zeros((Jnd,Ind,N+1,N+1))

fydata=np.zeros((Jnd,Ind,N+1,N+1))



tt=dt*(np.arange(1,Ind+1).reshape(Ind,1,1))

num, sigma =3,5


mdata=np.zeros((Jnd,3,1+num))




qq2=np.random.normal(0,sigma,2000*2*2*num**2)  # if Jnd=700, 800 qq2.
qq=qq2.reshape(2000,2,2,num,num)  # 10--> ampli=3, 3--> ampli=1

filename=f'./data/{equation}{eps}/train'

qq1=qq[:,0]+1j*qq[:,1]

X1=np.pi*(X+1)
Y1=np.pi*(Y+1)


for jnd in range(1,Jnd+1):       
    
    fx=exf2(tt,qq1[jnd-1,0],X1,Y1)
    fy=exf2(tt,qq1[jnd-1,1],X1,Y1)
    fxdata[jnd-1,]=fx
    fydata[jnd-1,]=fy
  

aa=fxdata[:,-1]**2+fydata[:,-1]**2
aa=aa.reshape(Jnd,-1)
aa1=np.max(aa,1)



x_rhs=fxdata[:,0,]+0.5*(4*u1-uun1)/dt
y_rhs=fydata[:,0,]+0.5*(4*v1-vvn1)/dt



cFx0=conv(x_rhs,phisets,lep)


cFy0=conv(y_rhs,phisets,lep)


cFx=cFx0-cpp0[:,0]
cFy=cFy0-cpp0[:,1]


cfdata0[:,0]=cFx
cfdata0[:,1]=cFy


cfdata[:,0,0]=cFx0
cfdata[:,1,0]=cFy0


t00 = time.time()
for ind in range(1,Ind+1):

    
    exfx=np.zeros((Jnd,N-1,N-1))
    exfy=np.zeros((Jnd,N-1,N-1))
    exfx=Ed.T@cFx
    exfy=Ed.T@cFy
    
    alx1=np.linalg.solve(ode_data, np.transpose(exfx,(1,2,0)))
    aly1=np.linalg.solve(ode_data, np.transpose(exfy,(1,2,0)))
    
   
    
    
    alx=Ed@np.transpose(alx1,(2,0,1))
    aly=Ed@np.transpose(aly1,(2,0,1))
   
    
    cFnx1=(Mmx@alx)@Mm.T  #second
   
    cFny1=(Mm@aly)@Mmx.T  #second
    
    
       
    
    
    cFn=1.5*((cFnx1)+(cFny1))/dt
    
    
    Pexfx=En.T@cFn
   
    phial1=np.linalg.solve(oden_data, np.transpose(Pexfx,(1,2,0)))
   
    phial=-En@np.transpose(phial1,(2,0,1))
    
    if ind<Ind:
        
        cFx0=conv(fxdata[:,ind,],phisets,lep)
        cFy0=conv(fydata[:,ind,],phisets,lep)
        
        cfdata[:,0,ind]=cFx0
        cfdata[:,1,ind]=cFy0
       
        
    phiall=phial.copy()
    
    
    ""
    cu_data[:,ind-1,]=alx    
    cv_data[:,ind-1,]=aly
    
    cp_data[:,ind-1,]=phial
    
   
    phiall[:,:,0]=0  
    px1=(Mxnd@phiall)@Mnd.T
    
 
    uu1=alx@Md.T
    
       
   
    vv1=(Mdxd@aly)@Mxdd.T
    phiall=phial.copy()
    
    phiall[:,0,:]=0
   
    py1=(Mnd@phiall)@Mxnd.T
    
   
    uuu1=(Mxdd@alx)@Mdxd.T
   
    vvv1=Md@aly
   
    
    
    
    cFx01=px1+cFx01+eps*(uu1+vv1)
    cFy01=py1+cFy01+eps*(uuu1+vvv1)
   
    

    
    
    al_unext3=(Md@alx)@Md
   
    phiall=phial.copy()  
  
    phiall[:,:,0]=0
    
    phixnext3=(Mxnd@phiall)@Mnd.T
   
    
    al_vnext3=(Md@aly)@Md
    phiall=phial.copy()
    phiall[:,0,:]=0
    
    phiynext3=(Mnd@phiall)@Mxnd.T
   
   

    al_unext=al_unext3-2*dt*phixnext3/3
    al_vnext=al_vnext3-2*dt*phiynext3/3
    
  
    auu=(iMd@al_unext)@iMd
    avv=(iMd@al_vnext)@iMd
   
   
    u1=phiset(auu,phisets)
    v1=phiset(avv,phisets)

   

    
    
    u_data[:,ind-1,]=u1
    v_data[:,ind-1,]=v1
 


    cuux1,cvvy1=nonlinear(u1,v1,phisets,lep,iMd,Mdxd)
    
    cFx=-cFx01+0.5*(4*al_unext-al_upre)/dt+cFx0-(2*(cuux1)-cuux0)
    
    cFy=-cFy01+0.5*(4*al_vnext-al_vpre)/dt+cFy0-(2*(cvvy1)-cvvy0)
   
  
    cuux0=cuux1
    cvvy0=cvvy1
    
    al_upre=al_unext
    al_vpre=al_vnext
    
  
print('compuational time',time.time() - t00)
print(u_data.shape)

data_uu=np.zeros((Jnd,2,Ind,N+1,N+1))
data_alp=np.zeros((Jnd,3,Ind,N-1,N-1))
fdata=np.zeros((Jnd,2,(Ind),N+1,N+1))

    #input('ddd')
# print('alphas',cu_data[0,0,1,])

data_alp[:,0]=cu_data[:,0:0+Ind,]
data_alp[:,1]=cv_data[:,0:0+Ind,]

data_alp[:,2]=cp_data[:,0:0+Ind,]

data_uu[:,0]=u_data[:,0:0+Ind,]
data_uu[:,1]=v_data[:,0:0+Ind,]


fdata[:,0,]=fxdata[:,0:0+Ind,]
fdata[:,1,]=fydata[:,0:0+Ind,]



Ind=1-1

for jnd in range(Jnd):
    data.append([data_alp[jnd,], fdata[jnd,],data_uu[jnd,],cfdata[jnd,],cfdata0[jnd]])
  

data = np.array(data, dtype=object)

print('max u',np.max(abs(u1)))
print('max v',np.max(abs(v1)))




with open(filename+f'/{Jnd}N{N}sigma{sigma}.pkl', 'wb') as f:
        
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


