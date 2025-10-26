import numpy as np
from numpy.linalg import inv
import pickle
import os
from sem import sem as sem
from tqdm import tqdm
import argparse
from mpl_toolkits import mplot3d 
from pprint import pprint
from funsjax import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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

Y,X=np.meshgrid(x,x)



D = sem.legslbdiff(N+1, x)

Md,sd_diag,Ed,eid=basic_mat(b,N,'dirichlet')

Mn,sn_diag,En,ein=basic_mat(bn,N,'neumann')


Mm=np.zeros((N-1,N-1))
Mmx=np.zeros((N-1,N-1))
phisets=np.zeros((N+1,N-1))
phinsets=np.zeros((N+1,N-1))


phixsets=np.zeros((N+1,N-1))
lep=lepolys[N]

for ii in range(N-1):
    phi=(lepolys[ii]- lepolys[ii+2])/(sd_diag[ii])**.5
    psi00=(lepolys[ii]+ bn[ii]*lepolys[ii+2])/(sn_diag[ii])**.5
    phix=(lepolysx[ii].T-lepolysx[ii+2].T)/(sd_diag[ii])**.5
    phisets[:,ii]=phi[:,0]
    phinsets[:,ii]=psi00[:,0]
    phixsets[:,ii]=phix[:,0]
    for jj in range(N-1):
        psi=(lepolys[jj]+ bn[jj]*lepolys[jj+2])/(sn_diag[jj])**.5
        Mm[jj,ii]=np.sum(psi*phi/(lepolys[N])**2)*(2/(N*(N+1)))
        Mmx[jj,ii]=np.sum(psi*phix/(lepolys[N])**2)*(2/(N*(N+1)))

Mm[abs(Mm)<10**-8]=0
Mmx[abs(Mmx)<10**-8]=0
SS=np.diag(sd_diag[:N-1])

Mxnd=np.zeros((N-1,N-1))
Mdxd=np.zeros((N-1,N-1))
Mxdd=np.zeros((N-1,N-1))
Mnd=np.zeros((N-1,N-1))
Mxnxd=np.zeros((N-1,N-1))

mnd1=np.zeros((N-1,))
mnd2=np.zeros((N-1,))
mnd3=np.zeros((N-1,))
mxnxd=np.zeros((N-1,))
mxdd=np.zeros((N-1,))

for ii in range(N-1):
    mnd2[ii]=2*(1/(2*ii+1)+b[ii]*bn[ii]/(2*ii+5))/(sd_diag[ii]*sn_diag[ii])**.5
    mnd1[ii]=(b[ii])*2/(2*ii+5)/(sd_diag[ii]*sn_diag[ii+2])**.5
    mnd3[ii]=(bn[ii])*2/(2*ii+5)/(sd_diag[2+ii]*sn_diag[ii])**.5
    if ii< N-2:
        diri = (lepolys[ii]-lepolys[ii+2])/(sd_diag[ii])**.5
        dirix = (lepolysx[ii+1].T-lepolysx[ii+3].T)/(sd_diag[ii+1])**.5
        qwe=diri*dirix/lepolys[N]**2
        mxdd[ii]=np.sum(qwe)*(2/(N*(N+1)))
    # mxnxd[ii]=-bn[ii]*(4*ii+6)/(sd_diag[ii]*sn_diag[ii])**.5
# mxnxd[0]=1/sd_diag[0]**.5

Mnd=  mnd2*np.eye(N-1)+np.diag(mnd1[0:N-3],2)+np.diag(mnd3[0:N-3],-2)
Mdxd=np.diag(mxdd[:N-2],1)-np.diag(mxdd[:N-2],-1)
Mxdd=Mdxd.T

for ii in range(N-1):
    # dirix = (lepolysx[ii].T-lepolysx[ii+2].T)/(sd_diag[ii])**.5
    
    neunx = (lepolysx[ii].T+ bn[ii]*lepolysx[ii+2].T)/(sn_diag[ii])**.5
    # neun = (lepolys[ii]+ bn[ii]*lepolys[ii+2])/(sn_diag[ii])**.5
    for jj in range(N-1):
        diri1=(lepolys[jj]-lepolys[jj+2])/(sd_diag[jj])**.5
        dirix1 = (lepolysx[jj].T  -lepolysx[jj+2].T)/(sd_diag[jj])**.5
        
        # psi_l_M = (lepolysx[jj].T  -lepolysx[jj+2].T)/(sd_diag[jj])**.5
        phi1=neunx*diri1/lepolys[N]**2
       
        
        # Mnd[jj,ii]=np.sum(phi4)*(2/(N*(N+1)))
        Mxnd[jj,ii]=np.sum(phi1)*(2/(N*(N+1)))
       

Mxnd[abs(Mxnd)<10**-8]=0  #diri*neumann

# Mxnxd[abs(Mxnxd)<10**-8]=0  #diri_x*neuman_x
# Mxdd[abs(Mxdd)<10**-8]=0  #diri_x*neuman_x
# Mnd[abs(Mnd)<10**-8]=0     #diri_x*diri
# Mdnx[abs(Mdnx)<10**-8]=0   #diri*neuman_x


t=0

# u0,v0,w0,_,_,_=ini(-dt,X,Y,Z)
# u1,v1,w1,px,py,pz=ini(0,X,Y,Z)

iMd=Ed@np.diag(1/eid)@Ed.T
iMn=En@np.diag(1/ein)@En.T

# tdata=np.reshape(dt*np.array(np.arange(0,11)),(11,1,1,1))





ode_data=np.zeros((N-1,N-1,N-1))
# ode_data2=np.zeros((N-1,N-1,N-1))

# iode_data=np.zeros((N-1,N-1,N-1,N-1))

for jj in range(N-1):
        for ii in range(N-1):
            ode_data[jj,]=(1.5*eid[jj]/dt+eps)*Md+eps*eid[jj]*np.eye(N-1)
            # ode_data2[jj,ii,]=eid[ii]*Md+eid[jj]*Md+ein[jj]*eid[ii]*np.eye(N-1)
            # iode_data[jj,ii,]=np.diag(1/np.diag(ode1[jj,ii,])**.5)
            # ode_data[jj,ii,]=(iode_data[jj,ii,]@ode1[jj,ii,])@iode_data[jj,ii,]

oden_data=np.zeros((N-1,N-1,N-1))
# ioden_data=np.zeros((N-1,N-1,N-1,N-1))
for jj in range(N-1):
        # ode1=(eie[jj]*3*.5/dt+1)*eie[0]*M+eie[jj]*M+eie[jj]*eie[0]*np.eye(N-1)
        for ii in range(N-1):
            oden_data[jj,]=oden_data[jj,]=Mn+ein[jj]*np.eye(N-1)
            # ioden_data[jj,ii,]=np.diag(1/np.diag(ode2[jj,ii,])**.5)
            # oden_data[jj,ii,]=(ioden_data[jj,ii,]@ode2[jj,ii,])@ioden_data[jj,ii,]

# phiset0=phiset(sd_diag,b,'dirichlet')
# phiset1=phiset(sd_diag,b,'dirichlet1')
# phiset2=phiset(sd_diag,b,'dirichlet2')





T=Ind*dt






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

filename=f'./data/{equation}{eps}/force'

qq1=qq[:,0]+1j*qq[:,1]


# qq1[:,:,0,0,0]=qq1[:,:,0,0,0]+(N+1)**3

X1=np.pi*(X+1)
Y1=np.pi*(Y+1)




# fx=np.sin(tt)*1.5*((1+np.cos(0*X1+1*Y1)-np.sin(1*X1+0*Y1)-np.sin(1*X1+1*Y1)).reshape((1,24,24)))
# fy=np.sin(tt)*1.5*((1+np.sin(0*X1+1*Y1)-np.cos(1*X1+0*Y1)-np.cos(1*X1+1*Y1)).reshape((1,24,24)))

# noi1=1


for jnd in range(1,Jnd+1):    
    
    
    fx=exf2(tt,qq1[jnd-1,0],X1,Y1)
    
    
    
    
    fy=exf2(tt,qq1[jnd-1,1],X1,Y1)

   
    fxdata[jnd-1,]=fx
    fydata[jnd-1,]=fy
  

aa=fxdata[:,-1]**2+fydata[:,-1]**2
aa=aa.reshape(Jnd,-1)
aa1=np.max(aa,1)


qq=np.where(aa1==np.max(aa1))





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
# ind=1
    
    exfx=np.zeros((Jnd,N-1,N-1))
    exfy=np.zeros((Jnd,N-1,N-1))
    
    # eee0[jj,]=np.reshape(Ed[:,jj],(N-1,1,1))*cFx
    exfx=Ed.T@cFx
    exfy=Ed.T@cFy
    
    alx1=np.linalg.solve(ode_data, np.transpose(exfx,(1,2,0)))
    aly1=np.linalg.solve(ode_data, np.transpose(exfy,(1,2,0)))
    
   
    
    
    alx=Ed@np.transpose(alx1,(2,0,1))
    aly=Ed@np.transpose(aly1,(2,0,1))
    # alz=Ed@alz1
    
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
    
    # phiall[0,:,:]=0
    phiall[:,:,0]=0
    # phiall[:,:,0]=0
    px1=(Mxnd@phiall)@Mnd.T
    
 
    uu1=alx@Md.T
    
       
   
    vv1=(Mdxd@aly)@Mxdd.T
       
   
    
    ""
    # py1=np.zeros((N-1,N-1,N-1))
    phiall=phial.copy()
    
    phiall[:,0,:]=0
   
    py1=(Mnd@phiall)@Mxnd.T
    
   
    uuu1=(Mxdd@alx)@Mdxd.T
   
    vvv1=Md@aly
   
    
    
    
    cFx01=px1+cFx01+eps*(uu1+vv1)
    cFy01=py1+cFy01+eps*(uuu1+vvv1)
   
    

    
    
    al_unext3=(Md@alx)@Md
   
    phiall=phial.copy()
   
    # phiall[0,:,:]=0
    phiall[:,:,0]=0
    
    phixnext3=(Mxnd@phiall)@Mnd.T
   
    
    al_vnext3=(Md@aly)@Md
    
    # phiznext3=np.zeros((N-1,N-1,N-1))
    phiall=phial.copy()
    phiall[:,0,:]=0
    
    phiynext3=(Mnd@phiall)@Mxnd.T
    """initial data pp"""
   
    # p_data[:,ind-1]=phiset(phixnext300,phinsets)
    """"""
   

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
# input('time')
#Ind=20
"""initial data saving """
# from scipy.io import savemat
# mdic = {"fxdata": fxdata[22,-1,:,:,:], "fydata": fydata[22,-1,:,:,:],"u":u_data[22,::10,:,:,:],"v":v_data[22,::10,:,:,:],"p":p_data[0,::10,:,:,:]}

# savemat("forcing data.mat", mdic)
# input('sdfsdf')
""""""
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
    
    # data.append([data_alp[jnd,:,Ind:], fdata[jnd,:,0:1],cfdata[jnd,:,Ind:],data_uu[jnd,:,Ind:]])


data = np.array(data, dtype=object)

print('max u',np.max(abs(u1)))
print('max v',np.max(abs(v1)))




with open(filename+f'/{Jnd}N{N}sigma{sigma}.pkl', 'wb') as f:
        
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


