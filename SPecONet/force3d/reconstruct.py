#reconstruct.py
import torch
import numpy as np
from sem.sem import legslbndm, lepoly, legslbdiff
import gc
# import pdb

# Check if CUDA is available and then use it.
def get_device():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    return torch.device(dev)

device = get_device()

def gen_lepolys(N, x):
    lepolys = {}
    for i in range(N):
        lepolys[i] = lepoly(i, x)
    return lepolys


def basis(N,x,eps, lepolys, equation):
    
    # NEWNEWNEW
    if equation == 'Standardb':
        phi = torch.empty((N-1,N))
        a, b = np.zeros((N,)), np.ones((N,))
        b *= -1
        for i in range(N-2):
            phi[i,:] = torch.from_numpy(lepolys[i] + a[i]*lepolys[i+1] + b[i]*lepolys[i+2]).reshape(1,N)
        # print(phi.shape)
        phi[N-2,:]=torch.from_numpy(1 - np.exp(-(1+x)/eps)  - (1 - np.exp(-2/eps))*(x+1)*.5).reshape(1,N)
    else: 
        phi = torch.empty((N-2,N))
    if equation in ('Standard','Standard1', 'Standard2D', 'NS3d'):
        a, b = np.zeros((N,)), np.ones((N,))
        b *= -1
    elif equation in ('Burgers', 'BurgersT'):
        a, b = np.zeros((N,)), np.ones((N,))
        b *= -1
    elif equation == 'Helmholtz':
        a, b = np.zeros((N,)), np.ones((N,))
        for k in range(N):
            b[k] = -k*(k+1)/((k+2)*(k+3))
    for i in range(N-2):
        phi[i,:] = torch.from_numpy(lepolys[i] + a[i]*lepolys[i+1] + b[i]*lepolys[i+2]).reshape(1,N)
    return phi.to(device).double(), a[0:N-2],b[0:N-2]


def dx(N, x, lepolys):
    def gen_diff_lepoly(N, n, x,lepolys):
        lepoly_x = np.zeros((N, 1))
        for i in range(n):
            if ((i+n) % 2) != 0:
                lepoly_x += (2*i+1)*lepolys[i]
        return lepoly_x
    Dx = {}
    for i in range(N):
        Dx[i] = gen_diff_lepoly(N, i, x, lepolys).reshape(1, N)
    return Dx


def basis_x(N,x,eps, phi, Dx, equation):
    if equation == 'Standardb':
        phi_x = phi.clone()
        a, b = np.zeros((N,)), np.ones((N,))
        b *= -1
        for i in range(N-2):
            phi_x[i,:] = torch.from_numpy(Dx[i] + a[i]*Dx[i+1] + b[i]*Dx[i+2]).reshape(1,N)
        phi_x[N-2,:]= torch.from_numpy(np.exp(-(1+x)/eps)/eps  - (1 - np.exp(-2/eps))*.5).reshape(1, N)
    else: 
        phi_x = phi.clone()    
    # NEWNEWNEW
    if equation in ('Standard','Standard1', 'Standard2D', 'NS3d'):
        a, b = np.zeros((N,)), np.ones((N,))
        b *= -1
    elif equation in ('Burgers', 'BurgersT'):
        a, b = np.zeros((N,)), np.ones((N,))
        b *= -1
    elif equation == 'Helmholtz':
        a, b = np.zeros((N,)), np.ones((N,))
        for k in range(N):
            b[k] = -k*(k+1)/((k+2)*(k+3))
    for i in range(N-2):
        phi_x[i,:] = torch.from_numpy(Dx[i] + a[i]*Dx[i+1] + b[i]*Dx[i+2]).reshape(1,N)
    return phi_x.to(device).double()


def dxx(N, x, lepolys):
    def gen_diff2_lepoly(N, n, x,lepolys):
        lepoly_xx = np.zeros((N,1))
        for i in range(n-1):
            if ((i+n) % 2) == 0:
                lepoly_xx += (i+1/2)*(n*(n+1)-i*(i+1))*lepolys[i]
        return lepoly_xx
    Dxx = {}
    for i in range(N):
        Dxx[i] = gen_diff2_lepoly(N, i, x, lepolys).reshape(1, N)
    return Dxx


def basis_xx(N,x,eps, phi, Dxx, equation):
    if equation == 'Standardb':        
        phi_xx = phi.clone()
        a, b = np.zeros((N,)), np.ones((N,))
        b *= -1
        for i in range(N-2):
            phi_xx[i,:] = torch.from_numpy(Dxx[i] + a[i]*Dxx[i+1] + b[i]*Dxx[i+2]).reshape(1,N)
        phi_xx[N-2,:]= torch.from_numpy(-np.exp(-(1+x)/eps)/eps**2).reshape(1, N)
    else: 
        phi_xx = phi.clone() 
    
    # NEWNEWNEW
    if equation in ('Standard','Standard1', 'Standard2D', 'NS3d'):
        a, b = np.zeros((N,)), np.ones((N,))
        b *= -1
    elif equation in ('Burgers', 'BurgersT'):
        a, b = np.zeros((N,)), np.ones((N,))
        b *= -1
    elif equation == 'Helmholtz':
        a, b = np.zeros((N,)), np.ones((N,))
        for k in range(N):
            b[k] = -k*(k+1)/((k+2)*(k+3))
    for i in range(N-2):
        phi_xx[i,:] = torch.from_numpy(Dxx[i] + a[i]*Dxx[i+1] + b[i]*Dxx[i+2]).reshape(1,N)
    return phi_xx.to(device).double()


def basis_vectors(N,eps, equation):
    xx = legslbndm(N)
    lepolys = gen_lepolys(N, xx)
    lepoly_x = dx(N, xx, lepolys)
    lepoly_xx = dxx(N, xx, lepolys)
    phi,aa,bb = basis(N,xx,eps, lepolys, equation)
    phi_x = basis_x(N,xx,eps, phi, lepoly_x, equation)
    phi_xx = basis_xx(N,xx,eps, phi_x, lepoly_xx, equation)
    D = legslbdiff(N, xx)
    return xx, lepolys, lepoly_x, lepoly_xx, phi, phi_x, phi_xx, D,aa,bb




def reconstruct(alphas, phi):
    alphas.to(device).double()
    phi.to(device).double()
    
    
    B,_, i1, j1,k1 = alphas.shape
    i,j=phi.shape
    
    P = torch.empty((B,1,1, j, i), requires_grad=False).to(device).double()
    
    
    P[:,:,:,:,:] = (phi).T
    # print(P.shape)
    T = alphas@ P
    
    PT1 = T.permute(0,1,4, 3, 2)
    T=PT1@P
    
    PT2 = T.permute(0,1,2, 4, 3)
    T=(PT2@P).permute(0,1,3, 4, 2)
  
    
    return T

def reconstruct2(alphas, phi,psi,dir):
    B,_, _, _,_ = alphas.shape
    i,j=phi.shape        
   
    P=(phi.T)
    Px=(psi.T)
   
    
    if dir=='z':
       
        T = alphas@ Px
        
      

        
        PT1 = T.permute((0,1,4, 3, 2))
        T=PT1@P
        
        PT2 = T.permute((0,1,2, 4, 3))
        T=(PT2@P).permute((0,1,3,4,2))
    elif dir=='x':
        
        T = alphas@ P
        
        PT1 = T.permute((0,1,4, 3, 2))
        T=PT1@Px
        
        PT2 = T.permute((0,1,2, 4, 3))
        T=(PT2@P).permute((0,1,3,4,2))
    elif dir=='y':
        
        T = alphas@ P
        
        PT1 = T.permute((0,1,4, 3, 2))
        T=PT1@P
        
        PT2 = T.permute((0,1,2, 4, 3))
        T=(PT2@Px).permute((0,1,3,4,2))
   
    return T



      





# def weak_form2(eps,aa,bb,dt, N,para,data_Mass, f, u, alphas, lepolys, phi, phi_x, equation, nbfuncs, lepolysx, D = None):
def weak_form0(al_data,cfx0,cfy0,cfz0, N,ode_data,pre_cond, Ed):
   
    
    ndata,_,ndt,_,_,_=al_data.shape
    
    alx=al_data[:,0]
    
    aly=al_data[:,1]
    alz=al_data[:,2]
    
    cFx1=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    cFy1=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    cFz1=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    
    cFx1[:,0,]=cfx0
    cFy1[:,0,]=cfy0
    cFz1[:,0,]=cfz0
    
   
    exfx0=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    exfy0=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    exfz0=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    
    # ode_data=np.zeros((N-1,N-1,N-1))
    
    for jj in range(N-1):
        exfx0[:,:,jj,:,:,0]=Ed.T@torch.sum(torch.reshape(Ed[:,jj],(1,1,N-1,1,1))*cFx1,2)
        exfy0[:,:,jj,:,:,0]=Ed.T@torch.sum(torch.reshape(Ed[:,jj],(1,1,N-1,1,1))*cFy1,2)
        exfz0[:,:,jj,:,:,0]=Ed.T@torch.sum(torch.reshape(Ed[:,jj],(1,1,N-1,1,1))*cFz1,2)
    
    exfx=torch.sum(pre_cond@exfx0,5)
    exfy=torch.sum(pre_cond@exfy0,5)    
    exfz=torch.sum(pre_cond@exfz0,5)
    
    
    
    alx00=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    alx00[:,:,:,:,:,0]=alx
    
    aly00=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    aly00[:,:,:,:,:,0]=aly
    
    alz00=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    alz00[:,:,:,:,:,0]=alz
    
    alxnew=(ode_data@alx00)[:,:,:,:,:,0]
    alynew=(ode_data@aly00)[:,:,:,:,:,0]
    alznew=(ode_data@alz00)[:,:,:,:,:,0]
    
    
    return alxnew,alynew,alznew,exfx,exfy,exfz

def weak_form1(eps,cu0,cv0,cw0,cu1,cv1,cw1,cuu0,cvv0,cww0,cuu1,cvv1,cww1,cFx01,cFy01,cFz01,cFx,cFy,cFz,dt, N,ode_data,pre_cond, al_data,Mxnd,Mnd,Md,Mxdd,Mdxd,Ed,Mm,Mmx ):
   
    
    ndata,_,ndt,_,_,_=al_data.shape
    
    
    alx=al_data[:,0]
    
    aly=al_data[:,1]
    alz=al_data[:,2]
    
    
    
    cFx1=-cFx01+0.5*(4*cu1-cu0)/dt+cFx-(2*cuu1-cuu0)
    cFy1=-cFy01+0.5*(4*cv1-cv0)/dt+cFy-(2*cvv1-cvv0)
    cFz1=-cFz01+0.5*(4*cw1-cw0)/dt+cFz-(2*cww1-cww0)
   
        
    exfx0=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    exfy0=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    exfz0=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    
    # ode_data=np.zeros((N-1,N-1,N-1))
    
    for jj in range(N-1):
        exfx0[:,:,jj,:,:,0]=Ed.T@torch.sum(torch.reshape(Ed[:,jj],(1,1,N-1,1,1))*cFx1,2)
        exfy0[:,:,jj,:,:,0]=Ed.T@torch.sum(torch.reshape(Ed[:,jj],(1,1,N-1,1,1))*cFy1,2)
        exfz0[:,:,jj,:,:,0]=Ed.T@torch.sum(torch.reshape(Ed[:,jj],(1,1,N-1,1,1))*cFz1,2)
    # print(cFz01[0,0,0,0,:10])
    # print(cw1[0,0,0,0,:10])
    # print(cFz1[0,0,0,0,:10])
    # input('kkkk')
    exfx=torch.sum(pre_cond@exfx0,5)
    exfy=torch.sum(pre_cond@exfy0,5)    
    exfz=torch.sum(pre_cond@exfz0,5)
    
    
    alx00=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    alx00[:,:,:,:,:,0]=alx
    
    aly00=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    aly00[:,:,:,:,:,0]=aly
    
    alz00=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    alz00[:,:,:,:,:,0]=alz
    
    alxnew=(ode_data@alx00)[:,:,:,:,:,0]
    alynew=(ode_data@aly00)[:,:,:,:,:,0]
    alznew=(ode_data@alz00)[:,:,:,:,:,0]
    
    
   
    return alxnew,alynew,alznew,exfx,exfy,exfz







def weak_combine(eps,alx,aly,alz,cFx00,cFy00,cFz00,dt, N,phial,Mxnd,Mnd,Md,Mxdd,Mdxd,Mm,Mmx,iMd,phisets,lep,s_diag ):
 
    ndata,ndt,_,_,_=alx.shape
    
    phiall=phial.clone()
    phiall[:,:,0,:,:]=0
   
    Px3=torch.transpose(torch.matmul(Mxnd,torch.transpose(phiall,3,2)),3,2)
    Px2=torch.matmul(Mnd,Px3) #second
    px1=torch.transpose(torch.matmul(Mnd,torch.transpose(Px2,4,3)),4,3) #first
    
   
    uu3=torch.transpose(torch.matmul(Md,torch.transpose(alx,4,3)),4,3)
    # uu1=torch.transpose(Md,torch.transpose(uu3,3,4),3,4)
    uu1=torch.matmul(Md,uu3)
    
       
    uu3=torch.transpose(torch.matmul(Md,torch.transpose(aly,4,3)),4,3)
    uu2=torch.transpose(torch.matmul(Mdxd,torch.transpose(uu3,3,2)),3,2)
    vv1=torch.matmul(Mxdd,uu2)
    
    uu3=torch.transpose(torch.matmul(Mxdd,torch.transpose(alz,4,3)),4,3)
    uu2=torch.transpose(torch.matmul(Mdxd,torch.transpose(uu3,3,2)),3,2)
    ww1=torch.matmul(Md,uu2)
    
    
    ""
    # py1=torch.empty((N-1,N-1,N-1))
    phiall=phial.clone()
    
    phiall[:,:,:,0,:]=0
   
    Py3=torch.transpose(torch.matmul(Mnd,torch.transpose(phiall,3,2)),3,2)
    Py2=torch.matmul(Mxnd,Py3) #second
    py1=torch.transpose(torch.matmul(Mnd,torch.transpose(Py2,4,3)),4,3)
    
   
    uu3=torch.transpose(torch.matmul(Md,torch.transpose(alx,4,3)),4,3)
    uu2=torch.matmul(Mdxd,uu3)
    uuu1=torch.transpose(torch.matmul(Mxdd,torch.transpose(uu2,3,2)),3,2)
    
    uu3=torch.transpose(torch.matmul(Md,torch.transpose(aly,4,3)),4,3)
    
    vvv1=torch.transpose(torch.matmul(Md,torch.transpose(uu3,3,2)),3,2)
    # vvv1=Md,uu3
    
    uu3=torch.transpose(torch.matmul(Mxdd,torch.transpose(alz,4,3)),4,3)
    uu2=torch.matmul(Mdxd,uu3)
    www1=torch.transpose(torch.matmul(Md,torch.transpose(uu2,3,2)),3,2)
    
    pz1=torch.empty((ndata,ndt,N-1,N-1,N-1))
    phiall=phial.clone()
    phiall[:,:,:,:,0]=0
    
    Pz3=torch.transpose(torch.matmul(Mnd,torch.transpose(phiall,3,2)),3,2)
    Pz2=torch.matmul(Mnd,Pz3) #second
    pz1=torch.transpose(torch.matmul(Mxnd,torch.transpose(Pz2,4,3)),4,3)
    
   
    
    uu3=torch.transpose(torch.matmul(Mdxd,torch.transpose(alx,4,3)),4,3)
    uu2=torch.matmul(Md,uu3)
    uuuu1=torch.transpose(torch.matmul(Mxdd,torch.transpose(uu2,3,2)),3,2)
    
    uu3=torch.transpose(torch.matmul(Mdxd,torch.transpose(aly,4,3)),4,3)
    uu2=torch.matmul(Mxdd,uu3)
    vvvv1=torch.transpose(torch.matmul(Md,torch.transpose(uu2,3,2)),3,2)
    # vvv1=Md,uu3
    
    
    uu2=torch.matmul(Md,alz)
    wwww1=torch.transpose(torch.matmul(Md,torch.transpose(uu2,3,2)),3,2)
    
    
    
    
    qqx=px1+eps*(uu1+vv1+ww1)
    qqy=py1+eps*(uuu1+vvv1+www1)
    qqz=pz1+eps*(uuuu1+vvvv1+wwww1)
    
  
    al_unext1=torch.matmul(Md,alx)
    al_unext2=torch.transpose(torch.matmul(Md,torch.transpose(al_unext1,3,2)),3,2)
    al_unext3=torch.transpose(torch.matmul(Md,torch.transpose(al_unext2,4,3)),4,3)
    
    # phixnext3=torch.empty((N-1,N-1,N-1))
    phiall=phial.clone()
    
    
    phiall[:,:,0,:,:]=0
   
    phixnext1=torch.transpose(torch.matmul(Mxnd,torch.transpose(phiall,3,2)),3,2)
    phixnext2=torch.matmul(Mnd,phixnext1)
    phixnext3=torch.transpose(torch.matmul(Mnd,torch.transpose(phixnext2,4,3)),4,3)
    
    al_vnext1=torch.matmul(Md,aly)
    al_vnext2=torch.transpose(torch.matmul(Md,torch.transpose(al_vnext1,3,2)),3,2)
    al_vnext3=torch.transpose(torch.matmul(Md,torch.transpose(al_vnext2,4,3)),4,3)
    
    # phiynext3=torch.empty((N-1,N-1,N-1))
    phiall=phial.clone()
    phiall[:,:,:,0,:]=0
   
    phiynext1=torch.transpose(torch.matmul(Mnd,torch.transpose(phiall,3,2)),3,2)
    phiynext2=torch.matmul(Mxnd,phiynext1)
    phiynext3=torch.transpose(torch.matmul(Mnd,torch.transpose(phiynext2,4,3)),4,3)
    
    al_wnext1=torch.matmul(Md,alz)
    al_wnext2=torch.transpose(torch.matmul(Md,torch.transpose(al_wnext1,3,2)),3,2)
    al_wnext3=torch.transpose(torch.matmul(Md,torch.transpose(al_wnext2,4,3)),4,3)
    
    # phiznext3=torch.empty((N-1,N-1,N-1))
    phiall=phial.clone()
    
    phiall[:,:,:,:,0]=0
    phiznext1=torch.transpose(torch.matmul(Mnd,torch.transpose(phiall,3,2)),3,2)
    phiznext2=torch.matmul(Mnd,phiznext1)
    phiznext3=torch.transpose(torch.matmul(Mxnd,torch.transpose(phiznext2,4,3)),4,3)
    
   
    cFx01=cFx00+qqx
    cFy01=cFy00+qqy
    cFz01=cFz00+qqz
    al_unext=al_unext3-2*dt*phixnext3/3
    al_vnext=al_vnext3-2*dt*phiynext3/3
    al_wnext=al_wnext3-2*dt*phiznext3/3
    
    alu13=iMd@al_unext
    alu12=torch.permute(iMd@torch.permute(alu13,(0,1,3,2,4)),(0,1,3,2,4))
    auu=torch.permute(iMd@torch.permute(alu12,(0,1,2,4,3)),(0,1,2,4,3)) #first
    
    alu13=iMd@al_vnext
    alu12=torch.permute(iMd@torch.permute(alu13,(0,1,3,2,4)),(0,1,3,2,4))
    avv=torch.permute(iMd@torch.permute(alu12,(0,1,2,4,3)),(0,1,2,4,3)) #first
    
    alu13=iMd@al_wnext
    alu12=torch.permute(iMd@torch.permute(alu13,(0,1,3,2,4)),(0,1,3,2,4))
    aww=torch.permute(iMd@torch.permute(alu12,(0,1,2,4,3)),(0,1,2,4,3)) #first
    
    u1=reconstruct(auu,phisets)
    v1=reconstruct(avv,phisets)
    w1=reconstruct(aww,phisets)
    
    
    cuux1,cvvy1,cwwz1=nonlinear(u1,v1,w1,phisets,lep,iMd,Mdxd)
    # cuux1,cvvy1,cwwz1=0,0,0
    
    
    
    
    
    return al_unext,al_vnext,al_wnext,cFx01,cFy01,cFz01,cuux1,cvvy1,cwwz1
    
def weak_pressure(alx,aly,alz,dt, N,oden_data,pre_condn, phial,En,Mm,Mmx ):
   
    # ndata,_,ndt,_,_,_=phial.shape
    ndata,ndt,_,_,_=alx.shape
    
    cFnx3=Mm@alx  #second
    cFnx2=torch.transpose(Mm@torch.transpose(cFnx3,4,3),4,3) #third
    cFnx1=torch.transpose(Mmx@torch.transpose(cFnx2,3,2),3,2)
    
   
    cFny3=torch.transpose(Mm@torch.transpose(aly,4,3),4,3)  #third
    cFny2=Mmx@cFny3 #second
    cFny1=torch.transpose(Mm@torch.transpose(cFny2,3,2),3,2) #first
   
    cFnz3=Mm@alz  #third
    cFnz2=torch.transpose(Mmx@torch.transpose(cFnz3,4,3),4,3)#second
    cFnz1=torch.transpose(Mm@torch.transpose(cFnz2,3,2),3,2) #first
    # cFny1=0
    # cFnz1=0
    cFn=1.5*((cFnx1)+(cFny1)+(cFnz1))/dt
    # cFn=((cFnx1)+(cFny1)+(cFnz1))
    
    Pf=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    Pexfx0=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    
    for jj in range(N-1):
        # G=Ft[jj,].T@E
        
        Pf[:,:,jj,]=torch.sum(torch.reshape(En[:,jj],(1,1,N-1,1,1))*cFn,2)
        
        Pexfx0[:,:,jj,:,:,0]=En.T@Pf[:,:,jj,]
    Pexfx=-torch.sum(pre_condn@Pexfx0,5)
    # Pexfx=torch.cat([Pexfx00,Pexfx00,Pexfx00],1).reshape(ndata,3,ndt,N-1,N-1,N-1)
    # phial2=-torch.transpose(En@torch.transpose(phial,2,3),2,3)
    
    
    # phial1=En@phial2
    
    phial11=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    phial11[:,:,:,:,:,0]=phial
    
    
    phial00=(oden_data@phial11)[:,:,:,:,:,0]
    
    
    return phial00,Pexfx





def sol(alx,aly,alz,dt,phial,Mxnd,Mnd,Md,iMd,phisets ):
 
    
    al_unext1=torch.matmul(Md,alx)
    al_unext2=torch.transpose(torch.matmul(Md,torch.transpose(al_unext1,3,2)),3,2)
    al_unext3=torch.transpose(torch.matmul(Md,torch.transpose(al_unext2,4,3)),4,3)
    
    # phixnext3=torch.empty((N-1,N-1,N-1))
    phiall=phial.clone()
    
    
    "ok"
    phiall[:,:,0,:,:]=0
   
    phixnext1=torch.transpose(torch.matmul(Mxnd,torch.transpose(phiall,3,2)),3,2)
    phixnext2=torch.matmul(Mnd,phixnext1)
    phixnext3=torch.transpose(torch.matmul(Mnd,torch.transpose(phixnext2,4,3)),4,3)
    
    


    al_vnext1=torch.matmul(Md,aly)
    al_vnext2=torch.transpose(torch.matmul(Md,torch.transpose(al_vnext1,3,2)),3,2)
    al_vnext3=torch.transpose(torch.matmul(Md,torch.transpose(al_vnext2,4,3)),4,3)
    
    # phiynext3=torch.empty((N-1,N-1,N-1))
    phiall=phial.clone()
    "ok"
    phiall[:,:,:,0,:]=0
   
    
    phiynext1=torch.transpose(torch.matmul(Mnd,torch.transpose(phiall,3,2)),3,2)
    phiynext2=torch.matmul(Mxnd,phiynext1)
    phiynext3=torch.transpose(torch.matmul(Mnd,torch.transpose(phiynext2,4,3)),4,3)
    
    al_wnext1=torch.matmul(Md,alz)
    al_wnext2=torch.transpose(torch.matmul(Md,torch.transpose(al_wnext1,3,2)),3,2)
    al_wnext3=torch.transpose(torch.matmul(Md,torch.transpose(al_wnext2,4,3)),4,3)
    
    # phiznext3=torch.empty((N-1,N-1,N-1))
    phiall=phial.clone()
    
    phiall[:,:,:,:,0]=0
    phiznext1=torch.transpose(torch.matmul(Mnd,torch.transpose(phiall,3,2)),3,2)
    phiznext2=torch.matmul(Mnd,phiznext1)
    phiznext3=torch.transpose(torch.matmul(Mxnd,torch.transpose(phiznext2,4,3)),4,3)
    
    
   
    al_unext=al_unext3-2*dt*phixnext3/3
    al_vnext=al_vnext3-2*dt*phiynext3/3
    al_wnext=al_wnext3-2*dt*phiznext3/3

   
    
    alu13=iMd@al_unext
    alu12=torch.transpose(torch.matmul(iMd,torch.transpose(alu13,3,2)),3,2)
    auu=torch.transpose(torch.matmul(iMd,torch.transpose(alu12,4,3)),4,3) 
    
    alu13=iMd@al_vnext
    alu12=torch.transpose(torch.matmul(iMd,torch.transpose(alu13,3,2)),3,2)
    avv=torch.transpose(torch.matmul(iMd,torch.transpose(alu12,4,3)),4,3) 
    
    alu13=iMd@al_wnext
    alu12=torch.transpose(torch.matmul(iMd,torch.transpose(alu13,3,2)),3,2)
    aww=torch.transpose(torch.matmul(iMd,torch.transpose(alu12,4,3)),4,3) 
    
    u1=reconstruct(auu,phisets)
    v1=reconstruct(avv,phisets)
    w1=reconstruct(aww,phisets)
    
    return u1,v1,w1



def psol(alx,aly,alz,phial,p0,phisets,phixsets,phinsets,eps,D ): 
    # u=reconstruct(alx, phisets)
    phi=reconstruct(phial, phinsets)
    ux=reconstruct2(alx, phisets,phixsets,'x')
    vx=reconstruct2(aly, phisets,phixsets,'y')
    wx=reconstruct2(alz, phisets,phixsets,'z')
    
    
  
    
    pp=phi+p0-eps*(ux+vx+wx)
    
    
    px1=torch.swapaxes(D@torch.swapaxes(pp,2,3),2,3)
    py1=D@pp
    pz1=torch.swapaxes(D@torch.swapaxes(pp,3,4),3,4)

    return px1,py1,pz1,pp

 

def conv(FF,phi,lep):    
    B,_ ,i1, j1,k1 = FF.shape
    i,j=phi.shape        
    P = torch.zeros((B,1,1, i, j), requires_grad=False).to(device).double()
    P[:,:,:,:]=(phi/lep**2)
    T = FF@ P
   
    PT1 = T.permute(0,1,4, 3, 2)
    T=PT1@P
    PT2 = T.permute(0,1,2, 4, 3)
    T=(2/((j-1)*j))**3*(PT2@P).permute(0,1,2, 4, 3)
   
    return T.permute(0,1,4,3,2)
    # return phi,lep



def nonlinear(uu0,vv0,ww0,phi,lep,iMd,Mdxd):        
    uu=uu0*uu0
    uv=uu0*vv0
    uw=uu0*ww0
    vv=vv0*vv0
    vw=vv0*ww0
    ww=vv0*ww0
    
    cuu=conv(uu,phi,lep)
    cuv=conv(uv,phi,lep)
    cuw=conv(uw,phi,lep)
    cvv=conv(vv,phi,lep)
    cvw=conv(vw,phi,lep)
    cww=conv(ww,phi,lep)
    
    "uux+uvy+uwz"
    cuux=torch.permute(Mdxd@(iMd@torch.permute(cuu,(0,1,3,2,4))),(0,1,3,2,4))
    
    cuvy=Mdxd@(iMd@cuv) #second
    
    
    cuwz=torch.permute(Mdxd@(iMd@torch.permute(cuw,(0,1,2,4,3))),(0,1,2,4,3)) #second
    
   
    
    "vux+vvy+vwz"
    cuvx=torch.permute(Mdxd@(iMd@torch.permute(cuv,(0,1,3,2,4))),(0,1,3,2,4))
    
    
    
    cvvy=Mdxd@(iMd@cvv) #second
    
    
    cvwz=torch.permute(Mdxd@(iMd@torch.permute(cvw,(0,1,2,4,3))),(0,1,2,4,3)) #second
    
    
    "wux+wvy+wwz"
    cuwx=torch.permute(Mdxd@(iMd@torch.permute(cuw,(0,1,3,2,4))),(0,1,3,2,4))
    
    
    
    cvwy=Mdxd@(iMd@cvw) #second
    
    
    cwwz=torch.permute(Mdxd@(iMd@torch.permute(cww,(0,1,2,4,3))),(0,1,2,4,3)) #second
    return cuux+cuvy+cuwz, cuvx+cvvy+cvwz,cuwx+cvwy+cwwz
    # return cuux+cuvy+cuwz




