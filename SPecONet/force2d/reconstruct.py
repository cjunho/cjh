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
    if equation in ('Standard','Standard1', 'Standard2D', 'NS2d'):
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
    if equation in ('Standard','Standard1', 'Standard2D', 'NS2d'):
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
    if equation in ('Standard','Standard1', 'Standard2D', 'NS2d'):
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
    # 1D case
    T = (phi@alphas)@phi.T 
  
    return T


def reconstructx(alphas, phi,phix):
   
    T = (phix@alphas)@phi.T 
    return T




def weak_form0(alx,aly,cfx0,cfy0, N,ode_eye,iode_data, Ed ):
   

    ndata,ndt,NN,_=alx.shape 
  
    exfx0=(Ed.T@cfx0).reshape((ndata,ndt,NN,NN,1))
    exfy0=(Ed.T@cfy0).reshape((ndata,ndt,NN,NN,1))
   

    exfx=torch.sum(iode_data@exfx0,4)
    exfy=torch.sum(iode_data@exfy0,4)    

    
  
   
   
    alx00=torch.empty((ndata,ndt,N-1,N-1,1)).to(device).double()
    alx00[:,:,:,:,0]=alx
    
    aly00=torch.empty((ndata,ndt,N-1,N-1,1)).to(device).double()
    aly00[:,:,:,:,0]=aly
    
    alxnew=torch.sum(ode_eye@alx00,4)
    alynew=torch.sum(ode_eye@aly00,4)
    
    
    return alxnew,alynew,exfx,exfy

def weak_form1(cu0,cv0,cu1,cv1,cuu0,cvv0,cuu1,cvv1,cFx01,cFy01,cFx,cFy,dt,ode_data,pre_cond, al_data,Ed ):
   
    
    ndata,_,ndt,NN,_=al_data.shape
    
    
    alx=al_data[:,0]
    
    aly=al_data[:,1]
    
    
    
    cfx0=-cFx01+0.5*(4*cu1-cu0)/dt+cFx-(2*cuu1-cuu0)
    cfy0=-cFy01+0.5*(4*cv1-cv0)/dt+cFy-(2*cvv1-cvv0)
    
    
    exfx0=(Ed.T@cfx0).reshape((ndata,ndt,NN,NN,1))
    exfy0=(Ed.T@cfy0).reshape((ndata,ndt,NN,NN,1))
    

    exfx=torch.sum(pre_cond@exfx0,4)
    exfy=torch.sum(pre_cond@exfy0,4)    

    
    
    alx00=torch.empty((ndata,ndt,NN,NN,1)).to(device).double()
    alx00[:,:,:,:,0]=alx
    
    aly00=torch.empty((ndata,ndt,NN,NN,1)).to(device).double()
    aly00[:,:,:,:,0]=aly
  
    
    alxnew=(ode_data@alx00)[:,:,:,:,0]
    alynew=(ode_data@aly00)[:,:,:,:,0]
  
    
   
    return alxnew,alynew,exfx,exfy




def weak_combine(eps,alx,aly,cFx00,cFy00,dt, N,phial,Mxnd,Mnd,Md,Mxdd,Mdxd,Mm,Mmx,iMd,phisets,lep,s_diag ):
 
    
    al_unext3=(Md@alx)@Md
    al_vnext3=(Md@aly)@Md

    phiall=phial.clone()
    phiall[:,:,:,0]=0
  
    
    px1=(Mxnd@phiall)@Mnd.T
   
    uu1=alx@Md.T
    
       
    vv1=(Mdxd@aly)@Mxdd.T
   
    phiall=phial.clone()
    
    phiall[:,:,0,:]=0
  
    py1=(Mnd@phiall)@Mxnd.T
    
    uuu1=(Mxdd@alx)@Mdxd.T
    
    vvv1=Md@aly
    
    qqx=px1+eps*(uu1+vv1)
    qqy=py1+eps*(uuu1+vvv1)
   
  
    phiall[:,:,:,0]=0
    
    phixnext3=(Mxnd@phiall)@Mnd.T

   
    phiall[:,:,0,:]=0
    
    phiynext3=(Mnd@phiall)@Mxnd.T

    al_unext=al_unext3-2*dt*phixnext3/3
    al_vnext=al_vnext3-2*dt*phiynext3/3
    
    cFx01=cFx00+qqx
    cFy01=cFy00+qqy
  
  
    auu=(iMd@al_unext)@iMd
    
    avv=(iMd@al_vnext)@iMd
   
    v1=reconstruct(avv,phisets)
    
    u1=reconstruct(auu,phisets)
   
    
    cuux1,cvvy1=nonlinear(u1,v1,phisets,lep,iMd,Mdxd)
    
    
   
    
    return al_unext,al_vnext,cFx01,cFy01,cuux1,cvvy1
    
def weak_pressure(alx,aly,p_pred,Mmx,Mm,dt, oden_eye,ioden_data, En ):
    ndata,ndt,N,_=alx.shape
    

    cFnx1=(Mmx@alx)@Mm.T
    
    
        
    cFny1=(Mm@aly)@Mmx.T
   
    
    cFn=1.5*((cFnx1)+(cFny1))/dt
   
    Pexfx=ioden_data@((En.T@cFn).reshape((ndata,ndt,N,N,1)))
    
    phial11=torch.empty((ndata,ndt,N,N,1)).to(device).double()
    phial11[:,:,:,:,0]=-p_pred
    
    
    phial00=torch.sum(oden_eye@phial11,4)
   
    return phial00,Pexfx[:,:,:,:,0]
    
    

def sol(alx,aly,dt,phial,Mxnd,Mnd,Md,iMd,phisets ):
    
    al_unext3=(Md@alx)@Md
    al_vnext3=(Md@aly)@Md

 
   
    phiall=phial.clone()
    
  
    phiall[:,:,:,0]=0
    
    phixnext3=(Mxnd@phiall)@Mnd.T

   
    phiall[:,:,0,:]=0
    
    phiynext3=(Mnd@phiall)@Mxnd.T

    al_unext=al_unext3-2*dt*phixnext3/3
    al_vnext=al_vnext3-2*dt*phiynext3/3
    
      
  
    auu=(iMd@al_unext)@iMd
    
    avv=(iMd@al_vnext)@iMd
   
    v1=reconstruct(avv,phisets)
    
    u1=reconstruct(auu,phisets)
    
    return u1,v1



def psol(alx,aly,phial,p0,phisets,phixsets,phinsets,eps,D ): 
    
    phi=reconstruct(phial, phinsets)
    ux=reconstructx(alx, phisets,phixsets)
    vx=reconstructx(aly,phixsets, phisets)
   
   

    
    pp=phi+p0-eps*(ux+vx)
    
    
    px=D@pp
    py=torch.swapaxes(D@torch.swapaxes(pp,2,3),2,3)
    
    return px,py,pp

 

def conv(FF,phi,lep): 
    
    B,_ ,i1, j1 = FF.shape
    i,j=phi.shape        
    P = torch.zeros((B,1, i, j), requires_grad=False).to(device).double()
    P[:,:,:,:]=(phi/lep**2)
    T = FF@ P
    
    PT1 = T.permute(0,1, 3, 2)
    T=(2/((i-1)*i))**2*PT1@P
   
   
    return T.permute(0,1, 3, 2)
   


def nonlinear(uu0,vv0,phi,lep,iMd,Mdxd):        
    uu=uu0*uu0
    uv=uu0*vv0
    
    vv=vv0*vv0
    
    
    cuu=conv(uu,phi,lep)
    cuv=conv(uv,phi,lep)
    
    cvv=conv(vv,phi,lep)
   
    "uux+uvy+uwz"
    cuux=Mdxd@(iMd@cuu)

  
    cuvy=torch.permute(Mdxd@(iMd@torch.permute(cuv,(0,1,3,2))),(0,1,3,2))
    
   
   
    
    "vux+vvy+vwz"
    cuvx=Mdxd@(iMd@cuv)
    
    
    cvvy=torch.permute(Mdxd@(iMd@torch.permute(cvv,(0,1,3,2))),(0,1,3,2)) #second
   
    return cuux+cuvy, cuvx+cvvy
   

