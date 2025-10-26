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


def L2(u_pred,u,legendre,N,DATASET):
    N -= 1
    denom = torch.square(torch.from_numpy(legendre).to(device).double())
    denom = torch.transpose(denom, 0, 1)
    diff=(u_pred-u)**2
    loss1=torch.sum(diff*2/(N*(N+1))/denom, axis=2)/torch.sum(u*u*2/(N*(N+1))/denom, axis=2)   
    loss=torch.sum(torch.sqrt(loss1))
    return loss

def reconstruct(alphas, phi):
    alphas.to(device).double()
    phi.to(device).double()
    # 1D case
    T = (phi@alphas)@phi.T 
  
    return T


def reconstructx(alphas, phi,phix):
    
        # Dim alphas: (B, 1, N-1, N-1)
    # B, _, i, j = alphas.shape
    # alphas = alphas[:,0,:,:].to(device).double()
    # P = torch.empty((B, j, j+2), requires_grad=False).to(device).double()
    # P[:,:,:] = phi
    # PT = P.permute(0, 2, 1)
    # print(PT.shape)
    # T = torch.bmm(PT, alphas)
    # print(T.shape)
    # T = torch.bmm(T, P)
    # print(T.shape)
    # input('sfdgsdf')
    # T = T.reshape(B, 1, (j+2), j+2) # Changed
    T = (phix@alphas)@phi.T 
    return T


      

def ODE2(eps, u, alphas, phi_x, phi_xx, equation):
    ux = reconstruct(alphas, phi_x)
    uxx = reconstruct(alphas, phi_xx)
    if equation == 'Standard':
        return -eps*uxx - ux
    elif equation == 'Burgers':
        return -eps*uxx + u*ux
    elif equation == 'BurgersT':
        return -eps*uxx + u*ux + u
    elif equation == 'Helmholtz':
        ku = 3.5
        return uxx + ku*u





# def weak_form2(eps,aa,bb,dt, N,para,data_Mass, f, u, alphas, lepolys, phi, phi_x, equation, nbfuncs, lepolysx, D = None):
def weak_form0(alx,aly,cfx0,cfy0, N,ode_eye,iode_data, Ed ):
   

    ndata,ndt,NN,_=alx.shape #NN=SHAPE-2
  
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
    # cFz1=-cFz01+0.5*(4*cw1-cw0)/dt+cFz-(2*cww1-cww0)
    
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

def weak_form11(eps,cu0,cv0,cw0,cu1,cv1,cw1,cuu0,cvv0,cww0,cuu1,cvv1,cww1,cFx01,cFy01,cFz01,cFx,cFy,cFz,dt, N,ode_data,pre_cond, al_data,Mxnd,Mnd,Md,Mxdd,Mdxd,Ed,Mm,Mmx ):
   
    
    ndata0,ndt,_,_,_=al_data.shape
    
    ndata=int(ndata0/4)
    alx=al_data[0::4,]
    
    aly=al_data[1::4,]
    alz=al_data[2::4,]
    
    cFx1=-cFx01+0.5*(4*cu1-cu0)/dt+cFx-(2*cuu1-cuu0)
    cFy1=-cFy01+0.5*(4*cv1-cv0)/dt+cFy-(2*cvv1-cvv0)
    cFz1=-cFz01+0.5*(4*cw1-cw0)/dt+cFz-(2*cww1-cww0)
   
    # print(al_unext[1,0,1,])
    # print(al_unext[0,0,1,])
    
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



def phi_combine(alx,aly,alz,dt, N,oden_data,En,Mm,Mmx,ndata,ndt):
    

    cFnx3=Mm@alx  #second
    
    cFnx2=torch.transpose(Mm@torch.transpose(cFnx3,4,3),4,3) #third
    
    # cFnx1=torch.transpose(Mmx@torch.transpose(cFnx2,3,2),3,2) #first
    
    cFnx1=torch.transpose(Mmx@torch.transpose(cFnx2,3,2),3,2)
    
    
    # cFny3=Mmx@aly  #third
        
    # cFny2=torch.transpose(Mm@torch.transpose(cFny3,4,3),4,3) #second
    
    # cFny1=torch.transpose(Mm@torch.transpose(cFny2,(1,2,0)),(1,2,0)) #first
    cFny3=torch.transpose(Mm@torch.transpose(aly,4,3),4,3)  #third
        
    cFny2=Mmx@cFny3 #second
    
    cFny1=torch.transpose(Mm@torch.transpose(cFny2,3,2),3,2) #first
   
    cFnz3=Mm@alz  #third
    
    cFnz2=torch.transpose(Mmx@torch.transpose(cFnz3,4,3),4,3)#second
    
    cFnz1=torch.transpose(Mm@torch.transpose(cFnz2,3,2),3,2) #first
    
    cFn=1.5*((cFnx1)+(cFny1)+(cFnz1))/dt
    

    Pf=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    Pexfx0=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    
    
    for jj in range(N-1):
        # G=Ft[jj,].T@E
        
        Pf[:,:,jj,]=torch.sum(torch.reshape(En[:,jj],(1,1,N-1,1,1))*cFn,2)
        
        Pexfx0[:,:,jj,:,:]=En.T@Pf[:,:,jj,]
    
    
    
    
    phi=torch.linalg.solve(oden_data, torch.permute(Pexfx0,(1,2,3,4,0)))
    
    return phi.permute((4,0,1,2,3))

def weak_combine(eps,alx,aly,cFx00,cFy00,dt, N,phial,Mxnd,Mnd,Md,Mxdd,Mdxd,Mm,Mmx,iMd,phisets,lep,s_diag ):
 
    # ndata,ndt,_,_=alx.shape
    al_unext3=(Md@alx)@Md
    al_vnext3=(Md@aly)@Md

    phiall=phial.clone()
    phiall[:,:,:,0]=0
    # phiall[:,:,0]=0
    
    px1=(Mxnd@phiall)@Mnd.T
   
    uu1=alx@Md.T
    
       
    vv1=(Mdxd@aly)@Mxdd.T
    
    
    ""
    # py1=torch.empty((N-1,N-1,N-1))
    phiall=phial.clone()
    
    phiall[:,:,0,:]=0
    # phiall[:,:,0,:]=0
    # phiall[:,:,:,0]=0
    py1=(Mnd@phiall)@Mxnd.T
    # Px3=alp,Mdxd
    # Px2=torch.transpose(torch.matmul(torch.transpose(Px3,(2,0,1)),Md,(1,2,0)) #second
    # px2=torch.transpose(torch.matmul(torch.transpose(Px2,(1,2,0)),Md,(2,0,1)) #first
    
    uuu1=(Mxdd@alx)@Mdxd.T
    
    vvv1=Md@aly
    
   
  
    
    
    qqx=px1+eps*(uu1+vv1)
    qqy=py1+eps*(uuu1+vvv1)
    # cFx01[0,]=qqx[0,]
    # cFy01[0,]=qqy[0,]
    # cFz01[0,]=qqz[0,]
  
    phiall[:,:,:,0]=0
    
    phixnext3=(Mxnd@phiall)@Mnd.T

   
    phiall[:,:,0,:]=0
    
    phiynext3=(Mnd@phiall)@Mxnd.T

    al_unext=al_unext3-2*dt*phixnext3/3
    al_vnext=al_vnext3-2*dt*phiynext3/3
    
    
    # for jj in range(1,ndt):
    
    cFx01=cFx00+qqx
    cFy01=cFy00+qqy
  
  
    auu=(iMd@al_unext)@iMd
    
    avv=(iMd@al_vnext)@iMd
   
    v1=reconstruct(avv,phisets)
    
    u1=reconstruct(auu,phisets)
   
    
    cuux1,cvvy1=nonlinear(u1,v1,phisets,lep,iMd,Mdxd)
    # cuux1,cvvy1,cwwz1=0,0,0
    
   
    
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
 
    
    # ndata,ndt,_,_=alx.shape
    al_unext3=(Md@alx)@Md
    al_vnext3=(Md@aly)@Md

 
    # py1=torch.empty((N-1,N-1,N-1))
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
    # u=reconstruct(alx, phisets)
    phi=reconstruct(phial, phinsets)
    ux=reconstructx(alx, phisets,phixsets)
    vx=reconstructx(aly,phixsets, phisets)
   
    
    # ux1=torch.swapaxes(D@torch.swapaxes(u,2,3),2,3) #'x'
    # ux1=D@u#'y'
    # ux1=torch.swapaxes(D@torch.swapaxes(u,3,4),3,4) #'z'

    
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
    # PT2 = T.permute(0,1,2, 4, 3)
    # T=(PT2@P).permute(0,1,2, 4, 3)
   
    return T.permute(0,1, 3, 2)
    # return phi,lep



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
    # return cuux+cuvy+cuwz




def weak_form_all(eps,alx0,aly0,alz0,cfx0,cfy0,cfz0,cFx,cFy,cFz,dt, N,ode_data,oden_data,pre_cond,pre_condn, al_data,Mxnd,Mnd,Md,Mxdd,Mdxd,Ed,En,Mm,Mmx ):
   
    
    ndata0,ndt,_,_,_=al_data.shape
    ndata=int(ndata0/4)
    # alx=al_data[0::4,]
    
    # aly=al_data[1::4,]
    # alz=al_data[2::4,]
    # phial=al_data[3::4,]
    
    alx=al_data[0:ndata,]
    
    aly=al_data[ndata:2*ndata,]
    alz=al_data[2*ndata:3*ndata,]
    phial=al_data[3*ndata:4*ndata,]
   
    phiall=phial.clone()
    phiall[:,:,:,0,:]=0
    # phiall[:,:,0]=0
    
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
    
    phiall[:,:,0,:,:]=0
    # phiall[:,:,0,:]=0
    # phiall[:,:,:,0]=0
    Py3=torch.transpose(torch.matmul(Mnd,torch.transpose(phiall,3,2)),3,2)
    Py2=torch.matmul(Mxnd,Py3) #second
    py1=torch.transpose(torch.matmul(Mnd,torch.transpose(Py2,4,3)),4,3)
    
    # Px3=alp,Mdxd
    # Px2=torch.transpose(torch.matmul(torch.transpose(Px3,(2,0,1)),Md,(1,2,0)) #second
    # px2=torch.transpose(torch.matmul(torch.transpose(Px2,(1,2,0)),Md,(2,0,1)) #first
    
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
    # phiall[:,:,:,0]=0
    # phiall[:,0,:,:]=0
    # phiall[:,:,0,:]=0
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
    
    cFx01=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    cFy01=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    cFz01=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    
    qqx=px1+eps*(uu1+vv1+ww1)
    qqy=py1+eps*(uuu1+vvv1+www1)
    qqz=pz1+eps*(uuuu1+vvvv1+wwww1)
    # cFx01[0,]=qqx[0,]
    # cFy01[0,]=qqy[0,]
    # cFz01[0,]=qqz[0,]
    cFx01[:,0,]=0
    cFy01[:,0,]=0
    cFz01[:,0,]=0
    
    # print(cFx01[0,0,1,])
    
    # cFz01=0*py1
  
    al_unext1=torch.matmul(Md,alx)
    al_unext2=torch.transpose(torch.matmul(Md,torch.transpose(al_unext1,3,2)),3,2)
    al_unext3=torch.transpose(torch.matmul(Md,torch.transpose(al_unext2,4,3)),4,3)
    
    # phixnext3=torch.empty((N-1,N-1,N-1))
    phiall=phial.clone()
    
    # phiall[:,0,:,:]=0
    phiall[:,:,:,0,:]=0
    # phiall[:,:,:,0]=0
    phixnext1=torch.transpose(torch.matmul(Mxnd,torch.transpose(phiall,3,2)),3,2)
    phixnext2=torch.matmul(Mnd,phixnext1)
    phixnext3=torch.transpose(torch.matmul(Mnd,torch.transpose(phixnext2,4,3)),4,3)
    
    al_vnext1=torch.matmul(Md,aly)
    al_vnext2=torch.transpose(torch.matmul(Md,torch.transpose(al_vnext1,3,2)),3,2)
    al_vnext3=torch.transpose(torch.matmul(Md,torch.transpose(al_vnext2,4,3)),4,3)
    
    # phiynext3=torch.empty((N-1,N-1,N-1))
    phiall=phial.clone()
    phiall[:,:,0,:,:]=0
    # phiall[:,:,0,:]=0
    # phiall[:,:,:,0]=0
    
    phiynext1=torch.transpose(torch.matmul(Mnd,torch.transpose(phiall,3,2)),3,2)
    phiynext2=torch.matmul(Mxnd,phiynext1)
    phiynext3=torch.transpose(torch.matmul(Mnd,torch.transpose(phiynext2,4,3)),4,3)
    
    al_wnext1=torch.matmul(Md,alz)
    al_wnext2=torch.transpose(torch.matmul(Md,torch.transpose(al_wnext1,3,2)),3,2)
    al_wnext3=torch.transpose(torch.matmul(Md,torch.transpose(al_wnext2,4,3)),4,3)
    
    # phiznext3=torch.empty((N-1,N-1,N-1))
    phiall=phial.clone()
    # phiall[:,0,:,:]=0
    # phiall[:,:,0,:]=0
    # phiall[:,:,:,0]=0
    phiznext1=torch.transpose(torch.matmul(Mnd,torch.transpose(phiall,3,2)),3,2)
    phiznext2=torch.matmul(Mnd,phiznext1)
    phiznext3=torch.transpose(torch.matmul(Mxnd,torch.transpose(phiznext2,4,3)),4,3)
    
    al_unext=torch.empty((ndata,ndt+1,N-1,N-1,N-1)).to(device).double()
    al_vnext=torch.empty((ndata,ndt+1,N-1,N-1,N-1)).to(device).double()
    al_wnext=torch.empty((ndata,ndt+1,N-1,N-1,N-1)).to(device).double()
    
    
    al_unext[:,0,]=alx0
    al_vnext[:,0,]=aly0
    al_wnext[:,0,]=alz0
    
    for jj in range(1,ndt):
        
        cFx01[:,jj,]=cFx01[:,jj-1,]+qqx[:,jj-1,]
        cFy01[:,jj,]=cFy01[:,jj-1,]+qqy[:,jj-1,]
        cFz01[:,jj,]=cFz01[:,jj-1,]+qqz[:,jj-1,]
   
    al_unext[:,1:,]=al_unext3-2*dt*phixnext3/3
    al_vnext[:,1:,]=al_vnext3-2*dt*phiynext3/3
    al_wnext[:,1:,]=al_wnext3-2*dt*phiznext3/3
    
    cFx1=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    cFy1=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    cFz1=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    
    cFx1[:,0,]=cfx0
    cFy1[:,0,]=cfy0
    cFz1[:,0,]=cfz0
    
    
    # cFx1[:,1:,]=-cFx01[:,1:,]+0.5*(4*al_unext[:,1:ndt,]-al_unext[:,0:ndt-1,])/dt+cFx[:,1:,]
    # cFy1[:,1:,]=-cFy01[:,1:,]+0.5*(4*al_vnext[:,1:ndt,]-al_vnext[:,0:ndt-1,])/dt+cFy[:,1:,]
    # cFz1[:,1:,]=-cFz01[:,1:,]+0.5*(4*al_wnext[:,1:ndt,]-al_wnext[:,0:ndt-1,])/dt+cFz[:,1:,]
    
    
    # print(al_unext[0,0,1,])
    
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
   
    cFnx3=Mm@alx  #second
    
    cFnx2=torch.transpose(Mm@torch.transpose(cFnx3,4,3),4,3) #third
    
    # cFnx1=torch.transpose(Mmx@torch.transpose(cFnx2,3,2),3,2) #first
    
    cFnx1=torch.transpose(Mmx@torch.transpose(cFnx2,3,2),3,2)
    
    
    # cFny3=Mmx@aly  #third
        
    # cFny2=torch.transpose(Mm@torch.transpose(cFny3,4,3),4,3) #second
    
    # cFny1=torch.transpose(Mm@torch.transpose(cFny2,(1,2,0)),(1,2,0)) #first
    cFny3=torch.transpose(Mm@torch.transpose(aly,4,3),4,3)  #third
        
    cFny2=Mmx@cFny3 #second
    
    cFny1=torch.transpose(Mm@torch.transpose(cFny2,3,2),3,2) #first
   
    cFnz3=Mm@alz  #third
    
    cFnz2=torch.transpose(Mmx@torch.transpose(cFnz3,4,3),4,3)#second
    
    cFnz1=torch.transpose(Mm@torch.transpose(cFnz2,3,2),3,2) #first
    
    cFn=1.5*((cFnx1)+(cFny1)+(cFnz1))/dt
   
    Pf=torch.empty((ndata,ndt,N-1,N-1,N-1)).to(device).double()
    Pexfx0=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    
    
    for jj in range(N-1):
        # G=Ft[jj,].T@E
        
        Pf[:,:,jj,]=torch.sum(torch.reshape(En[:,jj],(1,1,N-1,1,1))*cFn,2)
        
        Pexfx0[:,:,jj,:,:,0]=En.T@Pf[:,:,jj,]
    Pexfx=-torch.sum(pre_condn@Pexfx0,5)
    # phial2=-torch.transpose(En.T@torch.transpose(phial,2,3),2,3)
    
    
    # phial1=En.T@phial2
    
    phial11=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    phial11[:,:,:,:,:,0]=phial
    
    
    phial00=(oden_data@phial11)[:,:,:,:,:,0]
    
    # alx2=torch.transpose(torch.matmul(Ed.T,torch.transpose(alx,2,3)),2,3)
    # aly2=torch.transpose(torch.matmul(Ed.T,torch.transpose(aly,2,3)),2,3)
    # alz2=torch.transpose(torch.matmul(Ed.T,torch.transpose(alz,2,3)),2,3)
    
    # alx1=torch.matmul(Ed.T,alx2)
    # aly1=torch.matmul(Ed.T,aly2)
    # alz1=torch.matmul(Ed.T,alz2)
    
    # alx0=torch.sum(ipre_cond@(alx1.reshape((ndata,ndt,N-1,N-1,N-1,1))),5)
    # aly0=torch.sum(ipre_cond@(aly1.reshape((ndata,ndt,N-1,N-1,N-1,1))),5)
    # alz0=torch.sum(ipre_cond@(alz1.reshape((ndata,ndt,N-1,N-1,N-1,1))),5)
    
    
    alx00=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    alx00[:,:,:,:,:,0]=alx
    
    aly00=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    aly00[:,:,:,:,:,0]=aly
    
    alz00=torch.empty((ndata,ndt,N-1,N-1,N-1,1)).to(device).double()
    alz00[:,:,:,:,:,0]=alz
    
    alxnew=(ode_data@alx00)[:,:,:,:,:,0]
    alynew=(ode_data@aly00)[:,:,:,:,:,0]
    alznew=(ode_data@alz00)[:,:,:,:,:,0]
    
    
    # al1=torch.linalg.solve(ode_data, exfx[0])
    # al0=torch.linalg.solve(ode_data, alxnew[0])
    # print(al_data.shape)
    # print(alx.shape)
    # qwe=abs(al1-al0)
    # qwe0=abs(al0-alx)
    # print(torch.max(qwe))
    # print(torch.max(qwe0))
    # input('ggg')
    # print(alxnew[1,0,2,:])
    # print(exfx[1,0,2,:])
    
    # print(torch.max(abs(alxnew[1,]-exfx[1,])))
    # print(torch.max(abs(alynew[1,]-exfy[1,])))
    # print(torch.max(abs(alznew[1,]-exfz[1,])))
    # print(torch.max(abs(phial00[1,]-Pexfx[1,])))
    # input('ggggg')
    # rr=0
    # for obj in gc.get_objects():
    #         try:
    #             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #                 print(type(obj), obj.size())
    #                 qwe=obj.element_size() * obj.nelement()/1024**2
    #                 # if qwe>100:
    #                 #     print(obj.nelement())
    #                 #     print(obj.size())
    #                 #     input('10000')
    #                 # print(qwe)
    #                 rr+=qwe
    #         except:
    #             pass
    # print(rr/1024)
    return alxnew,alynew,alznew,exfx,exfy,exfz,phial00,Pexfx