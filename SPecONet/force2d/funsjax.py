import numpy as np
from numpy.linalg import inv
# import torch
import pickle
import os
# import LG_1d as lg
from sem import sem as sem
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from pprint import pprint
from mpl_toolkits import mplot3d 
from pprint import pprint
# from net.data_loader import *
# from reconstruct import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
# import jax
# import jax.numpy as jnp
# from jax import jit
from functools import partial
# jax.config.update("jax_enable_x64", True)

def gen_lepolys(N, x):
    lepolys = {}
    for i in range(N+3): # OLD
    #for i in range(N+3):
        lepolys[i] = sem.lepoly(i, x)
    return lepolys

def gen_lepolysx(N, x, lepolys):
    def gen_diff_lepoly(N, n, x, lepolys):
        
        lepoly_x = np.zeros((N, 1))
        for i in range(n):
            if ((i+n) % 2) != 0:
                lepoly_x += (2*i+1)*lepolys[i]
        return lepoly_x    
    Dx = {}
    for i in range(N+1):
        temp = gen_diff_lepoly(N+1, i, x, lepolys)
        Dx[i] = temp.reshape(1, N+1)
    return Dx

def dxx(N, x, lepolys):
    def gen_diff2_lepoly(N, n, x,lepolys):
        lepoly_xx = np.zeros((N,1))
        
        for i in range(n-1):
            if ((i+n) % 2) == 0:
                
                lepoly_xx += (i+1/2)*(n*(n+1)-i*(i+1))*lepolys[i]
        return lepoly_xx
    Dxx = {}
    for i in range(N+1):
        Dxx[i] = gen_diff2_lepoly(N+1, i, x, lepolys).reshape(N+1,1)
    return Dxx

def basic_mat(b,N,op):
    m_diag=np.zeros((N-1,))
    m11_diag=np.zeros((N-1,))    
    s_diag = -(4*np.arange(N+1)+6)*b
    if op=='neumann':
        s_diag[0]=1
    for ii in range(1, N):
        k = ii - 1
        
        m_diag[ii-1] = 2*(1/(2*k+1)+b[k]**2/(2*k+5))/(s_diag[ii-1])
        m11_diag[ii-1]=b[k]*2/(2*k+5)/(s_diag[ii-1]*s_diag[ii+1])**.5
        
    m1_diag=m11_diag[0:N-3]
    
    M=  m_diag*np.eye(N-1)+np.diag(m1_diag,2)+np.diag(m1_diag,-2)
    # Mth=s_diag*np.eye(N-1)
    if op=='dirichlet':
        eie,eiv=np.linalg.eig(M)
       
    elif op=='neumann':
        # eie,eiv=np.linalg.eig(M[1:,1:])        
        eie,eiv=np.linalg.eig(M)        
    eiv[abs(eiv)<10**-10]=0
    return M,s_diag, eiv, eie

N=int(24-1)

x = sem.legslbndm(N+1)

b=-np.ones((N+1,))

bn=-np.arange(N+1)*np.arange(1,N+2)/(np.arange(2,N+3)*np.arange(3,N+4))

lepolys = gen_lepolys(N, x)

lepolysx = gen_lepolysx(N, x, lepolys)

lepolysxx = dxx(N, x,lepolys)


    
def exf2(t,m,x,y):
    
    num=len(m)
    
    NN=len(x)
    qq=np.arange(0,0+num)
    
    # m1=np.zeros((NN,NN,NN),dtype=np.complex_)
    # m1[:num,:num,:num]=m
    
    ik=1j*qq
    ikx=ik.reshape(1,num,1,1)*x.reshape(1,1,NN,NN)
    iky=ik.reshape(num,1,1,1)*y.reshape(1,1,NN,NN)

   
    ff=2*np.real(np.sum(np.sum(m.reshape(num,num,1,1)*(np.exp(ikx))*(np.exp(iky)),axis=0),axis=0))/NN
  
    
 

    ff=np.sin(t)*ff
   
    return ff
inte=np.reshape((lepolys[N])*(lepolys[N].T),(N+1,N+1))
# print(F.shape)



# qwe=np.sum(F*F/inte**2)*(2/(N*(N+1)))**3
def convx(FF,s_diag,b,N):
    # if op=='dirichlet':
    Ft=np.zeros((N-1,N-1,N-1))
    for ii in range(N-1):
        
        for jj in range(N-1):
            phixy=np.reshape((lepolys[ii] + b[ii]*lepolys[ii+2])/(s_diag[ii])**.5\
        *(lepolys[jj].T + b[jj]*lepolys[jj+2].T)/(s_diag[jj])**.5,(N+1,N+1,1))
            for kk in range(N-1):
                phi=phixy*np.reshape(lepolys[kk]+ b[kk]*lepolys[kk+2],(1,1,N+1))/(s_diag[kk])**.5
                qwe=np.sum(FF*phi/inte**2)*(2/(N*(N+1)))**3
                
                Ft[ii,jj,kk]=qwe
    return Ft

# @jit
def conv(FF,phi,lep):    
    B, _, _ = FF.shape
    i,j=phi.shape        
    P = np.zeros((B,1, i, j))
    P=((phi/lep**2))
    T = FF@ P
   
    PT1 = T.transpose(0, 2, 1)
    T=(2/((i-1)*i))**2*PT1@P
    # PT2 = T.transpose(0,1, 3, 2)
    
   
    return T.transpose(0,2,1)
    # return phi,lep
# @jit
def phiset(alp,phi):        
    B, _, _ = alp.shape
    i,j=phi.shape        
    
    T = (phi@alp)@phi.T 
    
   
    return T

def phixset(alp,phi,phix,dir):        
    B, i1, j1,k1 = alp.shape
    i,j=phi.shape        
    P = np.empty((B,1, j, i))
    Px = np.empty((B,1, j, i))
    P=(phi.T)
    Px=(phix.T)
    # P[:,:,:,:] = phi.T
    
    if dir=='x':
       
        T = alp@ Px
       
        PT1 = T.transpose(0,3, 2, 1)
        T=PT1@P
        
        PT2 = T.transpose(0,1, 3, 2)
        T=(PT2@P).transpose(0,2,3,1)
    elif dir=='y':
        
        T = alp@ P
        
        PT1 = T.transpose(0,3, 2, 1)
        T=PT1@Px
        
        PT2 = T.transpose(0,1, 3, 2)
        T=(PT2@P).transpose(0,2,3,1)
    elif dir=='z':
        
        T = alp@ P
        
        PT1 = T.transpose(0,3, 2, 1)
        T=PT1@P
        
        PT2 = T.transpose(0,1, 3, 2)
        T=(PT2@Px).transpose(0,2,3,1)
    
    return T

# @partial(jit, static_argnames=['Jnd','N'])
# @jit
def nonlinear(uu0,vv0,phi,lep,iMd,Mdxd):
   
    uu=uu0*uu0
    uv=uu0*vv0
   
    vv=vv0*vv0
   
    
    cuu=conv(uu,phi,lep)
    cuv=conv(uv,phi,lep)
    
    cvv=conv(vv,phi,lep)
   
    
    "uux+uvy+uwz"
    # cuux=0.5*np.transpose(Mdxd@(iMd@np.transpose(cuu,(0,2,1))),(0,2,1))
    cuux=Mdxd@(iMd@cuu)
    
    cuvy=np.transpose(Mdxd@(iMd@np.transpose(cuv,(0,2,1))),(0,2,1)) #second
    
    
    
    
    
    
    "vux+vvy+vwz"
    cuvx=Mdxd@(iMd@cuv)
    
    
    
    cvvy=np.transpose(Mdxd@(iMd@np.transpose(cvv,(0,2,1))),(0,2,1)) #second
    
  
    
    return cuux+cuvy, cvvy+cuvx
   