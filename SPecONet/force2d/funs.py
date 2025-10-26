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

# import jax.numpy as jnp
# from jax import jit

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

def ini(t,x,y,z):
    
    uf=np.sin(t)*(np.cos(np.pi*x)+1)*np.sin(np.pi*y)*np.sin(np.pi*z)
    vf=-np.cos(t)*np.sin(np.pi*x)*(np.cos(np.pi*y)+1)*np.sin(np.pi*z)
    wf=-(np.sin(t)-np.cos(t))*np.sin(np.pi*x)*np.sin(np.pi*y)*(np.cos(np.pi*z)+1)
    px=-np.pi*np.sin(t)*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z)
    py=np.pi*np.sin(t)*np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
    pz=-np.pi*np.sin(t)*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    return uf,vf,wf,px,py,pz

def exf(t,x,y,z,dt,eps):
    fx=np.cos(t)*(np.cos(np.pi*x)+1)*np.sin(np.pi*y)*np.sin(np.pi*z)\
        +eps*np.pi**2*np.sin(t)*(np.cos(np.pi*x))*np.sin(np.pi*y)*np.sin(np.pi*z)\
        +eps*np.pi**2*np.sin(t)*(np.cos(np.pi*x)+1)*np.sin(np.pi*y)*np.sin(np.pi*z)\
        +eps*np.pi**2*np.sin(t)*(np.cos(np.pi*x)+1)*np.sin(np.pi*y)*np.sin(np.pi*z)\
        -np.pi*np.sin(t)*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z)
    # fx=np.pi*np.cos(t)*np.sin(np.pi*x)**2*np.sin(2*np.pi*y)*np.sin(np.pi*z)\
    #         -2*np.pi**3*np.sin(t)*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)*np.sin(np.pi*z)\
    #         +4*np.pi**3*np.sin(t)*np.sin(np.pi*x)**2*np.sin(2*np.pi*y)*np.sin(np.pi*z)\
    #         +np.pi**3*np.sin(t)*np.sin(np.pi*x)**2*np.sin(2*np.pi*y)*np.sin(np.pi*z)
        
    fy=np.sin(t)*np.sin(np.pi*x)*(np.cos(np.pi*y)+1)*np.sin(np.pi*z)\
        -eps*np.pi**2*np.cos(t)*np.sin(np.pi*x)*(np.cos(np.pi*y)+1)*np.sin(np.pi*z)\
        -eps*np.pi**2*np.cos(t)*np.sin(np.pi*x)*(np.cos(np.pi*y))*np.sin(np.pi*z)\
        -eps*np.pi**2*np.cos(t)*np.sin(np.pi*x)*(np.cos(np.pi*y)+1)*np.sin(np.pi*z)\
        +np.pi*np.sin(t)*np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
    # fy=-np.pi*np.cos(t)*np.sin(2*np.pi*x)*np.sin(np.pi*y)**2*np.sin(np.pi*z)\
    #     -4*np.pi**3*np.sin(t)*np.sin(np.pi*y)**2*np.sin(2*np.pi*x)*np.sin(np.pi*z)\
    #     +2*np.pi**3*np.sin(t)*np.cos(2*np.pi*y)*np.sin(2*np.pi*x)*np.sin(np.pi*z)\
    #     -np.pi**3*np.sin(t)*np.sin(2*np.pi*x)*np.sin(np.pi*y)**2*np.sin(np.pi*z)
    fz=-(np.cos(t)+np.sin(t))*np.sin(np.pi*x)*np.sin(np.pi*y)*(np.cos(np.pi*z)+1)\
        -eps*np.pi**2*(np.sin(t)-np.cos(t))*np.sin(np.pi*x)*np.sin(np.pi*y)*(np.cos(np.pi*z)+1)\
            -eps*np.pi**2*(np.sin(t)-np.cos(t))*np.sin(np.pi*x)*np.sin(np.pi*y)*(np.cos(np.pi*z)+1)\
                -eps*np.pi**2*(np.sin(t)-np.cos(t))*np.sin(np.pi*x)*np.sin(np.pi*y)*(np.cos(np.pi*z))\
                    -np.pi*np.sin(t)*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    return fx,fy,fz

def exf1(t,n,m,x,y,z,dt):
    # m=np.random.rand((9))-.5
    # ff=np.sin(t)*(m[0]*np.sin(m[1]*x)+m[2])*(m[3]*np.sin(m[4]*y)+m[5])*(m[6]*np.sin(m[7]*z)+m[8])
    # ff=np.sin(t)*(np.sin(10*n[0]*x+np.pi*(m[0]))+(m[1]-.5))*(np.sin(10*n[1]*y+np.pi*(m[2]))+(m[3]-.5))*(np.sin(10*n[2]*z+np.pi*(m[4]))+(m[5]-.5))
    # ff=np.sin(t)*n*(lepolys[m[0]]-lepolys[m[0]+2])*(lepolys[m[1]].T-lepolys[m[1]+2].T)*(lepolys[m[2]]-lepolys[m[2]+2]).reshape((1,1,N+1))
    # ff=np.sin(t)*(np.sin(np.pi*x))*(np.sin(np.pi*y))*(np.sin(np.pi*z))
    ff=np.sin(t)*((lepolys[m[0]])*(lepolys[m[1]].T)).reshape((N+1,N+1,1))*(lepolys[m[2]]).reshape((1,1,N+1))
    # ff=np.sin(t)*((lepolys[m]-lepolys[m+2])*(lepolys[m].T-lepolys[m+2].T))*((lepolys[m]-lepolys[m+2]).reshape((1,1,N+1)))
    # ff=np.sin(t)*(lepolys[m].T-lepolys[m+2].T)
    return ff
    

inte=np.reshape((lepolys[N])\
    *(lepolys[N].T),(N+1,N+1,1))\
    *np.reshape(lepolys[N],(1,1,N+1))

# inte=np.reshape((lepolys[N])\
#     *(lepolys[N].T),(N+1,N+1,1))
# print(F.shape)

def exf2(t,m,x,y,z):
    
    num=len(m)
    
    NN=len(x)
    qq=np.arange(0,num)
    
    
    # m1=np.zeros((NN,NN,NN),dtype=np.complex_)
    # m1[:num,:num,:num]=m
    
    ik=1j*qq
    ikx=ik.reshape(1,1,num,1,1,1)*x.reshape(1,1,1,NN,NN,NN)
    iky=ik.reshape(1,num,1,1,1,1)*y.reshape(1,1,1,NN,NN,NN)
    ikz=ik.reshape(num,1,1,1,1,1)*z.reshape(1,1,1,NN,NN,NN)
   
    # qa=np.real(np.sum(m.reshape(num,num,num,1,1,1)*np.exp(ikx)*np.exp(iky)*np.exp(ikz),axis=0))/NN**3
    
    
    ff=np.real(np.sum(np.sum(np.sum(m.reshape(num,num,num,1,1,1)*(np.exp(ikx))*(np.exp(iky))*(np.exp(ikz)),axis=0),axis=0),axis=0))/NN**3
    
    ff=np.sin(t)*ff
   
    return ff


# qwe=np.sum(F*F/inte**2)*(2/(N*(N+1)))**3
def convx(FF,s_diag,b,N):
    # if op=='dirichlet':
    Ft=np.zeros((N-1,N-1,N-1))
    for ii in range(N-1):
        
        for jj in range(N-1):
            phixy=np.reshape((lepolysx[ii].T + b[ii]*lepolysx[ii+2].T)/(s_diag[ii])**.5\
        *(lepolys[jj].T + b[jj]*lepolys[jj+2].T)/(s_diag[jj])**.5,(N+1,N+1,1))
            for kk in range(N-1):
                phi=phixy*np.reshape(lepolys[kk]+ b[kk]*lepolys[kk+2],(1,1,N+1))/(s_diag[kk])**.5
                qwe=np.sum(FF*phi/inte**2)*(2/(N*(N+1)))**3
                
                Ft[ii,jj,kk]=qwe
    return Ft


# def conv(FF,s_diag,b,N,op):
#     if op=='part':
#         Ft=np.zeros((N-1,N-1,N-1)) 
#         for ii in range(N-1):
            
#             for jj in range(N-1):
#                 phixy=np.reshape((lepolys[ii] + b[ii]*lepolys[ii+2])/(s_diag[ii])**.5\
#             *(lepolys[jj].T + b[jj]*lepolys[jj+2].T)/(s_diag[jj])**.5,(N+1,N+1,1))
#                 for kk in range(N-1):
#                     phi=phixy*np.reshape(lepolys[kk]+ b[kk]*lepolys[kk+2],(1,1,N+1))/(s_diag[kk])**.5
                    
#                     # qwe=np.sum(np.sum(np.sum(FF*phi/inte**2,axis=1),axis=1),axis=1)*(2/(N*(N+1)))**3
#                     qwe=np.sum(FF*phi/inte**2)*(2/(N*(N+1)))**3
                    
                    
#                     Ft[ii,jj,kk]=qwe
        
        
#     if op=='all':
#         Jnd,_,_,_=FF.shape
#         # FF=np.transpose(FF,(3,0,1,2))
#         Ft=np.zeros((Jnd,N-1,N-1,N-1)) 
#         for ii in range(N-1):
            
#             for jj in range(N-1):
#                 phixy=np.reshape((lepolys[ii] + b[ii]*lepolys[ii+2])/(s_diag[ii])**.5\
#             *(lepolys[jj].T + b[jj]*lepolys[jj+2].T)/(s_diag[jj])**.5,(N+1,N+1,1))
#                 for kk in range(N-1):
#                     phi=phixy*np.reshape(lepolys[kk]+ b[kk]*lepolys[kk+2],(1,1,N+1))/(s_diag[kk])**.5
                    
#                     qwe=np.sum(np.sum(np.sum(FF*phi/inte**2,axis=1),axis=1),axis=1)*(2/(N*(N+1)))**3
#                     # qwe=np.sum(FF*phi/inte**2)*(2/(N*(N+1)))**3
                    
                    
#                     Ft[:,ii,jj,kk]=qwe
   
#     # elif op=='neumann':
#     #     Ft=np.zeros((N-1,N-1,N-1))
#     #     for ii in range(1,N-1):
            
#     #         for jj in range(N-1):
                
#     #             phixy=np.reshape((lepolys[ii]+  b[ii]*lepolys[ii+2])/(s_diag[ii])**.5\
#     #         *(lepolys[jj].T+ b[jj]*lepolys[jj+2].T)/(s_diag[jj])**.5,(N+1,N+1,1))
#     #             for kk in range(N-1):
                    
#     #                 phi=phixy*np.reshape(lepolys[kk]+ b[kk]*lepolys[kk+2],(1,1,N+1))/(s_diag[kk])**.5
#     #                 qwe=np.sum(FF*phi/inte**2)*(2/(N*(N+1)))**3
                    
#     #                 Ft[ii,jj,kk]=qwe
#     return Ft
# @jit
def conv(FF,phi,lep):    
    B, i1, j1,k1 = FF.shape
    i,j=phi.shape        
    P = np.zeros((B,1, i, j))
    P[:,:,:,:]=((phi/lep**2))
    T = FF@ P
   
    PT1 = T.transpose(0,3, 2, 1)
    T=PT1@P
    PT2 = T.transpose(0,1, 3, 2)
    T=(2/((i-1)*i))**3*(PT2@P).transpose(0,1, 3, 2)
   
    return T.transpose(0,3,2,1)

def phiset(s_diag, b,op):
        if op=='1d':
            phiset=np.zeros((N-1,1),dtype=object)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolys[k]  + b[k]*lepolys[k+2])/(s_diag[k])**.5
                phiset[k,0]=phi_k_M
        elif op=='dirichlet':
            phiset=np.zeros((N-1,N-1,N-1),dtype=object)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolys[k]  + b[k]*lepolys[k+2])/(s_diag[k])**.5
                
                for jj in range(1,N):
                    l=jj-1
                    psi_l_M = (lepolys[l]  + b[l]*lepolys[l+2])/(s_diag[l])**.5
                    for kk in range(1,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolys[m]  + b[m]*lepolys[m+2])/(s_diag[m])**.5
                        
                        phiset[k,l,m]=np.reshape(phi_k_M*psi_l_M.T,(N+1,N+1,1))*np.reshape(ome_l_M,(1,1,N+1))
                        # phiset[k,l,m]=np.reshape(psi_l_M,(N+1,1,1))
                        # phiset[k,l,m]=ome_l_M
        elif op=='dirichlet0':
            phiset=np.zeros((N-1,N-1,N-1),dtype=object)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolys[k]  + b[k]*lepolys[k+2])/(s_diag[k])**.5
                
                for jj in range(1,N):
                    l=jj-1
                    psi_l_M = (lepolys[l]  + b[l]*lepolys[l+2])/(s_diag[l])**.5
                    for kk in range(1,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolys[m]  + b[m]*lepolys[m+2])/(s_diag[m])**.5
                        
                        # phiset[k,l,m]=np.reshape(phi_k_M*psi_l_M.T,(N+1,N+1,1))*np.reshape(ome_l_M,(1,1,N+1))
                        phiset[k,l,m]=np.reshape(phi_k_M,(N+1,1,1))
                        # phiset[k,l,m]=ome_l_M
        elif op=='dirichlet1':
            phiset=np.zeros((N-1,N-1,N-1),dtype=object)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolys[k]  + b[k]*lepolys[k+2])/(s_diag[k])**.5
                
                for jj in range(1,N):
                    l=jj-1
                    psi_l_M = (lepolys[l]  + b[l]*lepolys[l+2])/(s_diag[l])**.5
                    for kk in range(1,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolys[m]  + b[m]*lepolys[m+2])/(s_diag[m])**.5
                        
                        phiset[k,l,m]=np.reshape(psi_l_M,(1,N+1,1))
                        # phiset[k,l,m]=ome_l_M
        elif op=='dirichlet2':
            phiset=np.zeros((N-1,N-1,N-1),dtype=object)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolys[k]  + b[k]*lepolys[k+2])/(s_diag[k])**.5
                
                for jj in range(1,N):
                    l=jj-1
                    psi_l_M = (lepolys[l]  + b[l]*lepolys[l+2])/(s_diag[l])**.5
                    for kk in range(1,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolys[m]  + b[m]*lepolys[m+2])/(s_diag[m])**.5
                        
                        phiset[k,l,m]=np.reshape(ome_l_M,(1,1,N+1))
                        # phiset[k,l,m]=ome_l_M
        elif op=='dirichletx':
            phiset=np.zeros((N-1,N-1,N-1),dtype=object)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolysx[k].T  + b[k]*lepolysx[k+2].T)/(s_diag[k])**.5
                
                for jj in range(1,N):
                    l=jj-1
                    psi_l_M = (lepolys[l]  + b[l]*lepolys[l+2])/(s_diag[l])**.5
                    for kk in range(1,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolys[m]  + b[m]*lepolys[m+2])/(s_diag[m])**.5
                        
                        phiset[k,l,m]=np.reshape(phi_k_M*psi_l_M.T,(N+1,N+1,1))*np.reshape(ome_l_M,(1,1,N+1))
        elif op=='dirichlety':
            phiset=np.zeros((N-1,N-1,N-1),dtype=object)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolys[k]  + b[k]*lepolys[k+2])/(s_diag[k])**.5
                
                for jj in range(1,N):
                    l=jj-1
                    psi_l_M = (lepolysx[l].T  + b[l]*lepolysx[l+2].T)/(s_diag[l])**.5
                    for kk in range(1,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolys[m]  + b[m]*lepolys[m+2])/(s_diag[m])**.5
                        
                        phiset[k,l,m]=np.reshape(phi_k_M*psi_l_M.T,(N+1,N+1,1))*np.reshape(ome_l_M,(1,1,N+1))
        elif op=='dirichletz':
            phiset=np.zeros((N-1,N-1,N-1),dtype=object)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolys[k]  + b[k]*lepolys[k+2])/(s_diag[k])**.5
                
                for jj in range(1,N):
                    l=jj-1
                    psi_l_M = (lepolys[l]  + b[l]*lepolys[l+2])/(s_diag[l])**.5
                    for kk in range(1,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolysx[m].T  + b[m]*lepolysx[m+2].T)/(s_diag[m])**.5
                        
                        phiset[k,l,m]=np.reshape(phi_k_M*psi_l_M.T,(N+1,N+1,1))*np.reshape(ome_l_M,(1,1,N+1))
        elif op=='mix':
            phiset=np.zeros((N-1,N-1,N-1),dtype=object)
            s0_diag = (4*np.arange(N+1)+6)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolys[k]  -lepolys[k+2])/(s0_diag[k])**.5
                
                for jj in range(1,N):
                    l=jj-1
                    psi_l_M = (lepolys[l]  -lepolys[l+2])/(s0_diag[l])**.5
                    for kk in range(1,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolys[m]  + b[m]*lepolys[m+2])/(s_diag[m])**.5
                        
                        phiset[k,l,m]=np.reshape(phi_k_M*psi_l_M.T,(N+1,N+1,1))*np.reshape(ome_l_M,(1,1,N+1))
        elif op=='neumann':
            phiset=np.zeros((N-1,N-1,N-1),dtype=object)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolys[k]  + b[k]*lepolys[k+2])/(s_diag[k])**.5
                
                for jj in range(1,N):
                    l=jj-1
                    psi_l_M = (lepolys[l]  + b[l]*lepolys[l+2])/(s_diag[l])**.5
                    for kk in range(1,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolys[m]  + b[m]*lepolys[m+2])/(s_diag[m])**.5
                        
                        phiset[k,l,m]=np.reshape(phi_k_M*psi_l_M.T,(N+1,N+1,1))*np.reshape(ome_l_M,(1,1,N+1))
        elif op=='neumannxx':
            phiset=np.zeros((N-2,N-2,N-2),dtype=object)
            for ii in range(2, N):
                k = ii - 1
                phi_k_M = (lepolysxx[k] + b[k]*lepolysxx[k+2])/(s_diag[k])**.5
                
                for jj in range(2,N):
                    l=jj-1
                    psi_l_M = (lepolys[l]  + b[l]*lepolys[l+2])/(s_diag[l])**.5
                    for kk in range(2,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolys[m]  + b[m]*lepolys[m+2])/(s_diag[m])**.5
                        
                        phiset[k-1,l-1,m-1]=np.reshape(phi_k_M*psi_l_M.T,(N+1,N+1,1))*np.reshape(ome_l_M,(1,1,N+1))
        
        elif op=='neumannx':
            phiset=np.zeros((N-1,N-1,N-1),dtype=object)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolysx[k].T  + b[k]*lepolysx[k+2].T)/(s_diag[k])**.5
                
                for jj in range(1,N):
                    l=jj-1
                    psi_l_M = (lepolys[l]  + b[l]*lepolys[l+2])/(s_diag[l])**.5
                    for kk in range(1,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolys[m]  + b[m]*lepolys[m+2])/(s_diag[m])**.5
                        
                        phiset[k,l,m]=np.reshape(phi_k_M*psi_l_M.T,(N+1,N+1,1))*np.reshape(ome_l_M,(1,1,N+1))
        elif op=='neumanny':
            phiset=np.zeros((N-1,N-1,N-1),dtype=object)
            for ii in range(1, N):
                k = ii - 1
                phi_k_M = (lepolys[k]  + b[k]*lepolys[k+2])/(s_diag[k])**.5
                
                for jj in range(1,N):
                    l=jj-1
                    psi_l_M = (lepolysx[l].T  + b[l]*lepolysx[l+2].T)/(s_diag[l])**.5
                    for kk in range(1,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolys[m]  + b[m]*lepolys[m+2])/(s_diag[m])**.5
                        phiset[k,l,m]=np.reshape(phi_k_M*psi_l_M.T,(N+1,N+1,1))*np.reshape(ome_l_M,(1,1,N+1))
        elif op=='neumannz':
            phiset=np.zeros((N-2,N-2,N-2),dtype=object)
            for ii in range(2, N):
                k = ii - 1
                phi_k_M = (lepolys[k]  + b[k]*lepolys[k+2])/(s_diag[k])**.5
                
                for jj in range(2,N):
                    l=jj-1
                    psi_l_M = (lepolys[l]  + b[l]*lepolys[l+2])/(s_diag[l])**.5
                    for kk in range(2,N):
                        m=kk-1
                    # phiset[(N-1)*l+k,:,:]=phi_k_M.T*psi_l_M
                        ome_l_M = (lepolysx[m].T  + b[m]*lepolysx[m+2].T)/(s_diag[m])**.5
                        
                        phiset[k-1,l-1,m-1]=np.reshape(phi_k_M*psi_l_M.T,(N+1,N+1,1))*np.reshape(ome_l_M,(1,1,N+1))
    
        return phiset
