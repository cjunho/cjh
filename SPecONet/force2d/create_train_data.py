import numpy as np
from scipy.linalg import block_diag
import torch
import pickle
import os
import LG_1d as lg
from sem import sem as sem
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from pprint import pprint
from mpl_toolkits import mplot3d 
from pprint import pprint
from net.data_loader import *
from reconstruct import *

parser = argparse.ArgumentParser("SEM")
parser.add_argument("--equation", type=str, default='ConvDiff2D', choices=['Standard','Standard1','Standardb', 'Burgers', 'Helmholtz', 'Standard2D', 'ConvDiff2D']) #, 'BurgersT' 
parser.add_argument("--size", type=int, default=10000)
parser.add_argument("--N", type=int, default=15, choices=[int(2**i-1) for i in [4, 5, 6, 7, 8]]) 
parser.add_argument("--file", type=str, default='10000N15', help='Example: --file 2000N31') # 2^5-1, 2^6-1
parser.add_argument("--forcing", type=str, default='normal', choices=['normal', 'uniform','zero'])
parser.add_argument("--nbfuncs", type=int, default=1)
parser.add_argument("--A", type=float, default=0)
parser.add_argument("--F", type=float, default=0)
parser.add_argument("--U", type=float, default=1)
parser.add_argument("--WF", type=float, default=1) # 1 = include weaf form
parser.add_argument("--sd", type=float, default=1)
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--ndt", type=int, default=5)
parser.add_argument("--eps", type=float, default=1)
parser.add_argument("--kind", type=str, default='train', choices=['train', 'validate'])
parser.add_argument("--rand_eps", type=bool, default=False)
# parser.add_argument("--dt", type=float, default=0.01)

args = parser.parse_args()
gparams = args.__dict__
#pprint(gparams)
EQUATION = args.equation
epsilons = args.eps
EPSILON = epsilons


#GLOBALS
FILE = gparams['file']
dt = gparams['dt']
DATASET = int(FILE.split('N')[0])
SHAPE = int(FILE.split('N')[1]) + 1
NBFUNCS = int(gparams['nbfuncs'])
ndt = int(gparams['ndt'])+1

# FOLDER = f'{gparams["model"]}_{args.forcing}_epochs{EPOCHS}_{cur_time}'
# PATH = os.path.join('training', f"{EQUATION}", FILE, FOLDER)
# gparams['path'] = PATH
D_in = 1
device = get_device()

NORM = False
gparams['norm'] = False
transform_f = None

BATCH_SIZE,  D_out = DATASET, SHAPE
gparams['epsilon'] = EPSILON



# lg_dataset = get_data1(gparams, kind='train', transform_f=transform_f)
# trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=True)

# for sample_batch in trainloader:        
#         uu = sample_batch['u'].to(device)
N = args.N
x = sem.legslbndm(N+1)
# uu11=torch.load('tensor1.pt')

"nodes 32"
"data_new1=0.1*data_new2"
# with open('data_new1.npy', 'rb') as data_ex1:
#     uu11=np.load(data_ex1)
# with open('data_new2.npy', 'rb') as data_ex2:
#     uu12=np.load(data_ex2)

# "nodes 16"
# # with open('data20.npy', 'rb') as data_ex1:
# #     uu11=np.load(data_ex1)
# # with open('data21.npy', 'rb') as data_ex2:
# #     uu12=np.load(data_ex2)

uu1=np.zeros((DATASET,SHAPE))
# uu1[0:5000,:]=uu12
# uu1[5000:2*5000,:]=uu12
# uu1[2*5000:3*5000,:]=uu11[:,:,2]
# uu1[3*5000:4*5000,:]=uu11[:,:,3]
# uu1[4*5000:5*5000,:]=uu11[:,:,4]

# uu1=np.reshape(uu1,(DATASET,1,SHAPE))
# print(uu1.shape)
# input("fgs")
# uu1=uu11.detach().cpu().numpy()

equation = args.equation
size = args.size

epsilon = args.eps
eps_flag = args.rand_eps
kind = args.kind
sd = args.sd
forcing = args.forcing
dt=args.dt
# asd=load_obj('/Standard/train/2000N31sd1.0')
# uu=asd[:,0]
# input("creart")
# print(size,shape)
def gen_lepolys(N, x):
    lepolys = {}
    for i in range(N+1): # OLD
    #for i in range(N+3):
        lepolys[i] = sem.lepoly(i, x)
    return lepolys

# This is specifically for Standard2D
def dx(N, x, lepolys):
    def gen_diff_lepoly(N, n, x,lepolys):
        lepoly_x = np.zeros((N+1, 1))
        for i in range(n):
            if ((i+n) % 2) != 0:                
                lepoly_x += (2*i+1)*lepolys[i]
        return lepoly_x
    Dx = {}
    for i in range(N+1):
        
        Dx[i] = gen_diff_lepoly(N, i, x, lepolys).reshape(1, N+1)
    return Dx

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



# This is specifically for ConvDiff2D, from SandBoxJP3
def gen_lepolys_CD(N, x):
    lepolys = {}
    for i in range(N+3):
        lepolys[i] = lepoly(i, x)
    return lepolys

# This is specifically for ConvDiff2D, from SandBoxJP3
def gen_lepolysx_CD(N, x, lepolys):
    lepolysx = {}
    for i in range(N+3):
        lepolysx[i] = gen_diff_lepoly(N+1, i, x, lepolys).reshape(N+1, 1)
    return lepolysx

def gen_phi(N, x, lepolys):
    phi = {}
    for l in range(N): # OLD: N
        phi[l] = lepolys[l] - lepolys[l+2]
    return phi

def gen_phi_x(N, x, lepolysx):
    phi = {}
    for l in range(N): # OLD: N
        phi[l] = lepolysx[l] - lepolysx[l+2]
    return phi

def gen_phi_CD(N, x, lepolys):
    phi = {}
    for l in range(N-1): # OLD: N
        phi[l] = lepolys[l] - lepolys[l+2]
    return phi

def gen_phi_x_CD(N, x, lepolysx):
    phi = {}
    for l in range(N-1): # OLD: N
        phi[l] = lepolysx[l] - lepolysx[l+2]
    return phi


def func(x, equation, sd, forcing):
    if forcing == 'uniform':
        m = 3 + 2*np.random.rand(2)
        n = np.pi*(np.random.rand(2))
        f = m[0]*np.sin(n[0]*x) + m[1]*np.cos(n[1]*x)
        m = np.array([m[0], m[1], n[0], n[1]])
    elif forcing == 'normal':
        m = np.random.normal(0, sd, 4)
        f = m[0]*np.sin(m[1]*np.pi*x) + m[2]*np.cos(m[3]*np.pi*x)
    else: 
        m = np.zeros(4)
        f =np.zeros(x.shape) 
    return f, m


# Returing an extra 0???
# We replaced the m???
def func2D(x, y, equation, sd, forcing):
    # if forcing == 'uniform': # If change m/w in one, change other
    #     m = np.random.rand(2) + 1 # OLD: + 0
    #     w = np.random.rand(4)*(np.pi) # OLD: np.pi/2
    # elif forcing == 'normal': # If change m/w in one, change other
    #     m = np.random.normal(0, sd, 2) + 1
    #     w = np.random.normal(0, sd, 4)*(np.pi)

    # #return np.pi*np.cos(np.pi*x)*(np.cos(2*np.pi*y) - 1) + 4*(np.pi**2)*np.cos(2*np.pi*y)*np.sin(np.pi*x) + (np.pi**2)*np.sin(np.pi*x)*(np.cos(2*np.pi*y) - 1) - 4*np.pi*np.sin(np.pi*x)*np.sin(2*np.pi*y), m
    # # w = 2
    # f = m[0]*np.cos(w[0]*x + w[1]*y) + m[1]*np.sin(w[2]*x + w[3]*y)
    # #f = np.sin(x + y)
    # m = np.array([m[0], m[1], w[0], w[1], w[2], w[3]])
    m=np.zeros((5,))
    f=(x+1)**3*np.sin(np.pi*(1-y)*.5)+4*(x+1)*np.sin(3*np.pi*(1-y)*.5)
    return f, m


def standard(x, D, a, b,lepolysx, lepolys, epsilon, equation, sd, forcing,uu, f, params, s_diag):
    Mx = np.zeros((N-1,N-1))    
    m_diag=np.copy(s_diag)
    for ii in range(1, N):
        k = ii - 1
        phi_k_Mx = lepolysx[k] + a*lepolysx[k+1] + b*lepolysx[k+2]
        
        for jj in range(1,N):
            
                l = jj-1
                psi_l_M = lepolysx[l] + a*lepolysx[l+1] + b*lepolysx[l+2]
                Mx[jj-1,ii-1] = np.sum((x*psi_l_M.T*phi_k_Mx.T)*2/(N*(N+1))/(lepolys[N]**2))
    # print(Mx)
    with open(f'Mxx.npy', 'wb') as data_ex:
        np.save(data_ex, Mx)
    # k=3
    # l=3
    # phi_k_Mx = lepolysx[k].T + a*lepolysx[k+1].T + b*lepolysx[k+2].T
    # psi_l_M = lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2]
    # qwe = np.sum((x*psi_l_M*phi_k_Mx)*2/(N*(N+1))/(lepolys[N]**2))
    # print(x.shape,phi_k_Mx.shape,psi_l_M.shape)
    # print(qwe)
    input('dfgsfdg')
    for ii in range(1, N):
        k = ii - 1
        s_diag[ii-1] = -(4*k+6)*b
        m_diag[ii-1] = 2*(1/(2*k+1)+1/(2*k+5))        
        phi_k_Mx = D@(lepolys[k] + a*lepolys[k+1] + b*lepolys[k+2])
        
        for jj in range(1,N):
            if np.abs(ii-jj) <=2:
                l = jj-1
                psi_l_M = lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2]
                Mx[jj-1,ii-1] = np.sum((psi_l_M*phi_k_Mx)*2/(N*(N+1))/(lepolys[N]**2))
    
    # print(m1_diag.shape)
    # print(m11_diag.shape)
    S = s_diag*np.eye(N-1)
    # print((m_diag*np.eye(N-1)).shape)
    # print((np.diag(m1_diag,2)).shape)    
    g = np.zeros((N+1,))    
    fu=np.reshape(f,(N+1,1))
    for i in range(1,N+1):
        k = i - 1
        g[i-1] = (2*k+1)/(N*(N+1))*np.sum(fu*(lepolys[k])/(lepolys[N]**2))
    g[N-1] = 1/(N+1)*np.sum(fu/lepolys[N])

    bar_f = np.zeros((N-1,))
    for i in range(1,N):
        k = i-1
        bar_f[i-1] = g[i-1]/(k+1/2) + a*g[i]/(k+3/2) + b*g[i+1]/(k+5/2)

    Mass = epsilon*S-Mx
    u = np.linalg.solve(Mass, bar_f)
    alphas = np.copy(u)
    g[0], g[1] = u[0], u[1] + a*u[0]

    for i in range(3, N):
        k = i - 1
        g[i-1] = u[i-1] + a*u[i-2] + b*u[i-3]

    g[N-1] = a*u[N-2] + b*u[N-3]
    g[N] = b*u[N-2]
    u = np.zeros((N+1,))
    for i in range(1,N+2):
        _ = 0
        for j in range(1, N+2):
            k = j-1
            L = lepolys[k]
            _ += g[j-1]*L[i-1]
        u[i-1] = _[0]
    return u,uu, f, alphas,params

def standard1(x, D, a, b, lepolys, epsilon, equation, sd, forcing,uu, f, params, s_diag):
    # input("start standard1")
    
    Mx = np.zeros((N-1,N-1))
    M = np.zeros((N-1, N-1))
    m_diag=np.copy(s_diag)
    m11_diag=np.copy(s_diag)
    for ii in range(1, N):
        k = ii - 1
        s_diag[ii-1] = -(4*k+6)*b
        m_diag[ii-1] = 2*(1/(2*k+1)+1/(2*k+5))
        m11_diag[ii-1]=-2/(2*k+5)
        phi_k_Mx = D@(lepolys[k] + a*lepolys[k+1] + b*lepolys[k+2])
        
        for jj in range(1,N):
            if np.abs(ii-jj) <=2:
                l = jj-1
                psi_l_M = lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2]
                Mx[jj-1,ii-1] = np.sum((psi_l_M*phi_k_Mx)*2/(N*(N+1))/(lepolys[N]**2))
    m11_diag=np.reshape(m11_diag,(N-1,))
    m1_diag=m11_diag[0:N-3]
    # print(m1_diag.shape)
    # print(m11_diag.shape)
    S = s_diag*np.eye(N-1)
    # print((m_diag*np.eye(N-1)).shape)
    # print((np.diag(m1_diag,2)).shape)
    M=  m_diag*np.eye(N-1)+np.diag(m1_diag,2)+np.diag(m1_diag,-2)
    g = np.zeros((N+1,))    
    fu=np.reshape(f+uu/dt,(N+1,1))
    for i in range(1,N+1):
        k = i - 1
        g[i-1] = (2*k+1)/(N*(N+1))*np.sum(fu*(lepolys[k])/(lepolys[N]**2))
    g[N-1] = 1/(N+1)*np.sum(fu/lepolys[N])

    bar_f = np.zeros((N-1,))
    for i in range(1,N):
        k = i-1
        bar_f[i-1] = g[i-1]/(k+1/2) + a*g[i]/(k+3/2) + b*g[i+1]/(k+5/2)

    Mass = epsilon*S-Mx+M/dt
    u = np.linalg.solve(Mass, bar_f)
    alphas = np.copy(u)
    g[0], g[1] = u[0], u[1] + a*u[0]

    for i in range(3, N):
        k = i - 1
        g[i-1] = u[i-1] + a*u[i-2] + b*u[i-3]

    g[N-1] = a*u[N-2] + b*u[N-3]
    g[N] = b*u[N-2]
    u = np.zeros((N+1,))
    for i in range(1,N+2):
        _ = 0
        for j in range(1, N+2):
            k = j-1
            L = lepolys[k]
            _ += g[j-1]*L[i-1]
        u[i-1] = _[0]
    return u,uu, f, alphas,params

def int_exp(n,ep):
    if n==0:
        sol=ep*(-np.exp(-2/ep)+1)
    elif n==1: 
        sol=ep**2*((1-1/ep)-(1+1/ep)*np.exp(-2/ep))
    else:    
        aa0=ep*(-np.exp(-2/ep)+1)
        aa1=ep**2*((1-1/ep)-(1+1/ep)*np.exp(-2/ep))
        for ii in range(1,n):    
            sol=(2*ii+1)*ep*aa1+aa0
            aa0=aa1;aa1=sol
    
    return sol

def standardb(x, D, a, b, lepolys,lepolysx, eps, equation, sd, forcing,uu, f, params, s_diag):
    # input("start standard1")
    
    Mx = np.zeros((N-1,N-1))
    Mass=np.zeros((N,N))
    m_diag=np.copy(s_diag)
    m11_diag=np.copy(s_diag)
    cor=1 - np.exp(-(1+x)/eps)  - (1 - np.exp(-2/eps))*(x+1)*.5
    res=.5*(1-np.exp(-2/eps))
    a_12=np.zeros(N-1,)
    a_21=np.zeros(N-1,)
    # print(params)
    # input("sfgs")
    for ii in range(1, N):
        k = ii - 1
        s_diag[ii-1] = -(4*k+6)*b  
        phi_k_M = lepolys[k] + a*lepolys[k+1] + b*lepolys[k+2]
        phi_k_Mx = lepolysx[k] + a*lepolysx[k+1] + b*lepolysx[k+2]
        
        # phi_k_Mxx = lepolysxx[k] + a*lepolysxx[k+1] + b*lepolysxx[k+2]
        a_12[ii-1] = np.sum(res*phi_k_M*2/(N*(N+1))/(lepolys[N]**2));
        a_21[ii-1] = -res*np.sum( phi_k_M*2/(N*(N+1))/(lepolys[N]**2))\
            +2*(int_exp(k,eps)-int_exp(k+2,eps))/eps
        
        for jj in range(1,N):
            if np.abs(ii-jj) <=2:
                l = jj-1
                psi_l_M = lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2]
                Mx[jj-1,ii-1] = np.sum((psi_l_M*phi_k_Mx.T)*2/(N*(N+1))/(lepolys[N]**2))    
    # print(m1_diag.shape)
    # print(m11_diag.shape)
    a_22 =0.5*(-0.5*np.exp(-4/eps)+.5*(1-np.exp(-2/eps))**2+eps*np.exp(-2/eps)*(1-np.exp(-2/eps))
               +0.5-eps*(1-np.exp(-2/eps)))
    S = s_diag*np.eye(N-1)
    # print(Mx)
    # input("cvb")
    # print((m_diag*np.eye(N-1)).shape)
    # print((np.diag(m1_diag,2)).shape)
    
    g = np.zeros((N+1,))    
    fu=np.reshape(f,(N+1,1))
    
    for i in range(1,N+1):
        k = i - 1
        g[i-1] = (2*k+1)/(N*(N+1))*np.sum(fu*(lepolys[k])/(lepolys[N]**2))
    g[N] = 1/(N+1)*np.sum(fu/lepolys[N])

    bar_f = np.zeros((N,))
    for i in range(1,N):
        k = i-1
        bar_f[i-1] = g[i-1]/(k+1/2) + a*g[i]/(k+3/2) + b*g[i+1]/(k+5/2)
    
    Mass[0:N-1,0:N-1] = eps*S-Mx    
    Mass[0:N-1,N-1]=a_12
    Mass[N-1,0:N-1]=a_21
    Mass[N-1,N-1]=a_22
    # print(params.shape)
    # input("fsgs")
    m1=params[0]
    m2=params[1]
    w1=params[2]
    w2=params[3]
    # m1=4
    # m2=3
    # w1=3
    # w2=4
    q1=m2*(np.sin(w2)-np.sin(-w2))/w2;
    q2=-m2*(np.exp(-2/eps)*(-np.cos(w2)/eps+w2*np.sin(w2))-(-np.cos(-w2)/eps+w2*np.sin(-w2)))/(w2**2+1/eps**2);
    q3=-m1*(np.exp(-2/eps)*(-np.sin(w1)/eps-w1*np.cos(w1))-(-np.sin (-w1)/eps-w1*np.cos(-w1)))/(w1**2+1/eps**2);
    q5=m1*(np.sin(w1)-w1*np.cos(w1)-(np.sin(-w1)+w1*np.cos(-w1)))/w1**2;
    
    bar_f_end =q1*(1-.5*(1-np.exp(-2/eps)))+q2+q3-.5*(1-np.exp(-2/eps))*(q5)
    
    
    bar_f[N-1] = bar_f_end    

    
    u = np.linalg.solve(Mass, bar_f)
    alphas = np.copy(u)
    g[0], g[1] = u[0], u[1] + a*u[0]

    for i in range(3, N):
        k = i - 1
        g[i-1] = u[i-1] + a*u[i-2] + b*u[i-3]

    g[N-1] = a*u[N-2] + b*u[N-3]
    g[N] = b*u[N-2]
    u = np.zeros((N+1,))
    for i in range(1,N+2):
        _ = 0
        for j in range(1, N+2):
            k = j-1
            L = lepolys[k]
            _ += g[j-1]*L[i-1]
        u[i-1] = _[0]
    u=u+alphas[N-1]*np.reshape(cor,(N+1,))
    return u,uu, f, alphas,params,Mass


# -\Delta u + 1/dt*u = f + 1/dt*u_{n-1} = F
# >>> -\Delta u + 1/dt*u = F

def standard2D(x, D, a, b, lepolys, lepolysx, epsilon, equation, sd, forcing, f, params, s_diag):
    D2 = D@D
    D2 = D2[1:-1, 1:-1]
    D1 = D[1:-1, 1:-1]

    I = np.eye(x.shape[0]-2)

    L = np.kron(I, D2) + np.kron(D2, I)
    
    f_ = f[1:-1, 1:-1]
    f_ = f_.reshape(-1)
    
    u = np.linalg.solve(-L, f_)

    x1, y1 = np.meshgrid(x, x)
    xx, yy = np.meshgrid(x[1:-1], x[1:-1])
    uu = np.zeros_like(x1)
    
    # uu = np.zeros((u.shape[0]+2, u.shape[0]+2))
    uu[1:-1, 1:-1] = np.reshape(u, (x.shape[0]-2, x.shape[0]-2))



    alphas = 0
    return uu, f, alphas, params



def ConvDiff2D(x, D, a, b, lepolys, epsilon, equation, sd, forcing,uu, f, params, s_diag,Mxx,Mx,Mth):
    ff=np.zeros((N-1,N-1))
    Mass=block_diag(Mxx+Mx+Mth[0,0]*np.eye(N-1))
    for ii in range(1, N):
        k = ii - 1
        phi_k_M = lepolys[k] + a*lepolys[k+1] + b*lepolys[k+2]
        f1=f*phi_k_M.T
        f2=np.sum((f1)*2/(N*(N+1))/(lepolys[N].T**2),axis=1)
        if  ii<N-1:
            Mass=block_diag(Mass,Mxx+Mx+Mth[ii,ii]*np.eye(N-1))
        for jj in range(1,N):
            l=jj-1
            psi_l_M = lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2]
            f3=np.sum((f2*psi_l_M[:,0])*2/(N*(N+1))/(lepolys[N][0,]**2))
            
            ff[k,l]=f3
    bar_f=np.reshape(ff.T,((N-1)**2,1))
    
    alphas = np.linalg.solve(Mass, bar_f)
    return uu, f, alphas, params




def burgers(x, D, a, b, lepolys, epsilon, equation, sd, forcing,uu, f, params, s_diag):
    for ii in range(1, N):
        k = ii - 1
        s_diag[k] = -(4*k+6)*b
    S = s_diag*np.eye(N-1)
    Mass = epsilon*S
    error, tolerance, u_old, force = 1, 1E-9, 0*f.copy(), f.copy()
    iterations = 0
    while error > tolerance:        
        f_ = force - u_old*(D@u_old)
        # print(D.shape)
        # print(f_ .shape)
        # input("fdg")
        g = np.zeros((N+1,))
        for i in range(1,N+1):
            k = i-1
            g[k] = (2*k+1)/(N*(N+1))*np.sum(f_*(lepolys[k])/(lepolys[N]**2))
        g[N-1] = 1/(N+1)*np.sum(f_/lepolys[N])

        bar_f = np.zeros((N-1,))
        for i in range(1,N):
            k = i-1
            bar_f[k] = g[k]/(k+1/2) + a*g[k+1]/(k+3/2) + b*g[k+2]/(k+5/2)
        
        alphas = np.linalg.solve(Mass, bar_f)
        
        u_sol = np.zeros((N+1, 1))
        for ij in range(1, N):
            i_ind = ij - 1
            u_sol += alphas[i_ind]*(lepolys[i_ind] + a*lepolys[i_ind+1] + b*lepolys[i_ind+2])

        error = np.max(u_sol - u_old)
        u_old = u_sol
        iterations += 1
    u = np.reshape(u_sol,(N+1, ))
    
    return u,uu, f, alphas, params


def burgersT(x, D, a, b, lepolys, epsilon, equation, sd, forcing, f, params, s_diag):
    M = np.zeros((N-1, N-1))
    tol, T, dt = 1E-9, 5E-4, 1E-4
    t_f = int(T/dt)
    u_pre, u_ans, f_ans, alphas_ans = np.sin(np.pi*x), [], [], []
    for ii in range(1, N):
        k = ii - 1
        s_diag[k] = -(4*k + 6)*b
        phi_k_M = lepolys[k] + a*lepolys[k+1] + b*lepolys[k+2]
        for jj in range(1, N):
            if np.abs(ii-jj) <= 2:
                l = jj-1
                psi_l_M = lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2]
                entry = psi_l_M*phi_k_M*2/(N*(N+1))/(lepolys[N]**2)
                M[l, k] = np.sum(entry)

    S = s_diag*np.eye(N-1)
    Mass = epsilon*S + (1/dt)*M
    
    for t_idx in np.linspace(1, t_f, t_f, endpoint=True):
        error, tolerance, u_old, force = 1, tol, u_pre, np.cos(t_idx*dt)*f

        iterations = 0
        while error > tolerance:
            f_ = force - u_old*(D@u_old) + (1/dt)*u_pre
            g = np.zeros((N+1,))
            for i in range(1,N+1):
                k = i-1
                g[k] = (2*k+1)/(N*(N+1))*np.sum(f_*(lepolys[k])/(lepolys[N]**2))
            g[N-1] = 1/(N+1)*np.sum(f_/lepolys[N])

            bar_f = np.zeros((N-1,))
            for i in range(1,N):
                k = i-1
                bar_f[k] = g[k]/(k+1/2) + a*g[k+1]/(k+3/2) + b*g[k+2]/(k+5/2)

            alphas = np.linalg.solve(Mass, bar_f)
            u_sol = np.zeros((N+1, 1))
            for ij in range(1, N):
                i_ind = ij - 1
                u_sol += alphas[i_ind]*(lepolys[i_ind] + a*lepolys[i_ind+1] + b*lepolys[i_ind+2])

            error = np.max(u_sol - u_old)
            u_old = u_sol.copy()
            iterations += 1

        u_ans.append(u_sol)
        f_ans.append(force)
        alphas_ans.append(alphas)
        u_pre = u_sol
    u, f, alphas = u_ans, f_ans, alphas_ans
    return u, f, alphas, params


def helmholtz(x, D, a, b, lepolys, epsilon, equation, sd, forcing,uu, f, params, s_diag):
    ku = 3.5
    M = np.zeros((N-1, N-1))
    for ii in range(1, N):
        k = ii - 1
        s_diag[k] = -(4*k + 6)*b[k]
        phi_k_M = lepolys[k] + a[k]*lepolys[k+1] + b[k]*lepolys[k+2]
        for jj in range(1, N):
            if np.abs(ii-jj) <= 2:
                l = jj-1
                psi_l_M = lepolys[l] + a[l]*lepolys[l+1] + b[l]*lepolys[l+2]
                entry = psi_l_M*phi_k_M*2/(N*(N+1))/(lepolys[N]**2)
                M[l, k] = np.sum(entry)

    S = s_diag*np.eye(N-1)
    g = np.zeros((N+1,))
    for i in range(1,N+1):
        k = i-1
        g[k] = (2*k+1)/(N*(N+1))*np.sum(f*(lepolys[k])/(lepolys[N]**2))
    g[N] = 1/(N+1)*np.sum(f/lepolys[N])

    bar_f = np.zeros((N-1,))
    for i in range(1,N):
        k = i-1
        bar_f[k] = g[k]/(k+1/2) + a[k]*g[k+1]/(k+3/2) + b[k]*g[k+2]/(k+5/2)

    Mass = -S + ku*M
    alphas = np.linalg.solve(Mass, bar_f)

    u = np.zeros((N+1, 1))
    for ij in range(1,N):
        i_ind = ij-1
        u += alphas[i_ind]*(lepolys[i_ind] + a[i_ind]*lepolys[i_ind+1] + b[i_ind]*lepolys[i_ind+2])
    u=np.reshape(u,(N+1,))
    return u,uu, f, alphas, params


def generate(x,uu, D, a, b,lepolysx, lepolys, epsilon, equation, sd, forcing,Mxx,Mx,Mth):
    # input("gener")
    if equation in ('Standard2D', 'ConvDiff2D'):
        x1, y1 = np.meshgrid(x, x)
        f, params = func2D(x1, y1, equation, sd, forcing)
    else:
        f, params = func(x, equation, sd, forcing)
    s_diag = np.zeros((N-1,1))
    if equation == 'Standard':
        u,uu, f, alphas, params = standard(x, D, a, b,lepolysx, lepolys, epsilon, equation, sd, forcing,uu, f, params, s_diag)
    elif equation == 'Standard1':
        u,uu, f, alphas, params = standard1(x, D, a, b, lepolys, epsilon, equation, sd, forcing,uu, f, params, s_diag)
    elif equation == 'Standardb':
        u,uu, f, alphas, params = standardb(x, D, a, b, lepolys,lepolysx, epsilon, equation, sd, forcing,uu, f, params, s_diag)
    elif equation == 'Standard2D':
        u, f, alphas, params = standard2D(x, D, a, b, lepolys, lepolysx, epsilon, equation, sd, forcing, f, params, s_diag)
        # plot3D(u)
    elif equation == 'ConvDiff2D':
        u, f, alphas, params = ConvDiff2D(x, D, a, b, lepolys, epsilon, equation, sd, forcing,uu, f, params, s_diag,Mxx,Mx,Mth)
        # plot3D(u)
    elif equation == 'Burgers':
       
        u,uu, f, alphas, params = burgers(x, D, a, b, lepolys, epsilon, equation, sd, forcing,uu, f, params, s_diag)
    elif equation == 'BurgersT':
        u, f, alphas, params = burgersT(x, D, a, b, lepolys, epsilon, equation, sd, forcing, f, params, s_diag)        
    elif equation == 'Helmholtz':
        u,uu, f, alphas, params = helmholtz(x, D, a, b, lepolys, epsilon, equation, sd, forcing,uu, f, params, s_diag)
    return u,uu, f, alphas, params


def plot3D(u):
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    
    # u = np.reshape(u, (N-1, N-1))
    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, x)
    # Z = 0*X.copy()
    # Z[1:-1,1:-1] = u
    Z = u
    # print(u[0,:])
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

# OLD: equation='Standard'
def create_fast(N, epsilon, size, eps_flag=False, equation='Standard', sd=1, forcing='uniform'):
    if equation == 'Helmholtz':
        a, b = np.zeros((N+1,)), np.zeros((N+1,))
        for i in range(1, N+2):
            k = i-1
            b[k] = -k*(k+1)/((k+2)*(k+3))
    else:
        a, b = 0, -1
    return loop(N, epsilon, size, lepolysx,lepolys, eps_flag, equation, a, b, forcing)

# print(uu1.shape)
# input("sfdgs")
def loop(N, epsilon, size,lepolysx, lepolys, eps_flag, equation, a, b, forcing):
    # input("loop")
    if eps_flag == True:
        epsilons = np.random.uniform(1E0, 1E-6, size)
    data = []
    U, F, ALPHAS, PARAMS = [],[], [], []
    Mxx = np.zeros((N-1,N-1))
    Mx = np.zeros((N-1,N-1))
    s_diag=np.zeros((N-1,1))
    for ii in range(1, N):
        k = ii - 1
        phi_k_Mx = lepolysx[k] + a*lepolysx[k+1] + b*lepolysx[k+2]
        s_diag[ii-1] = -(4*k+6)*b
        for jj in range(1,N):
                l = jj-1
                psix_l_M = lepolysx[l] + a*lepolysx[l+1] + b*lepolysx[l+2]
                psi_l_M = lepolys[l] + a*lepolys[l+1] + b*lepolys[l+2]
                Mxx[jj-1,ii-1] = np.sum(((x+1)*psix_l_M.T*phi_k_Mx.T)*2/(N*(N+1))/(lepolys[N]**2))
                Mx[jj-1,ii-1] = np.sum(((x+1)*psi_l_M*phi_k_Mx.T)*2/(N*(N+1))/(lepolys[N]**2))
    Mxx[abs(Mxx)<10**-10]=0
    Mx[abs(Mx)<10**-10]=0
    Mxx*=8
    Mx*=-4
    Mth=64*np.pi**2*s_diag*np.eye(N-1)
    for n in tqdm(range(size)):
        if equation  in ('Standard2D', 'ConvDiff2D'):
            data_uu=np.zeros((D_out,D_out,ndt))
            uu=0
        else:        
            data_uu=np.zeros((D_out,ndt))
            # uu=np.reshape(uu1[n,0,:],(N+1,1))
            data_uu[:,0]=uu1[n,:]
            uu=np.reshape(uu1[n,:],(N+1,1))
        
        # print(uu.shape)
        # input("fdgs")
        # uu=uu1
        for kk in range(1,ndt):
            if eps_flag == True:
                epsilon = epsilons[n]
            if equation == 'Standard':
                u,uu, f, alphas, params = generate(x,uu, D, a, b,lepolysx, lepolys, epsilon, equation, sd, forcing)
            elif equation == 'Standardb':
                u,uu, f, alphas, params = generate(x,uu, D, a, b, lepolys, epsilon, equation, sd, forcing)    
            elif equation == 'Standard1':
                u,uu, f, alphas, params = generate(x,uu, D, a, b, lepolys, epsilon, equation, sd, forcing)
            elif equation == 'Helmholtz':
                u,uu, f, alphas, params = generate(x,uu, D, a, b, lepolys, epsilon, equation, sd, forcing)
            elif equation == 'Standard2D':
                u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd, forcing)
            elif equation == 'ConvDiff2D':
                u,uu, f, alphas, params = generate(x,uu, D, a, b, lepolys, epsilon, equation, sd, forcing,Mxx,Mx,Mth)
            elif equation == 'BurgersT':
                u, f, alphas, params = generate(x, D, a, b, lepolys, epsilon, equation, sd, forcing)
                for i, u_ in enumerate(u):
                    if i < len(u):
                        data.append([u[i],uu[i], f[i], alphas[i], params, epsilon])
            elif equation == 'Burgers':
               
                u,uu, f, alphas, params = generate(x,uu, D, a, b, lepolys, epsilon, equation, sd, forcing)
                
            else:
                u,uu, f, alphas, params = generate(x,uu, D, a, b, lepolys, epsilon, equation, sd, forcing)
            if equation  in ('Standard2D', 'ConvDiff2D'):
                data_uu[:,:,kk]=u
            else:
                data_uu[:,kk]=u
                uu=np.reshape(u,(SHAPE,1))
                
            # print("uu",uu.shape)
       
        data.append([data_uu, f, alphas, params, epsilon])
    return data,Mxx,Mx,Mth



D = sem.legslbdiff(N+1, x)
lepolys = gen_lepolys(N, x)
lepolysx = dx(N, x, lepolys)

# print(f.shape)
# pprint(lepolys)
if equation in ('Standard2D', 'ConvDiff2D'):
    lepolysx = gen_lepolysx(N, x, lepolys)
# pprint(lepolysx)
# input('start creat_f')
data,Mxx0,Mx0,Mth0 = create_fast(N, epsilon, size, eps_flag, equation, sd, forcing)

data = np.array(data, dtype=object)



def save_obj(data, name, equation, kind):
    cwd = os.getcwd()
    path = os.path.join(cwd,'data', equation, kind)
   
    if os.path.isdir(path) == False:
        os.makedirs(f'data/{equation}{epsilon}/{kind}')
    with open(f'Mth.npy', 'wb') as data_ex:
        np.save(data_ex, Mth)
    with open(f'Mxx.npy', 'wb') as data_ex:
        np.save(data_ex, Mxx)
    with open(f'Mx.npy', 'wb') as data_ex:
        np.save(data_ex, Mx)
    with open(f'data/{equation}{epsilon}/{kind}/'+ name + '.pkl', 'wb') as f:
        
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
       

if forcing == 'normal':
    save_obj(data, f'{size}N{N}sd{sd}', equation, kind)
elif forcing == 'uniform':
    save_obj(data, f'{size}N{N}uniform', equation, kind)
else:save_obj(data, f'{size}N{N}zero', equation, kind)

