from firedrake import *

def scaling_to_nondim(t0,ds,tan0,tildeA,Ly,H0,ep):
    t0 = t0/(H0/9.8/ep)**.5                 # initial time in seconds
    dt = ds/(H0/9.8/ep)**.5
    tan0 = tan0                             # angle 
    tildeA = tildeA/((4/3)**(1/3)*H0*ep)    # amplitude
    Ly = Ly/(H0/ep**.5)
    
    return(t0,dt,tan0,tildeA,Ly)

def scaling_to_dim(eta01,eta0,H0,ep):
    eta01.assign(H0*ep*eta0)
    
    return(eta01)
