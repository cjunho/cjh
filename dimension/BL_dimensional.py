" This file implements a simulation of Benney-Luke equations on a 2D periodic domain. "
" The simulation tracks the evolution of two or three line solitons as they interact, "
" to produce a four or eight times higher splash than the initial height of each soliton. "
" The numerical solution is scaled into dimensional variables. "

" The user can modify the following simulation parameters in section 'Switches' below: "
" soliton_number (two or three), domain_type (single periodic, both periodc), and basis_type (CG1, CG2, CG3). "
" In order to modify the initial conditions, the user can change values of variables in sections 'Parameters' and 'Parameters for k_i'. "

from firedrake import *
from initial_data import *
from boundary_point import *
from scaling import *
from firedrake.petsc import PETSc
import numpy as np
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"

op2.init()
parameters["coffee"]["O2"] = False

# Timing
t00 = time.time()

""" ________________ Switches ________________ """
soliton_number = "SP3"      # Choices: ["SP2", "SP3"]
                            # "SP2": Two-soliton interaction
                            # "SP3": Three-soliton interaction
domain_type = "single"      # Choices: ["both", "sigle"]
                            # "both": doubly periodic domain in x- and y-directions
                            # "single": singly periodic domain in x-direction only
basis_type = int(1)         # Choices: ["1", "2", "3"]
                            # "1": CG1
                            # "2": CG2
                            # "3": CG3

""" ________________ Parameters ________________ """
ep = 0.05                   # Small amplitude parameter 
mu = 0.0025                 # Small dispersion parameter
ds = 0.031943828249996996   # Time step
lam = 1                     # lambda parameter

""" ________________ Dimensional variables ________________ """
H0 = 20                            # Depth of water (in meters)
if soliton_number == "SP3":
    s0 = -1277.75312999988         # Initial time (in seconds)
    tan0 = 0.25                    # Angle 
    tildeA0 = 0.10396012211594852  # Initial height of line solitons (in meters)
    Ly0 = 4203.807797699605        # Domain width (in meters)
elif soliton_number == "SP2":
    s0 = 0                         # Initial time (in seconds)
    tan0 = 0.25                    # Angle
    tildeA0 = 0.8333333333333334   # Initial height of line solitons (in meters)
    Ly0 = 3577.7087639996635       # Domain width (in meters)

""" ________ Scaling dimensional variables to non-dimensional form ________ """
t,dt,tan1,tildeA,Ly = scaling_to_nondim(s0,ds,tan0,tildeA0,Ly0,H0,ep)
tan = (2/9)**(1/6)*tan1/np.sqrt(ep)

""" ________________ Parameters for k_i ________________ """
if soliton_number == "SP3":    
    k1 = -0.5*tan-np.sqrt(tildeA*0.5/lam)
    k2 = -0.5*tan+np.sqrt(tildeA*0.5/lam)
    k3 = -np.sqrt(tildeA*0.5)
    k4 = -k3
    k5 = -k2
    k6 = -k1
    x_shift = 5     # Shifting in x
elif soliton_number == "SP2":
    k1 = -np.sqrt(tildeA)
    k2 = 0
    k3 = 0
    k4 = -k1
    x_shift = 0     # Shifting in x
    
""" ___________________ Mesh ___________________ """
                     
if soliton_number == "SP3":
    if domain_type == "single":
        Ly = Ly
        direction = "x"
        y_shift = 0
    else:
        Ly = Ly*2
        direction = "both"
        y_shift = 0.5*Ly
        
    y_ast = yast(k1,k2,k3,k4,k5,k6) # y_ast
    y2 = y_ast+(Ly-y_shift)         # upper boundary y=y2
    xb1 = bd_x1(y2,x_shift,k1,k2,k3,k4,k5,k6,t,ep,mu) # left wall
    xb2 = bd_x2(y2,x_shift,k1,k2,k3,k4,k5,k6,t,ep,mu) # right wall 
    Lx = xb2-xb1
    
elif soliton_number == "SP2":
    if domain_type == "single":
        Ly = Ly
        direction = "x"
        y_shift = 0
    else:
        Ly = Ly*2
        direction = "both"
        y_shift = 0.5*Ly
        
    y_ast = -20                        # y_ast
    y2 = y_ast+(Ly-y_shift)            # upper boundary y=y2
    xb1 = bd_x21(y2,x_shift,k1,k2,k3,k4,t,ep,mu) # left wall
    xb2 = bd_x22(y2,x_shift,k1,k2,k3,k4,t,ep,mu) # right wall 
    Lx = xb2-xb1

Nx = int(12*np.ceil(Lx)+1)
Ny = int(12*np.ceil(Ly)+1)

mesh = PeriodicRectangleMesh(Nx, Ny, Lx, Ly, direction=direction,
                          quadrilateral=True, reorder=None,
                          distribution_parameters=None,
                          diagonal=None,
                          comm=COMM_WORLD)

coords = mesh.coordinates

coords.dat.data[:,0] = H0/ep**0.5*(coords.dat.data[:,0] + xb1)
coords.dat.data[:,1] = H0/ep**0.5*(coords.dat.data[:,1] + y_ast - y_shift)

""" ______________ Function Space ______________ """
V = FunctionSpace(mesh, "CG", basis_type)   

""" _____________ Define functions _____________ """

eta0 = Function(V, name="eta")          # eta(n)
phi0 = Function(V, name="phi")          # phi(n)
eta1 = Function(V, name="eta_next")     # eta(n+1)
phi1 = Function(V, name="phi_next")     # phi(n+1)
phi2 = Function(V, name="phi_periodic") # phi_periodic

phi21 = Function(V, name="save_phi2")   # makes phi2 dimesional
eta01 = Function(V, name="save_eta0")   # makes eta dimesional
phi01 = Function(V, name="save_phi0")   # makes phi dimesional

q0 = Function(V)                        # q(n)    
q1 = Function(V)                        # q(n+1) 
phi_h = Function(V)                     # phi(n+1/2)
q_h = Function(V)                       # q(n+1/2)

U1 = Function(V, name="corrector")      # corrector
Ux = Function(V)
Uy = Function(V)

q = TrialFunction(V)
v = TestFunction(V)

""" ____________ Initial conditions _____________ """
x = SpatialCoordinate(mesh)*ep**0.5/H0
xx = (x[0]-x_shift)*(4.5)**(1/6)*(ep/mu)**(1/2) # Scaling x into X

# Initialization
if soliton_number == "SP3":
    yy = x[1]*(4.5)**(1/3)*ep*(1/mu)**(1/2)     # Scaling y into Y 
    eta0 = initial_eta(xx,yy,eta0,k1,k2,k3,k4,k5,k6,t,ep,mu)
    phi0 = initial_phi(xx,yy,phi0,k1,k2,k3,k4,k5,k6,t,ep,mu)
elif soliton_number == "SP2":
    yy = (abs(x[1]-y_ast)+y_ast)*(4.5)**(1/3)*ep*(1/mu)**(1/2)  # Scaling y into Y 
    eta0 = initial_eta2(xx,yy,eta0,k1,k2,k3,k4,t,ep,mu)
    phi0 = initial_phi2(xx,yy,phi0,k1,k2,k3,k4,t,ep,mu)

""" _____________ Corrector plane _____________ """
if soliton_number == "SP3":
    # U1 is a corrector plane connecting (x11, phib1) and (x22, phib2)
    x11 = (xb1-x_shift)*(4.5)**(1/6)*(ep/mu)**(1/2)
    x22 = (xb2-x_shift)*(4.5)**(1/6)*(ep/mu)**(1/2)
    
    phib1 = initial_phib(x11,yy,k1,k2,k3,k4,k5,k6,t,ep,mu)
    phib2 = initial_phib(x22,yy,k1,k2,k3,k4,k5,k6,t,ep,mu)
    
    U = (phib2-phib1)/(x22-x11)
    U1.interpolate(U*(xx-x11) + phib1)  # Corrector plane
    
    # Uy is partial_y(U1)
    Uy1 = initial_phiby(x11,yy,k1,k2,k3,k4,k5,k6,t,ep,mu)
    Uy2 = initial_phiby(x22,yy,k1,k2,k3,k4,k5,k6,t,ep,mu)
    Uy.interpolate((Uy2-Uy1)*(xx-x11)/(x22-x11) + Uy1)
    
    # Ux is partial_x(U1)
    Ux.interpolate(Ux+U*(4.5)**(1/6)*(ep/mu)**(1/2))
    
    # Correct phi0 to make phi0 periodic, phi2(xb1)=phi2(xb2)
    phi2.interpolate(phi0-U1)
elif soliton_number == "SP2":
    # U1 is a corrector plane connecting (x11, phib1) and (x22, phib2).
    x11 = (xb1-x_shift)*(4.5)**(1/6)*(ep/mu)**(1/2)
    x22 = (xb2-x_shift)*(4.5)**(1/6)*(ep/mu)**(1/2)
    
    phib1 = initial_phib2(x11,yy,k1,k2,k3,k4,t,ep,mu)
    phib2 = initial_phib2(x22,yy,k1,k2,k3,k4,t,ep,mu)
    
    U = (phib2-phib1)/(x22-x11)
    U1.interpolate(U*(xx-x11) + phib1)  # Corrector plane
    
    # Uy is partial_y(U1)
    Uy1 = initial_phiby2(x11,yy,k1,k2,k3,k4,t,ep,mu)
    Uy2 = initial_phiby2(x22,yy,k1,k2,k3,k4,t,ep,mu)
    Uy.interpolate((Uy2-Uy1)*(xx-x11)/(x22-x11) + Uy1)
    
    # Ux is partial_x(U1)
    Ux.interpolate(Ux + U*(4.5)**(1/6)*(ep/mu)**(1/2))
    
    # Correct phi0 to make phi0 periodic, phi2(xb1)=phi2(xb2)
    phi2.interpolate(phi0-U1)

""" _____________ Weak formulations _____________ """
# Get phi(n+1/2)
Fphi_h = ( v*(phi_h-phi2)/(0.5*dt) + 0.5*mu*inner(grad(v),grad((phi_h-phi2)/(0.5*dt)))
            + v*eta0 + 0.5*ep*(inner(grad(phi_h),grad(phi_h))
                                    + 2*grad(phi_h)[0]*Ux + 2*grad(phi_h)[1]*Uy
                                    + Ux*Ux+Uy**2)*v )*dx

phi_problem_h = NonlinearVariationalProblem(Fphi_h,phi_h)
phi_solver_h = NonlinearVariationalSolver(phi_problem_h)

# Get q(n+1/2)
aq_h = v*q*dx
Lq_h = 2.0/3.0*( inner(grad(v),grad(phi_h)) + grad(v)[0]*Ux + grad(v)[1]*Uy )*dx

q_problem_h = LinearVariationalProblem(aq_h,Lq_h,q_h)
q_solver_h = LinearVariationalSolver(q_problem_h)

# Get eta(n+1)
Feta = ( v*(eta1-eta0)/dt + 0.5*mu*inner(grad(v),grad((eta1-eta0)/dt))
          - 0.5*((1+ep*eta0)+(1+ep*eta1))*(inner(grad(v),grad(phi_h))+grad(v)[0]*Ux+grad(v)[1]*Uy)         
          - mu*inner(grad(v),grad(q_h)) )*dx
eta_problem = NonlinearVariationalProblem(Feta,eta1)
eta_solver = NonlinearVariationalSolver(eta_problem)

# Get phi(n+1)
Fphi = ( v*(phi1-phi_h)/(0.5*dt) + 0.5*mu*inner(grad(v),grad((phi1-phi_h)/(0.5*dt)))
          + v*eta1 + 0.5*ep*(inner(grad(phi_h),grad(phi_h))
                                  + 2*grad(phi_h)[0]*Ux + 2*grad(phi_h)[1]*Uy
                                  + Ux*Ux+Uy**2)*v )*dx

phi_problem = NonlinearVariationalProblem(Fphi,phi1)
phi_solver = NonlinearVariationalSolver(phi_problem)

# Get q(n+1)
Lq = 2.0/3.0*( inner(grad(v),grad(phi1)) + grad(v)[0]*Ux+grad(v)[1]*Uy )*dx
q_problem = LinearVariationalProblem(aq_h,Lq,q1)
q_solver = LinearVariationalSolver(q_problem)

""" _________ Rescaling BLE into dimensional variables _____________ """
eta01 = scaling_to_dim(eta01,eta0,H0,ep)
phi01 = scaling_to_dim(phi01,phi0,H0,ep)
phi21 = scaling_to_dim(phi21,phi2,H0,ep)

""" ________________ Compute energy and max(eta) ________________ """
max_eta = np.zeros(1)
with eta01.dat.vec_ro as vv:
    L_inf = vv.max()[1]
max_eta[0] = L_inf
PETSc.Sys.Print(s0, L_inf)    

""" ________________ Saving data ________________ """
output1 = File('data/output.pvd')
output1.write(phi21, eta01, phi01, time=s0)

""" _____________ Time loop _____________ """
s1 = s0                   # Initial time
s = s0                    # Time	
step = int(0)             # Number of steps
S = 0.6388765649999399    # Time duration

while s < s1+S:        
      s += ds
      
      # Solve the weak formulations
      phi_solver_h.solve()
      q_solver_h.solve()
      eta_solver.solve()
      phi_solver.solve()
      q_solver.solve()
      
      # Update the solutions
      eta0.assign(eta1)
      phi2.assign(phi1)
      
      # Compute energy and max(eta)
      with eta0.dat.vec_ro as v:
        L_inf = H0*ep*v.max()[1]
      max_eta = np.r_[max_eta,[L_inf]]
      step += int(1)
      
      # Save data every 100 steps
      if step % 100 == 0:  
        phi0.assign(phi2+U1)
        
        eta01 = scaling_to_dim(eta01,eta0,H0,ep)  # rescaling eta0
        phi01 = scaling_to_dim(phi01,phi0,H0,ep)  # rescaling phi0
        phi21 = scaling_to_dim(phi21,phi2,H0,ep)  # rescaling phi2
        output1.write(phi21, eta01, phi01, time=s)        
        np.savetxt('data/max.csv', max_eta)
        PETSc.Sys.Print(s, L_inf)

# Print computational time
print(time.time() - t00)
