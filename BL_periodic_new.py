" This file implements a simulation of Benney-Luke equations on a 2D periodic domain. "
" The simulation tracks the evolution of two or three line solitons as they interact, "
" to produce a four or eight times higher splash than the initial height of each soliton. "

" The user can modify the following simulation parameters in section 'Switches' below: "
" soliton_number (two or three), domain_type (single periodic, both periodc), and basis_type (CG1, CG2, CG3). "
" In order to modify the initial conditions, the user can change values of variables in sections 'Parameters' and 'Parameters for k_i'. "

from firedrake import *
from initial_data import *
from boundary_point import *
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
basis_type = int(4)         # Choices: ["1", "2", "3", "4"]
                            # "1": CG1
                            # "2": CG2
                            # "3": CG3
                            # "4": CG4

""" ________________ Parameters ________________ """
ep = 0.05             # Small amplitude parameter 
mu = 0.0025           # Small dispersion parameter
dt = 0.005            # Time step

""" ________________ Initial time ________________ """
if soliton_number == "SP3":
    t = -60
elif soliton_number == "SP2":
    t = 0
    
""" ________________ Parameters for k_i ________________ """
tan0 = 0.25                         # tan(theta) for BL
tan = (2/9)**(1/6)*tan0/np.sqrt(ep) # tan(theta) for KP 
lam = 1                             # lambda  

if soliton_number == "SP3":
    tildeA = 0.5*(3/4)**(1/3)
    k1 = -tildeA**.5*((2/lam)**.5+.5**.5+del1)
    k2 = -tildeA**.5*(.5**.5+del1)
    k3 = -np.sqrt(tildeA*0.5)
    k4 = -k3
    k5 = -k2
    k6 = -k1
    
    x_shift = 0     # Shifting in x
    a=(k6*(k6**2-k4**2)/(k5*(k5**2-k4**2)))**.5
    b=1
    cc=1/a

elif soliton_number == "SP2":
    tildeA = 0.7571335803467248
    k1 = -np.sqrt(tildeA)
    k2 = 0
    k3 = 0
    k4 = -k1
    x_shift = 0     # Shifting in x
    
""" ___________________ Mesh ___________________ """
                     
if soliton_number == "SP3":
    if domain_type == "single":
        Ly = 30
        direction = "x"
        y_shift = 0
    else:
        Ly = 47*2
        direction = "both"
        y_shift = 0.5*Ly
        
    y_ast = yast(k1,k2,k3,k4,k5,k6) # y_ast
    y2 = y_ast+(Ly-y_shift)         # upper boundary y=y2
    xb1 = bd_x1(y2,x_shift,k1,k2,k3,k4,k5,k6,t,ep,mu) # left wall
    xb2 = bd_x2(y2,x_shift,k1,k2,k3,k4,k5,k6,t,ep,mu) # right wall 
    Lx = xb2-xb1
    
elif soliton_number == "SP2":
    if domain_type == "single":
        Ly = 40
        direction = "x"
        y_shift = 0
    else:
        Ly = 40*2
        direction = "both"
        y_shift = 0.5*Ly
        
    y_ast = -20                        # y_ast
    y2 = y_ast+(Ly-y_shift)            # upper boundary y=y2
    xb1 = bd_x21(y2,x_shift,k1,k2,k3,k4,t,ep,mu) # left wall
    xb2 = bd_x22(y2,x_shift,k1,k2,k3,k4,t,ep,mu) # right wall 
    Lx = xb2-xb1
    
c=16*4/basis_type
# else: c=12
Nx = int(c*np.ceil(Lx))
Ny = int(c*np.ceil(Ly))

mesh = PeriodicRectangleMesh(Nx, Ny, Lx, Ly, direction=direction,
                          quadrilateral=True, reorder=None,
                          distribution_parameters=None,
                          diagonal=None,
                          comm=COMM_WORLD,name='mesh')

coords = mesh.coordinates

coords.dat.data[:,0] = coords.dat.data[:,0] + xb1
coords.dat.data[:,1] = coords.dat.data[:,1] + y_ast - y_shift

""" ______________ Function Space ______________ """
V = FunctionSpace(mesh, "CG", basis_type)   

""" _____________ Define functions _____________ """

eta0 = Function(V, name="eta")          # eta(n)
phi0 = Function(V, name="phi")          # phi(n)
eta1 = Function(V, name="eta_next")     # eta(n+1)
phi1 = Function(V, name="phi_next")     # phi(n+1)
phi2 = Function(V, name="phi_periodic") # phi_periodic

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
x = SpatialCoordinate(mesh)
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

solver_parameters_lin={'ksp_type': 'cg', 'pc_type': 'none'}
# solver_parameters={'snes_type': 'newtonls','ksp_type': 'preonly','pc_type': 'lu'}
solver_parameters={ 'pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC','star_sub_sub_pc_type': 'lu', 'sub_sub_pc_factor_mat_ordering_type': 'rcm'}


""" _____________ Weak formulations _____________ """
# Get phi(n+1/2)
Fphi_h = ( v*(phi_h-phi2)/(0.5*dt) + 0.5*mu*inner(grad(v),grad((phi_h-phi2)/(0.5*dt)))
            + v*eta0 + 0.5*ep*(inner(grad(phi_h),grad(phi_h))
                                    + 2*grad(phi_h)[0]*Ux + 2*grad(phi_h)[1]*Uy
                                    + Ux*Ux+Uy**2)*v )*dx

phi_problem_h = NonlinearVariationalProblem(Fphi_h,phi_h)
phi_solver_h = NonlinearVariationalSolver(phi_problem_h, solver_parameters=solver_parameters)


# Get q(n+1/2)
aq_h = v*q*dx
Lq_h = 2.0/3.0*( inner(grad(v),grad(phi_h)) + grad(v)[0]*Ux + grad(v)[1]*Uy )*dx


q_problem_h = LinearVariationalProblem(aq_h,Lq_h,q_h)
q_solver_h = LinearVariationalSolver(q_problem_h,solver_parameters=solver_parameters_lin)

# Get eta(n+1)
Feta = ( v*(eta1-eta0)/dt + 0.5*mu*inner(grad(v),grad((eta1-eta0)/dt))
          - 0.5*((1+ep*eta0)+(1+ep*eta1))*(inner(grad(v),grad(phi_h))+grad(v)[0]*Ux+grad(v)[1]*Uy)         
          - mu*inner(grad(v),grad(q_h)) )*dx
eta_problem = NonlinearVariationalProblem(Feta,eta1)
eta_solver = NonlinearVariationalSolver(eta_problem,solver_parameters=solver_parameters)

# Get phi(n+1)
Fphi_a = ( v*q + 0.5*mu*inner(grad(v),grad(q)))*dx


Fphi_L = ( v*phi_h + 0.5*mu*inner(grad(v),grad(phi_h))
          - 0.5*dt*v*eta1 - 0.25*dt*ep*(inner(grad(phi_h),grad(phi_h))
                                  + 2*grad(phi_h)[0]*Ux + 2*grad(phi_h)[1]*Uy
                                  + Ux*Ux+Uy**2)*v )*dx



phi_problem = LinearVariationalProblem(Fphi_a,Fphi_L,phi1)
phi_solver = LinearVariationalSolver(phi_problem,solver_parameters=solver_parameters_lin)


""" ________________ Compute energy and max(eta) ________________ """
E_data1 = np.zeros(1)
max_eta = np.zeros(1)
t_data = np.zeros(1)

phi1.assign(phi2)
q_solver.solve()        # Compute Delta(phi)

E1 = assemble( (0.5*eta0**2 + 0.5*(1+ep*eta0)*((grad(phi2))**2
                                               + 2*grad(phi2)[0]*Ux + 2*grad(phi2)[1]*Uy
                                               + Ux*Ux+Uy*Uy) + 0.75*mu*q1**2)*dx )

with eta0.dat.vec_ro as vv:
    L_inf = vv.max()[1]
E_data1[0] = E1
max_eta[0] = L_inf
t_data[0]  = t

PETSc.Sys.Print(t, L_inf, E1)    

""" ________________ Saving data ________________ """
output1 = File('data/output.pvd')
output1.write(phi2, eta0, phi0, time=t)

""" _____________ Time loop _____________ """
t1 = t          # Initial time
T = 80         # Time duration
step = int(0)   # Number of steps

while t < t1+T:        
      t += dt
      
      # Solve the weak formulations
      phi_solver_h.solve()
      q_solver_h.solve()
      eta_solver.solve()
      phi_solver.solve()
      # q_solver.solve()
      
      # Update the solutions
      eta0.assign(eta1)
      phi2.assign(phi1)
      
      # Compute energy and max(eta)
      E1 = assemble( (0.5*eta0**2 + 0.5*(1+ep*eta0)*((grad(phi2))**2
                     + 2*grad(phi2)[0]*Ux + 2*grad(phi2)[1]*Uy
                     + Ux*Ux+Uy*Uy) + 0.75*mu*q1**2)*dx )
      E_data1 = np.r_[E_data1,[E1]]
      with eta0.dat.vec_ro as v:
        L_inf = v.max()[1]
      max_eta = np.r_[max_eta,[L_inf]]
      t_data = np.r_[t_data,[t]]
      step += int(1)
      
      # Save data every 100 steps
      if step % 100 == 0:  
        phi0.assign(phi2+U1)
      
        output1.write(phi2, eta0, phi0, time=t)
        np.savetxt('data/energy.csv', E_data1)
        np.savetxt('data/max.csv', max_eta)
        np.savetxt('data/time.csv', t_data)
        PETSc.Sys.Print(t, L_inf, E1)

# Print computational time
print(time.time() - t00)

with CheckpointFile(save_path+"data.h5", 'w') as afile:
    afile.save_mesh(mesh)  # optional
    afile.save_function(phi2)
    afile.save_function(eta0)
    # afile.save_function(phi0)
    afile.save_function(phi_h)
    afile.save_function(q_h)