"This file demonstrates a numerical solution of Benney-Luke equations with dimensional variables."
"The solution is two or three-line-soliton interaction."
"The code consits of three files:"
"• BL_dimension.py is the main file that solves BLE;"
"• initial_data.py defines the initial conditions eta_0(x,y), and Phi_0(x,y);"
"• boundary_point.py computes points to design a computational domain."
"For the demesional varible, please refer to the attached paper."

"The users can choose in section Opsions"
"soltion_type (two or three), doamin_type (both periodc, single periodic), and basis type (CG1, CG2, CG3)." 
"In order to modify the initial conditions, the user can change values of variables in sections 'Parameters' and Parameters for k_i"


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

""" ________________ Options ________________ """
soliton_number="SP3" # choices=["SP2", "SP3"]
#SP2: Two solitons interaction;
#SP3: Three solitons interaction 
domain_type="single"  # choices=["both","sigle"] 
#both: x and y directional periodic domain;
#single:  the upper half of the both domain about y=y_ast, and x directional periodic.
basis_type=int(1)   # choices=["1","2", "3"] 1:CG1, 2:CG2, 3:CG3

dimensional="true"   # choices=["1","2", "3"] 1:CG1, 2:CG2, 3:CG3


""" ________________ Parameters ________________ """
ep=0.05               # Small amplitude parameter 
mu = 0.0025           # Small dispersion parameter
dt = 0.005            # Time step
H0 = 20               # Depth of water
""" ________________ Initial time ________________ """
if soliton_number=="SP3":
    t=-150
elif soliton_number=="SP2":
    t=0
""" ________________ Parameters for k_i ________________ """
tan0=.25                          #tan(theta) for BL
tan=(2/9)**(1/6)*tan0/np.sqrt(ep) #tan(theta) for KP 

lam=1                             #lambda  

if soliton_number=="SP3":
    tildeA=0.094454039365117
    k1=-.5*tan-np.sqrt(tildeA*.5/lam)
    k2=-.5*tan+np.sqrt(tildeA*.5/lam)
    k3=-np.sqrt(tildeA*.5)
    k4=-k3
    k5=-k2
    k6=-k1
    x_shift=5           #shifting in x for convenience
elif soliton_number=="SP2":
    tildeA=0.7571335803467248
    k1 = -np.sqrt(tildeA)
    k2 = 0
    k3 = 0
    k4 = -k1
    x_shift=0           #shifting in x for convenience
""" ___________________ Mesh ___________________ """
                     
if soliton_number=="SP3":
    if domain_type=="single":
        Ly=47
        direction="x"
        y_shift=0
    else:
        Ly=47*2
        direction="both"
        y_shift=Ly*.5
    y_ast=yast(k1,k2,k3,k4,k5,k6) #y_\ast
    y2=y_ast+(Ly-y_shift)         #upper boundary y=y2
    xb1=bd_x1(y2,x_shift,k1,k2,k3,k4,k5,k6,t,ep,mu) #left wall
    xb2=bd_x2(y2,x_shift,k1,k2,k3,k4,k5,k6,t,ep,mu) #right wall 
    Lx=xb2-xb1
    
elif soliton_number=="SP2":
    if domain_type=="single":
        Ly=40
        direction="x"
        y_shift=0
    else:
        Ly=40*2
        direction="both"
        y_shift=Ly*.5
    y_ast=-20                        #y_ast
    y2=y_ast+(Ly-y_shift)            #upper boundary y=y2
    xb1=bd_x21(y2,x_shift,k1,k2,k3,k4,t,ep,mu) #left wall
    xb2=bd_x22(y2,x_shift,k1,k2,k3,k4,t,ep,mu) #right wall 
    Lx=xb2-xb1

Nx=int(12*np.ceil(Lx)+1)
Ny=int(12*np.ceil(Ly)+1)
# print(xb1, xb2,y_ast) #10.52400712883123+10.369306212961614
# input("sdgsd") #-6.971574122470538
mesh = PeriodicRectangleMesh(Nx, Ny, Lx, Ly, direction=direction,
                          quadrilateral=True, reorder=None,
                          distribution_parameters=None,
                          diagonal=None,
                          comm=COMM_WORLD)

coords = mesh.coordinates    # access to coordinate


coords.dat.data[:,0] =H0/ep**.5*(coords.dat.data[:,0]+xb1)           # turn x into demesional variable
coords.dat.data[:,1] = H0/ep**.5*(coords.dat.data[:,1]+y_ast-y_shift)# turn y into demesional variable
""" ______________ Function Space ______________ """
V = FunctionSpace(mesh, "CG", basis_type)   

""" ___________ Define the functions ___________ """

eta0 = Function(V, name="eta")          # eta(n)
phi0 = Function(V, name="phi")          # phi(n)
eta1 = Function(V, name="eta_next")     # eta(n+1)
phi1 = Function(V, name="phi_next")     # phi(n+1)
phi2 = Function(V, name="phi_periodic") # phi_periodic

q0 = Function(V)                        # q(n)    
q1 = Function(V)                        # q(n+1) 
phi_h = Function(V)                     # phi(n+1/2)
q_h = Function(V)                       # q(n+1/2)

U1 = Function(V,name="corrector")       # corrector
Ux = Function(V)
Uy=Function(V)

q = TrialFunction(V)
v = TestFunction(V)

""" _____________ Initial condition _____________ """
x = SpatialCoordinate(mesh)*ep**.5/H0
xx= (x[0]-x_shift)*(4.5)**(1/6)*(ep/mu)**(1/2)   #scaling x into X
 

## Initialization
if soliton_number=="SP3":
    yy= x[1]*(4.5)**(1/3)*ep*(1/mu)**(1/2)           #scaling y into Y 
    eta0=initial_eta(xx,yy,eta0,k1,k2,k3,k4,k5,k6,t,ep,mu)
    phi0=initial_phi(xx,yy,phi0,k1,k2,k3,k4,k5,k6,t,ep,mu)
elif soliton_number=="SP2":
    yy= (abs(x[1]-y_ast)+y_ast)*(4.5)**(1/3)*ep*(1/mu)**(1/2)           #scaling y into Y 
    eta0=initial_eta2(xx,yy,eta0,k1,k2,k3,k4,t,ep,mu)
    phi0=initial_phi2(xx,yy,phi0,k1,k2,k3,k4,t,ep,mu)


""" _____________ corrector plane _____________ """
if soliton_number=="SP3":
    #U1 is a corrector plane connecting (x11, phib1) and (x22, phib2).
    x11=(xb1-x_shift)*(4.5)**(1/6)*(ep/mu)**(1/2)
    x22=(xb2-x_shift)*(4.5)**(1/6)*(ep/mu)**(1/2)
    
    phib1=initial_phib(x11,yy,k1,k2,k3,k4,k5,k6,t,ep,mu)
    phib2=initial_phib(x22,yy,k1,k2,k3,k4,k5,k6,t,ep,mu)
    
    U=(phib2-phib1)/(x22-x11)
    U1.interpolate((U*(xx-x11)+phib1))  #corrector plane
    
    #Uy is partial_y U1
    Uy1=initial_phiby(x11,yy,k1,k2,k3,k4,k5,k6,t,ep,mu)
    Uy2=initial_phiby(x22,yy,k1,k2,k3,k4,k5,k6,t,ep,mu)
    Uy.interpolate(((Uy2-Uy1)*(xx-x11)/(x22-x11)+Uy1))  
    
    #Ux is partial_x U1
    Ux.interpolate(Ux+U*(4.5)**(1/6)*(ep/mu)**(1/2))
    
    #correct phi0 to make phi0 periodic, phi2(xb1)=phi2(xb2)
    phi2.interpolate(phi0-U1)
elif soliton_number=="SP2":
    #U1 is a corrector plane connecting (x11, phib1) and (x22, phib2).
    x11=(xb1-x_shift)*(4.5)**(1/6)*(ep/mu)**(1/2)
    x22=(xb2-x_shift)*(4.5)**(1/6)*(ep/mu)**(1/2)
    
    phib1=initial_phib2(x11,yy,k1,k2,k3,k4,t,ep,mu)
    phib2=initial_phib2(x22,yy,k1,k2,k3,k4,t,ep,mu)
    
    U=(phib2-phib1)/(x22-x11)
    U1.interpolate((U*(xx-x11)+phib1))  #corrector plane
    
    #Uy is partial_y U1
    Uy1=initial_phiby2(x11,yy,k1,k2,k3,k4,t,ep,mu)
    Uy2=initial_phiby2(x22,yy,k1,k2,k3,k4,t,ep,mu)
    Uy.interpolate(((Uy2-Uy1)*(xx-x11)/(x22-x11)+Uy1))  
    
    #Ux is partial_x U1
    Ux.interpolate(Ux+U*(4.5)**(1/6)*(ep/mu)**(1/2))
    
    #correct phi0 to make phi0 periodic, phi2(xb1)=phi2(xb2)
    phi2.interpolate(phi0-U1)


""" _____________ Weak formulations _____________ """#,time=t
# Get phi(n+1/2)
Fphi_h = ( v*(phi_h-phi2)/(0.5*dt) + 0.5*mu*inner(grad(v),grad((phi_h-phi2)/(0.5*dt)))
            + v*eta0 + 0.5*ep*(inner(grad(phi_h),grad(phi_h))
                                    +2*grad(phi_h)[0]*Ux+2*grad(phi_h)[1]*Uy
                                    +Ux*Ux+Uy**2)*v )*dx

phi_problem_h = NonlinearVariationalProblem(Fphi_h,phi_h)
phi_solver_h = NonlinearVariationalSolver(phi_problem_h)

# Get q(n+1/2)
aq_h = v*q*dx
Lq_h = 2.0/3.0*(inner(grad(v),grad(phi_h))+grad(v)[0]*Ux+grad(v)[1]*Uy)*dx



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
                                  +2*grad(phi_h)[0]*Ux+2*grad(phi_h)[1]*Uy
                                    +Ux*Ux+Uy**2)*v )*dx

phi_problem = NonlinearVariationalProblem(Fphi,phi1)
phi_solver = NonlinearVariationalSolver(phi_problem)

# Get q(n+1)
Lq = 2.0/3.0*(inner(grad(v),grad(phi1))+grad(v)[0]*Ux+grad(v)[1]*Uy)*dx
q_problem = LinearVariationalProblem(aq_h,Lq,q1)
q_solver = LinearVariationalSolver(q_problem)


""" ________________ computing max(eta) ________________ """
max_eta = np.zeros(1)

phi1.assign(phi2)
q_solver.solve()        #computing Delta phi


with eta0.dat.vec_ro as vv:
    L_inf = vv.max()

max_eta[0]=H0*ep*L_inf[1]
PETSc.Sys.Print((H0/9.8/ep)**.5*t,H0*ep*L_inf[1])    

""" ________________ Saving data ________________ """
output1 = File('data1234/output.pvd')  
phi21 = Function(V, name="save_phi2") # turn phi2 into demesional variable
eta01 = Function(V, name="save_eta0") # turn eta into demesional variable
phi01 = Function(V, name="save_phi0") # turn phi into demesional variable

phi21.assign(H0*ep*phi2)
eta01.assign(H0*ep*eta0)
phi01.assign(H0*ep*phi0)
output1.write(phi21, eta01, phi01,  time=(H0/9.8/ep)**.5*t)



""" _____________ Time loop _____________ """
t1=t        # initial time
T = .1      # duration time
step=int(0) # number of steps
while t < t1+T:        
      t += dt
      
      # Solve the weak formulations 
      phi_solver_h.solve()
      q_solver_h.solve()
      eta_solver.solve()
      phi_solver.solve()
      q_solver.solve()
      
      #Update the solutions
      eta0.assign(eta1)
      phi2.assign(phi1)
      
      with eta0.dat.vec_ro as v:
        L_inf = v.max()
      max_eta=np.r_[max_eta,[H0*ep*L_inf[1]]]
      step +=int(1)
      
      #Saving data every 100 steps
      if step % 100 == 0:  
        phi0.assign(phi2+U1)
      
        phi21.assign(H0*ep*phi2)
        eta01.assign(H0*ep*eta0)
        phi01.assign(H0*ep*phi0)
        output1.write(phi21, eta01, phi01,  time=(H0/9.8/ep)**.5*t)
        np.savetxt('data1234/max.csv', max_eta)
        PETSc.Sys.Print((H0/9.8/ep)**.5*t,H0*ep*L_inf[1])

print(time.time() - t00)     # Print computational time (s)

