# cjh
Program solving Benney-Luke equations(BLE) on a 2d periodic domain
with Stoermer-Verlet scheme derived by variational approach.
This simulation shows three-line-soliton interaction to produce a 8 times higher splash than an initial height of each soliton. 
For more information, please refer to the paper "Numerical experiments on extreme waves through
oblique-soliton interactions" written by J. Choi, O. Bokhove, A. Kalogirou, and M. A. Kelmanson.

USAGE

The codes consists of three files, BL_periodic.py, initial_data.py, and boundary_point.py. 
BL_periodic.py is the main file to solve BLE. initial_data.py yields initial condition eta_0(x,y), and Phi_0(x,y).
Finally, boundary_point.py computes points to design a computational domain.

Before running the simulation, we set initial condition, and a domain. To design initial condition, change variables on sections “Parameters”, and “Parameters for k_i” in BL_periodic.py.
For specific calculation about the parameters, refer to the paper. After setting the variables on “Parameters”, and “Parameters for k_i”, 
the periodic domain corresponding to the variables gets made at section “Mesh”.  

After the computation is finished, three types of data are saved, the results for visualization in Paraview in "data/output.pvd", energy profile against time in "data/energy.csv",
and maximum of eta against time in "data/max.csv"
