# User instructions

Program solving Benney-Luke equations(BLE) on a 2d periodic domain
with Stoermer-Verlet scheme derived by variational approach.
This simulation shows three-line-soliton interaction to produce a 8 times higher splash than an initial height of each soliton. 
For more information, please refer to the paper "Numerical experiments on extreme waves through
oblique-soliton interactions" written by J. Choi, O. Bokhove, A. Kalogirou, and M. A. Kelmanson.

USAGE

The code consists of three files:

• BL_periodic.py is the main file that solves BLE;

• initial_data.py defines the initial conditions eta_0(x,y), and Phi_0(x,y);

• boundary_point.py computes points to design a computational domain.

Before running the simulation, we set the initial condition, and define the computational domain. To modify the initial condition, the user can change the values of the variables in sections “Parameters”, and “Parameters for k_i” in BL_periodic.py.
For specific details on the calculation of the parameters, please refer to the paper. After setting the variables, 
the periodic domain corresponding to the chosen parameters is created insection “Mesh”.  

After the computation is finished, three sets of data are produced:

• solutions eta and Phi in "data/output.pvd"(these are later used for visualisation in Paraview);

• evolution of energy against time in "data/energy.csv";

• evolution of maximum of eta against time in "data/max.csv"
