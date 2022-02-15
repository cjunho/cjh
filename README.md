# User instructions

Program that solves the Benney-Luke equations (BLE) on a 2D periodic domain using the Stormer-Verlet scheme.

The simulation tracks the evolution of two or three line solitons as they interact, 
to produce a four or eight times higher splash than the initial height of each soliton.

For more information, please refer to the paper "Numerical experiments on extreme waves through
oblique-soliton interactions", written by J. Choi, O. Bokhove, A. Kalogirou, and M. A. Kelmanson.


FILES

The code consists of five files:

• BL_periodic.py is the main file that solves BLE;

• initial_data.py defines the initial conditions eta_0(x,y), and Phi_0(x,y);

• boundary_point.py computes points to design a computational domain;

• call_energy.py and call_maximum.py are for plotting energy evolution and maximum evolution, respectively.

After the computation is finished, three sets of data are produced:

• solutions eta and Phi in "data/output.pvd";

• evolution of energy against time in "data/energy.csv";

• evolution of maximum of eta against time in "data/max.csv".

The computational data are visualised as follows:

• Solutions eta and Phi are pvd files that are read with Paraview.

• Evolution of energy against time are plotted by running file "call_energy.py".

• Evolution of maximum of eta against time are plotted by running file "call_maximum.py".


USAGE

There are three 'Switches' in BL_periodic.py which the user can modify:

• soliton number: SP2 or SP3, corresponding to simulations of two- or three-soliton interations;

• domain type: single or both, corresponding to singly (x-direction) or doubly (x- and y-directions) periodic domain;

• basis type: 1, 2 or 3, corresponding to continuous Galerkin basis of the respective order.

In order to obtain the results as the figures in the paper, please choose the Sitches as follows. 
-- | soliton number | domain type | basis type | dimensional variable
--- | ---           | ---         | ---        |--- 
Fig 8 | SP3       | single and both         | CG2        | x
Fig 9-11 | SP2       | single         | CG1, CG2, and CG3        | x
Fig 12-15 | SP3       | single         | CG1, CG2, and CG3        | x


To modify the initial condition, the user can change the values of the variables in sections “Parameters”, and “Parameters for k_i” in BL_periodic.py.
For specific details on the calculation of the parameters, please refer to the paper. 


 
