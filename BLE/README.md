# User instructions

Program that solves the Benney-Luke equations (BLE) on a 2D periodic domain using the Stormer-Verlet scheme.

The simulation tracks the evolution of two or three line solitons as they interact, 
to produce a four or eight times higher splash than the initial height of each soliton.

For more information, please refer to the paper "Numerical experiments on extreme waves through
oblique-soliton interactions", written by J. Choi, O. Bokhove, A. Kalogirou, and M. A. Kelmanson.


## FILES

The code consists of five files:
- `BL_periodic.py` is the main file that solves BLE;
- `initial_data.py` defines the initial conditions eta_0(x,y), and Phi_0(x,y);
- `boundary_point.py` computes points to design a computational domain;
- `call_energy.py` and `call_maximum.py` are used for plotting the evolution of energy and maximum, respectively.

After the computation is finished, four sets of data are produced:
- solutions eta and Phi in "*data/output.pvd*";
- evolution of energy against time in "*data/energy.csv*";
- evolution of maximum of eta against time in "*data/max.csv*";
- time in "*data/time.csv*".

The computational data can be visualised as follows:
- Solutions eta and Phi are pvd files that can be read with Paraview.
- Time evolution of energy can be plotted by running file `call_energy.py`.
- Time evolution of maximum of eta can be plotted by running file `call_maximum.py`.


## USAGE

There are three ***Switches*** in `BL_periodic.py` which the user can modify:
- soliton number: SP2 or SP3, corresponding to simulations of two- or three-soliton interations;
- domain type: single or both, corresponding to singly (x-direction) or doubly (x- and y-directions) periodic domain;
- basis type: 1, 2 3, or 4 corresponding to continuous Galerkin basis of the respective order.

To obtain the results and figures in the paper, the ***Switches*** should be chosen as follows:
Figure | soliton number | domain type | basis type
--- | ---           | ---         | ---
8 | SP3       | single and both         | CG2
9-11 | SP2       | single         | CG1, CG2, and CG3
12-15 | SP3       | single         | CG1, CG2, CG3, and CG4

To modify the initial condition, the user can change the values of the variables in sections ***Parameters***, and ***Parameters for k_i*** in `BL_periodic.py`.
