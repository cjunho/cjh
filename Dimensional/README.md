# User instructions

Program that solves the Benney-Luke equations (BLE) on a 2D periodic domain using the Stormer-Verlet scheme.
The numerical solutions are scaled to become dimensional.

The simulation tracks the evolution of two or three line solitons as they interact, 
to produce a four or eight times higher splash than the initial height of each soliton.

For more information, please refer to the paper "Numerical experiments on extreme waves through
oblique-soliton interactions", written by J. Choi, O. Bokhove, A. Kalogirou, and M. A. Kelmanson.


## FILES

The code consists of four files:
- `BL_dimension.py` is the main file that solves BLE, and then scales the BLE solutions into dimensional variables;
- `initial_data.py` defines the initial conditions eta_0(x,y), and Phi_0(x,y);
- `boundary_point.py` computes points to design a computational domain;
- `scaling.py` scales the dimensional variables into non-dimensional form, and vice versa.
- `call_maximum.py` can be used for plotting the evolution of maximum.


After the computation is finished, three sets of data are produced:
- solutions eta and Phi in "*data/output.pvd*";
- evolution of maximum of eta against time in "*data/max.csv*";
- time in "*data/time.csv*".

The computational data can be visualised with Paraview.

## USAGE
The usage of the files is the same to `BL_periodic.py`, except that the variables are scaled to dimensional units.
