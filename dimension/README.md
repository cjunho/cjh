# User instructions

These files are for scaling non-dimensional variables of BLE into dimensional variables.


## FILES

The code consists of four files:

- 'BL_dimension.py' is the main file that solves BLE, and then scales the BLE solutions into dimensional variables;

- 'initial_data.py' defines the initial conditions eta_0(x,y), and Phi_0(x,y);

- 'boundary_point.py' computes points to design a computational domain;

- 'scaling.py' are for scaling dimensional vatiables to non-dimensionals, and rescales vice versa.

How to scale is written in the paper. 

After the computation is finished, three sets of data are produced:

- solutions eta and Phi in "*data/output.pvd";

- evolution of maximum of eta against time in "*data/max.csv".

The computational data with dimesional variables can be visualised with Paraview.

## USAGE
Except one thing, the usage of the files are the same to 'BL_periodic.py', that is, for scaling, three sections are added to 'BL_periodic.py': 'Dimensional variabels', 'Scaling the dimensional variables into non-dimensional', and 'Rescaling BLE into dimensional variables'. First, set dimensional variables in section Dimensional variabels. Subsequently, in section Scaling the dimensional variables into non-dimensional, a function 'scaling_to_nondim' turns the dimensional variables into non-dimensionals. And then, the codes solves BLE like BL_periodic.py. Finally, a function scaling_to_dim scales the BLE solution into the dimensional variables.
