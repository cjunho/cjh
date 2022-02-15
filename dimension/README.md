# User instructions

These files are for scaling non-dimensional variables of BLE into dimensional variables.


FILES

The code consists of five files:

• BL_dimension.py is the main file that scales non-dimensional variables into dimensional variables;

• initial_data.py defines the initial conditions eta_0(x,y), and Phi_0(x,y);

• boundary_point.py computes points to design a computational domain;

The way to scale is followed by scaling.pdf.

After the computation is finished, three sets of data are produced:

• solutions eta and Phi in "data/output.pvd";

• evolution of maximum of eta against time in "data/max.csv".

The user can see BLE solution with  dimensional variables with Paraview.