# User instructions

Program that solves the Potential Flow Equations (PFE) on a 3D periodic domain using the modified mid-point scheme.

The simulation tracks the evolution of two line solitons as they interact, 
to produce a four times higher splash than the initial height of each soliton.

## FILES

The code consists of five files:
- `pot_sp2.py` is the main file that solves PFE;
- `PFE_sp2energy.py`, `PFE_sp2_max.py`, `PFE_sp2_A.py`, and `PFE_sp2_maxA.py` are used for plotting the evolution of energy and maximum, amplitude of solitons in a far field (denoted by A), and maximum divided by A, respectively.

After the computation of `pot_sp2.py` is finished, four sets of data are produced:
- solutions eta and Phi in "*data/height.pvd*", "*data/psi.pvd*", "*data/varphi.pvd*";
- evolution of energy and maximum against time is saved in "*data/potflow3dperenergy.txt";


The computational data can be visualised as follows:
- Solutions eta and Phi are pvd files that can be read with Paraview.
- Time evolution of energy can be plotted by running file PFE_sp2energy.py`.
- Time evolution of maximum of eta can be plotted by running file `PFE_sp2_max.py`.
- Time evolution of A can be plotted by running file `PFE_sp2_A.py`.
- Time evolution of maximum/A can be plotted by running file `PFE_sp2_maxA.py`.

Simulation | $L_x$ (m) | $L_y$ (m) |$L_z (m)$ | $T$ (s) | $N_x$ | $N_y$ |$N_z$|running time (min)
---        | ---       | ---       | ---      | ---     |---    |---    | --- | ---

