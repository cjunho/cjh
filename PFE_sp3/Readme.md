# User instructions

Program that solves the Potential Flow Equations (PFE) on a 3D periodic domain using the modified mid-point scheme.

The simulation tracks the evolution of three line solitons as they interact, 
to produce an eight times higher splash than the initial height of each soliton.

## FILES

The code consists of five files:
- `pot_sp3.py` is the main file that solves PFE;
- `PFE_sp3energy.py`, `PFE_sp3_max.py`, `PFE_sp3_A.py`, and `PFE_sp3_maxA.py` are used for plotting the evolution of energy and maximum, amplitude of solitons in a far field (denoted by A), and maximum divided by A, respectively.

After the computation of `pot_sp3.py` is finished, four sets of data are produced:
- solutions eta and Phi in "*data/height.pvd*", "*data/psi.pvd*", "*data/varphi.pvd*";
- evolution of energy and maximum against time is saved in "*data/potflow3dperenergy.txt";


The computational data can be visualised as follows:
- Solutions eta and Phi are pvd files that can be read with Paraview.
- Time evolution of energy can be plotted by running file PFE_sp3energy.py`.
- Time evolution of maximum of eta can be plotted by running file `PFE_sp3_max.py`.
- Time evolution of A can be plotted by running file `PFE_sp3_A.py`.
- Time evolution of maximum/A can be plotted by running file `PFE_sp3_maxA.py`.

Running time table when varying spatial resolution $\Delta y(1,1/2,1/4)$ where $\Delta x\approx\Delta y=200$. $T$ is the total simulation time defined as $T=t_{end}-t_{0}$ with $t_0=0{\rm s}$. $\Delta t=0.7139{\rm s}$ or $\Delta t BLE=0.005$.  In order to change order of basis, modify nCG (currently setting nCG=2). For $N_y$, modify multiple (currently setting mutiple=1), for $\Delta t BLE$ modify dtBLE(currently setting dtBLE=1/200). All simuations were run on 40 cores of Leeds' arc4-HPC.
Simulation |$\epsilon$|$\delta$| $L_x$ (m) | $L_y$ (m) |$L_z=H_0$ (m) | $T$ (s) |$\Delta t BLE$ | $N_x$ | $N_y$ |$N_z$| run-time  
:---        | :---      | :---    | :---       | :---       |:---           |:---      | :---           | :---    |:---   |:---| :---
PFE-SP3-001-CG2- $\Delta y$ | $0.01$|$10^{-5}$ | 22624.6|50000 | 20|7149.9|0.005  | 120 | 250 | 4|715mins
PFE-SP3-001-CG2- $\frac{\Delta y}{2}$ | $0.01$|$10^{-5}$ | 22624.6|50000 | 20|2856.4|0.005  | 226 | 500 | 4|848.5mins
PFE-SP3-001-CG2- $\frac{\Delta y}{4}$ | $0.01$|$10^{-5}$ | 22624.6|50000 | 20|2285.2|0.005  | 452 | 1000 | 4|2days
PFE-SP3-001-CG4- $\frac{\Delta y}{2}$ | $0.01$|$10^{-5}$ | 22624.6|50000 | 20|1235.7|0.005  | 226 | 500 | 4|2days
