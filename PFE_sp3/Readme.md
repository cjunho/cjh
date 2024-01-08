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
- Solutions eta and Phi are pvd files that can be read with Paraview, which reproduce Fig.11 in referene [1].
- Time evolution of energy can be plotted by running file `PFE_sp3energy.py`, which reproduces Fig.13(d) in [1].
- Time evolution of maximum of eta can be plotted by running file `PFE_sp3_max.py`, which reproduces Fig.13(a) in [1].
- Time evolution of A can be plotted by running file `PFE_sp3_A.py`, which reproduces Fig.13(b) in [1].
- Time evolution of maximum/A can be plotted by running file `PFE_sp3_maxA.py`, which reproduces Fig.13(c) in [1].

We provide a table in which parameters were uses for setting simulations as follows. We denote spatial resolution by $\Delta x\approx\Delta y=200$. $T$ is the total simulation time defined as $T=t_{end}-t_{0}$ with $t_0=0{\rm s}$. $\Delta t=0.7139{\rm s}$ or $\Delta t BLE=0.005$.  In order to change order of basis, modify nCG (currently setting nCG=2). For $N_y$, modify multiple (currently setting mutiple=3), for $\Delta t BLE$ modify dtBLE(currently setting dtBLE=1/200).  All simuations were run on 40 cores of Leeds' arc4-HPC.
  
Simulation |$\epsilon$|$\delta$| $L_x$ (m) | $L_y$ (m) |$L_z=H_0$ (m) | $T$ (s) |$\Delta t_{BLE}$ | $N_x$ | $N_y$ |$N_z$|DoFs |Run time (min)  
:---        | :---      | :---    | :---       | :---       |:---           |:---      | :---           | :---    |:---   |:---| :---
PFE-SP3-CG2- $\frac{\Delta y}{3}$- $\Delta t$ | $0.01$|$10^{-5}$ | 17725.6|40000 | 20|6855|0.005  | 226 | 600 | 4|5,750,388|2880
PFE-SP3-CG4- $\frac{2\Delta y}{3}$- $\Delta t$ | $0.01$|$10^{-5}$ | 17725.6|40000 | 20|6855|0.005  | 133 | 300 | 4|5,750,388|5588
PFE-SP3-CG2- $\frac{\Delta y}{4}$- $\Delta t$ | $0.01$|$10^{-5}$ | 17725.6|40000 | 20|6855|0.005  | 355 | 800 | 4|10,230,390|5383
PFE-SP3-CG2- $\frac{\Delta y}{3}$- $\frac{\Delta t}{2}$ | $0.01$|$10^{-5}$ | 17725.6|40000 | 20|6855|0.0025  | 226 | 600 | 4|5,750,388|6094.6

## Reference
[1] Choi, J., Kalogirou, A., Kelmanson, M., Lu, Y., & Bokhove, O. (2023). A study of extreme water waves using a hierarchy of models based on potential-flow theory. Eartharxiv.
