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

  
Running time table when setting several numerical parameters as in the table for each simulation set, where $T$ is the total simulation time defined as $T=t_{end}-t_{0}$ with $t_0=0{\rm s}$. In addition, we set $\Delta t=0.2855{\rm s}$, $\epsilon=0.05$, and $\tan\theta=(\frac{2}{9})^{1/6}\frac{1}{4\sqrt{\epsilon}}$. Degrees of freedom $n_{DOF}$ scale as $n_{CG} N_x(n_{CG} N_y+1)$ with $N_x$ cells and $N_y$ cells in the $x,y$ directions. Hence, when $n_{CG} N_x$ and $n_{CG} N_Y$ are fixed $n_{DOF}$ remains the same. All simuations were run on 40 cores of Leeds' arc4-HPC (Macbook with 12 cores using $\hat{\phi}=1$; $1055{\rm min}$ with $\hat{\phi}(z)$ GLL2
Simulation | $L_x$ (m) | $L_y$ (m) |$L_z (m)$ | $T$ (s) | $N_x$ | $N_y$ |$N_z$|running time (min)
---        | ---       | ---       | ---      | ---     |---    |---    | --- | ---
PFE-SP2-CG2- $\Delta t$| $4110.90$ | 16000|20 | 1713.4  | 124 | 480|4|956 (1044)
PFE-SP2-CG2- $\frac{\Delta t}{2}$ | $4110.90$ | 16000|20 | 1713.4  | 124 | 480|4|1350
PFE-SP2-CG2- $\frac{\Delta t}{4}$ | $4110.90$ | 16000|20 | 1713.4  | 124 | 480|4|2622
PFE-SP2-CG4- $\Delta t$ | $4110.90$ | 16000|20 | 1713.4  | 62 | 240 |2|1423


